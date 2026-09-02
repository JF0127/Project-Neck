extern "C" {
#include "ethercat.h"
#include "motor_control.h"
#include "transmit.h"
}

#include "queue.h"
#include <sys/time.h>
#include <chrono>
#include <cinttypes>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <mutex>
#include "time.h"
#define EC_TIMEOUTM

spsc_queue<Queue_Msg_ptr, capacity<10>> messages[SLAVE_NUMBER];
std::atomic<bool> running{false};
std::thread runThread;


char IOmap[4096];
OSAL_THREAD_HANDLE checkThread;
int expectedWKC;
boolean needlf;
volatile int wkc;
boolean inOP;
uint8 currentgroup = 0;
uint64_t num;
bool isConfig[SLAVE_NUMBER]{false};

static std::mutex neckFrameMutex;
static EtherCAT_Msg neckFrames[SLAVE_NUMBER]{};
static bool neckFrameActive[SLAVE_NUMBER]{false};

struct MotorFeedbackCache {
    std::atomic<int> motor_id{0};
    std::atomic<double> angle_deg{0.0};
    std::atomic<std::int64_t> received_ns{0};
    std::atomic<std::uint64_t> sequence{0};
};

static MotorFeedbackCache neckFeedback[SLAVE_NUMBER][6];

static bool finiteDouble(double value)
{
    return value == value && value <= DBL_MAX && value >= -DBL_MAX;
}

extern "C" bool NeckFramePublish(int slave_id, const EtherCAT_Msg* frame)
{
    if (frame == nullptr || !running || slave_id < 0 ||
        slave_id >= ec_slavecount || slave_id >= SLAVE_NUMBER)
    {
        return false;
    }
    std::lock_guard<std::mutex> lock(neckFrameMutex);
    neckFrames[slave_id] = *frame;
    neckFrameActive[slave_id] = true;
    return true;
}

extern "C" void NeckFrameStop(int slave_id)
{
    if (slave_id < 0 || slave_id >= SLAVE_NUMBER)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(neckFrameMutex);
    neckFrameActive[slave_id] = false;
}

extern "C" bool NeckFeedbackRead(int slave_id, const int passages[3],
                                  const int motor_ids[3], double angles_deg[3],
                                  std::uint64_t sequences[3])
{
    if (slave_id < 0 || slave_id >= SLAVE_NUMBER || passages == nullptr ||
        motor_ids == nullptr || angles_deg == nullptr || sequences == nullptr)
    {
        return false;
    }

    const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    for (int index = 0; index < 3; ++index)
    {
        const int passage_index = passages[index] - 1;
        if (passage_index < 0 || passage_index >= 6)
        {
            return false;
        }
        const MotorFeedbackCache& feedback =
            neckFeedback[slave_id][passage_index];
        const std::uint64_t sequence_before =
            feedback.sequence.load(std::memory_order_acquire);
        if (sequence_before == 0 || (sequence_before & 1U) != 0)
        {
            return false;
        }
        const int motor_id = feedback.motor_id.load(std::memory_order_relaxed);
        const double angle_deg = feedback.angle_deg.load(std::memory_order_relaxed);
        const std::int64_t received_ns =
            feedback.received_ns.load(std::memory_order_relaxed);
        const std::uint64_t sequence_after =
            feedback.sequence.load(std::memory_order_acquire);
        const double age_sec = static_cast<double>(now_ns - received_ns) / 1e9;
        if (sequence_before != sequence_after || motor_id != motor_ids[index] ||
            !finiteDouble(angle_deg) || age_sec < 0.0 || age_sec > 0.1)
        {
            return false;
        }
        angles_deg[index] = angle_deg;
        sequences[index] = sequence_after / 2;
    }
    return true;
}

#define EC_TIMEOUTMON 500

void EtherCAT_Data_Get();

void EtherCAT_Command_Set();

static void degraded_handler()
{
    printf("[EtherCAT Error] Logging error...\n");
    time_t current_time = time(NULL);
    char* time_str = ctime(&current_time);
    printf("ESTOP. EtherCAT became degraded at %s.\n", time_str);
    printf("[EtherCAT Error] Stopping RT process.\n");
}

static int run_ethercat(const char* ifname)
{
    int i;
    int oloop, iloop, chk;
    needlf = FALSE;
    inOP = FALSE;

    num = 1;

    /* initialise SOEM, bind socket to ifname */
    if (ec_init(ifname))
    {
        printf("[EtherCAT Init] Initialization on device %s succeeded.\n", ifname);
        /* find and auto-config slaves */

        if (ec_config_init(FALSE) > 0)
        {
            printf("[EtherCAT Init] %d slaves found and configured.\n", ec_slavecount);
            if (ec_slavecount < SLAVE_NUMBER)
            {
                printf("[RT EtherCAT] Warning: Expected %d slaves, found %d.\n", SLAVE_NUMBER, ec_slavecount);
            }

            for (int slave_idx = 0; slave_idx < ec_slavecount; slave_idx++)
                ec_slave[slave_idx + 1].CoEdetails &= ~ECT_COEDET_SDOCA;

            ec_config_map(&IOmap);
            ec_configdc();

            printf("[EtherCAT Init] Mapped slaves.\n");
            /* wait for all slaves to reach SAFE_OP state */
            ec_statecheck(0, EC_STATE_SAFE_OP, EC_TIMEOUTSTATE * SLAVE_NUMBER);

            for (int slave_idx = 0; slave_idx < ec_slavecount; slave_idx++)
            {
                printf("[SLAVE %d]\n", slave_idx);
                printf("  IN  %d bytes, %d bits\n", ec_slave[slave_idx].Ibytes, ec_slave[slave_idx].Ibits);
                printf("  OUT %d bytes, %d bits\n", ec_slave[slave_idx].Obytes, ec_slave[slave_idx].Obits);
                printf("\n");
            }

            oloop = ec_slave[0].Obytes;
            if ((oloop == 0) && (ec_slave[0].Obits > 0))
                oloop = 1;
            if (oloop > 8)
                oloop = 8;
            iloop = ec_slave[0].Ibytes;
            if ((iloop == 0) && (ec_slave[0].Ibits > 0))
                iloop = 1;
            if (iloop > 8)
                iloop = 8;

            printf("[EtherCAT Init] segments : %d : %d %d %d %d\n", ec_group[0].nsegments, ec_group[0].IOsegment[0],
                   ec_group[0].IOsegment[1], ec_group[0].IOsegment[2], ec_group[0].IOsegment[3]);

            printf("[EtherCAT Init] Requesting operational state for all slaves...\n");
            expectedWKC = (ec_group[0].outputsWKC * 2) + ec_group[0].inputsWKC;
            printf("[EtherCAT Init] Calculated workcounter %d\n", expectedWKC);
            ec_slave[0].state = EC_STATE_OPERATIONAL;
            /* send one valid process data to make outputs in slaves happy*/
            ec_send_processdata();
            ec_receive_processdata(EC_TIMEOUTRET);
            /* request OP state for all slaves */
            ec_writestate(0);
            chk = 40;
            /* wait for all slaves to reach OP state */
            do
            {
                ec_send_processdata();
                ec_receive_processdata(EC_TIMEOUTRET);
                ec_statecheck(0, EC_STATE_OPERATIONAL, 50000);
            }
            while (chk-- && (ec_slave[0].state != EC_STATE_OPERATIONAL));

            if (ec_slave[0].state == EC_STATE_OPERATIONAL)
            {
                printf("[EtherCAT Init] Operational state reached for all slaves.\n");
                inOP = TRUE;
                return 1;
            }
            else
            {
                printf("[EtherCAT Error] Not all slaves reached operational state.\n");
                ec_readstate();
                for (i = 1; i <= ec_slavecount; i++)
                {
                    if (ec_slave[i].state != EC_STATE_OPERATIONAL)
                    {
                        printf("[EtherCAT Error] Slave %d State=0x%2.2x StatusCode=0x%4.4x : %s\n",
                               i, ec_slave[i].state, ec_slave[i].ALstatuscode,
                               ec_ALstatuscode2string(ec_slave[i].ALstatuscode));
                    }
                }
            }
        }
        else
        {
            printf("[EtherCAT Error] No slaves found!\n");
        }
    }
    else
    {
        printf("[EtherCAT Error] No socket connection on %s - are you running run.sh?\n", ifname);
    }
    return 0;
}

static int err_count = 0;
static int err_iteration_count = 0;
/**@brief EtherCAT errors are measured over this period of loop iterations */
#define K_ETHERCAT_ERR_PERIOD 100

/**@brief Maximum number of etherCAT errors before a fault per period of loop iterations */
#define K_ETHERCAT_ERR_MAX 20

static OSAL_THREAD_FUNC ecatcheck(void* ptr)
{
    (void)ptr;
    int slave = 0;
    while (1)
    {
        // count errors
        if (err_iteration_count > K_ETHERCAT_ERR_PERIOD)
        {
            err_iteration_count = 0;
            err_count = 0;
        }

        if (err_count > K_ETHERCAT_ERR_MAX)
        {
            // possibly shut down
            printf("[EtherCAT Error] EtherCAT connection degraded.\n");
            printf("[Simulink-Linux] Shutting down....\n");
            degraded_handler();
            break;
        }
        err_iteration_count++;

        if (inOP && ((wkc < expectedWKC) || ec_group[currentgroup].docheckstate))
        {
            if (needlf)
            {
                needlf = FALSE;
                printf("\n");
            }
            /* one ore more slaves are not responding */
            ec_group[currentgroup].docheckstate = FALSE;
            ec_readstate();
            for (slave = 1; slave <= ec_slavecount; slave++)
            {
                if ((ec_slave[slave].group == currentgroup) && (ec_slave[slave].state != EC_STATE_OPERATIONAL))
                {
                    ec_group[currentgroup].docheckstate = TRUE;
                    if (ec_slave[slave].state == (EC_STATE_SAFE_OP + EC_STATE_ERROR))
                    {
                        printf("[EtherCAT Error] Slave %d is in SAFE_OP + ERROR, attempting ack.\n", slave);
                        ec_slave[slave].state = (EC_STATE_SAFE_OP + EC_STATE_ACK);
                        ec_writestate(slave);
                        err_count++;
                    }
                    else if (ec_slave[slave].state == EC_STATE_SAFE_OP)
                    {
                        printf("[EtherCAT Error] Slave %d is in SAFE_OP, change to OPERATIONAL.\n", slave);
                        ec_slave[slave].state = EC_STATE_OPERATIONAL;
                        ec_writestate(slave);
                        err_count++;
                    }
                    else if (ec_slave[slave].state > 0)
                    {
                        if (ec_reconfig_slave(slave, EC_TIMEOUTMON))
                        {
                            ec_slave[slave].islost = FALSE;
                            printf("[EtherCAT Status] Slave %d reconfigured\n", slave);
                        }
                    }
                    else if (!ec_slave[slave].islost)
                    {
                        /* re-check state */
                        ec_statecheck(slave, EC_STATE_OPERATIONAL, EC_TIMEOUTRET);
                        if (!ec_slave[slave].state)
                        {
                            ec_slave[slave].islost = TRUE;
                            printf("[EtherCAT Error] Slave %d lost\n", slave);
                            err_count++;
                        }
                    }
                }
                if (ec_slave[slave].islost)
                {
                    if (!ec_slave[slave].state)
                    {
                        if (ec_recover_slave(slave, EC_TIMEOUTMON))
                        {
                            ec_slave[slave].islost = FALSE;
                            printf("[EtherCAT Status] Slave %d recovered\n", slave);
                        }
                    }
                    else
                    {
                        ec_slave[slave].islost = FALSE;
                        printf("[EtherCAT Status] Slave %d found\n", slave);
                    }
                }
            }
            if (!ec_group[currentgroup].docheckstate)
                printf("[EtherCAT Status] All slaves resumed OPERATIONAL.\n");
        }
        osal_usleep(50000);
    }
}

void EtherCAT_Init(char* ifname)
{
    int i;
    int rc;
    printf("[EtherCAT] Initializing EtherCAT\n");
    osal_thread_create((void*)&checkThread, 128000, (void*)&ecatcheck, (void*)&ctime);
    for (i = 1; i < 10; i++)
    {
        printf("[EtherCAT] Attempting to start EtherCAT, try %d of 10.\n", i);
        rc = run_ethercat(ifname);
        if (rc)
            break;
        osal_usleep(1000000);
    }
    if (rc)
        printf("[EtherCAT] EtherCAT successfully initialized on attempt %d \n", i);
    else
    {
        printf("[EtherCAT Error] Failed to initialize EtherCAT after 100 tries. \n");
    }
}

void EtherCAT_Transmit(EtherCAT_Msg* MasterCommand)
{
    for (int i = 0; i < ec_slavecount; i++)
    {
        memcpy((void*)(ec_slave[0].outputs + i * sizeof(EtherCAT_Msg)), (void*)&(MasterCommand[i]),
               sizeof(EtherCAT_Msg));
    }
    ec_send_processdata();
}

static int wkc_err_count = 0;
static int wkc_err_iteration_count = 0;

//数组大小根据从站数量确定
EtherCAT_Msg Rx_Message[SLAVE_NUMBER];
EtherCAT_Msg Tx_Message[SLAVE_NUMBER];

OD_Motor_Msg Rx_Motor_Msg[SLAVE_NUMBER][6];
/**
 * @description:
 * @return {*}
 */
void EtherCAT_Run()
{
    if (wkc_err_iteration_count > K_ETHERCAT_ERR_PERIOD)
    {
        wkc_err_count = 0;
        wkc_err_iteration_count = 0;
    }
    if (wkc_err_count > K_ETHERCAT_ERR_MAX)
    {
        printf("[EtherCAT Error] Error count too high!\n");
        degraded_handler();
    }
    // send
    EtherCAT_Command_Set();
    ec_send_processdata();
    // receive
    wkc = ec_receive_processdata(EC_TIMEOUTRET);
    EtherCAT_Data_Get();
    //  check for dropped packet
    if (wkc < expectedWKC)
    {
        printf("\x1b[31m[EtherCAT Error] Dropped packet (Bad WKC!)\x1b[0m\n");
        wkc_err_count++;
    }
    else
    {
        needlf = TRUE;
    }
    wkc_err_iteration_count++;
}

/**
 * @description: slave data get
 * @return {*}
 * @author: Kx Zhang
 */
void EtherCAT_Data_Get()
{
    for (int slave = 0; slave < ec_slavecount; ++slave)
    {
        EtherCAT_Msg* slave_src = (EtherCAT_Msg*)(ec_slave[slave + 1].inputs);
        if (slave_src)
            Rx_Message[slave] = *(EtherCAT_Msg*)(ec_slave[slave + 1].inputs);

        RV_can_data_repack(&Rx_Message[slave], comm_ack, Rx_Motor_Msg[slave], slave, isConfig[slave]);

        const auto feedback_time_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        for (int passage = 0; passage < 6; ++passage)
        {
            const Motor_Msg& raw = Rx_Message[slave].motor[passage];
            if (raw.dlc == 0 || raw.id == 0x7FF)
            {
                continue;
            }
            const int ack_status = raw.data[0] >> 5;
            double angle_deg = 0.0;
            if (ack_status == 1)
            {
                angle_deg = Rx_Motor_Msg[slave][passage].angle_actual_rad *
                            180.0 / 3.14159265358979323846;
            }
            else if (ack_status == 2)
            {
                angle_deg = Rx_Motor_Msg[slave][passage].angle_actual_float;
            }
            else
            {
                continue;
            }
            if (!finiteDouble(angle_deg))
            {
                continue;
            }
            MotorFeedbackCache& feedback = neckFeedback[slave][passage];
            feedback.sequence.fetch_add(1, std::memory_order_acq_rel);
            feedback.motor_id.store(
                Rx_Motor_Msg[slave][passage].motor_id,
                std::memory_order_relaxed);
            feedback.angle_deg.store(angle_deg, std::memory_order_relaxed);
            feedback.received_ns.store(feedback_time_ns,
                                       std::memory_order_relaxed);
            feedback.sequence.fetch_add(1, std::memory_order_release);
        }

        if (isConfig[slave])
        {
            isConfig[slave] = false;
        }
    }
}

/**
 * @description: slave command set
 * @return {*}
 * @author: Kx Zhang
 */
#define frequency 1000
#define POS_SPD (3.14/frequency)
float pos_set = 0, delta_pos = POS_SPD;
static int i = 0;


// 使用命令行控制电机的例程
void EtherCAT_Command_Set()
{
    static int state[SLAVE_NUMBER];
    for (int slave = 0; slave < ec_slavecount; ++slave)
    {
        bool frame_active = false;
        {
            std::lock_guard<std::mutex> lock(neckFrameMutex);
            if (neckFrameActive[slave])
            {
                Tx_Message[slave] = neckFrames[slave];
                frame_active = true;
            }
        }

        Queue_Msg_ptr msg;
        if (!frame_active && state[slave] == 0)
        {
            if (messages[slave].pop(msg))
            {
                Tx_Message[slave].motor[msg->passage - 1] = msg->motor;
                state[slave] = 1;
            }
        }
        else if (!frame_active && state[slave]++ == 10)
        {
            isConfig[slave] = true;
            state[slave] = 0;
        }

        EtherCAT_Msg* slave_dest = (EtherCAT_Msg*)(ec_slave[slave + 1].outputs);
        if (slave_dest)
            *(EtherCAT_Msg*)(ec_slave[slave + 1].outputs) = Tx_Message[slave];
    }
}

// 一个从站控制一个电机
// void EtherCAT_Command_Set() {

//     static int state;

//     // int slave = 0;

//     set_motor_speed(&Tx_Message[0], 2, 1, 50, 50, 2);

//     // if(state == 0) {
//     //     // 设置电机零点
//     //     MotorSetting(&Tx_Message[0], 1, 0x03);
//     //     state = 1;
//     // } else if (state < 2000) {
//     //     // 小于2000也就是2000ms内转到90度的位置
//     //     state++;
//     //     set_motor_position(&Tx_Message[0], 1, 1, 90, 100, 50, 2);
//     // } else {
//     //     // 2s后转到180度的位置
//     //     set_motor_position(&Tx_Message[0], 1, 1, 180, 100, 50, 2);
//     // }

//     isConfig[0] = true;

//     EtherCAT_Msg *slave_dest = (EtherCAT_Msg *) (ec_slave[1].outputs);
//     if (slave_dest)
//         *(EtherCAT_Msg *) (ec_slave[1].outputs) = Tx_Message[0];
// }


// 一个从站控制多个电机
// void EtherCAT_Command_Set() {

//     // 最多可以控制6个电机（1，2，3在CAN1上面，4，5，6在CAN2上面）
//     set_motor_speed(&Tx_Message[0], 1, 1, 10, 50, 2);
//     set_motor_speed(&Tx_Message[0], 2, 3, 20, 50, 2);
//     set_motor_speed(&Tx_Message[0], 3, 5, 30, 50, 2);
//     set_motor_speed(&Tx_Message[0], 4, 7, 40, 50, 2);
//     set_motor_speed(&Tx_Message[0], 5, 9, 50, 50, 2);
//     set_motor_speed(&Tx_Message[0], 6, 11, 60, 50, 2);

//     EtherCAT_Msg *slave_dest = (EtherCAT_Msg *) (ec_slave[1].outputs);
//     if (slave_dest)
//         *(EtherCAT_Msg *) (ec_slave[1].outputs) = Tx_Message[0];
// }

// 多个从站控制多个电机
// void EtherCAT_Command_Set() {
//     static int state[SLAVE_NUMBER];
//     // ec_slavecount是主站识别到从站的数量
//     for(int slave = 0; slave < ec_slavecount; ++slave) {

//         if(slave == 0) { // 第一个从站
//             set_motor_speed(&Tx_Message[slave], 1, 1, 10, 50, 2);
//         } else if(slave == 1) { // 第二个从站
//             if(state[slave] == 0) {
//                 // 设置电机零点
//                 MotorSetting(&Tx_Message[slave], 1, 0x03);
//                 state[slave] = 1;
//             } else {
//                 set_motor_position(&Tx_Message[slave], 1, 1, 90, 100, 50, 2);
//             }
//         }// 如果还有其他从站就继续else if，使用switch也可以

//         EtherCAT_Msg *slave_dest = (EtherCAT_Msg *) (ec_slave[1 + slave].outputs);
//         if (slave_dest)
//             *(EtherCAT_Msg *) (ec_slave[1 + slave].outputs) = Tx_Message[slave];
//     }
// }


void runImpl()
{
    while (running)
    {
        EtherCAT_Run();
        usleep(1000);
    }
}


void startRun()
{
    running = true;
    runThread = std::thread(runImpl);
}
