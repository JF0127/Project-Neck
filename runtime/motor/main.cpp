/*
 * @Description:
 * @Author: kx zhang
 * @Date: 2022-09-13 19:00:55
 * @LastEditTime: 2022-11-13 17:09:03
 */

#include <cstdio>
#include <readline/readline.h>
#include <map>

#include "Console.hpp"
#include "command.h"
#include "neck/measured_rpy.h"
#include "neck/model_socket.h"
#include "neck/neck_motion.h"
#include "time.h"

extern "C"
{
#include "ethercat.h"
}

namespace cr = CppReadline;
using ret = cr::Console::ReturnCode;
std::thread comThread;

void comImpl()
{
    cr::Console c(">");

    std::map<std::string, cr::Console::CommandFunction> commands = {
        {"help", help},
        {"MotorIdGet", motorIdGet},
        {"MotorIdSet", motorIdSet},
        {"MotorIdReset", motorIdReset},
        {"MotorZeroSet", motorZeroSet},
        {"MotorStop", motorStop},
        {"MotorSpeedSet", motorSpeedSet},
        {"MotorPositionSet", motorPositionSet},
        {"NeckPoseSet", neckPoseSet},
        {"NeckSequence", neckSequence},
        {"NeckSequenceStop", neckSequenceStop}};

    for (const auto &cmd : commands)
    {
        c.registerCommand(cmd.first, cmd.second);
    }

    c.executeCommand("help");
    int retCode;
    do
    {
        retCode = c.readLine();
        // We can also change the prompt based on last return value:
        if (retCode == ret::Ok)
            c.setGreeting(">");
        else
            c.setGreeting("!>");

        if (retCode == 1)
        {
            std::cout << "Received error code 1\n";
        }
        else if (retCode == 2)
        {
            std::cout << "Received error code 2\n";
        }

        usleep(100000);
        std::cout << std::endl;
    } while (retCode != ret::Quit);
}

int main()
{
    printf("SOEM 主站测试\n");

    MeasurementSocketServer measurement_socket;
    std::string socket_error;
    if (!measurement_socket.start(socket_error))
    {
        std::cerr << "测量 Socket 启动失败: " << socket_error << "\n";
        return 1;
    }

    ModelSocketServer model_socket;
    if (!model_socket.start(socket_error))
    {
        std::cerr << "模型 Socket 启动失败: " << socket_error << "\n";
        return 1;
    }

    // 这里填自己电脑上的网卡
    EtherCAT_Init((char *)"enp4s0"); // enp4s0:

    if (ec_slavecount <= 0)
    {
        printf("未找到从站, 程序退出！\n");
        return 1;
    }
    else
        printf("从站数量： %d\n", ec_slavecount);

    startRun();

    comThread = std::thread(comImpl);
    comThread.join();

    model_socket.stop();
    measurement_socket.stop();
    if (isTrajectoryExecuting())
    {
        std::string stop_error;
        if (!stopTrajectory(stop_error))
            std::cerr << "颈部轨迹停止失败: " << stop_error << "\n";
    }
    running = false;
    runThread.join();
    return 0;
}
