/*
 * @Description:
 * @Author: kx zhang
 * @Date: 2022-09-13 19:00:55
 * @LastEditTime: 2022-11-13 17:09:03
 *
 * 服务模式(--server): 监听 Unix socket /tmp/neck_ctl.sock, 行协议逐条执行
 * CLI 命令, 响应以 "###END###" 结尾。供 L1 一体化主控(l1_demo.py)驱动。
 * 安全语义与交互 CLI 完全一致(三把锁/急停/watchdog 不变)。
 */

#include <cstdio>
#include <readline/readline.h>
#include <map>

#include "Console.hpp"
#include "command.h"
#include "time.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>

extern "C"
{
#include "ethercat.h"
}

namespace cr = CppReadline;
using ret = cr::Console::ReturnCode;
std::thread comThread;

// --------------------------------------------------------------------------- //
// 命令表(交互 CLI 与服务模式共用)
// --------------------------------------------------------------------------- //
static std::map<std::string, cr::Console::CommandFunction> buildCommands()
{
    std::map<std::string, cr::Console::CommandFunction> commands = {
        {"help", help},
        {"MotorIdGet", motorIdGet},
        {"MotorIdSet", motorIdSet},
        {"MotorIdReset", motorIdReset},
        {"MotorAngleGet", motorAngleGet},
        {"MotorZeroSet", motorZeroSet},
        {"MotorStop", motorStop},
        {"MotorSpeedSet", motorSpeedSet},
        {"MotorPositionSet", motorPositionSet},
        {"NeckPoseSet", neckPoseSet},
        {"NeckSquence", neckSquence},
        {"NeckNpy", neckNpy},
        {"NeckTrajDryRun", neckTrajDryRun},
        {"NeckTrajMock", neckTrajMock},
        {"NeckTrajRun", neckTrajRun},
        {"NeckTrajStop", neckTrajStop},
        {"NeckTrajEStop", neckTrajEStop},
        {"NeckTrajAck", neckTrajAck},
        {"NeckTrajStatus", neckTrajStatus},
        {"NeckStaticCheck", neckStaticCheck},
        {"NeckCalibMove", neckCalibMove},
        {"NeckCalibPose", neckCalibPose},
        {"NeckDisable", neckDisable}};
    return commands;
}

// --------------------------------------------------------------------------- //
// 交互 CLI
// --------------------------------------------------------------------------- //
void comImpl()
{
    cr::Console c(">");

    auto commands = buildCommands();
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

// --------------------------------------------------------------------------- //
// 服务模式: Unix socket 行协议
// 每行一条命令(与 CLI 相同格式); 执行后 stdout 捕获, 响应以 "###END###" 结尾。
// --------------------------------------------------------------------------- //
static const char *SERVER_SOCK_PATH = "/tmp/neck_ctl.sock";
static const char *RESP_END = "###END###\n";

// 执行一条命令并捕获其 stdout 输出(串行调用; 命令内部不并发写 stdout)。
static std::string runCommandCapture(const std::string &line,
                                     std::map<std::string, cr::Console::CommandFunction> &commands)
{
    std::vector<std::string> args;
    {
        std::istringstream iss(line);
        std::string tok;
        while (iss >> tok) args.push_back(tok);
    }
    std::string out;
    if (args.empty()) return out;

    auto it = commands.find(args[0]);
    if (it == commands.end())
    {
        out = "[server] 未知命令: " + args[0] + "\n";
        return out;
    }

    // 捕获 stdout: dup 重定向到管道
    fflush(stdout);
    int saved = dup(STDOUT_FILENO);
    if (saved < 0) return "[server] dup 失败\n";
    int fds[2];
    if (pipe(fds) != 0)
    {
        close(saved);
        return "[server] pipe 失败\n";
    }
    dup2(fds[1], STDOUT_FILENO);
    close(fds[1]);

    try
    {
        it->second(args);       // 执行命令(输出进入管道)
    }
    catch (...)
    {
        out = "[server] 命令执行异常\n";
    }
    fflush(stdout);
    dup2(saved, STDOUT_FILENO);
    close(saved);

    char buf[4096];
    ssize_t n;
    while ((n = read(fds[0], buf, sizeof(buf))) > 0)
        out.append(buf, (size_t)n);
    close(fds[0]);
    return out;
}

static std::mutex gCmdMutex;   // 命令串行执行（stdout 捕获互斥；阻塞命令期间其他连接排队）

static void handleConnection(int cfd,
                             std::map<std::string, cr::Console::CommandFunction> &commands)
{
    std::string line;
    char ch;
    while (true)
    {
        ssize_t n = read(cfd, &ch, 1);
        if (n <= 0) break;                     // 连接关闭
        if (ch == '\n')
        {
            std::string resp;
            {
                std::lock_guard<std::mutex> lock(gCmdMutex);
                resp = runCommandCapture(line, commands);
            }
            std::string full = resp + RESP_END;
            size_t off = 0;
            while (off < full.size())
            {
                ssize_t w = write(cfd, full.data() + off, full.size() - off);
                if (w <= 0) break;
                off += (size_t)w;
            }
            line.clear();
        }
        else
        {
            line.push_back(ch);
        }
    }
    close(cfd);
    printf("[server] 客户端断开\n");
    fflush(stdout);
}

static void serverLoop()
{
    auto commands = buildCommands();

    unlink(SERVER_SOCK_PATH);
    int sfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sfd < 0)
    {
        perror("socket");
        return;
    }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SERVER_SOCK_PATH, sizeof(addr.sun_path) - 1);
    if (bind(sfd, (struct sockaddr *)&addr, sizeof(addr)) != 0)
    {
        perror("bind");
        printf("[server] 绑定失败: 残留 socket 文件无法删除(可能属主是其他用户)。\n"
               "       请手动清理: sudo rm -f %s\n", SERVER_SOCK_PATH);
        close(sfd);
        return;
    }
    chmod(SERVER_SOCK_PATH, 0666);   // 非 root 客户端可连接
    if (listen(sfd, 8) != 0)
    {
        perror("listen");
        close(sfd);
        return;
    }
    printf("[server] 监听 %s (Ctrl+C 退出)\n", SERVER_SOCK_PATH);
    fflush(stdout);

    while (true)
    {
        int cfd = accept(sfd, nullptr, nullptr);
        if (cfd < 0)
        {
            if (errno == EINTR) continue;
            perror("accept");
            break;
        }
        printf("[server] 客户端连接\n");
        fflush(stdout);
        std::thread(handleConnection, cfd, std::ref(commands)).detach();
    }
    close(sfd);
    unlink(SERVER_SOCK_PATH);
}

int main(int argc, char **argv)
{
    bool serverMode = false, noEthercat = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--server") == 0) serverMode = true;
        else if (strcmp(argv[i], "--no-ethercat") == 0) noEthercat = true;
    }

    printf("SOEM 主站测试%s\n", serverMode ? " [服务模式]" : "");

    if (!noEthercat) {
        // 这里填自己电脑上的网卡
        EtherCAT_Init((char *)"enp4s0"); // ens33:

        if (ec_slavecount <= 0)
        {
            printf("未找到从站！\n");
            printf("警告: 电机命令不可用；NeckTrajDryRun / NeckTrajMock 仍可运行（不连接硬件）。\n");
        }
        else
        {
            printf("从站数量： %d\n", ec_slavecount);
            startRun();
        }
    }
    else
    {
        printf("跳过 EtherCAT 初始化(--no-ethercat)；仅 dry-run/mock 可用。\n");
    }

    if (serverMode)
    {
        serverLoop();
    }
    else
    {
        comThread = std::thread(comImpl);
        comThread.join();
    }

    if (runThread.joinable())
        runThread.join();
    return 0;
}
