cmake_minimum_required(VERSION 3.10)
project(MasterStack)

add_subdirectory(thirdparty)

find_package(Boost REQUIRED)
find_package(Threads REQUIRED)

include_directories(${Boost_INCLUDE_DIR})
include_directories(
        thirdparty/soem/soem
        thirdparty/soem/osal
        thirdparty/soem/osal/linux
        thirdparty/soem/oshw/linux
        thirdparty/cpp-readline/src
)
include_directories(${CMAKE_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES motor_control.c
    math_ops.c
    transmit.cpp
    command.cpp
    neck_control/hardware_adapter.cpp
)
LINK_LIBRARIES(m)
add_library(motor ${SOURCES})
target_link_libraries(motor soem)

# ---- Neck Trajectory Executor 核心库（不依赖 EtherCAT，供测试二进制使用）----
set(NECK_CORE_SOURCES
    neck_config.cpp
    neck_control/json_parser.cpp
    neck_control/trajectory_io.cpp
    neck_control/config.cpp
    neck_control/coordinate_calibration.cpp
    neck_control/inverse_kinematics.cpp
    neck_control/trajectory_retimer.cpp
    neck_control/trajectory_validation.cpp
    neck_control/state_machine.cpp
    neck_control/telemetry.cpp
    neck_control/mock_adapter.cpp
    neck_control/executor.cpp
)
add_library(neckcore STATIC ${NECK_CORE_SOURCES})
target_link_libraries(neckcore Threads::Threads)

add_executable(master_stack_test main.cpp)
target_link_libraries(master_stack_test motor neckcore soem cpp-readline)

# 实测电机角 → 模型坐标换算工具（离线，纯计算）
add_executable(neck_meas2model tools/neck_meas2model.cpp)
target_link_libraries(neck_meas2model neckcore Threads::Threads)

# ---- 自动测试（dry-run / mock 级别；无需网卡与 root）----
enable_testing()
add_executable(neck_control_tests test/neck_control_tests.cpp)
target_link_libraries(neck_control_tests neckcore Threads::Threads)
add_test(NAME neck_control_tests COMMAND neck_control_tests)
