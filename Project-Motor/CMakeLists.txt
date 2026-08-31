cmake_minimum_required(VERSION 3.10)
project(MasterStack)

add_subdirectory(thirdparty)

find_package(Boost REQUIRED)

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

set(SOURCES motor_control.c
    math_ops.c
    transmit.cpp
    command.cpp
    neck/model_socket.cpp
    neck/neck_config.cpp
    neck/neck_motion.cpp
    neck/neck_kinematics.cpp
    neck/neck_sequence.cpp
    neck/trajectory.cpp
    neck/trajectory_io.cpp
)
LINK_LIBRARIES(m)
add_library(motor ${SOURCES})
target_link_libraries(motor soem)

add_executable(master_stack_test main.cpp)
target_link_libraries(master_stack_test motor soem cpp-readline)