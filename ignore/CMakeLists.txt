cmake_minimum_required(VERSION 3.1)

project(
  ZeroMQTest VERSION 1.0
  DESCRIPTION "ZeroMQ linking test"
  LANGUAGES CXX
)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules/")

## load in pkg-config support
# find_package(PkgConfig)
# pkg_check_modules(PC_ZeroMQ QUIET zmq)
#find_package(cppzmq)

## use the hint from above to find where 'zmq.hpp' is located
#set(PC_ZeroMQ_INCLUDE_DIRS /usr/local/Cellar/zeromq/4.3.4/include)
#message(${PC_ZeroMQ_INCLUDE_DIRS})
#find_path(ZeroMQ_INCLUDE_DIR NAMES zmq.hpp PATHS ${PC_ZeroMQ_INCLUDE_DIRS})
# set(ZeroMQ_INCLUDE_DIR /usr/local/Cellar/zeromq/4.3.4/include)

## use the hint from above to find the location of libzmq
# find_library(ZeroMQ_LIBRARY NAMES zmq PATHS ${PC_ZeroMQ_LIBRARY_DIRS})

find_package(CPPZMQ REQUIRED)

add_executable(zmq_test zeromq.cpp)

# message("ZMQ Include:${ZeroMQ_INCLUDE_DIR}")
# message("ZMQ Libs:${ZeroMQ_LIBRARY}")

# target_include_directories(zmq_test PUBLIC ${ZeroMQ_INCLUDE_DIR})
# target_link_libraries(zmq_test PUBLIC ${ZeroMQ_LIBRARY})

target_include_directories(zmq_test PUBLIC ${CPPZMQ_INCLUDE_DIRS})
target_link_libraries(zmq_test PUBLIC ${CPPZMQ_LIBRARIES})

# find_package(ZeroMQ REQUIRED)