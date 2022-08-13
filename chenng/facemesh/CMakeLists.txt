cmake_minimum_required(VERSION 3.0.0)
project(main VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(ncnn_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../build/install/lib/cmake/ncnn)

find_package(ncnn REQUIRED)
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library: ${OpenCV_INSTALL_PATH}")
message(STATUS "OpenCV include_dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")


include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main main.cpp)

target_link_libraries(main ncnn ${OpenCV_LIBS})
