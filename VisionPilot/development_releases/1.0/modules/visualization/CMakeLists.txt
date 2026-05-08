cmake_minimum_required(VERSION 3.22.1)

find_package(OpenCV REQUIRED)

add_library(visualization
        src/visualization.cpp
        src/visualization_to_rtsp.cpp
)

target_include_directories(visualization
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(visualization
        PUBLIC
        ${OpenCV_LIBRARIES}
)