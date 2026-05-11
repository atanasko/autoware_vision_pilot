cmake_minimum_required(VERSION 3.22.1)

project(v4l2_interface LANGUAGES CXX)

# Find required packages
find_package(OpenCV REQUIRED COMPONENTS core videoio)

# Create the v4l2_interface library
add_library(v4l2_interface
        src/v4l2_reader.cpp
)

# Set C++ standard
set_target_properties(v4l2_interface PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(v4l2_interface
        PUBLIC
        include
)

# Link OpenCV libraries
target_link_libraries(v4l2_interface
        PUBLIC
        ${OpenCV_LIBS}
)

# Link threading support for std::mutex
find_package(Threads REQUIRED)
target_link_libraries(v4l2_interface
        PUBLIC
        Threads::Threads
)