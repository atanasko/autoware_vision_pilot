cmake_minimum_required(VERSION 3.22.1)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

pkg_check_modules(WEBCONN REQUIRED IMPORTED_TARGET
        gstreamer-1.0
        gstreamer-app-1.0
        gstreamer-sdp-1.0
        gstreamer-webrtc-1.0
        libsoup-2.4
        json-glib-1.0
)

add_library(visualization
        src/visualization.cpp
        src/visualization_to_webrtc.cpp
)

target_compile_definitions(visualization PRIVATE GST_USE_UNSTABLE_API=1)

target_include_directories(visualization
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(visualization
        PUBLIC
        ${OpenCV_LIBRARIES}
        PkgConfig::WEBCONN
        Threads::Threads
)