add_executable(VisionPilot vision_pilot.cpp)

add_dependencies(VisionPilot generate_config)

if (ENABLE_ROS2_INTERFACE)
    target_link_libraries(VisionPilot
            PRIVATE
            camera_subscriber
    )

    target_compile_definitions(VisionPilot PRIVATE ENABLE_ROS2_INTERFACE)
endif ()

target_link_libraries(VisionPilot
        PRIVATE
        can_interface
        camera_interface
        image_preprocessing
        models
        visualization
)

