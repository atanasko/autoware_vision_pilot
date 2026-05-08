#include <camera_subscriber/ros2_to_opencv.hpp>
#include <rclcpp/rclcpp.hpp>

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <visualization/visualization.hpp>

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    std::cout << "Hello and welcome to  VisionPilot!\n";

    std::string topic = "/camera/image";
    if (argc > 1) {
        topic = argv[1];
    }

    auto camera_node = std::make_shared<camera_subscriber::ROS2ImageSubscriber>(
        topic,
        "vision_pilot_camera_subscriber"
    );

    RCLCPP_INFO(camera_node->get_logger(), "VisionPilot camera loop started");
    RCLCPP_INFO(camera_node->get_logger(), "  topic: %s", topic.c_str());

    rclcpp::Rate loop_rate(30);

    while (rclcpp::ok()) {
        rclcpp::spin_some(camera_node);

        auto [has_frame, frame] = camera_node->get_latest_frame();
        if (has_frame && !frame.empty()) {
            auto stats = camera_node->get_stats();
            std::vector<std::string> overlay_strs = {
                "topic: " + topic,
                "frames received: " + std::to_string(stats.frames_received),
                "frames dropped: " + std::to_string(stats.frames_dropped),
                "conversion errors: " + std::to_string(stats.conversion_errors)
            };

            visualization::render_frame(
                frame, 
                "VisionPilot", 
                overlay_strs
            );
        }

        loop_rate.sleep();
    }

    visualization::close_windows();
    rclcpp::shutdown();

    return 0;
}
