#ifndef VISIONPILOT_ROS2_TO_OPENCV_HPP
#define VISIONPILOT_ROS2_TO_OPENCV_HPP

#include <rclcpp/rclcpp.hpp>

namespace camera_subscriber {

    /**
    * @class ROS2ImageSubscriber
    * @brief ROS2 node that subscribes to `sensor_msgs/image` topics and 
    *        converts ROS2 image message to OpenCV image format (cv::Mat).
    * 
    * Features:
    * - Subscribes to ROS2 image topics (from any source - simulator, camera, hardware, etc.)
    * - Converts ROS2 image messages to OpenCV cv::Mat format
    * - Thread-safe conversion and data handling
    * - Supports various image encodings (RGB, BGR, grayscale, etc.)
    */

    

};

#endif //VISIONPILOT_ROS2_TO_OPENCV_HPP