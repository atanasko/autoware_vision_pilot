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

    class ROS2ImageSubscriber : public rclcpp::Node {

        public:

            /**
            * @brief Constructor for ROS2ImageSubscriber
            *
            * @param topic_name The name of the ROS2 topic to subscribe to ("/camera/image_raw", etc.)
            * @param queue_size Max number of frames to buffer for incoming images (default: 10)
            * @param node_name The name of the ROS2 node (default: "ros2_image_subscriber")
            *
            * The node automatically inits the ROS2 subscription and begins listening for incoming messages.
            */
            explicit ROS2ImageSubscriber(
                const std::string& topic_name,
                size_t queue_size = 10,
                const std::string& node_name = "ros2_image_subscriber"
            );

        private:

    };

};

#endif //VISIONPILOT_ROS2_TO_OPENCV_HPP