#ifndef VISIONPILOT_ROS2_TO_OPENCV_HPP
#define VISIONPILOT_ROS2_TO_OPENCV_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <queue>
#include <tuple>

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
            * @param queue_size Max number of frames to buffer for incoming images (default: 1)
            * @param node_name The name of the ROS2 node (default: "ros2_image_subscriber")
            *
            * The node automatically inits the ROS2 subscription and begins listening for incoming messages.
            */
            explicit ROS2ImageSubscriber(
                const std::string& topic_name,
                size_t queue_size = 1,
                const std::string& node_name = "ros2_image_subscriber"
            );


            /**
            * @brief Destructor for ROS2ImageSubscriber
            * 
            * Cleans up ROS2 subscriptions and resources when the node is destroyed.
            */
            ~ROS2ImageSubscriber() override = default;


            // FRAME HANDLINGS
      
            
            /**
            * @brief Get latest frame with corresponding frame metadata
            *
            * @return cv::Mat containing the latest image frame received from the ROS2 topic.
            *
            * This method is thread-safe and can be called from multiple threads without causing data corruption.
            * Returns empty cv::Mat if no frames have been received yet.
            */
            std::tuple<bool, cv::Mat> get_latest_frame();


            // /**
            // * @brief Get latest frame with frame metadata, via timestamp and frame index
            // * 
            // * @param frame_index Output parameter: frame sequence number from ROS2 message
            // * @param timestamp_sec Output parameter: ROS2 timestamp in seconds
            // * @return cv::Mat The image frame, or empty if none available
            // * 
            // * Provides additional timing information along with the frame for synchronization purposes.
            // */
            // cv::Mat get_latest_frame_with_timestamp(
            //     uint32_t &frame_index,
            //     double &timestamp_sec
            // );


            /**
            * @brief Check if any frames are currently in the queue
            * 
            * @return true if frames are available, false otherwise
            */
            bool has_frames() const;


            /**
            * @brief Get current queue size (number of buffered frames)
            * 
            * @return Current number of frames in the internal queue
            */
            size_t get_queue_size() const;


            /**
            * @brief Get the maximum queue size
            * 
            * @return Maximum allowed frames in buffer
            */
            size_t get_max_queue_size() const;


            /**
            * @brief Reset the frame buffer (clear all queued frames)
            * 
            * Useful for resetting state or handling error conditions
            */
            void clear_frame_buffer();


            /**
            * @brief Check if ROS2 stream is active
            * 
            * @return true if stream has started receiving frames, false otherwise
            */
            bool is_stream_active() const;


            // STATISTICS


            /**
            * @brief Get statistics about subscription
            * 
            * @return A struct containing:
            *         - frames_received: total frames received from ROS2
            *         - frames_dropped: frames dropped when queue was full
            *         - conversion_errors: failed ROS2 => OpenCV conversions
            */
            struct SubscriptionStats {
                uint64_t frames_received = 0;
                uint64_t frames_dropped = 0;
                uint64_t conversion_errors = 0;
                std::string last_encoding;
                std::string node_name;
            };


            SubscriptionStats get_stats() const;


            /**
            * @brief Reset statistics counters
            */
            void reset_stats();


        private:


            /**
            * @struct FrameMetadata
            * @brief Metadata associated with each frame
            */
            struct FrameMetadata {
                uint32_t sequence = 0;
                double timestamp = 0.0;
            };


            /**
            * @brief Internal callback function invoked when a new ROS2 image message arrives
            * 
            * Handles thread-safe conversion from sensor_msgs::msg::Image to cv::Mat
            * and queuing of frames for retrieval by the application.
            */
            void image_callback(
                const sensor_msgs::msg::Image::SharedPtr msg
            );


            /**
            * @brief Thread-safe conversion from ROS2 Image message to cv::Mat
            * 
            * @param msg The ROS2 image message
            *
            * @return cv::Mat : The converted image, or empty Mat if conversion failed
            * 
            * Uses cv_bridge library to handle various ROS2 image encodings.
            */
            cv::Mat convert_ros2_image_to_opencv(
                const sensor_msgs::msg::Image::SharedPtr &msg
            );


            // ROS2 subscription
            rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription;


            // Frame buffer with thread safety
            mutable std::mutex frame_mutex;
            std::queue<cv::Mat> frame_queue;
            std::queue<FrameMetadata> metadata_queue;
            size_t max_queue_size;


            // Flag for started stream
            bool is_stream_started = false;


            // Frame timestamp for synchronization
            std::queue<uint32_t> timestamp_queue;


            // Statistics tracking
            mutable std::mutex stats_mutex;
            SubscriptionStats stats;

    };

};  // namespace camera_subscriber

#endif //VISIONPILOT_ROS2_TO_OPENCV_HPP