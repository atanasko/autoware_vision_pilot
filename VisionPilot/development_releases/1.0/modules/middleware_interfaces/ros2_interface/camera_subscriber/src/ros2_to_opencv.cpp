#include <camera_subscriber/ros2_to_opencv.hpp>

namespace camera_subscriber {


    ROS2ImageSubscriber::ROS2ImageSubscriber(
        const std::string &topic_name,
        size_t queue_size,
        const std::string &node_name
    ) : (
        rclcpp::Node(node_name),
        max_queue_size_(queue_size)
    ) {
        
        RCLCPP_INFO(get_logger(), "Initializing ROS2 Image Subscriber");
        RCLCPP_INFO(get_logger(), "  Topic: %s", topic_name.c_str());
        RCLCPP_INFO(get_logger(), "  Queue Size: %zu", queue_size);

        stats_.node_name = node_name;

        // Create subscription to the image topic
        // Using rclcpp::QoS profile for best effort delivery (suitable for camera streams)
        // Reference:
        //      - https://docs.ros.org/en/iron/Concepts/Intermediate/About-Quality-of-Service-Settings.html
        //      - https://docs.ros2.org/foxy/api/rclcpp/classrclcpp_1_1QoS.html

        auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(queue_size))
            .best_effort()
            .durability_volatile();

        image_subscription_ = create_subscription<sensor_msgs::msg::Image>(
            topic_name,
            qos_profile,
            [this](const sensor_msgs::msg::Image::SharedPtr msg) {
                this->image_callback(msg);
            }
        );

        RCLCPP_INFO(get_logger(), "ROS2 Image Subscriber initialized successfully");

    };


    void ROS2ImageSubscriber::image_callback(
        const sensor_msgs::msg::Image::SharedPtr msg
    ) {

        // Check incoming msg
        if (!msg) {
            RCLCPP_WARN(get_logger(), "Received null image message");
            return;
        };
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.frames_received++;
            stats_.last_encoding = msg->encoding;
        };

        // Convert ROS2 msg => OpenCV img
        cv::Mat cv_image = convert_ros2_image_to_opencv(msg);

        if (cv_image.empty()) {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.conversion_errors++;
            RCLCPP_WARN(
                get_logger(), 
                "Failed to convert ROS2 image to OpenCV (encoding: %s)",
                msg->encoding.c_str()
            );
            return;
        };

        // Store frame in queue with thread safety
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);

            // Check if queue at cap
            // If full, drop oldest frame and metadata
            if (frame_queue_.size() >= max_queue_size_) {
                frame_queue_.pop();
                metadata_queue_.pop();
                
                std::lock_guard<std::mutex> stats_lock(stats_mutex_);
                stats_.frames_dropped++;
            }

            // Add new frame to queue
            frame_queue_.push(cv_image.clone()); // Clone to ensure independent memory
            
            // Store metadata
            FrameMetadata metadata;
            metadata.sequence = msg->header.stamp.sec;
            metadata.timestamp = static_cast<double>(msg->header.stamp.sec) +
                                 static_cast<double>(msg->header.stamp.nanosec) / 1e9;
            metadata_queue_.push(metadata);
        }

    };


    cv::Mat ROS2ImageSubscriber::convert_ros2_image_to_opencv(
        const sensor_msgs::msg::Image::SharedPtr &msg
    ) {

        try {
            // Here I just use cv_bridge to convert ROS2 image message to OpenCV Mat
            // Ref: https://docs.ros.org/en/diamondback/api/cv_bridge/html/c++/namespacecv__bridge.html
            // For desired output encoding:
            //      - "bgr8"  : commonly used for color images
            //      - "mono8" : for grayscale
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
            return cv_ptr->image;

        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(
                get_logger(),
                "cv_bridge exception: %s, encoding: %s",
                e.what(),
                msg->encoding.c_str()
            );
            return cv::Mat(); // Return empty mat on error

        } catch (const std::exception &e) {
            RCLCPP_ERROR(
                get_logger(),
                "Unexpected exception during image conversion: %s",
                e.what()
            );
            return cv::Mat();
        }

    };


    cv::Mat ROS2ImageSubscriber::get_latest_frame() 
    {

        std::lock_guard<std::mutex> lock(frame_mutex_);

        // Return empty Mat if no frames available
        if (frame_queue_.empty()) {
            return cv::Mat();
        }

        // Fetch latest frame and corresponding metadata
        cv::Mat latest_frame = frame_queue_.front();
        frame_queue_.pop();
        metadata_queue_.pop();

        return latest_frame;

    };


    cv::Mat ROS2ImageSubscriber::get_latest_frame_with_timestamp(
        uint32_t &frame_index,
        double &timestamp_sec
    ) {

        std::lock_guard<std::mutex> lock(frame_mutex_);

        // Return empty Mat if no frames available
        if (frame_queue_.empty()) {
            return cv::Mat();
        }

        // Fetch latest frame and corresponding metadata
        cv::Mat latest_frame = frame_queue_.front();
        frame_queue_.pop();

        // Get metadata for synchronization
        const FrameMetadata &metadata = metadata_queue_.front();
        frame_index = metadata.sequence;
        timestamp_sec = metadata.timestamp;
        metadata_queue_.pop();

        return latest_frame;

    };


    // Bunch of frame queue handlings

    bool ROS2ImageSubscriber::has_frames() const {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        return !frame_queue_.empty();
    };

    size_t ROS2ImageSubscriber::get_queue_size() const {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        return frame_queue_.size();
    };

    size_t ROS2ImageSubscriber::get_max_queue_size() const {
        return max_queue_size_;
    };

    void ROS2ImageSubscriber::clear_frame_buffer() {

        std::lock_guard<std::mutex> lock(frame_mutex_);

        while (!frame_queue_.empty()) {
            frame_queue_.pop();
        }

        while (!metadata_queue_.empty()) {
            metadata_queue_.pop();
        }

        RCLCPP_INFO(get_logger(), "Frame buffer cleared");

    };

}; // namespace camera_subscriber