#include <chrono>
#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
#include <thread>

#include <camera_interface/v4l2_camera_interface.hpp>

#ifdef ENABLE_ROS2_INTERFACE
#include <camera_subscriber/ros2_to_opencv.hpp>
#endif

#include <visualization/visualization.hpp>
#include <visualization/visualization_to_webrtc.hpp>

int main(int argc, char **argv) {
    std::unique_ptr<camera_interface::CameraInterface> camera_interface;

#ifdef ENABLE_ROS2_INTERFACE
    camera_interface = std::make_unique<camera_interface::ROS2ImageSubscriber>("/camera/image");
#else
    camera_interface = std::make_unique<camera_interface::V4L2CameraInterface>("/dev/video0", 10);
#endif

    if (!camera_interface->is_device_open()) {
        std::cerr << "Failed to open camera interface!" << std::endl;
        return 1;
    };


    std::unordered_map<std::string, std::string> args_info;


    // =================================== WEBRTC INIT ===================================
    bool start_webrtc = false;
    uint16_t webrtc_port = 8080;

    if (argc > 4) {
        start_webrtc = (std::stoi(argv[4]) != 0);
    };
    if (argc > 5) {
        webrtc_port = static_cast<uint16_t>(std::stoi(argv[5]));
    };

    std::unique_ptr<visualization::WebRTCStreamer> webrtc_streamer;
    // Disable local preview if WebRTC is enabled to avoid X11/xcb threading issues
    const bool show_local_preview = !start_webrtc;

    if (start_webrtc) {
        std::cout << "Starting WebRTC streamer on port: " << webrtc_port << "\n";

        // Init WebRTC streamer instance (one-liner - Atanasko's request)
        webrtc_streamer = std::make_unique<visualization::WebRTCStreamer>();

        if (!webrtc_streamer->init(webrtc_port)) {
            std::cerr << "Failed to start WebRTC streamer." << std::endl;
            return 1;
        }

        std::cout << "Local OpenCV preview is disabled while WebRTC is enabled.\n";
    } else {
        std::cout << "WebRTC streamer disabled.\n";
    };

    //  MAIN LOOP
    while (true) {
        bool has_frame = false;
        cv::Mat frame;

        auto frame_result = camera_interface->get_latest_frame();
        has_frame = std::get<0>(frame_result);
        frame = std::get<1>(frame_result);

        if (has_frame && !frame.empty()) {
            std::vector<std::string> overlay = camera_interface->get_overlay();

            // Render out frame ONLY WHEN not WebRTC streaming to avoid X11/xcb threading issues
            if (show_local_preview) {
                visualization::render_frame(frame, "VisionPilot", overlay);
            }

            // Push frame to WebRTC streamer if enabled
            if (webrtc_streamer != nullptr) {
                webrtc_streamer->push_frame(frame);
            };
        };

        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    };
}
