#ifdef ENABLE_ROS2_INTERFACE
#include <camera_subscriber/ros2_to_opencv.hpp>
#endif
#include <camera_interface/v4l2_camera_interface.hpp>

#include <chrono>
#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
#include <thread>

#include <visualization/visualization.hpp>
#include <visualization/visualization_to_webrtc.hpp>

int main(int argc, char **argv) {
    // For now, first argument is either 0 (for ROS2) or 1 (for V4L2):
    // - If 0: second argument is ROS2 topic name (default: "/camera/image")
    // - If 1:
    //      - Second argument is V4L2 device path (default: "/dev/video0")
    //      - Third argument is FPS (default: 10)
    //
    // Then the following arguments are for the WebRTC streamer config:
    // - Next argument is whether to start WebRTC streamer (0: False or 1: True, default: 0)
    // - Next argument is browser port (default: 8080)

    std::unique_ptr<camera_interface::CameraInterface> camera_interface;

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " [mode] [args...]\n";
        std::cout << "  mode: 0 for ROS2, 1 for V4L2\n";
        std::cout << "  For ROS2 mode: [topic_name]\n";
        std::cout << "  For V4L2 mode: [device_path] [fps]\n";
        return 1;
    } else {
        int mode = std::stoi(argv[1]);
        std::unordered_map<std::string, std::string> args_info;


        // ==================================== INPUT MODES ====================================


        // 1. ROS2

#ifdef ENABLE_ROS2_INTERFACE
            std::string topic = "/camera/image";
            if (argc > 2) {
                topic = argv[2];
            };
            std::cout << "Starting in ROS2 mode with topic: " << topic << "\n";

            args_info["topic"] = topic;

            // Init reader instance
            camera_interface = std::make_unique<camera_interface::ROS2ImageSubscriber>(topic);
#else
        // 2. V4L2
            std::string device_path = "/dev/video0";
            uint32_t target_fps = 10;
            if (argc > 2) {
                device_path = argv[2];
            };
            if (argc > 3) {
                target_fps = static_cast<uint32_t>(std::stoi(argv[3]));
            };
            std::cout << "Starting in V4L2 mode with device: " << device_path
                    << " and FPS: " << target_fps << "\n";

            args_info["device_path"] = device_path;
            args_info["target_fps"] = std::to_string(target_fps);

            // Init reader instance
            camera_interface = std::make_unique<camera_interface::V4L2CameraInterface>(device_path, target_fps);
            if (!camera_interface->is_device_open()) {
                std::cerr << "Failed to open V4L2 device: " << device_path << std::endl;
                return 1;
            };
#endif


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


        // ==================================== MAIN LOOP ====================================


        while (true) {
            // Get latest frame and flag
            bool has_frame = false;
            cv::Mat frame;

            auto frame_result = camera_interface->get_latest_frame();
            has_frame = std::get<0>(frame_result);
            frame = std::get<1>(frame_result);


            if (has_frame && !frame.empty()) {
                // Overlap stats strings
                std::vector<std::string> overlay_strs;

                auto stats = camera_interface->get_stats();

#ifdef ENABLE_ROS2_INTERFACE
                    overlay_strs = {
                        "topic: " + args_info["topic"],
                        "frames received: " + std::to_string(stats.frames_received),
                        "frames dropped: " + std::to_string(stats.frames_dropped),
                        "conversion errors: " + std::to_string(stats.conversion_errors)
                    };
#else
                    overlay_strs = {
                        "device: " + args_info["device_path"],
                        "frames captured: " + std::to_string(stats.frames_captured),
                        "capture errors: " + std::to_string(stats.capture_errors),
                        "resolution: " + std::to_string(stats.current_width) + "x" +
                        std::to_string(stats.current_height),
                        "fps: " + std::to_string(static_cast<int>(stats.current_fps))
                    };
#endif

                // Render out frame ONLY WHEN not WebRTC streaming to avoid X11/xcb threading issues
                if (show_local_preview) {
                    visualization::render_frame(
                        frame,
                        "VisionPilot",
                        overlay_strs
                    );
                }

                // Push frame to WebRTC streamer if enabled
                if (webrtc_streamer != nullptr) {
                    webrtc_streamer->push_frame(frame);
                };
            };

            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        };

        visualization::close_windows();
    }

    return 0;
}
