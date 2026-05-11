#include <v4l2_interface/v4l2_reader.hpp>

#include <sstream>
#include <iomanip>
#include <chrono>
#include <thread>


namespace v4l2_interface {

    V4L2Reader::V4L2Reader(
        const std::string& device_path,
        uint32_t fps
    ) : device_path(device_path),
        target_fps(fps),
        is_stream_started(false),
        has_latest_frame(false)
    {
        
        log_info("Initializing V4L2 Reader");
        log_info("  Device Path: " + device_path);
        log_info("  Target FPS: " + std::to_string(fps));

}