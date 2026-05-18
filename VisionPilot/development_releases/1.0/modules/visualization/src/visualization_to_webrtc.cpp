#include <visualization/visualization_to_webrtc.hpp>

// GStreamer headers for WebRTC streaming
#include <gst/app/gstappsrc.h>
#include <gst/sdp/gstsdpmessage.h>
#include <gst/webrtc/webrtc.h>

// libsoup for WebSocket signaling
#include <libsoup/soup.h>

// JSON-GLib for JSON handling in signaling messages
#include <json-glib/json-glib.h>

#include <algorithm>
#include <cmath>
#include <string>


namespace visualization {


    namespace {


        // Helper func to initialize GStreamer once (thread-safe)
        void init_gstreamer_once() {

            static std::once_flag once;
            std::call_once(once, []() {
                gst_init(nullptr, nullptr);
            });

        }


        // Helper func to escape JSON strings for signaling messages
        std::string escape_json(
            const std::string& value
        ) {
            gchar * escaped = g_strescape(value.c_str(), nullptr);
            std::string result = escaped != nullptr ? escaped : "";
            g_free(escaped);

            return result;
        };


        // Helper func to generate JSON signaling message for SDP offer
        std::string make_offer_message(
            const std::string& sdp_offer
        ) {

            return  std::string{
                        "{ \"type\": \"offer\", \"sdp\": \""
                    } +                                                 \
                    escape_json(sdp_offer) + "\" }";

        };


        // Helper func to generate JSON signaling message for ICE candidate
        std::string make_candidate_message(
            int sdp_mline_index,
            const std::string& candidate
        ) {

            return  std::string{
                    "{ \"type\": \"candidate\", \"sdpMLineIndex\": "
                    } +                                                 \
                    std::to_string(sdp_mline_index) +                   \
                    ", \"candidate\": \"" +                             \
                    escape_json(candidate) +                            \
                    "\" }";

        };


        // Helper func to ensure all received frames are in BGR format for WebRTC streaming
        cv::Mat ensure_bgr_frame(
            const cv::Mat& frame
        ) {

            // Return empty frame as is
            if (frame.empty()) {
                return frame;
            };

            // If already in BGR format, return as is (or clone if not continuous)
            if (frame.type() == CV_8UC3) {
                return frame.isContinuous() ? frame : frame.clone();
            };

            // Convert grayscale or BGRA frames to BGR format for WebRTC streaming
            cv::Mat converted;
            if (frame.type() == CV_8UC1) {          // If grayscale
                cv::cvtColor(
                    frame, 
                    converted, 
                    cv::COLOR_GRAY2BGR
                );
            } else if (frame.type() == CV_8UC4) {   // If BGRA
                cv::cvtColor(
                    frame,
                    converted,
                    cv::COLOR_BGRA2BGR
                );
            } else {                                // For other formats, try direct conversion
                frame.convertTo(
                    converted,
                    CV_8UC3
                );
                if (converted.channels() != 3) {
                    return cv::Mat();
                };
            };

            return converted;

        };


    };  // namespace


    struct WebRTCStreamer::Impl {

        // Implementation details for WebRTC streaming (e.g., GStreamer pipeline, signaling, etc.)
        // This struct shall contain members and methods to manage WebRTC connections, encode frames, handle signaling, etc.

        explicit Impl(Config config_in) : config(std::move(config_in)) {};


        // Helper funcs for WebRTC streaming handling
        bool start();
        bool stop();
        bool push_frame(const cv::Mat& frame);
        bool has_clients() const;
        void queue_signal(const std::string& signal);
        void flush_pending_signals();
        void queue_remote_candidate(
            int sdp_mline_index,
            const std::string& candidate
        );
        void flush_pending_remote_candidates();


        // Config for WebRTC streaming
        Config config;
        SoupServer *server = nullptr;
        GMainLoop *main_loop = nullptr;
        std::thread server_thread;


        // GStreamer elements for WebRTC streaming
        GstElement *pipeline = nullptr;
        GstElement *appsrc = nullptr;
        GstElement *webrtc = nullptr;


        // State management for signaling and streaming
        mutable std::mutex signal_mutex;
        SoupWebsocketConnection *client_connection = nullptr;
        std::vector<std::string> pending_signals;


        // State management for remote ICE candidates received before remote description is set
        std::mutex remote_candidate_mutex;
        std::vector<std::pair<int, std::string>> pending_remote_candidates;
        std::atomic<bool> remote_description_ready{false};


        // State management for streaming
        std::atomic<bool> running{false};
        std::atomic<uint64_t> frame_index{0};
        bool caps_configured = false;
        int configured_width = 0;
        int configured_height = 0;

    };

}