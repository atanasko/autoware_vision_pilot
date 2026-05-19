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

            return  std::string{"{ \"type\": \"offer\", \"sdp\": \""} + \
                    escape_json(sdp_offer) + "\" }";

        };


        // Helper func to generate JSON signaling message for ICE candidate
        std::string make_candidate_message(
            int sdp_mline_index,
            const std::string& candidate
        ) {

            return  std::string{"{ \"type\": \"candidate\", \"sdpMLineIndex\": "} + \
                    std::to_string(sdp_mline_index) + \
                    ", \"candidate\": \"" + \
                    escape_json(candidate) + \
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


    // Implementation of WebRTCStreamer class methods
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


    // HTTP handler for root path to serve browser page with WebRTC client
    void root_http_handler(
        SoupServer *server,
        SoupMessage *msg,
        const char *path,
        GHashTable *query,
        SoupClientContext *client,
        gpointer user_data
    ) {

        (void)server;
        (void)path;
        (void)query;
        (void)user_data;

        // Server a simple HTML page with JS to link WS and display video stream via WebRTC
        soup_message_set_response(
            msg,
            "text/html", 
            SOUP_MEMORY_STATIC,
            kBrowserHtml,
            std::strlen(kBrowserHtml)
        );

    };


    // Websocket handler for closing connection to clean up client state
    void on_websocket_closed(
        SoupWebsocketConnection *connection,
        gpointer user_data
    ) {

        auto *impl = static_cast<WebRTCStreamer::Impl*>(user_data);
        std::lock_guard<std::mutex> lock(impl->signal_mutex);

        if (impl->client_connection == connection) {
            g_object_unref(impl->client_connection);
            impl->client_connection = nullptr;
        }

    };


    // Websocket handler for remote candidate
    void handle_remote_candidate(
        WebRTCStreamer::Impl *impl,
        int sdp_mline_index,
        const std::string& candidate
    ) {

        g_signal_emit_by_name(
            impl->webrtc,
            "add-ice-candidate",
            static_cast<guint>(sdp_mline_index),
            candidate.c_str()
        );

    };


    // Websocket handler for incoming messages (SDP offers and ICE candidates)
    void on_websocket_message(
        SoupWebsocketConnection *connection,
        gint type,
        GBytes *message,
        gpointer user_data
    ) {

        (void)connection;
        (void)type;

        auto *impl = static_cast<WebRTCStreamer::Impl*>(user_data);
        gsize size = 0;
        const gchar *data = static_cast<const gchar *>(g_bytes_get_data(message, &size));
        
        // Handle empty messages gracefully
        if (data == nullptr || size == 0) {
            return;
        };

        JsonParser *parser = json_parser_new();
        GError *error = nullptr;
        
        // Handle JSON parsing errors gracefully
        if (!json_parser_load_from_data(parser, data, static_cast<gsize>(size), &error)) {
            if (error != nullptr) {
                g_error_free(error);
            }
            g_object_unref(parser);
            return;
        };

        JsonNode *root = json_parser_get_root(parser);
        JsonObject *object = json_node_get_object(root);
        
        // Handle missing object or type field gracefully
        if (object == nullptr || !json_object_has_member(object, "type")) {
            g_object_unref(parser);
            return;
        };

        
        // Handle signaling messages based on their type (SDP offers and ICE candidates)
        const gchar *signal_type = json_object_get_string_member(object, "type");

        // Handle SDP answer messages to set remote description
        if (
            (g_strcmp0(signal_type, "answer") == 0) && 
            (json_object_has_member(object, "sdp"))
        ) {
            handle_remote_description(
                impl, 
                json_object_get_string_member(object, "sdp")
            );
        // Handle ICE candidate messages
        } else if (
            (g_strcmp0(signal_type, "candidate") == 0) && 
            (json_object_has_member(object, "candidate")) && 
            (json_object_has_member(object, "sdpMLineIndex"))
        ) {
            const int sdp_mline_index = json_object_get_int_member(object, "sdpMLineIndex");
            const std::string candidate = json_object_get_string_member(object, "candidate");

            // If remote description is already set, add candidate immediately
            if (impl->remote_description_ready.load(std::memory_order_acquire)) {
                handle_remote_candidate(impl, sdp_mline_index, candidate);
            // Otherwise, queue candidate to be added once remote description is set
            } else {
                impl->queue_remote_candidate(sdp_mline_index, candidate);
            }
        };

        g_object_unref(parser);

    };


    // Websocket handler for new connections to set up signaling handlers and manage client state
    void websocket_handler(
        SoupServer *server,
        SoupWebsocketConnection *connection,
        const char *path,
        SoupClientContext *client,
        gpointer user_data
    ) {
        
        (void)server;
        (void)path;

        auto *impl = static_cast<WebRTCStreamer::Impl *>(user_data);

        // Set up handlers for incoming messages and connection closure
        g_signal_connect(
            connection, 
            "message", 
            G_CALLBACK(on_websocket_message), 
            impl
        );
        g_signal_connect(
            connection, 
            "closed", 
            G_CALLBACK(on_websocket_closed), 
            impl
        );

        // Set keepalive interval to detect dead connections faster
        soup_websocket_connection_set_keepalive_interval(connection, 15);

        g_object_ref(connection);

        // Update client connection state in a thread-safe manner
        {
            std::lock_guard<std::mutex> lock(impl->signal_mutex);
            if (impl->client_connection_ != nullptr) {
                g_object_unref(impl->client_connection_);
            }
            impl->client_connection_ = connection;
        }

        impl->flush_pending_signals();

    };


    // Callback for when SDP offer is created to set local description and send offer to client
    void on_offer_created(
        GstPromise *promise, 
        gpointer user_data
    ) {

        auto *impl = static_cast<WebRTCStreamer::Impl *>(user_data);
        const GstStructure *reply = gst_promise_get_reply(promise);
        GstWebRTCSessionDescription *offer = nullptr;

        if (reply != nullptr) {
            gst_structure_get(
                reply, 
                "offer", 
                GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &offer, 
                NULL
            );
        }

        if (offer == nullptr) {
            gst_promise_unref(promise);
            return;
        }

        g_signal_emit_by_name(
            impl->webrtc_, 
            "set-local-description", 
            offer, 
            nullptr
        );

        gchar *sdp_text = gst_sdp_message_as_text(offer->sdp);
        if (sdp_text != nullptr) {
            impl->queue_signal(make_offer_message(sdp_text));
            g_free(sdp_text);
        }

        gst_webrtc_session_description_free(offer);
        gst_promise_unref(promise);

    };


    // Callback for when negotiation is needed to create a new SDP offer
    void on_negotiation_needed(
        GstElement *element, 
        gpointer user_data
    ) {

        (void)element;

        auto *impl = static_cast<WebRTCStreamer::Impl *>(user_data);
        GstPromise *promise = gst_promise_new_with_change_func(
            on_offer_created, 
            impl, 
            nullptr
        );

        g_signal_emit_by_name(
            impl->webrtc_, 
            "create-offer", 
            nullptr, 
            promise
        );

    };

    
    // Callback for when an ICE candidate is gathered to send it to the client
    void on_ice_candidate(
        GstElement *element, 
        guint mline_index, 
        gchar *candidate, 
        gpointer user_data
    ) {
        
        (void)element;
        auto *impl = static_cast<WebRTCStreamer::Impl *>(user_data);

        if (candidate != nullptr) {
            impl->queue_signal(make_candidate_message(
                static_cast<int>(mline_index), 
                candidate
            ));
        };
        
    };

}