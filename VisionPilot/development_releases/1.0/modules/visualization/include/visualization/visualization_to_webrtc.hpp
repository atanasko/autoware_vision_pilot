//
// Created by atanasko on 1.5.26.
// Developed by TranHuuNhatHuy on 18.5.26.
//

#ifndef VISIONPILOT_VISUALIZATION_TO_WEBRTC_H
#define VISIONPILOT_VISUALIZATION_TO_WEBRTC_H

#include <opencv2/opencv.hpp>
#include <cstdint>
#include <memory>
#include <string>


namespace visualization {

    class WebRTCStreamer {

        public:
            
            /**
            * @brief Config options for the WebRTC streamer.
            * Provides parameters for WebRTC connection and streaming behavior.
            * 
            * Includes:
            * - host: WebRTC signaling server host (default: "127.0.0.1")
            * - port: WebRTC signaling server port (default: 8080)
            * - websocket_path: WebRTC signaling server WebSocket path (default: "/ws")
            * - frame_rate: desired streaming frame rate in FPS (default: 10.0 FPS)
            */
            struct Config {
                std::string host = "127.0.0.1"; // Default to IPv4 localhost
                uint16_t port = 8080;
                std::string websocket_path = "/ws";
                double frame_rate = 10.0;       // Default to 10 FPS
            };


            /**
            * @brief Constructor for WebRTCStreamer.
            * Inits WebRTC streamer with specified config.
            *
            * @param config Config options for WebRTC connection and streaming behavior.
            */
            WebRTCStreamer(
                Config config = Config()
            );


            /**
            * @brief Destructor for WebRTCStreamer.
            * Cleans up WebRTC resources and connections.
            */
            ~WebRTCStreamer();
            WebRTCStreamer(const WebRTCStreamer&) = delete;
            WebRTCStreamer& operator=(const WebRTCStreamer&) = delete;


            // STREAM HANDLING FUNCS

            
            /**
            * @brief Start the WebRTC streaming session.
            * Establishes connection to signaling server and prepares for streaming.
            *
            * @return true if streaming started successfully, false otherwise
            */
            bool start();


            /**
            * @brief Stop the WebRTC streaming session.
            * Closes WebRTC connections and cleans up resources.
            *
            * @return true if streaming stopped successfully, false otherwise
            */
            bool stop();

            
            /**
            * @brief Push a new video frame to the WebRTC stream.
            * Provides a thread-safe way to send frames to connected WebRTC clients.
            * 
            * @param frame The video frame to stream (as cv::Mat)
            *
            * @return true if frame was successfully pushed to the stream, false otherwise
            */
            bool push_frame(
                const cv::Mat& frame
            );


            /**
            * @brief Check if WebRTC stream is currently running.
            *
            * @return true if streaming is active, false otherwise
            */
            bool is_running() const;


            /**
            * @brief Check if there are any connected WebRTC clients.
            *
            * @return true if at least one client is connected, false otherwise
            */
            bool has_client() const;


            /**
            * @brief Get URL for browser to connect to WebRTC stream.
            *
            * @return Browser URL for WebRTC stream
            */
            std::string browser_url() const;

        private:
            // Internal implementation details (e.g., WebRTC connection, encoding, etc.)
            struct Impl;
            std::unique_ptr<Impl> impl_;
    
        };

}


#endif //VISIONPILOT_VISUALIZATION_TO_WEBRTC_H
