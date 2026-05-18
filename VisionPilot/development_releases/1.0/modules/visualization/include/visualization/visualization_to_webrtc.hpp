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

        private:
            // Internal implementation details (e.g., WebRTC connection, encoding, etc.)
            struct Impl;
            std::unique_ptr<Impl> impl_;
    
        };

}


#endif //VISIONPILOT_VISUALIZATION_TO_WEBRTC_H
