# VISUALIZATION MODULE

## Acknowledgement

I would like to thank [Ethan](https://dev.to/ethand91) and his blog post of [Streaming Camera with C++ WebRTC GStreamer](https://dev.to/ethand91/streaming-camera-with-c-webrtc-gstreamer-pof).

Your implementation was truly helpful and inspiring for me to complete this module.

## I. Overview

The WebRTC Visualization Module provides a real-time video streaming capability for the VisionPilot pipeline via WebRTC protocol. It serves the following core functions:

1. **Real-time frame capture and encoding** which accepts OpenCV `cv::Mat` frames and encodes them to VP8 video codec via GStreamer.
2. **WebRTC peer-to-peer streaming** which establishes a WebRTC peer connection between the server (VisionPilot app) and browser clients, enabling live video delivery over the internet or LAN.
3. **Lightweight browser client** which serves a minimal, self-contained HTML5 page with built-in WebRTC JavaScript client without external dependencies required for the browser.
4. Implements WebSocket-based signaling for SDP (Session Description Protocol) offer/answer negotiation and ICE (Interactive Connectivity Establishment) candidate exchange.
5. **Thread-safe frame streaming** that manages concurrent frame pushes from the main application thread while running a GStreamer pipeline and event loop in separate threads.

This module is essential for downstream remote monitoring, debugging, and visualization of autonomous driving pipelines during development and testing phases.

## II. Architecture && Module structure

### 1. Architecture

```

┌─────────────────────────────────────────────────────────────────┐
│                    VisionPilot Application                      │
│                  (vision_pilot.cpp main thread)                 │
│                                                                 │
│   DATA CAPTURE                                                  │
│   ┌──────────────────────┐                                      │
│   │  V4L2/ROS2 Camera    │                                      │
│   │      Source          │                                      │
│   └──────────┬───────────┘                                      │
│              │                                                  │
│              │ cv::Mat frames (33ms loop)                       │
│              │                                                  │
│              ▼                                                  │
│   ┌──────────────────────────────────────────┐                  │
│   │  (Various upstream modules, like         │                  │
│   │  model inference, processing, calc etc.) │                  │
│   └──────────┬───────────────────────────────┘                  │
│              │                                                  │
│              │ Frames & longitudinal/lateral planning results   │
│              │                                                  │
│              ▼                                                  │
│   VISUALIZATION                                                 │
│   ┌──────────────────────────────────────────┐                  │
│   │  visualization::render_frame()           │                  │
│   │  (draw frame + planning results)         │                  │
│   └──────────┬───────────────────────────────┘                  │
│              │                                                  │
│              ▼                                                  │
│   ┌──────────────────────────────────────────┐                  │
│   │  WebRTCStreamer::push_frame()            │                  │
│   │  (stream to endpoint via WebRTC)         │                  │
│   └──────────┬───────────────────────────────┘                  │
│              │                                                  │
└──────────────┼──────────────────────────────────────────────────┘
               │
               │ BGR frames + metadata
               │
    ┌──────────▼───────────────────────────────────────────┐
    │         WebRTCStreamer::Impl (Internal)              │
    │                                                      │
    │  ┌────────────────────────────────────────────────┐  │
    │  │  GStreamer Pipeline (separate thread)          │  │
    │  │                                                │  │
    │  │  appsrc => queue => videoconvert => vp8enc =>  │  │
    │  │  rtpvp8pay => webrtcbin                        │  │
    │  │                                                │  │
    │  │  ┌──────────────────────────────────────────┐  │  │
    │  │  │ WebRTC peer connection (GStreamer)       │  │  │
    │  │  │  - Manages media stream                  │  │  │
    │  │  │  - Generates SDP offers                  │  │  │
    │  │  │  - Gathers ICE candidates                │  │  │
    │  │  └──────────────────────────────────────────┘  │  │
    │  └────────────────────────────────────────────────┘  │
    │                                                      │
    │  ┌────────────────────────────────────────────────┐  │
    │  │  Signaling Layer (SoupServer + WebSocket)      │  │
    │  │                                                │  │
    │  │  HTTP handler:                                 │  │
    │  │    GET / => serves kBrowserHtml                │  │
    │  │                                                │  │
    │  │  WebSocket handler:                            │  │
    │  │    - Receives: SDP answer, ICE candidates      │  │
    │  │    - Sends: SDP offer, ICE candidates          │  │
    │  │    - Queue + flush mechanism for ordering      │  │
    │  │                                                │  │
    │  └────────────────────────────────────────────────┘  │
    │                                                      │
    └───────────────┬──────────────────────────────────────┘
                    │
        ┌───────────┴──────────────┐
        │                          │
        ▼                          ▼
   ┌─────────────────┐        ┌──────────────────┐
   │  Browser Client │        │  Network         │
   │  (HTML5 + JS)   │ <====> │  (Internet/LAN)  │
   │                 │        └──────────────────┘
   │ ┌─────────────┐ │        
   │ │ RTCPeerConn │ │
   │ │ (signaling) │ │
   │ ├─────────────┤ │
   │ │ WebSocket   │ │
   │ │ (SDP/ICE)   │ │
   │ ├─────────────┤ │
   │ │ <video>     │ │
   │ │ (playback)  │ │
   │ └─────────────┘ │
   └─────────────────┘

```

### 2. Flow summary

1. `VisionPilot` application calls `webrtc_streamer->push_frame(cv::Mat)` in its main loop. This frame is generated from `visualization::render_frame()`.
2. Frame is validated, converted to BGR, and pushed to the GStreamer pipeline's `appsrc` element.
3. GStreamer encodes the frame using VP8 codec and feeds it to the `webrtcbin` element.
4. On the first frame, `webrtcbin` triggers `on-negotiation-needed`, which creates an SDP offer.
5. SDP offer is queued and sent to the browser client via WebSocket.
6. Browser responds with SDP answer and ICE candidates.
7. Server receives answer, sets remote description, and flushes any pending ICE candidates.
8. Media stream begins flowing from server to browser via the established peer connection.

### 3. Module structure

```

visualization/
├── CMakeLists.txt
├── README.md (this file)
├── include/
│   └── visualization/
│       ├── visualization.hpp           (visualization header)
│       └── visualization_to_webrtc.hpp (WebRTC header)
└── src/
    ├── visualization.cpp               (visualization drawing, OpenCV window management)
    └── visualization_to_webrtc.cpp     (WebRTC implementation)

```