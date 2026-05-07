# VisionPilot Auxiliaries - Camera Viewer Node

This directory contains the standalone ROS2 => OpenCV bridge visualization side of the VisionPilot closed-loop test flow.

## I. Features

- A lightweight `camera_viewer_node` that subscribes to a ROS2 image topic, converts each message to `cv::Mat`, and displays the stream with OpenCV.
- The viewer prints runtime diagnostics in the terminal, including frame dimensions, channels, queue state, `is_stream_active()`, `has_frames()`, and a rolling received frame rate.

## II. Image source options

You can publish ROS2 images without modifying VisionPilot:

### Option 1: `image_publisher`

Install the package and publish a video file directly as a ROS2 image stream.

```bash

# 1. Install package, assuming you are using ROS2 Humble
sudo apt update
sudo apt install ros-humble-image-publisher

# 2. Establish ROS2 topic (assuming the topic is `/camera/image`)
ros2 run image_publisher image_publisher_node --ros-args \
-p filename:="<video absolute path>" \
-p publish_rate:=30.0 \
-p frame_id:="camera_link" \
-r image_raw:=/camera/image

```

### Option 2: `v4l2_camera` with a virtual device

Use `v4l2loopback` to create a fake camera device, stream video into it with `ffmpeg`, and read it with `v4l2_camera`.

(TODO: not yet tested, will check it properly later)

## III. Build and run

The viewer is built from the auxiliaries tree and reuses the existing `camera_subscriber` module from `development_releases/1.0`.

```bash

# 1. From repo root
cd VisionPilot/development_releases/auxiliaries

# 2. Build and config
mkdir -p build && cd build
source /opt/ros/humble/setup.bash
cmake ..
make -j$(nproc)

```

Run the viewer after publishing images:

```bash

./camera_viewer_node /camera/image_raw 1

```

The first argument is the ROS2 image topic. The second argument is the frame queue size, which defaults to `1` for low-latency behavior.

## IV. Notes

- The viewer expects the `camera_subscriber` ROS2-to-OpenCV pipeline from `development_releases/1.0`.