<!--
===============================================
vidgear library source-code is deployed under the Apache 2.0 License:

Copyright (c) 2019 Abhishek Thakur(@abhiTronix) <abhi.una12@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
===============================================
-->

# Introduction

<figure>
  <img src="../assets/images/gears_fbd.png" loading="lazy" alt="@Vidgear Functional Block Diagram" class="shadow2" />
  <figcaption>Gears: generalized workflow</figcaption>
</figure>

## Gears :octicons-gear-24:, What are these?

VidGear is built on Standalone APIs - also known as **Gears**, each with some unique functionality. Each Gears is designed exclusively to handle/control/process different data-specific & device-specific video streams, network streams, and media encoders/decoders. 

Gears allows users to work with an inherently optimized, easy-to-use, extensible, and exposed API Framework on top of many state-of-the-art libraries, while silently delivering robust error handling and unmatched real-time performance.

## Gears Classification

These Gears can be classified as follows:

### A. VideoCapture Gears

> **Basic Function:** Retrieves [`numpy.ndarray`](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) frames from various sources.

* [CamGear](camgear/overview/): Multi-Threaded API targeting various IP-USB-Cameras/Network-Streams/Streaming-Sites-URLs.
* [PiGear](pigear/overview/): Multi-Threaded API targeting various Raspberry-Pi Camera Modules.
* [ScreenGear](screengear/overview/): High-performance API targeting rapid Screencasting Capabilities.
* [VideoGear](videogear/overview/): Common Video-Capture API with internal [Video Stabilizer](stabilizer/overview/) wrapper. 

### B. VideoWriter Gears

> **Basic Function:** Writes [`numpy.ndarray`](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) frames to a video file or network stream.

* [WriteGear](writegear/introduction/): Handles Lossless Video-Writer for file/stream/frames Encoding and Compression.

### C. Streaming Gears

> **Basic Function:** Transcodes/Broadcasts files and [`numpy.ndarray`](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) frames for streaming.

!!! tip "You can also use [WriteGear](writegear/introduction/) for :material-video-wireless: streaming with traditional protocols such as RTMP, RTSP/RTP."

* [StreamGear](streamgear/introduction/): Handles Transcoding of High-Quality, Dynamic & Adaptive Streaming Formats.

* **Asynchronous I/O Streaming Gear:**

    * [WebGear](webgear/overview/): ASGI Video-Server that broadcasts Live MJPEG-Frames to any web-browser on the network.

    * [WebGear_RTC](webgear_rtc/overview/): Real-time Asyncio WebRTC media server for streaming directly to peer clients over the network.

### D. Network Gears

> **Basic Function:** Sends/Receives data and [`numpy.ndarray`](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) frames over connected networks.

* [NetGear](netgear/overview/): Handles High-Performance Video-Frames & Data Transfer between interconnecting systems over the network.

* **Asynchronous I/O Network Gear:**

    * [NetGear_Async](netgear_async/overview/): Immensely Memory-Efficient Asyncio Video-Frames Network Messaging Framework.

&thinsp;