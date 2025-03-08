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
  <figcaption>Gears: Generalized Workflow</figcaption>
</figure>

## Gears :octicons-gear-24:, What Are These?

VidGear is built on standalone APIs—also known as **Gears :fontawesome-solid-gears:**—each with some unique functionality. Each Gear is designed exclusively to handle/control/process different data-specific and device-specific video streams, network streams, and media encoders/decoders. 

Gears allow users to work with an inherently optimized, easy-to-use, extensible, and exposed API framework on top of many state-of-the-art libraries, while silently delivering robust error handling and unmatched real-time performance.

## Gears Classification

These Gears can be classified as follows:

### A. VideoCapture Gears

> **Basic Function:** Retrieves [`numpy.ndarray`](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) frames from various sources.

* [CamGear](camgear/overview/): Multi-threaded API targeting various IP-USB-Cameras/Network-Streams/Streaming-Sites-URLs.
* [PiGear](pigear/overview/): Multi-threaded API targeting various Camera Modules and (limited) USB cameras on Raspberry Pis :fontawesome-brands-raspberry-pi:.
* [ScreenGear](screengear/overview/): High-performance API targeting rapid screencasting capabilities.
* [VideoGear](videogear/overview/): Common video-capture API with internal [Video Stabilizer](stabilizer/overview/) wrapper. 

### B. VideoWriter Gears

> **Basic Function:** Writes [`numpy.ndarray`](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) frames to a video file or network stream.

* [WriteGear](writegear/introduction/): Handles lossless video writer for file/stream/frame encoding and compression.

### C. Streaming Gears

> **Basic Function:** Transcodes/broadcasts files and [`numpy.ndarray`](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) frames for streaming.

!!! tip "You can also use [WriteGear](writegear/introduction/) for :material-video-wireless: streaming with traditional protocols such as RTMP, RTSP/RTP."

* [StreamGear](streamgear/introduction/): Handles transcoding of high-quality, dynamic, and adaptive streaming formats.

* **Asynchronous I/O Streaming Gear:**

    * [WebGear](webgear/overview/): ASGI video server that broadcasts live MJPEG-frames to any web browser on the network.

    * [WebGear_RTC](webgear_rtc/overview/): Real-time asyncio WebRTC media server for streaming directly to peer clients over the network.

### D. Network Gears

> **Basic Function:** Sends/receives data and [`numpy.ndarray`](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) frames over connected networks.

* [NetGear](netgear/overview/): Handles high-performance video-frames and data transfer between interconnecting systems over the network.

* **Asynchronous I/O Network Gear:**

    * [NetGear_Async](netgear_async/overview/): Immensely memory-efficient asyncio video-frames network messaging framework.
