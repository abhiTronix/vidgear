<!--
===============================================
vidgear library source-code is deployed under the Apache 2.0 License:

Copyright (c) 2019-2020 Abhishek Thakur(@abhiTronix) <abhi.una12@gmail.com>

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
  <img src="../assets/images/gears_fbd.webp" alt="@Vidgear Functional Block Diagram"/>
  <figcaption>Gears generalized workflow diagram</figcaption>
</figure>

## Gears, What are these?

VidGear is built on standalone classes - also known as **Gears**, each with some unique functionality. These Gears provides a powerful, easy-to-use, highly extensible, Multi-Threaded + Asyncio layer above many state-of-the-art specialized libraries to exploit their internal properties flexibly, while providing robust error-free and unparalleled real-time performance.

## Gears Classification

These Gears can be classified as follows:

### A. VideoCapture Gears

> **Basic Function:** Retrieves [`numpy.ndarray`](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) frames from various sources.

* [CamGear](camgear/overview/): Multi-threaded API targeting various IP-USB-Cameras/Network-Streams/YouTube-Video-URLs.
* [PiGear](pigear/overview/): Multi-threaded API targeting  various Raspberry Pi Camera Modules.
* [ScreenGear](screengear/overview/): Multi-threaded ultra-fast Screencasting.    
* [VideoGear](videogear/overview/): Common API with internal [Video Stabilizer](stabilizer/overview/) wrapper.  

### B. VideoWriter Gears

> **Basic Function:** Writes [`numpy.ndarray`](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) frames to a video file.

* [WriteGear](writegear/introduction/): Handles Flexible Lossless Video Encoding and Compression.

### C. Streaming Gears

> **Basic Function:** Transcodes videos/audio files & [`numpy.ndarray`](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) frames for HTTP streaming.

* [StreamGear](streamgear/overview/): Handles Ultra-Low Latency, High-Quality, Dynamic & Adaptive Streaming Formats.

### D. Network Gears

> **Basic Function:** Sends/Receives [`numpy.ndarray`](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) frames over the network.

* [NetGear](netgear/overview/): Handles high-performance video-frames & data transfer between interconnecting systems over the network.

* **Asynchronous I/O Network Gears:**

    * [WebGear](webgear/overview/): ASGI Video Server that can send live video-frames to any web browser on the network.
    * [NetGear_Async](netgear_async/overview/): Immensely Memory-efficient Asyncio video-frames network messaging framework.

&thinsp;