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

# WriteGear API 


## Overview

<p align="center">
  <img src="../../../assets/images/writegear.png" alt="WriteGear Functional Block Diagram" width="75%" />
  <br>
  <sub><i>WriteGear API generalized workflow</i></sub>
</p>

WriteGear API provides a complete, flexible and robust wrapper around [**FFmpeg**](https://ffmpeg.org/), a leading multimedia framework. With WriteGear, we can process real-time frames into a lossless compressed video-file with any suitable specification in just few easy lines of codes. These specifications include setting video/audio properties such as `bitrate, codec, framerate, resolution, subtitles,  etc.`, and also performing complex tasks such as multiplexing video with audio in real-time _(see this [doc](../compression/usage/#using-compression-mode-with-live-audio-input)_, while handling all errors robustly. 

Best of all, WriteGear grants the complete freedom to play with any FFmpeg parameter with its exclusive ==Custom Commands function== _(see this [doc](../compression/advanced/cciw/))_, without relying on any Third-party library.

In addition to this, WriteGear also provides flexible access to [**OpenCV's VideoWriter API**](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html) which provides some basic tools for video frames encoding but without compression.


&nbsp; 

## Modes of Operation

WriteGear primarily operates in following modes:

* [**Compression Mode**](../compression/overview/): In this mode, WriteGear utilizes powerful **FFmpeg** inbuilt encoders to encode lossless multimedia files. This mode provides us the ability to exploit almost any parameter available within FFmpeg, effortlessly and flexibly, and while doing that it robustly handles all errors/warnings quietly.

* [**Non-Compression Mode**](../non_compression/overview/): In this mode, WriteGear utilizes basic **OpenCV's inbuilt VideoWriter API** tools. This mode also supports all parameters manipulation available within VideoWriter API, but it lacks the ability to manipulate encoding parameters and other important features like video compression, audio encoding, etc.


&nbsp; 


!!! tip "Helpful Tips"

  * It is advised to enable logging(`logging = True`) on the first run for easily identifying any runtime errors.

  * Compression Mode must require FFmpeg, Follow these [Installation Instructions ➶](../compression/advanced/ffmpeg_install/#ffmpeg-installation-instructions) for its installation.


&nbsp; 

## Importing

You can import WriteGear API in your program as follows:

```python
from vidgear.gears import WriteGear
```

&nbsp; 