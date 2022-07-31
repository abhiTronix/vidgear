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

# WriteGear API 

<figure>
  <img src="../../../assets/images/writegear.png" alt="WriteGear Functional Block Diagram" loading="lazy" class="center-small shadow"/>
  <figcaption>WriteGear API generalized workflow</figcaption>
</figure>

## Overview

> *WriteGear handles various powerful Video-Writer Tools that provide us the freedom to do almost anything imaginable with multimedia data.*

WriteGear API provides a complete, flexible, and robust wrapper around [**FFmpeg**](https://ffmpeg.org/), a leading multimedia framework. WriteGear can process real-time frames into a lossless compressed video-file with any suitable specifications _(such as`bitrate, codec, framerate, resolution, subtitles,  etc.`)_. 

WriteGear also supports streaming with traditional protocols such as [RTSP/RTP](../../../help/writegear_ex/#using-writegears-compression-mode-for-rtsprtp-live-streaming), RTMP. It is powerful enough to perform complex tasks such as [Live-Streaming](../compression/usage/#using-compression-mode-for-live-streaming) _(such as for Twitch, YouTube etc.)_ and [Multiplexing Video-Audio](../compression/usage/#using-compression-mode-with-live-audio-input) with real-time frames in just few lines of code.

Best of all, WriteGear grants users the complete freedom to play with any FFmpeg parameter with its exclusive ==Custom Commands function== _(see this [doc](../compression/advanced/cciw/))_ without relying on any third-party API.

In addition to this, WriteGear also provides flexible access to [**OpenCV's VideoWriter API**](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html) tools for video-frames encoding without compression.

&thinsp; 

## Modes of Operation

WriteGear primarily operates in following modes:

* [**Compression Mode**](../compression/overview/): In this mode, WriteGear utilizes powerful **FFmpeg** inbuilt encoders to encode lossless multimedia files. This mode provides us the ability to exploit almost any parameter available within FFmpeg, effortlessly and flexibly, and while doing that it robustly handles all errors/warnings quietly.

* [**Non-Compression Mode**](../non_compression/overview/): In this mode, WriteGear utilizes basic **OpenCV's inbuilt VideoWriter API** tools. This mode also supports all parameter transformations available within OpenCV's VideoWriter API, but it lacks the ability to manipulate encoding parameters and other important features like video compression, audio encoding, etc.


&thinsp; 


!!! tip "Helpful Tips"

	* If you're already familar with [OpenCV](https://github.com/opencv/opencv) library, then see [Switching from OpenCV ➶](../../../switch_from_cv/#switching-videowriter-api)

	* It is advised to enable logging(`logging = True`) on the first run for easily identifying any runtime errors.

&thinsp; 

## Importing

You can import WriteGear API in your program as follows:

```python
from vidgear.gears import WriteGear
```

&thinsp; 

## FAQs

<div>
<a href="../../../help/writegear_faqs/">See here 🚀</a>
</div>  

&thinsp; 