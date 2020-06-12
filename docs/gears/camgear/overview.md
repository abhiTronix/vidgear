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

# CamGear API 


## Overview

<p align="center">
  <img src="../../../assets/images/camgear.png" alt="CamGear Functional Block Diagram" title="Designed by Abhishek Thakur(@abhiTronix), under CC-BY-NC-SA 4.0 License"/>
  <br>
  <sub><i>Functional block diagram depicts CamGear API's generalized workflow</i></sub>
</p>


CamGear supports a diverse range of video streams which can handle/control video stream almost any IP/USB Cameras,  multimedia video file format ([_upto 4k tested_](https://github.com/abhiTronix/vidgear/blob/62f32ad6663c871ec6aa4890ca1b55cd1286511a/vidgear/tests/benchmark_tests/test_benchmark_playback.py#L31-L71)), any network stream URL such as `http(s), rtp, rstp, rtmp, mms, etc.` In addition to this, it also supports live [Gstreamer's RAW pipelines](https://gstreamer.freedesktop.org/documentation/frequently-asked-questions/using.html) and [YouTube video/livestreams URLs](/gears/camgear/usage/#using-camgear-with-youtube-videos).

CamGear API provides a flexible, high-level multi-threaded wrapper around OpenCV's *[VideoCapture API](https://docs.opencv.org/master/d8/dfe/classcv_1_1VideoCapture.html#a57c0e81e83e60f36c83027dc2a188e80)* with direct access to almost all of its available [*parameters*](/gears/camgear/source_params/) and also internally employs `pafy` with `youtube-dl` backend for seamless live [*YouTube streaming*](/gears/camgear/usage/#using-camgear-with-youtube-videos). 

CamGear relies exclusively on [**Threaded Queue mode**](/bonus/TQM/) for threaded, error-free and synchronized frame handling.

&nbsp; 


!!! tip "Helpful Tips"

  * It is advised to enable logging(`logging = True`) on the first run for easily identifying any runtime errors.

  * You can use `framerate` class variable to retrieve framerate of the input source. Its usage example can be found [here ➶](/gears/writegear/compression/usage/#using-compression-mode-with-controlled-framerate)


&nbsp; 

## Importing

You can import CamGear API in your program as follows:

```python
from vidgear.gears import CamGear
```

&nbsp; 