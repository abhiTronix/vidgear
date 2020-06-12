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

<img src="assets/images/vidgear.png" alt="VidGear" title="Logo designed by Abhishek Thakur(@abhiTronix), under CC-BY-NC-SA 4.0 License"/>

<h2 align="center">
	<img src="assets/images/tagline.svg" alt="VidGear tagline", width="50%"/>
</h2>

&nbsp;

VidGear is a High-Performance Framework that provides an all-in-one complete Video Processing solution for building real-time applications in python.

VidGear provides an easy-to-use, highly extensible, **Multi-Threaded + Asyncio wrapper** around many state-of-the-art specialized libraries like *[OpenCV][opencv], [FFmpeg][ffmpeg], [ZeroMQ][zmq], [picamera][picamera], [starlette][starlette], [pafy][pafy] and [python-mss][mss]* at its backend, and enable us to flexibly exploit their internal parameters and methods, while silently delivering robust error-handling and unparalleled real-time performance. 

VidGear primarily focuses on simplicity, and thereby lets programmers and software developers to easily integrate and perform Complex Video Processing Tasks, in just a few lines of code.

&nbsp;

## Getting Started

If this is your first time using VidGear, head straight to the [Installation ➶](installation.md) to install VidGear.

Once you have VidGear installed, checkout its [Gears ➶](gears.md)

Also, if you're already familar with [OpenCV][opencv] library, then see [Switching from OpenCV ➶](switch_from_cv.md)

&nbsp;

## Gears

VidGear is built on multiple [Gears ➶](gears) (APIs), which are exclusively designed to handle/control different device-specific video streams, network streams, and powerful media encoders. 

These Gears can be classified as follows:

**VideoCapture Gears**

* [CamGear](../gears/camgear/overview/): Multi-threaded API targeting various IP-USB-Cameras/Network-Streams/YouTube-Video-URLs.
* [PiGear](../gears/pigear/overview/): Multi-threaded API targeting  various Raspberry Pi Camera Modules.
* [ScreenGear](../gears/screengear/overview/): Multi-threaded ultra-fast Screencasting.    
* [VideoGear](../gears/videogear/overview/): Common API with internal [Video Stabilizer](../gears/stabilizer/overview/) wrapper.  

**VideoWriter Gears**

* [WriteGear](../gears/writegear/introduction/): Handles Flexible Lossless Video Encoding and Compression.

**Network Gears**

* [NetGear](../gears/netgear/overview/): Handles high-performance video-frames & data transfer between interconnecting systems over the network.

* **Asynchronous I/O Network Gears:**

    * [WebGear](../gears/webgear/overview/): ASGI Video Server that can send live video-frames to any web browser on the network.
    * [NetGear_Async](../gears/netgear_async/overview/): Immensely Memory-efficient Asyncio video-frames network messaging framework. 

&nbsp;

## Contributions

Contributions are welcome, and greatly appreciated!  

Please see our [Contribution Guidelines ➶](contribution.md) for more details.

&nbsp;

## Community Channel

If you've come up with some new idea, or looking for the fastest way troubleshoot your problems, then please join our [Gitter community channel ➶][gitter]

&nbsp; 

## Become a Stargazer

You can be a [Stargazer :star2:][stargazer] by starring us on Github, it helps us a lot and **you're making it easier for others to find & trust this library.** Thanks!

&nbsp;

## Love using VidGear? 

**VidGear is free, but rely on your support.** 

Sending a donation using link below is **extremely** helpful in keeping VidGear development alive:

<script type='text/javascript' src='https://ko-fi.com/widgets/widget_2.js'></script><script type='text/javascript'>kofiwidget2.init('Support Me on Ko-fi', '#eba100', 'W7W8WTYO');kofiwidget2.draw();</script> 

&nbsp;

## Citation

Here is a Bibtex entry you can use to cite this project in a publication:


```BibTeX
@misc{vidgear,
    author = {Abhishek Thakur},
    title = {vidgear},
    howpublished = {\url{https://github.com/abhiTronix/vidgear}},
    year = {2019}
  }
```

&nbsp;

<!--
External URLs
-->

[opencv]:https://github.com/opencv/opencv
[picamera]:https://github.com/waveform80/picamera
[pafy]:https://github.com/mps-youtube/pafy
[zmq]:https://zeromq.org/
[mss]:https://github.com/BoboTiG/python-mss
[gitter]: https://gitter.im/vidgear/community
[starlette]:https://www.starlette.io/
[stargazer]: https://github.com/abhiTronix/vidgear/stargazers
[ffmpeg]:https://www.ffmpeg.org/