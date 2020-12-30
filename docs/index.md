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

<img src="assets/images/vidgear.png" alt="VidGear" title="Logo designed by Abhishek Thakur(@abhiTronix), under CC-BY-NC-SA 4.0 License" loading="lazy" width="90%" />

<h2 align="center">
	<img src="assets/images/tagline.svg" alt="VidGear tagline" loading="lazy" width="40%"/>
</h2>

&thinsp;

> VidGear is a High-Performance **Video-Processing** Framework for building complex real-time media applications in python :fire:

VidGear provides an easy-to-use, highly extensible, **Multi-Threaded + Asyncio Framework** on top of many state-of-the-art specialized libraries like *[OpenCV][opencv], [FFmpeg][ffmpeg], [ZeroMQ][zmq], [picamera][picamera], [starlette][starlette], [streamlink][streamlink], [pafy][pafy], [pyscreenshot][pyscreenshot] and [python-mss][mss]* at its backend, and enable us to flexibly exploit their internal parameters and methods, while silently delivering robust error-handling and unparalleled real-time performance.

> _"Write Less and Accomplish More"_ — VidGear's Motto

VidGear focuses on simplicity, and thereby lets programmers and software developers to easily integrate and perform Complex Video Processing Tasks, in just a few lines of code.

&thinsp;

## Getting Started

- [x] If this is your first time using VidGear, head straight to the [Installation ➶](installation.md) to install VidGear.

- [x] Once you have VidGear installed, **Checkout its Function-Specific [Gears ➶](gears.md)**

- [x] Also, if you're already familar with [OpenCV][opencv] library, then see [Switching from OpenCV Library ➶](switch_from_cv.md)

- [x] Or, if you're just getting started with OpenCV with Python, then see [here ➶](../help/general_faqs/#im-new-to-python-programming-or-its-usage-in-computer-vision-how-to-use-vidgear-in-my-projects)

&thinsp;

## Gears

> VidGear is built with multiple APIs a.k.a [Gears](gears), each with some unique functionality.

Each Gear is designed exclusively to handle/control/process different data-specific & device-specific video streams, network streams, and media encoders/decoders.

These Gears can be classified as follows:

#### VideoCapture Gears

* [CamGear](gears/camgear/overview/): Multi-Threaded API targeting various IP-USB-Cameras/Network-Streams/Streaming-Sites-URLs.
* [PiGear](gears/pigear/overview/): Multi-Threaded API targeting various Raspberry-Pi Camera Modules.
* [ScreenGear](gears/screengear/overview/): Multi-Threaded API targeting ultra-fast Screencasting.    
* [VideoGear](gears/videogear/overview/): Common Video-Capture API with internal [Video Stabilizer](gears/stabilizer/overview/) wrapper.

#### VideoWriter Gears

* [WriteGear](gears/writegear/introduction/): Handles Lossless Video-Writer for file/stream/frames Encoding and Compression.

#### Streaming Gears

* [StreamGear](gears/streamgear/overview/): Handles Transcoding of High-Quality, Dynamic & Adaptive Streaming Formats.

* **Asynchronous I/O Streaming Gear:**

    * [WebGear](gears/webgear/overview/): ASGI Video-Server that broadcasts Live Video-Frames to any web-browser on the network.

#### Network Gears

* [NetGear](gears/netgear/overview/): Handles High-Performance Video-Frames & Data Transfer between interconnecting systems over the network.

* **Asynchronous I/O Network Gear:**

    * [NetGear_Async](gears/netgear_async/overview/): Immensely Memory-Efficient Asyncio Video-Frames Network Messaging Framework. 

&thinsp;

## Contributions

Contributions are welcome, and greatly appreciated!  

Please see our [Contribution Guidelines ➶](contribution.md) for more details.

&thinsp;

## Community Channel

If you've come up with some new idea, or looking for the fastest way troubleshoot your problems. Please checkout our [Gitter community channel ➶][gitter]

&thinsp; 

## Become a Stargazer

You can be a [Stargazer :star2:][stargazer] by starring us on Github, it helps us a lot and *you're making it easier for others to find & trust this library.* Thanks!

&thinsp;

## Support Us

> VidGear relies on your support :heart:

Donations help keep VidGear's Development alive. Giving a little means a lot, even the smallest contribution can make a huge difference.

<script type='text/javascript' src='https://ko-fi.com/widgets/widget_2.js'></script><script type='text/javascript'>kofiwidget2.init('Support Me on Ko-fi', '#eba100', 'W7W8WTYO');kofiwidget2.draw();</script> 

&thinsp;

## Citation

Here is a Bibtex entry you can use to cite this project in a publication:


```BibTeX
@misc{vidgear,
    author = {Abhishek Thakur},
    title = {vidgear},
    howpublished = {\url{https://github.com/abhiTronix/vidgear}},
    year = {2019-2020}
  }
```

&thinsp;

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
[pyscreenshot]:https://github.com/ponty/pyscreenshot
[streamlink]:https://streamlink.github.io/