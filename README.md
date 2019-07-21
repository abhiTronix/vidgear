<!--
============================================
vidgear library code is placed under the MIT license
Copyright (c) 2019 Abhishek Thakur

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
===============================================
-->

<h1 align="center">
  <img src="https://abhitronix.github.io/img/vidgear/vidgear logo.svg" alt="VidGear Logo" width="70%"/>
</h1>
<h2 align="center">
  <img src="https://abhitronix.github.io/img/vidgear/vidgear banner.svg" alt="VidGear Banner" width="40%"/>
</h2>

<div align="center">

[Releases][release]&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Gears](#gears)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Wiki Documentation][wiki]&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Installation](#installation)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[License](#license)

[![PyPi version][pypi-badge]][pypi] [![PRs Welcome][prs-badge]][prs] [![Build Status][travis-cli]][travis] [![Build Status][appveyor]][app] [![Say Thank you][Thank-you]][thanks] [![Twitter][twitter-badge]][twitter-intent] 

[![Buy Me A Coffee][Coffee-badge]][coffee]

</div>

&nbsp;

VidGear is a powerful python Video Processing library built with multi-threaded [**Gears**]()(_a.k.a APIs_) each with a unique set of trailblazing features. These APIs provides a easy-to-use, highly extensible, and multi-threaded wrapper around many underlying state-of-the-art python libraries such as *[OpenCV ➶](https://github.com/opencv/opencv), [FFmpeg ➶](https://ffmpeg.org/), [picamera ➶](https://github.com/waveform80/picamera), [pafy ➶](https://github.com/mps-youtube/pafy), [pyzmq ➶](https://github.com/zeromq/pyzmq) and [python-mss ➶](https://github.com/BoboTiG/python-mss)*

&nbsp;

The following **functional block diagram** clearly depicts the functioning of VidGear library:

<p align="center">
  <img src="https://abhitronix.github.io/img/vidgear/vidgear_function2-01.svg" alt="@Vidgear Functional Block Diagram" />
</p>

&nbsp;

## Table of Contents

[**TL;DR**](#tldr)

[**New Release : VidGear 0.1.5**](#new-release-sneekpeak--vidgear-015)

[**Installation Options**](#installation)
  * [**Prerequisites**](#prerequisites)
  * [**1 - PyPI Install**](#option-1-pypi-install)
  * [**2 - Release Archive Download**](#option-2-release-archive-download)
  * [**3 - Clone Repo**](#option-3-clone-the-repo)

[**Gears: What are these?**](#gears)
  * [**CamGear**](#camgear)
  * [**PiGear**](#pigear)
  * [**VideoGear**](#videogear)
  * [**ScreenGear**](#screengear)
  * [**WriteGear**](#writegear)
  * [**NetGear**](#netgear)

**For Developers/Contributors**
  * [**Testing**](#testing)
  * [**Contributing**](#contibuting)

[**Documentation**](#documentation)

[**Project Motivation**](#project-motivation)

**Additional Info**
  * [**Supported Python legacies**](#supported-python-legacies)
  * [**Changelog**](#changelog)
  * [**License**](#license)

&nbsp;

## TL;DR
   *VidGear is an [ultrafast➶](https://github.com/abhiTronix/vidgear/wiki/FAQ-&-Troubleshooting#2-vidgear-is-ultrafast-but-how), compact, flexible and easy-to-adapt complete Video Processing Python Library.*

   Built with simplicity in mind, VidGear lets programmers and software developers to easily integrate and perform complex Video Processing tasks in their existing or new applications, without going through various underlying python library's documentation and using just a few lines of code. Beneficial for both, if you're new to Programming with Python language or pro at it. 

   For more advanced information see the [Wiki Documentation ➶][wiki].

&nbsp;

## New Release SneekPeak : VidGear 0.1.5
  * Released new ScreenGear API, supports Live ScreenCasting.
  * Released new NetGear API, aids real-time frame transfer through messaging(ZmQ) over network.
  * Released new Stabilizer Class, for minimum latency Video Stabilization with OpenCV.
  * Updated VideoGear API to be used as internal wrapper around Stabilizer Class.
  * Implemented exclusive Threaded Queue Mode for blazingly fast, synchronized and error-free multi-threading APIs.
  * Added Option to use VidGear API's standalone.
  * Several Performance enhancements and Bugs exterminated.
  * Revamped Docs and [many more...](changelog.md)

&nbsp;

## Installation

### Prerequisites

To use VidGear in your python application, you must have check the following dependencies before you install VidGear :

* Must support [these Python legacies](#supported-python-legacies) and [pip](https://pip.pypa.io/en/stable/installing/) already installed.


* **`OpenCV:`** VidGear must require OpenCV(3.0+) python enabled binaries to be installed on your machine for its core functions. For its installation, you can follow these online tutorials for [linux][OpenCV-linux] and [raspberry pi][OpenCV-pi], otherwise, install it via pip:

    ```sh
      pip install opencv-python
    ```

* **`FFmpeg:`** VidGear must requires FFmpeg for its powerful video compression and encoding capabilities. :star2: Follow this [**WiKi Page**](https://github.com/abhiTronix/vidgear/wiki/FFmpeg-Installation) for its installation. :star2:

* **`picamera:`** Required for using Raspberry Pi Camera Modules(such as OmniVision OV5647 Camera Module) on your Raspberry Pi machine. You can easily install it via pip:

    ```sh
      pip install picamera
    ``` 
  Also, make sure to enable Raspberry Pi hardware specific settings prior using this library.

* **`mss:`** Required for Screen Casting. Install it via pip:

    ```sh
      pip install mss
    ```
* **`pyzmq:`** Required for transferring video frames through ZeroMQ messaging system over the network. Install it via pip:

    ```sh
      pip install pyzmq
    ```

* **`pafy:`** For direct YouTube Video streaming, Vidgear needs `pafy` and latest `youtube-dl`(as pafy's backend) python libraries installed. Install it via pip:

    ```sh
      pip install pafy
      pip install -U youtube-dl
    ```
&nbsp;

### Option 1: PyPI Install

> Best option for **quickly** getting VidGear installed.

```sh
  pip install vidgear
```
&nbsp;

### Option 2: Release Archive Download

> Best option if you want an **compressed archive**.

VidGear releases are available for download as packages in the [latest release](https://github.com/abhiTronix/vidgear/releases/latest)

&nbsp;

### Option 3: Clone the Repo

> Best option for **automatically installing required dependencies**(_except FFmpeg_), or for **latest patches**(_maybe experimental_), or **contributing** to development.

You can clone this repository's `testing` branch for development and thereby can install as follows:
```sh
 git clone https://github.com/abhiTronix/vidgear.git
 cd vidgear
 git checkout testing
 pip install .
```

&nbsp;

## Gears:

VidGear is built with multi-threaded **Gears** each with some unique function/mechanism. Each **Gear** is designed exclusively to handle/control different device-specific video streams, network streams, and media encoders. These APIs provides an easy-to-use, highly extensible, and a multi-threaded wrapper around many underlying various python libraries to exploit their features and functions directly while providing robust error-handling. 

**These Gears can be classified as follows:**

**A. VideoCapture Gears:**

  * [**CamGear:**](#camgear) _Targets various IP-USB-Cameras/Network-Streams/YouTube-Video-URL._
  * [**PiGear:**](#pigear) _Targets various Raspberry Pi Camera Modules._
  * [**ScreenGear:**](#screengear) _Enables ultra-fast Screen Casting._    
  * [**VideoGear:**](#videogear) _A common API with Video Stabilizer wrapper._  

**B. VideoWriter Gear:**

  * [**WriteGear:**](#writegear) _Handles easy Lossless Video Encoding and Compression._

**C. Network Gear:**

  * [**NetGear:**](#netgear) _Targets synchronous video frames transferring between interconnecting systems over the network._

&nbsp;

### CamGear

CamGear supports a diverse range of video streams which can handle/control video stream almost any IP/USB Cameras, multimedia video file format, network stream URL such as `http(s), rtp, rstp, mms, etc.` In addition to this, it also supports live Gstreamer's RAW pipelines and YouTube video/livestreams URLs. CamGear provides a flexible, high-level multi-threaded wrapper around `OpenCV's` [VideoCapture class](https://docs.opencv.org/master/d8/dfe/classcv_1_1VideoCapture.html#a57c0e81e83e60f36c83027dc2a188e80) with access almost all of its available parameters and also employs `pafy's` APIs for live [YouTube streaming](https://github.com/abhiTronix/vidgear/wiki/CamGear-Class#2-camgear-class-with-live-youtube-piplineing-using-video-url). Furthermore, CamGear relies exclusively on [**Threaded Queue mode**](https://github.com/abhiTronix/vidgear/wiki/Threaded-Queue-Mode) for ultra-fast, error-free and synchronized frame handling.

**Following simplified functional block diagram depicts CamGear API's generalized working:**

<p align="center">
  <img src="https://github.com/abhiTronix/Imbakup/raw/master/Images/CamGear.png" alt="CamGear Functional Block Diagram" width=60%/>
</p>

**CamGear API Guide:**

[>>> Usage Guide](https://github.com/abhiTronix/vidgear/wiki/CamGear#camgear-api)

&nbsp;

### VideoGear

VideoGear API provides a special internal wrapper around VidGear's exclusive [**Video Stabilizer**](https://github.com/abhiTronix/vidgear/wiki/Stabilizer-Class) class. Furthermore, VideoGear API can provide internal access to both [CamGear](https://github.com/abhiTronix/vidgear/wiki/CamGear)) and [PiGear](https://github.com/abhiTronix/vidgear/wiki/PiGear) APIs separated by a special flag. Thereby, _this API holds the exclusive power for any incoming VideoStream from any source, whether it is live or not, to stabilize it directly with minimum latency and memory requirements._

**Below is a snapshot of a VideoGear Stabilizer in action:**

<p align="center">
  <img src="https://github.com/abhiTronix/Imbakup/raw/master/Images/stabilizer.gif" alt="VideoGear Stabilizer in action!" />
  <br>
  <sub><i>Original Video Courtesy<a href="http://liushuaicheng.org/SIGGRAPH2013/database.html" title="opensourced video samples database">@SIGGRAPH2013</a></i></sub>
</p>



Code to generate above VideoGear API Stabilized Video(_See more detailed usage examples [here](https://github.com/abhiTronix/vidgear/wiki/Real-time-Video-Stabilization#usage)_): 

```python
# import libraries
from vidgear.gears import VideoGear
import numpy as np
import cv2

stream_stab = VideoGear(source='test.mp4', stabilize = True).start() # To open any valid video stream with `stabilize` flag set to True.
stream_org = VideoGear(source='test.mp4').start() # open same stream without stabilization for comparison

# infinite loop
while True:
  
  frame_stab = stream_stab.read()
  # read stabilized frames

  # check if frame is None
  if frame_stab is None:
    #if True break the infinite loop
    break
  
  #read original frame
  frame_org = stream_org.read()

  #concatenate both frames
  output_frame = np.concatenate((frame_org, frame_stab), axis=1)

  #put text
  cv2.putText(output_frame, "Before", (10, output_frame.shape[0] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
  cv2.putText(output_frame, "After", (output_frame.shape[1]//2+10, frame.shape[0] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
  
  cv2.imshow("Stabilized Frame", output_frame)
  # Show output window

  key = cv2.waitKey(1) & 0xFF
  # check for 'q' key-press
  if key == ord("q"):
    #if 'q' key-pressed break out
    break

cv2.destroyAllWindows()
# close output window
stream_org.stop()
stream_stab.stop()
# safely close video streams.
```
 

**VideoGear API Guide:**

[>>> Usage Guide](https://github.com/abhiTronix/vidgear/wiki/VideoGear#videogear-api)

&nbsp;

### PiGear

PiGear is similar to CamGear but made to support various Raspberry Pi Camera Modules (such as [OmniVision OV5647 Camera Module](https://github.com/techyian/MMALSharp/wiki/OmniVision-OV5647-Camera-Module) and [Sony IMX219 Camera Module](https://github.com/techyian/MMALSharp/wiki/Sony-IMX219-Camera-Module)). To interface with these modules correctly, PiGear provide a flexible multi-threaded wrapper around complete [picamera](https://github.com/waveform80/picamera) python library, and provides us the ability to exploit its various features like `brightness, saturation, sensor_mode, etc.` effortlessly. 

**Following simplified functional block diagram depicts PiGear API:**

<p align="center">
  <img src="https://github.com/abhiTronix/Imbakup/raw/master/Images/PiGear.png" alt="PiGear Functional Block Diagram" width=40%/>
</p>

**PiGear API Guide:**

[>>> Usage Guide](https://github.com/abhiTronix/vidgear/wiki/PiGear#pigear-api)

&nbsp;

### ScreenGear

With ScreenGear, we can easily define an area on the computer screen or an open window to record the live screen frames in real-time at the expense of insignificant latency. To achieve this, ScreenGear provides a high-level multi-threaded wrapper around [**`mss`**](https://github.com/BoboTiG/python-mss) python library API and also supports the flexible direct parameter manipulation. 

**Below is a snapshot of a ScreenGear API in action:**

<p align="center">
  <img src="https://github.com/abhiTronix/Imbakup/raw/master/Images/screengear.gif" alt="ScreenGear in action!" />
</p>

Code to generate above result:

```python
# import libraries
from vidgear.gears import ScreenGear
import cv2

stream = ScreenGear().start()

# infinite loop
while True:
  
  frame = stream.read()
  # read frames

  # check if frame is None
  if frame is None:
    #if True break the infinite loop
    break
  
  cv2.imshow("Output Frame", frame)
  # Show output window

  key = cv2.waitKey(1) & 0xFF
  # check for 'q' key-press
  if key == ord("q"):
    #if 'q' key-pressed break out
    break

cv2.destroyAllWindows()
# close output window

stream.stop()
# safely close video stream.
```

**ScreenGear API Usage:**

[>>> Usage Guide](https://github.com/abhiTronix/vidgear/wiki/ScreenGear-Class#usage-hammer)


&nbsp;


### WriteGear

WriteGear is undoubtedly the most powerful Video Processing Gear of them all. It solely handles various powerful FFmpeg tools that allow us to do almost anything you can imagine with multimedia files. With WriteGear API, you can process real-time video frames into a lossless format and specification suitable for our playback in just a few lines of codes. These specifications include setting bitrate, codec, framerate, resolution, subtitles, compression, etc. Furthermore, we can multiplex extracted audio at the output with compression and all that in real-time(see this example). In addition to this, WriteGear also provides flexible access to OpenCV's VideoWriter API which provide some basic tools for video frames encoding but without compression.

**WriteGear primarily operates in the following two modes:**

  * **Compression Mode:** In this mode, WriteGear utilizes [**`FFmpeg's`**](https://ffmpeg.org/) inbuilt encoders to encode lossless multimedia files. It provides us the ability to exploit almost any available parameters available within FFmpeg, with so much ease and flexibility and while doing that it robustly handles all errors/warnings quietly. **You can find more about this mode [here](https://github.com/abhiTronix/vidgear/wiki/Compression-Mode:-FFmpeg)**.

  * **Non-Compression Mode:** In this mode, WriteGear utilizes basic OpenCV's inbuilt [**VideoWriter API**](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html). Similar to compression mode, WriteGear also supports all parameters manipulation available within OpenCV's VideoWriter API. But this mode lacks the ability to manipulate encoding parameters and other important features like video compression, audio encoding, etc. **You can learn about this mode [here](https://github.com/abhiTronix/vidgear/wiki/Non-Compression-Mode:-OpenCV)**.


**Following functional block diagram depicts WriteGear API's generalized working:**

<p align="center">
  <img src="https://github.com/abhiTronix/Imbakup/raw/master/Images/WriteGear.png" alt="WriteGear Functional Block Diagram" width=70%/>
</p>

**WriteGear API Guide:**

[>>> Usage Guide](https://github.com/abhiTronix/vidgear/wiki/WriteGear#writegear-api)

&nbsp;

### NetGear

NetGear is exclusively designed to transfer video frames synchronously between interconnecting systems over the network in real-time. This is achieved by implementing a high-level wrapper around [PyZmQ](https://github.com/zeromq/pyzmq) python library that contains python bindings for [ZeroMQ](http://zeromq.org/) - a high-performance asynchronous distributed messaging library that aim to be used in distributed or concurrent applications.  It provides a message queue, but unlike message-oriented middleware, a ZeroMQ system can run without a dedicated message broker. Furthermore, NetGear currently supports two ZeroMQ messaging patterns: i.e `zmq.PAIR` and `zmq.REQ and zmq.REP`.

**NetGear API has two modes of operations:**
  * **Send Mode:**(a.k.a Server configuration) which employs `send()` function to send frames to the client(s). You can use this function to send messages to client(s) too.
  * **Receive Mode:**(a.k.a Client configuration) which employs `recv()` function to receive frames sent by server. Client send back confirmation when frame is received successfully.


**Following functional block diagram depicts NetGear API:**

<p align="center">
  <img src="https://github.com/abhiTronix/Imbakup/raw/master/Images/NetGear.png" alt="NetGear Functional Block Diagram" width=80%/>
</p>

**NetGear API Guide:**

[>>> Usage Guide](https://github.com/abhiTronix/vidgear/wiki/NetGear#netgear-api)

&nbsp;


## Documentation

The full documentation for all VidGear classes and functions can be find in link below:

* [Wiki Documentation - English][wiki]

&nbsp;

## Testing

* **Prerequisites:** Testing VidGear require some *additional dependencies & data* which can be downloaded manually as follows:

  * **Clone & Install [Testing Branch](#option-3-clone-the-repo)**

  * **Download few additional python libraries:**
    ```sh
     pip install six
     pip install pytest
    ```
  
  * **Download Test Dataset:** To perform tests, additional *test dataset* is required, which can be downloaded by running [*bash script*](https://github.com/abhiTronix/vidgear/blob/testing/scripts/bash/prepare_dataset.sh) as follows:

    ```sh
     chmod +x scripts/prepare_dataset.sh
     ./scripts/prepare_dataset.sh               #for windows, use `sh scripts/pre_install.sh`
    ```

* **Run Tests:** Then various VidGear tests can be run with `pytest`(*in VidGear's root folder*) as below:

  ```sh
   pytest -sv                                   #-sv for verbose output.
  ```

&nbsp; 

## Contributing

See [contributing.md](contributing.md)

&nbsp; 

## Project Motivation

See [Wiki: Project Motivation][wiki-vidgear-purpose]

&nbsp;

## Supported Python legacies

  * **Python 2.7 legacies:** *VidGear v0.1.5 is officially the last Python 2.7 legacies supporting version.* Kindly migrate your source code to Python 3 as soon as possible.

  * **Python 3.x legacies:** follows the [numpy](https://github.com/numpy/numpy) releases.

&nbsp;

## Changelog

See [changelog.md](changelog.md)

&nbsp; 

## License

Copyright © abhiTronix 2019

This project is licensed under the [MIT][license] license.

<!--
Badges
-->

[appveyor]:https://img.shields.io/appveyor/ci/abhitronix/vidgear.svg?style=popout-square&logo=appveyor
[travis-cli]:https://img.shields.io/travis/abhiTronix/vidgear.svg?style=popout-square&logo=travis
[prs-badge]:https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=popout-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAABC0lEQVRYhdWVPQoCMRCFX6HY2ghaiZUXsLW0EDyBrbWtN/EUHsHTWFnYyCL4gxibVZZlZzKTnWz0QZpk5r0vIdkF/kBPAMOKeddE+CQPKoc5Yt5cTjBMdQSwDQToWgBJAn3jmhqgltapAV6E6b5U17MGGAUaUj07TficMfIBZDV6vxowBm1BP9WbSQE4o5h9IjPJmy73TEPDDxVmoZdQrQ5jRhly9Q8tgMUXkIIWn0oG4GYQfAXQzz1PGoCiQndM7b4RgJay/h7zBLT3hASgoKjamQJMreKf0gfuAGyYtXEIAKcL/Dss15iq6ohXghozLYiAMxPuACwtIT4yeQUxAaLrZwAoqGRKGk7qDSYTfYQ8LuYnAAAAAElFTkSuQmCC
[twitter-badge]:https://img.shields.io/twitter/url/http/shields.io.svg?style=popout-square&logo=twitter
[pypi-badge]:https://img.shields.io/pypi/v/vidgear.svg?style=popout-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAABC0lEQVRYhdWVPQoCMRCFX6HY2ghaiZUXsLW0EDyBrbWtN/EUHsHTWFnYyCL4gxibVZZlZzKTnWz0QZpk5r0vIdkF/kBPAMOKeddE+CQPKoc5Yt5cTjBMdQSwDQToWgBJAn3jmhqgltapAV6E6b5U17MGGAUaUj07TficMfIBZDV6vxowBm1BP9WbSQE4o5h9IjPJmy73TEPDDxVmoZdQrQ5jRhly9Q8tgMUXkIIWn0oG4GYQfAXQzz1PGoCiQndM7b4RgJay/h7zBLT3hASgoKjamQJMreKf0gfuAGyYtXEIAKcL/Dss15iq6ohXghozLYiAMxPuACwtIT4yeQUxAaLrZwAoqGRKGk7qDSYTfYQ8LuYnAAAAAElFTkSuQmCC
[Thank-you]:https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg?style=popout-square&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz48c3ZnIGlkPSJzdmcyIiB3aWR0aD0iNjQ1IiBoZWlnaHQ9IjU4NSIgdmVyc2lvbj0iMS4wIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPiA8ZyBpZD0ibGF5ZXIxIj4gIDxwYXRoIGlkPSJwYXRoMjQxNyIgZD0ibTI5Ny4zIDU1MC44N2MtMTMuNzc1LTE1LjQzNi00OC4xNzEtNDUuNTMtNzYuNDM1LTY2Ljg3NC04My43NDQtNjMuMjQyLTk1LjE0Mi03Mi4zOTQtMTI5LjE0LTEwMy43LTYyLjY4NS01Ny43Mi04OS4zMDYtMTE1LjcxLTg5LjIxNC0xOTQuMzQgMC4wNDQ1MTItMzguMzg0IDIuNjYwOC01My4xNzIgMTMuNDEtNzUuNzk3IDE4LjIzNy0zOC4zODYgNDUuMS02Ni45MDkgNzkuNDQ1LTg0LjM1NSAyNC4zMjUtMTIuMzU2IDM2LjMyMy0xNy44NDUgNzYuOTQ0LTE4LjA3IDQyLjQ5My0wLjIzNDgzIDUxLjQzOSA0LjcxOTcgNzYuNDM1IDE4LjQ1MiAzMC40MjUgMTYuNzE0IDYxLjc0IDUyLjQzNiA2OC4yMTMgNzcuODExbDMuOTk4MSAxNS42NzIgOS44NTk2LTIxLjU4NWM1NS43MTYtMTIxLjk3IDIzMy42LTEyMC4xNSAyOTUuNSAzLjAzMTYgMTkuNjM4IDM5LjA3NiAyMS43OTQgMTIyLjUxIDQuMzgwMSAxNjkuNTEtMjIuNzE1IDYxLjMwOS02NS4zOCAxMDguMDUtMTY0LjAxIDE3OS42OC02NC42ODEgNDYuOTc0LTEzNy44OCAxMTguMDUtMTQyLjk4IDEyOC4wMy01LjkxNTUgMTEuNTg4LTAuMjgyMTYgMS44MTU5LTI2LjQwOC0yNy40NjF6IiBmaWxsPSIjZGQ1MDRmIi8%2BIDwvZz48L3N2Zz4%3D
[Coffee-badge]:https://abhitronix.github.io/img/vidgear/orange_img.png
[wiki-vidgear-purpose]:https://github.com/abhiTronix/vidgear/wiki/Project-Motivation#why-is-vidgear-a-thing

<!--
Internal URLs
-->

[release]:https://github.com/abhiTronix/vidgear/releases
[wiki]:https://github.com/abhiTronix/vidgear/wiki
[prs]:http://makeapullrequest.com "Make a Pull Request (external link) ➶"
[travis]:https://travis-ci.org/abhiTronix/vidgear
[app]:https://ci.appveyor.com/project/abhiTronix/vidgear
[pypi]:https://pypi.org/project/vidgear/
[thanks]:https://saythanks.io/to/abhiTronix
[coffee]:https://www.buymeacoffee.com/2twOXFvlA
[twitter-intent]:https://twitter.com/intent/tweet?url=https%3A%2F%2Fgithub.com%2FabhiTronix%2Fvidgear&via%20%40abhi_una12&text=VidGear%20-%20A%20simple%2C%20powerful%2C%20flexible%20%26%20threaded%20Python%20Video%20Processing%20Library&hashtags=vidgear%20%23multithreaded%20%23python%20%23video-processing%20%23github
[license]:https://github.com/abhiTronix/vidgear/blob/master/LICENSE

<!--
External URLs
-->

[OpenCV-linux]:https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/
[OpenCV-pi]:https://www.pyimagesearch.com/2018/09/26/install-opencv-4-on-your-raspberry-pi/