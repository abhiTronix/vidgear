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

<h1 align="center">
  <img src="https://abhitronix.github.io/img/vidgear/vidgear logo.svg" alt="VidGear Logo" width="70%"/>
</h1>
<h2 align="center">
  <img src="https://abhitronix.github.io/img/vidgear/vidgear banner.svg" alt="VidGear Banner" width="40%"/>
</h2>

<div align="center">

[Releases][release]&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Gears](#gears)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Wiki Documentation][wiki]&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Installation](#installation)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[License](#license)

[![Build Status][travis-cli]][travis] [![Codecov branch][codecov]][code] [![Build Status][appveyor]][app]

[![PyPi version][pypi-badge]][pypi] [![Say Thank you][Thank-you]][thanks] [![Twitter][twitter-badge]][twitter-intent] 

[![Buy Me A Coffee][Coffee-badge]][coffee]

</div>

&nbsp;

VidGear is a powerful python Video Processing library built with multi-threaded [**Gears**](#gears) each with a unique set of trailblazing features. These APIs provides a easy-to-use, highly extensible, and multi-threaded wrapper around many underlying state-of-the-art libraries such as *[OpenCV ➶][opencv], [FFmpeg ➶][ffmpeg], [picamera ➶][picamera], [pafy ➶][pafy], [pyzmq ➶][pyzmq] and [python-mss ➶][mss]*

&nbsp;

The following **functional block diagram** clearly depicts the functioning of VidGear library:

<p align="center">
  <img src="https://abhitronix.github.io/img/vidgear/vidgear_function2-01.svg" alt="@Vidgear Functional Block Diagram" />
</p>

&nbsp;

# Table of Contents

[**TL;DR**](#tldr)

[**Gears: What are these?**](#gears)
  * [**CamGear**](#camgear)
  * [**PiGear**](#pigear)
  * [**VideoGear**](#videogear)
  * [**ScreenGear**](#screengear)
  * [**WriteGear**](#writegear)
  * [**NetGear**](#netgear)

[**Installation**](#installation)
  * [**Prerequisites**](#prerequisites)
  * [**1 - PyPI Install**](#option-1-pypi-install)
  * [**2 - Release Archive Download**](#option-2-release-archive-download)
  * [**3 - Clone Repo**](#option-3-clone-the-repo)

[**New-Release SneekPeak: v0.1.6**](#new-release-sneekpeak--vidgear-016)

[**Documentation**](#documentation)

**For Developers/Contributors**
  * [**Testing**](#testing)
  * [**Contributing**](#contributing)

**Additional Info**
  * [**Supported Python legacies**](#supported-python-legacies)
  * [**Changelog**](#changelog)
  * [**Citing**](#citing)
  * [**License**](#license)


&nbsp;


# TL;DR
  
  #### What is vidgear?

   > ***"VidGear is an [ultrafast➶][ultrafast-wiki], compact, flexible and easy-to-adapt complete Video Processing Python Library."***

  #### What does it do?
   > ***"VidGear can read, write, process, send & receive video frames from various devices in real-time."***

  #### What is its purpose?
   > ***"Built with simplicity in mind, VidGear lets programmers and software developers to easily integrate and perform complex Video Processing tasks in their existing or new applications, without going through various underlying library's documentation and using just a few lines of code. Beneficial for both, if you're new to programming with Python language or already a pro at it."***

   **For more advanced information, see the [*Wiki Documentation ➶*][wiki].**


&nbsp;


# Gears

> **VidGear is built with various **Multi-Threaded APIs** *(a.k.a Gears)* each with some unique function/mechanism.**

Each of these API is designed exclusively to handle/control different device-specific video streams, network streams, and media encoders. These APIs provides an easy-to-use, highly extensible, and a multi-threaded wrapper around various underlying libraries to exploit their features and functions directly while providing robust error-handling. 

**These Gears can be classified as follows:**

**A. VideoCapture Gears:**

  * [**CamGear:**](#camgear) _Targets various IP-USB-Cameras/Network-Streams/YouTube-Video-URL._
  * [**PiGear:**](#pigear) _Targets various Raspberry Pi Camera Modules._
  * [**ScreenGear:**](#screengear) _Enables ultra-fast Screen Casting._    
  * [**VideoGear:**](#videogear) _A common API with Video Stabilizer wrapper._  

**B. VideoWriter Gear:**

  * [**WriteGear:**](#writegear) _Handles easy Lossless Video Encoding and Compression._

**C. Network Gear:**

  * [**NetGear:**](#netgear) _Targets synchronous/asynchronous video frames transferring between interconnecting systems over the network._

&nbsp;

## CamGear

> *CamGear can grab ultra-fast frames from diverse range of devices/streams, which includes almost any IP/USB Cameras, multimedia video file format ([_upto 4k tested_][test-4k]), various network stream protocols such as `http(s), rtp, rstp, rtmp, mms, etc.`, plus support for live Gstreamer's stream pipeline and YouTube video/livestreams URLs.*

CamGear provides a flexible, high-level multi-threaded wrapper around `OpenCV's` [VideoCapture class][opencv-vc] with access almost all of its available parameters and also employs [`pafy`][pafy] python APIs for live [YouTube streaming][youtube-wiki]. Furthermore, CamGear implements exclusively on [**Threaded Queue mode**][TQM-wiki] for ultra-fast, error-free and synchronized frame handling.


**Following simplified functional block diagram depicts CamGear API's generalized working:**

<p align="center">
  <img src="https://github.com/abhiTronix/Imbakup/raw/master/Images/CamGear.png" alt="CamGear Functional Block Diagram" width=60%/>
</p>

### CamGear API Guide:

[**>>> Usage Guide**][camgear-wiki]

&nbsp;

## VideoGear

> *VideoGear API provides a special internal wrapper around VidGear's exclusive [**Video Stabilizer**][stablizer-wiki] class.*

Furthermore, VideoGear API can provide internal access to both [CamGear](#camgear) and [PiGear](#pigear) APIs separated by a special flag. Thereby, _this API holds the exclusive power for any incoming VideoStream from any source, whether it is live or not, to access and stabilize it directly with minimum latency and memory requirements._

**Below is a snapshot of a VideoGear Stabilizer in action:**

<p align="center">
  <img src="https://github.com/abhiTronix/Imbakup/raw/master/Images/stabilizer.gif" alt="VideoGear Stabilizer in action!" />
  <br>
  <sub><i>Original Video Courtesy <a href="http://liushuaicheng.org/SIGGRAPH2013/database.html" title="opensourced video samples database">@SIGGRAPH2013</a></i></sub>
</p>

Code to generate above VideoGear API Stabilized Video(_See more detailed usage examples [here][stablizer-wiki-ex]_): 

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
 

### VideoGear API Guide:

[**>>> Usage Guide**][videogear-wiki]

&nbsp;

## PiGear

> *PiGear is similar to CamGear but made to support various Raspberry Pi Camera Modules *(such as [OmniVision OV5647 Camera Module][OV5647-picam] and [Sony IMX219 Camera Module][IMX219-picam])*.*

PiGear provides a flexible multi-threaded wrapper around complete [**picamera**][picamera] python library to interface with these modules correctly, and also grants the ability to exploit its various features like `brightness, saturation, sensor_mode, etc.` effortlessly. 

Best of all, PiGear API provides excellent Error-Handling with features like a threaded internal timer that keeps active track of any frozen threads and handles hardware failures/frozen threads robustly thereby will exit safely if any failure occurs. So now if you accidently pulled your camera module cable out when running PiGear API in your script, instead of going into possible kernel panic/frozen threads, API exit safely to save resources. 

**Following simplified functional block diagram depicts PiGear API:**

<p align="center">
  <img src="https://github.com/abhiTronix/Imbakup/raw/master/Images/PiGear.png" alt="PiGear Functional Block Diagram" width=40%/>
</p>

### PiGear API Guide:

[**>>> Usage Guide**][pigear-wiki]

&nbsp;

## ScreenGear

> *ScreenGear API act as Screen Recorder, that can grab frames from your monitor in real-time either by define an area on the computer screen or fullscreen at the expense of insignificant latency. It also provide seemless support for capturing frames from multiple monitors.*

ScreenGear provides a high-level multi-threaded wrapper around [**python-mss**][mss] python library API and also supports a easy and flexible direct internal parameter manipulation. 

**Below is a snapshot of a ScreenGear API in action:**

<p align="center">
  <img src="https://github.com/abhiTronix/Imbakup/raw/master/Images/screengear.gif" alt="ScreenGear in action!" />
</p>

Code to generate the above result:

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

### ScreenGear API Guide:

[**>>> Usage Guide**][screengear-wiki]


&nbsp;


## WriteGear

> *WriteGear handles various powerful Writer Tools that provide us the freedom to do almost anything imagine with multimedia files.*

WriteGear API provide a complete, flexible & robust wrapper around [**FFmpeg**][ffmpeg], a leading multimedia framework. With WriteGear, we can process real-time video frames into a lossless compressed format with any suitable specification in just few easy [lines of codes][compression-mode-ex]. These specifications include setting any video/audio property such as `bitrate, codec, framerate, resolution, subtitles,  etc.` easily as well complex tasks such as multiplexing video with audio in real-time(see this [example wiki][live-audio-wiki]). Best of all, WriteGear grants the freedom to play with any FFmpeg parameter with its exclusive custom Command function(see this [example wiki][custom-command-wiki]), while handling all errors robustly. 

In addition to this, WriteGear also provides flexible access to [**OpenCV's VideoWriter API**][opencv-writer] which provides some basic tools for video frames encoding but without compression.

**WriteGear primarily operates in the following two modes:**

  * **Compression Mode:** In this mode, WriteGear utilizes [**`FFmpeg's`**][ffmpeg] inbuilt encoders to encode lossless multimedia files. It provides us the ability to exploit almost any available parameters available within FFmpeg, with so much ease and flexibility and while doing that it robustly handles all errors/warnings quietly. **You can find more about this mode [here][cm-writegear-wiki]**.

  * **Non-Compression Mode:** In this mode, WriteGear utilizes basic OpenCV's inbuilt [**VideoWriter API**][opencv-vw]. Similar to compression mode, WriteGear also supports all parameters manipulation available within OpenCV's VideoWriter API. But this mode lacks the ability to manipulate encoding parameters and other important features like video compression, audio encoding, etc. **You can learn about this mode [here][ncm-writegear-wiki]**.

**Following functional block diagram depicts WriteGear API's generalized working:**

<p align="center">
  <img src="https://github.com/abhiTronix/Imbakup/raw/master/Images/WriteGear.png" alt="WriteGear Functional Block Diagram" width=70%/>
</p>

### WriteGear API Guide:

[**>>> Usage Guide**][writegear-wiki]

&nbsp;

## NetGear

> *NetGear is exclusively designed to transfer video frames synchronously and asynchronously between interconnecting systems over the network in real-time.*

NetGear implements a high-level wrapper around [**PyZmQ**][pyzmq] python library that contains python bindings for [ZeroMQ](http://zeromq.org/) - a high-performance asynchronous distributed messaging library that aim to be used in distributed or concurrent applications. It provides a message queue, but unlike message-oriented middleware, a ZeroMQ system can run without a dedicated message broker. 

NetGear provides seamless support for bidirectional data transmission between receiver(client) and sender(server) through bi-directional synchronous messaging patterns such as zmq.PAIR _(ZMQ Pair Pattern)_ & zmq.REQ/zmq.REP _(ZMQ Request/Reply Pattern)_. 

NetGear also supports real-time frame Encoding/Decoding compression capabilities for optimizing performance while sending the frames directly over the network, by encoding the frame before sending it and decoding it on the client's end automatically in real-time. 

For security, NetGear implements easy access to ZeroMQ's powerful, smart & secure Security Layers, that enables strong encryption on data, and unbreakable authentication between the Server and the Client with the help of custom certificates/keys and brings easy, standardized privacy and authentication for distributed systems over the network. 

Best of all, NetGear can robustly handle Multiple Servers devices at once, thereby providing access to seamless Live Streaming of the multiple device in a network at the same time.


**NetGear as of now seamlessly supports three ZeroMQ messaging patterns:**

* [**`zmq.PAIR`**][zmq-pair] _(ZMQ Pair Pattern)_ 
* [**`zmq.REQ/zmq.REP`**][zmq-req-rep] _(ZMQ Request/Reply Pattern)_
* [**`zmq.PUB/zmq.SUB`**][zmq-pub-sub] _(ZMQ Publish/Subscribe Pattern)_


**Following functional block diagram depicts generalized functioning of NetGear API:**

<p align="center">
  <img src="https://github.com/abhiTronix/Imbakup/raw/master/Images/NetGear.png" alt="NetGear Functional Block Diagram" width=80%/>
</p>

### NetGear API Guide:

[**>>> Usage Guide**][netgear-wiki]


&nbsp;


# New Release SneekPeak : VidGear 0.1.6

* ***:warning: Python 2.7 legacy support [dropped in v0.1.6][drop27] !***

* **NetGear API:**
  * Added powerful ZMQ Authentication & Data Encryption features for NetGear API
  * Added robust Multi-Server support for NetGear API.
  * Added exclusive Bi-Directional Mode for bidirectional data transmission.
  * Added frame-compression support with on-the-fly flexible encoding/decoding.
  * Implemented new *Publish/Subscribe(`zmq.PUB/zmq.SUB`)* pattern for seamless Live Streaming in NetGear API.

* **PiGear API:**
  * Added new threaded internal timing function for PiGear to handle any hardware failures/frozen threads
  * PiGear will not exit safely with `SystemError` if Picamera ribbon cable is pulled out to save resources.

* **WriteGear API:** Added new `execute_ffmpeg_cmd` function to pass a custom command to its internal FFmpeg pipeline.

* **Stabilizer class:** Added new _Crop and Zoom_ feature.

* ***Added VidGear's official native support for MacOS environment and [many more...](changelog.md)***



&nbsp;




# Installation

## Prerequisites:

Before installing VidGear, you must verify that the following dependencies are met:

* :warning: Must be using only [**supported Python legacies**](#supported-python-legacies) and also [**pip**][pip] already installed and configured.


* **`OpenCV:`** VidGear must require OpenCV(3.0+) python enabled binaries to be installed on your machine for its core functions. For its installation, you can follow these online tutorials for [linux][OpenCV-linux] and [raspberry pi][OpenCV-pi], otherwise, install it via pip:

    ```sh
      pip3 install -U opencv-python       #or install opencv-contrib-python similarly
    ```

* **`FFmpeg:`** VidGear must require FFmpeg for its powerful video compression and encoding capabilities. :star2: Follow this [**FFmpeg wiki page**][ffmpeg-wiki] for its installation. :star2:

* **`picamera:`** Required if using Raspberry Pi Camera Modules(_such as OmniVision OV5647 Camera Module_) with your Raspberry Pi machine. You can easily install it via pip:

    ```sh
      pip3 install picamera
    ``` 
  Also, make sure to enable Raspberry Pi hardware-specific settings prior to using this library.

* **`mss:`** Required for using Screen Casting. Install it via pip:

    ```sh
      pip3 install mss
    ```
* **`pyzmq:`** Required for transferring live video frames through _ZeroMQ messaging system_ over the network. Install it via pip:

    ```sh
      pip3 install pyzmq
    ```

* **`pafy:`** Required for direct YouTube Video streaming capabilities. Both [`pafy`][pafy] and latest only [`youtube-dl`][yt-dl](_as pafy's backend_) libraries must be installed via pip as follows:

    ```sh
      pip3 install pafy
      pip3 install -U youtube-dl
    ```

&nbsp;

## Available Installation Options:

### Option 1: PyPI Install

> Best option for **quickly** getting VidGear installed.

```sh
  pip3 install vidgear
```


### Option 2: Release Archive Download

> Best option if you want a **compressed archive**.

VidGear releases are available for download as packages in the [latest release][release].



### Option 3: Clone the Repository

> Best option for trying **latest patches(_maybe experimental_), Pull Requests**, or **contributing** to development.

You can clone this repository's `testing` branch for development and thereby can install as follows:
```sh
 git clone https://github.com/abhiTronix/vidgear.git
 cd vidgear
 git checkout testing
 sudo pip3 install .
```


&nbsp;



# Documentation

The full documentation for all VidGear classes and functions can be found in the link below:

* [Wiki Documentation - English][wiki]

&nbsp;

# Testing

* **Prerequisites:** Testing VidGear require some *additional dependencies & data* which can be downloaded manually as follows:

  * **Clone & Install [Testing Branch](#option-3-clone-the-repo)**

  * **Download few additional python libraries:**
    ```sh
     pip3 install six
     pip3 install pytest
    ```
  
  * **Download Test Dataset:** To perform tests, additional *test dataset* is required, which can be downloaded *(to temp dir)* by running [*bash script*][bs_script_dataset] as follows:

    ```sh
     chmod +x scripts/bash/prepare_dataset.sh
     .scripts/bash/prepare_dataset.sh               #for Windows, use `sh scripts/bash/prepare_dataset.sh`
    ```

* **Run Tests:** Then various VidGear tests can be run with `pytest`(*in VidGear's root folder*) as below:

  ```sh
   pytest -sv                                   #-sv for verbose output.
  ```

&nbsp; 

# Contributing

See [**contributing.md**](contributing.md).

&nbsp;

# Supported Python legacies

  * **Python 3+ are only supported legacies for installing Vidgear v0.1.6 and above.**
  * **:warning: Python 2.7 legacy support [dropped in v0.1.6][drop27].**

&nbsp;

# Changelog

See [**changelog.md**](changelog.md)

&nbsp;

# Citing

**Here is a Bibtex entry you can use to cite this project in a publication:**

```tex
@misc{vidgear,
    Title = {vidgear},
    Author = {Abhishek Thakur},
    howpublished = {\url{https://github.com/abhiTronix/vidgear}}   
  }
```

&nbsp;  

# License

**Copyright © abhiTronix 2019**

This library is licensed under the **[Apache 2.0 License][license]**.




<!--
Badges
-->

[appveyor]:https://img.shields.io/appveyor/ci/abhitronix/vidgear.svg?style=for-the-badge&logo=appveyor
[codecov]:https://img.shields.io/codecov/c/github/abhiTronix/vidgear/testing?style=for-the-badge&logo=codecov
[travis-cli]:https://img.shields.io/travis/abhiTronix/vidgear.svg?style=for-the-badge&logo=travis
[prs-badge]:https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAABC0lEQVRYhdWVPQoCMRCFX6HY2ghaiZUXsLW0EDyBrbWtN/EUHsHTWFnYyCL4gxibVZZlZzKTnWz0QZpk5r0vIdkF/kBPAMOKeddE+CQPKoc5Yt5cTjBMdQSwDQToWgBJAn3jmhqgltapAV6E6b5U17MGGAUaUj07TficMfIBZDV6vxowBm1BP9WbSQE4o5h9IjPJmy73TEPDDxVmoZdQrQ5jRhly9Q8tgMUXkIIWn0oG4GYQfAXQzz1PGoCiQndM7b4RgJay/h7zBLT3hASgoKjamQJMreKf0gfuAGyYtXEIAKcL/Dss15iq6ohXghozLYiAMxPuACwtIT4yeQUxAaLrZwAoqGRKGk7qDSYTfYQ8LuYnAAAAAElFTkSuQmCC
[twitter-badge]:https://img.shields.io/twitter/url/http/shields.io.svg?style=for-the-badge&logo=twitter
[pypi-badge]:https://img.shields.io/pypi/v/vidgear.svg?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAABC0lEQVRYhdWVPQoCMRCFX6HY2ghaiZUXsLW0EDyBrbWtN/EUHsHTWFnYyCL4gxibVZZlZzKTnWz0QZpk5r0vIdkF/kBPAMOKeddE+CQPKoc5Yt5cTjBMdQSwDQToWgBJAn3jmhqgltapAV6E6b5U17MGGAUaUj07TficMfIBZDV6vxowBm1BP9WbSQE4o5h9IjPJmy73TEPDDxVmoZdQrQ5jRhly9Q8tgMUXkIIWn0oG4GYQfAXQzz1PGoCiQndM7b4RgJay/h7zBLT3hASgoKjamQJMreKf0gfuAGyYtXEIAKcL/Dss15iq6ohXghozLYiAMxPuACwtIT4yeQUxAaLrZwAoqGRKGk7qDSYTfYQ8LuYnAAAAAElFTkSuQmCC
[Thank-you]:https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg?style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz48c3ZnIGlkPSJzdmcyIiB3aWR0aD0iNjQ1IiBoZWlnaHQ9IjU4NSIgdmVyc2lvbj0iMS4wIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPiA8ZyBpZD0ibGF5ZXIxIj4gIDxwYXRoIGlkPSJwYXRoMjQxNyIgZD0ibTI5Ny4zIDU1MC44N2MtMTMuNzc1LTE1LjQzNi00OC4xNzEtNDUuNTMtNzYuNDM1LTY2Ljg3NC04My43NDQtNjMuMjQyLTk1LjE0Mi03Mi4zOTQtMTI5LjE0LTEwMy43LTYyLjY4NS01Ny43Mi04OS4zMDYtMTE1LjcxLTg5LjIxNC0xOTQuMzQgMC4wNDQ1MTItMzguMzg0IDIuNjYwOC01My4xNzIgMTMuNDEtNzUuNzk3IDE4LjIzNy0zOC4zODYgNDUuMS02Ni45MDkgNzkuNDQ1LTg0LjM1NSAyNC4zMjUtMTIuMzU2IDM2LjMyMy0xNy44NDUgNzYuOTQ0LTE4LjA3IDQyLjQ5My0wLjIzNDgzIDUxLjQzOSA0LjcxOTcgNzYuNDM1IDE4LjQ1MiAzMC40MjUgMTYuNzE0IDYxLjc0IDUyLjQzNiA2OC4yMTMgNzcuODExbDMuOTk4MSAxNS42NzIgOS44NTk2LTIxLjU4NWM1NS43MTYtMTIxLjk3IDIzMy42LTEyMC4xNSAyOTUuNSAzLjAzMTYgMTkuNjM4IDM5LjA3NiAyMS43OTQgMTIyLjUxIDQuMzgwMSAxNjkuNTEtMjIuNzE1IDYxLjMwOS02NS4zOCAxMDguMDUtMTY0LjAxIDE3OS42OC02NC42ODEgNDYuOTc0LTEzNy44OCAxMTguMDUtMTQyLjk4IDEyOC4wMy01LjkxNTUgMTEuNTg4LTAuMjgyMTYgMS44MTU5LTI2LjQwOC0yNy40NjF6IiBmaWxsPSIjZGQ1MDRmIi8%2BIDwvZz48L3N2Zz4%3D
[Coffee-badge]:https://abhitronix.github.io/img/vidgear/orange_img.png

<!--
Internal URLs
-->

[release]:https://github.com/abhiTronix/vidgear/releases/latest
[pypi]:https://pypi.org/project/vidgear/
[thanks]:https://saythanks.io/to/abhiTronix
[twitter-intent]:https://twitter.com/intent/tweet?url=https%3A%2F%2Fgithub.com%2FabhiTronix%2Fvidgear&via%20%40abhi_una12&text=VidGear%20-%20A%20simple%2C%20powerful%2C%20flexible%20%26%20threaded%20Python%20Video%20Processing%20Library&hashtags=vidgear%20%23multithreaded%20%23python%20%23video-processing%20%23github
[coffee]:https://www.buymeacoffee.com/2twOXFvlA
[license]:https://github.com/abhiTronix/vidgear/blob/master/LICENSE
[travis]:https://travis-ci.org/abhiTronix/vidgear
[app]:https://ci.appveyor.com/project/abhiTronix/vidgear
[code]:https://codecov.io/gh/abhiTronix/vidgear

[test-4k]:https://github.com/abhiTronix/vidgear/blob/e0843720202b0921d1c26e2ce5b11fadefbec892/vidgear/tests/benchmark_tests/test_benchmark_playback.py#L65
[bs_script_dataset]:https://github.com/abhiTronix/vidgear/blob/testing/scripts/bash/prepare_dataset.sh

[wiki]:https://github.com/abhiTronix/vidgear/wiki
[wiki-vidgear-purpose]:https://github.com/abhiTronix/vidgear/wiki/Project-Motivation#why-is-vidgear-a-thing
[ultrafast-wiki]:https://github.com/abhiTronix/vidgear/wiki/FAQ-&-Troubleshooting#2-vidgear-is-ultrafast-but-how
[compression-mode-ex]:https://github.com/abhiTronix/vidgear/wiki/Compression-Mode:-FFmpeg#1-writegear-bare-minimum-examplecompression-mode
[live-audio-wiki]:https://github.com/abhiTronix/vidgear/wiki/Working-with-Audio#a-live-audio-input-to-writegear-class
[ffmpeg-wiki]:https://github.com/abhiTronix/vidgear/wiki/FFmpeg-Installation
[youtube-wiki]:https://github.com/abhiTronix/vidgear/wiki/CamGear#2-camgear-api-with-live-youtube-piplineing-using-video-url
[TQM-wiki]:https://github.com/abhiTronix/vidgear/wiki/Threaded-Queue-Mode
[camgear-wiki]:https://github.com/abhiTronix/vidgear/wiki/CamGear#camgear-api
[stablizer-wiki]:https://github.com/abhiTronix/vidgear/wiki/Stabilizer-Class
[stablizer-wiki-ex]:https://github.com/abhiTronix/vidgear/wiki/Real-time-Video-Stabilization#usage
[videogear-wiki]:https://github.com/abhiTronix/vidgear/wiki/VideoGear#videogear-api
[pigear-wiki]:https://github.com/abhiTronix/vidgear/wiki/PiGear#pigear-api
[cm-writegear-wiki]:https://github.com/abhiTronix/vidgear/wiki/Compression-Mode:-FFmpeg
[ncm-writegear-wiki]:https://github.com/abhiTronix/vidgear/wiki/Non-Compression-Mode:-OpenCV
[screengear-wiki]:https://github.com/abhiTronix/vidgear/wiki/ScreenGear#screengear-api
[writegear-wiki]:https://github.com/abhiTronix/vidgear/wiki/WriteGear#writegear-api
[netgear-wiki]:https://github.com/abhiTronix/vidgear/wiki/NetGear#netgear-api
[drop27]:https://github.com/abhiTronix/vidgear/issues/29
[custom-command-wiki]:https://github.com/abhiTronix/vidgear/wiki/Custom-FFmpeg-Commands-in-WriteGear-API#custom-ffmpeg-commands-in-writegear-api

<!--
External URLs
-->
[ffmpeg]:https://www.ffmpeg.org/
[opencv-writer]:https://docs.opencv.org/master/dd/d9e/classcv_1_1VideoWriter.html#ad59c61d8881ba2b2da22cff5487465b5
[OpenCV-linux]:https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/
[OpenCV-pi]:https://www.pyimagesearch.com/2018/09/26/install-opencv-4-on-your-raspberry-pi/
[prs]:http://makeapullrequest.com
[opencv]:https://github.com/opencv/opencv
[picamera]:https://github.com/waveform80/picamera
[pafy]:https://github.com/mps-youtube/pafy
[pyzmq]:https://github.com/zeromq/pyzmq
[mss]:https://github.com/BoboTiG/python-mss
[pip]:https://pip.pypa.io/en/stable/installing/
[opencv-vc]:https://docs.opencv.org/master/d8/dfe/classcv_1_1VideoCapture.html#a57c0e81e83e60f36c83027dc2a188e80
[OV5647-picam]:https://github.com/techyian/MMALSharp/wiki/OmniVision-OV5647-Camera-Module
[IMX219-picam]:https://github.com/techyian/MMALSharp/wiki/Sony-IMX219-Camera-Module
[opencv-vw]:https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html
[yt-dl]:https://github.com/ytdl-org/youtube-dl/
[numpy]:https://github.com/numpy/numpy
[zmq-pair]:https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/pair.html
[zmq-req-rep]:https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/client_server.html
[zmq-pub-sub]:https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/pubsub.html