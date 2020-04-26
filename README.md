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

[![Glitter chat][gitter-bagde]][gitter] [![PyPi version][pypi-badge]][pypi] [![Twitter][twitter-badge]][twitter-intent]

[![Buy Me A Coffee][Coffee-badge]][coffee]

</div>

&nbsp;

VidGear is a powerful python Video Processing library built with multiple APIs *(a.k.a [**Gears**](#gears))* each with a unique set of trailblazing features. These APIs provides an easy-to-use, highly extensible, multi-threaded & asyncio wrapper around many underlying state-of-the-art libraries such as *[OpenCV ➶][opencv], [FFmpeg ➶][ffmpeg], [ZeroMQ ➶][zmq], [picamera ➶][picamera], [starlette ➶][starlette], [pafy ➶][pafy] and [python-mss ➶][mss]*

&nbsp;

The following **functional block diagram** clearly depicts the generalized functioning of VidGear library:

<p align="center">
  <img src="https://abhitronix.github.io/img/vidgear/vidgear_function2020_3.svg" alt="@Vidgear Functional Block Diagram" />
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
* [**WebGear**](#webgear)
* [**NetGear_Async**](#netgear_async)

[**New-Release SneekPeak: v0.1.7**](#new-release-sneekpeak--vidgear-017)

[**Documentation**](#documentation)

[**Installation**](#installation)
* [**Prerequisites**](#prerequisites)
  * [**Supported Systems**](#supported-systems)
  * [**Supported Python legacies**](#supported-python-legacies)
  * [**Pip Dependencies**](#pip-dependencies)
* [**Available Installation Options**](#available-installation-options)
  * [**PyPI Install**](#option-1-pypi-installrecommended)
  * [**Release Download**](#option-2-release-download)
  * [**Build from source**](#option-3-build-from-source)
  
[**Testing, Formatting & Linting**](#testing-formatting--linting)
* [**Requirements**](#requirements)
* [**Running Tests**](#running-tests)
* [**Formatting & Linting**](#formatting--linting)

[**Contributions & Support**](#contributions--support)
* [**Support**](#support)
* [**Contributors**](#contributors)

[**Community Channel**](#community-channel)

[**Citing**](#citing)

[**Copyright**](#copyright)


&nbsp;

&nbsp;



# TL;DR
  
#### What is vidgear?

> ***"VidGear is an [ultrafast➶][ultrafast-wiki], compact, flexible and easy-to-adapt complete Video Processing Python Library."***

#### What does it do?
> ***"VidGear can read, write, process, send & receive video frames from/to various devices in real-time."***

#### What is its purpose?
> ***"Built with simplicity in mind, VidGear lets programmers and software developers to easily integrate and perform complex Video Processing tasks in their existing or new applications, without going through various underlying library's documentation and using just a [few lines of code][flic]. Beneficial for both, if you're new to programming with Python language or already a pro at it."***

&nbsp;

**For more information, see [*Frequently Asked Questions ➶*][faq].**


&nbsp;

&nbsp;


# Gears

> **VidGear is built with multiple APIs *(a.k.a Gears)*, each with some unique function/mechanism.**

Each of these APIs is exclusively designed to handle/control different device-specific video streams, network streams, and media encoders. These APIs provide an easy-to-use, highly extensible, multi-threaded and asyncio wrapper around state-of-the-art libraries under the hood to exploit their internal parameters and methods flexibly while providing robust error-handling and unparalleled performance. 

**These Gears can be classified as follows:**

**A. VideoCapture Gears:**

  * [**CamGear:**](#camgear) _Targets various IP-USB-Cameras/Network-Streams/YouTube-Video-URL._
  * [**PiGear:**](#pigear) _Targets various Raspberry Pi Camera Modules._
  * [**ScreenGear:**](#screengear) _Enables ultra-fast Screen Casting._    
  * [**VideoGear:**](#videogear) _Common API with Video Stabilizer wrapper._  

**B. VideoWriter Gear:**

  * [**WriteGear:**](#writegear) _Handles easy Lossless Video Encoding and Compression._

**C. Network Gears:**

  * [**NetGear:**](#netgear) _Targets flexible video-frames and data transfer between interconnecting systems over the network._

  * **Asynchronous Network Gears:**

    * [**WebGear:**](#webgear) _ASGI Video Server that transfers live video frames to any web browser on the network._
    * [**NetGear_Async:**](#netgear_sync) _Fast, Memory-Efficient Asyncio video-frame messaging framework._ 


&nbsp;

&nbsp;


## CamGear

> *CamGear can grab ultra-fast frames from diverse range of devices/streams, which includes almost any IP/USB Cameras, multimedia video file format ([_upto 4k tested_][test-4k]), various network stream protocols such as `http(s), rtp, rstp, rtmp, mms, etc.`, plus support for live Gstreamer's stream pipeline and YouTube video/live-streams URLs.*

CamGear provides a flexible, high-level multi-threaded wrapper around `OpenCV's` [VideoCapture class][opencv-vc] with access almost all of its available parameters and also employs [`pafy`][pafy] python APIs for live [YouTube streaming][youtube-wiki]. Furthermore, CamGear implements exclusively on [**Threaded Queue mode**][TQM-wiki] for ultra-fast, error-free and synchronized frame handling.


**Following functional block diagram depicts CamGear API's generalized workflow:**

<p align="center">
  <img src="https://github.com/abhiTronix/Imbakup/raw/master/Images/vidgear/camgearz2.png" alt="CamGear Functional Block Diagram"  width="70%"/>
</p>

### CamGear API Guide:

[**>>> Usage Guide**][camgear-wiki]

&nbsp;

&nbsp;


## VideoGear

> *VideoGear API provides a special internal wrapper around VidGear's exclusive [**Video Stabilizer**][stablizer-wiki] class.*

Furthermore, VideoGear API can provide internal access to both [CamGear](#camgear) and [PiGear](#pigear) APIs separated by a special flag. Thereby, _this API holds the exclusive power for any incoming VideoStream from any source, whether it is live or not, to access and stabilize it directly with minimum latency and memory requirements._

**Below is a snapshot of a VideoGear Stabilizer in action  (_See its detailed usage [here][stablizer-wiki-ex]_):**

<p align="center">
  <img src="https://github.com/abhiTronix/Imbakup/raw/master/Images/stabilizer.gif" alt="VideoGear Stabilizer in action!"/>
  <br>
  <sub><i>Original Video Courtesy <a href="http://liushuaicheng.org/SIGGRAPH2013/database.html" title="opensourced video samples database">@SIGGRAPH2013</a></i></sub>
</p>

**Code to generate above result:**

```python
# import required libraries
from vidgear.gears import VideoGear
import numpy as np
import cv2

# open any valid video stream with stabilization enabled(`stabilize = True`)
stream_stab = VideoGear(source = "test.mp4", stabilize = True).start()

# open same stream without stabilization for comparison
stream_org = VideoGear(source = "test.mp4").start()

# loop over
while True:

    # read stabilized frames
    frame_stab = stream_stab.read()

    # check for stabilized frame if Nonetype
    if frame_stab is None:
        break

    # read un-stabilized frame
    frame_org = stream_org.read()

    # concatenate both frames
    output_frame = np.concatenate((frame_org, frame_stab), axis=1)

    # put text over concatenated frame
    cv2.putText(
        output_frame, "Before", (10, output_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 255, 0), 2,
    )
    cv2.putText(
        output_frame, "After", (output_frame.shape[1] // 2 + 10, output_frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 255, 0), 2,
    )

    # Show output window
    cv2.imshow("Stabilized Frame", output_frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close both video streams
stream_org.stop()
stream_stab.stop()
```

### VideoGear API Guide:

[**>>> Usage Guide**][videogear-wiki]

&nbsp;

&nbsp;


## PiGear

> *PiGear is similar to CamGear but made to support various Raspberry Pi Camera Modules *(such as [OmniVision OV5647 Camera Module][OV5647-picam] and [Sony IMX219 Camera Module][IMX219-picam])*.*

PiGear provides a flexible multi-threaded wrapper around complete [**picamera**][picamera] python library to interface with these modules correctly, and also grants the ability to exploit its various parameters like `brightness, saturation, sensor_mode, etc.` effortlessly. 

Best of all, PiGear API provides excellent Error-Handling with features like a threaded internal timer that keeps active track of any frozen threads and handles hardware failures/frozen threads robustly thereby will exit safely if any failure occurs. So now if someone accidentally pulled your Raspi-camera module cable out when you're running PiGear API in your script, instead of going into possible kernel panic/frozen threads, this API will exit safely to save resources. 

**Code to open Picamera stream with variable parameters in PiGear API:**

```python
# import required libraries
from vidgear.gears import PiGear
import cv2

# add various Picamera tweak parameters to dictionary
options = {"hflip": True, "exposure_mode": "auto", "iso": 800, "exposure_compensation": 15, "awb_mode": "horizon", "sensor_mode": 0}

# open pi video stream with defined parameters
stream = PiGear(resolution = (640, 480), framerate = 60, logging = True, **options).start() 

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break


    # {do something with the frame here}


    # Show output window
    cv2.imshow("Output Frame", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
```
### PiGear API Guide:

[**>>> Usage Guide**][pigear-wiki]

&nbsp;

&nbsp;


## ScreenGear

> *ScreenGear API act as Screen Recorder, that can grab frames from your monitor in real-time either by define an area on the computer screen or fullscreen at the expense of insignificant latency. It also provide seamless support for capturing frames from multiple monitors.*

ScreenGear provides a high-level multi-threaded wrapper around [**python-mss**][mss] python library API and also supports a easy and flexible direct internal parameter manipulation. 

**Below is a snapshot of a ScreenGear API in action:**

<p align="center">
  <img src="https://github.com/abhiTronix/Imbakup/raw/master/Images/screengear.gif" alt="ScreenGear in action!" />
</p>

**Code to generate the above results:**

```python
# import required libraries
from vidgear.gears import ScreenGear
import cv2

# open video stream with default parameters
stream = ScreenGear().start()

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break


    # {do something with the frame here}


    # Show output window
    cv2.imshow("Output Frame", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
```

### ScreenGear API Guide:

[**>>> Usage Guide**][screengear-wiki]


&nbsp;

&nbsp;



## WriteGear

> *WriteGear handles various powerful Writer Tools that provide us the freedom to do almost anything imagine with multimedia files.*

WriteGear API provide a complete, flexible & robust wrapper around [**FFmpeg**][ffmpeg], a leading multimedia framework. With WriteGear, we can process real-time video frames into a lossless compressed format with any suitable specification in just few easy [lines of codes][compression-mode-ex]. These specifications include setting any video/audio property such as `bitrate, codec, framerate, resolution, subtitles,  etc.` easily as well complex tasks such as multiplexing video with audio in real-time(see this [example wiki][live-audio-wiki]). Best of all, WriteGear grants the freedom to play with any FFmpeg parameter with its exclusive custom Command function(see this [example wiki][custom-command-wiki]), while handling all errors robustly. 

In addition to this, WriteGear also provides flexible access to [**OpenCV's VideoWriter API**][opencv-writer] which provides some basic tools for video frames encoding but without compression.

**WriteGear primarily operates in the following two modes:**

  * **Compression Mode:** In this mode, WriteGear utilizes [**FFmpeg**][ffmpeg] inbuilt encoders to encode lossless multimedia files. It provides us the ability to exploit almost any available parameters available within FFmpeg, with so much ease and flexibility and while doing that it robustly handles all errors/warnings quietly. **You can find more about this mode [here][cm-writegear-wiki]**.

  * **Non-Compression Mode:** In this mode, WriteGear utilizes basic [**OpenCV's inbuilt VideoWriter API**][opencv-vw]. Similar to compression mode, WriteGear also supports all parameters manipulation available within this API. But this mode lacks the ability to manipulate encoding parameters and other important features like video compression, audio encoding, etc. **You can learn about this mode [here][ncm-writegear-wiki]**.

**Following functional block diagram depicts WriteGear API's generalized workflow:**

<p align="center">
  <img src="https://github.com/abhiTronix/Imbakup/raw/master/Images/vidgear/writegear.png" alt="WriteGear Functional Block Diagram"/>
</p>

### WriteGear API Guide:

[**>>> Usage Guide**][writegear-wiki]

&nbsp;

&nbsp;


## NetGear

> *NetGear is exclusively designed to transfer video frames synchronously and asynchronously between interconnecting systems over the network in real-time.*

NetGear implements a high-level wrapper around [**PyZmQ**][pyzmq] python library that contains python bindings for [ZeroMQ](http://zeromq.org/) - a high-performance asynchronous distributed messaging library that aim to be used in distributed or concurrent applications. It provides a message queue, but unlike message-oriented middleware, a ZeroMQ system can run without a dedicated message broker. 

NetGear provides seamless support for [*Bi-directional data transmission*][netgear_bidata_wiki] between receiver(client) and sender(server) through bi-directional synchronous messaging patterns such as zmq.PAIR _(ZMQ Pair Pattern)_ & zmq.REQ/zmq.REP _(ZMQ Request/Reply Pattern)_. 

NetGear also supports real-time [*Frame Compression capabilities*][netgear_compression_wiki] for optimizing performance while sending the frames directly over the network, by encoding the frame before sending it and decoding it on the client's end automatically in real-time. 

For security, NetGear implements easy access to ZeroMQ's powerful, smart & secure Security Layers, that enables [*Strong encryption on data*][netgear_security_wiki], and unbreakable authentication between the Server and the Client with the help of custom certificates/keys and brings easy, standardized privacy and authentication for distributed systems over the network. 

Best of all, NetGear can robustly handle [*Multiple Servers devices*][netgear_multi_wiki] at once, thereby providing access to seamless Live Streaming of the multiple device in a network at the same time.


**NetGear as of now seamlessly supports three ZeroMQ messaging patterns:**

* [**`zmq.PAIR`**][zmq-pair] _(ZMQ Pair Pattern)_ 
* [**`zmq.REQ/zmq.REP`**][zmq-req-rep] _(ZMQ Request/Reply Pattern)_
* [**`zmq.PUB/zmq.SUB`**][zmq-pub-sub] _(ZMQ Publish/Subscribe Pattern)_

Whereas supported protocol are: `tcp` and `ipc`.

**Following functional block diagram depicts generalized workflow of NetGear API in its Multi-Servers Mode:**

<p align="center">
  <img src="https://github.com/abhiTronix/Imbakup/raw/master/Images/vidgear/netgearz.png" alt="NetGear Multi-Servers Mode Functional Block Diagram" width="90%" />
</p>

### NetGear API Guide:

[**>>> Usage Guide**][netgear-wiki]

&nbsp;

&nbsp;


## WebGear

> *WebGear is a powerful ASGI Video Streamer API, that transfers live video frames to any web browser on the network in real-time.*

WebGear API provides a flexible abstract asyncio wrapper around [Starlette][starlette] ASGI library and easy access to its various components independently. Thereby implementing the ability to flexibly interact with the Starlette's ecosystem of shared middle-ware and mountable applications & seamless access to its various Response classes, Routing tables, Static Files, Template engine(with Jinja2), etc.

WebGear can acts as robust _Live Video Streaming Server_ that can stream live video frames to any web browser on a network in real-time. It also auto-generates necessary data files for its default template and provides us the freedom to easily alter its [_performance parameters and routing tables_][advanced-webgear-wiki] according to our applications while handling errors robustly.

In addition to this, WebGear provides a special internal wrapper around [VideoGear](#videogear) API, which itself provides internal access to both [CamGear](#camgear) and [PiGear](#pigear) APIs thereby granting it exclusive power for streaming frames incoming from any device/source. Also on the plus side, since WebGear has access to all functions of [VideoGear](#videogear) API, therefore it can [stabilize video frames][stabilize_webgear_wiki] even while streaming live.

**Below is a snapshot of a WebGear Video Server in action on the Mozilla Firefox browser:**

<p align="center">
  <img src="https://github.com/abhiTronix/Imbakup/raw/master/Images/webgear.gif" alt="WebGear in action!" width=120%/>
  <br>
  <sub><i>WebGear Video Server at <a href="http://0.0.0.0:8000/" title="default address">http://0.0.0.0:8000/</a> address.</i></sub>
</p>

**Code to generate the above result:**

```python
# import required libraries
import uvicorn
from vidgear.gears.asyncio import WebGear

#various performance tweaks
options = {"frame_size_reduction": 40, "frame_jpeg_quality": 80, "frame_jpeg_optimize": True, "frame_jpeg_progressive": False}

#initialize WebGear app  
web = WebGear(source = "foo.mp4", logging = True, **options)

#run this app on Uvicorn server at address http://0.0.0.0:8000/
uvicorn.run(web(), host='0.0.0.0', port=8000)

#close app safely
web.shutdown()
```

### WebGear API Guide:

[**>>> Usage Guide**][webgear-wiki]


&nbsp;

&nbsp;

## NetGear_Async 

> _NetGear_Async can performance boost upto 1.2~2x times as compared to [NetGear API](#netgear) at about 1/3 of memory consumption but only at the expense of limited modes and features._

NetGear_Async is an asynchronous videoframe messaging framework built on [**AsyncIO ZmQ**][asyncio-zmq] and powered by high-performance asyncio event loop called [**`uvloop`**][uvloop] to achieve unmatchable high-speed and lag-free video streaming over the network with minimal resource constraint. Basically, this API is able to transfer thousands of frames in just a few seconds without causing any significant load on your system.

NetGear_Async provides complete server-client handling and options to use variable protocols/patterns similar to [NetGear API](#netgear) but doesn't support any [*NetGear Exclusive modes*][netgear-exm] yet. NetGear_Async also allows you to easily define your own custom Source at Server-end that you want to use to manipulate your frames before sending them onto the network(See this [Wiki-example][netgear_Async-cs]).

NetGear_Async as of now supports [all four ZeroMQ messaging patterns](#attributes-and-parameters-wrench):
* [**`zmq.PAIR`**][zmq-pair] _(ZMQ Pair Pattern)_ 
* [**`zmq.REQ/zmq.REP`**][zmq-req-rep] _(ZMQ Request/Reply Pattern)_
* [**`zmq.PUB/zmq.SUB`**][zmq-pub-sub] _(ZMQ Publish/Subscribe Pattern)_ 
* [**`zmq.PUSH/zmq.PULL`**][zmq-pull-push] _(ZMQ Push/Pull Pattern)_

Whereas supported protocol are: `tcp` and `ipc`.

**Code for NetGear_Async Server-Client API:**

<img src="https://github.com/abhiTronix/Imbakup/raw/master/Images/vidgear/netgear_async.png"/>


### NetGear_Async API Guide:

[**>>> Usage Guide**][webgear-wiki]

&nbsp;

&nbsp;

# New Release SneekPeak : VidGear 0.1.7

:warning: Dropped support for Python 3.5 and below legacies.

* **WebGear API:**
  * _Added a robust Live Video Server API that can transfer live video frames to any web browser on the network in real-time._
  * _Implemented a flexible asyncio wrapper around [`starlette`][starlette] ASGI Application Server._
  * _Added seamless access to various starlette's Response classes, Routing tables, Static Files, Templating engine(with Jinja2), etc._
  * _Added a special internal access to VideoGear API and all its parameters._
  * _Implemented a new Auto-Generation Workflow to generate/download & thereby validate WebGear API data files from its GitHub server automatically._
  * _Added on-the-go dictionary parameter in WebGear to tweak performance, Route Tables and other internal properties easily._
  * _Added new simple & elegant default Bootstrap Cover Template for WebGear Server._
  * _Added `__main__.py` to directly run WebGear Server through the terminal._


* **NetGear_Async API** 
  * _Designed NetGear_Async asynchronous network API built upon ZeroMQ's asyncio API._
  * _Implemented support for state-of-the-art asyncio event loop [`uvloop`][uvloop] at its backend._
  * _Achieved Unmatchable high-speed and lag-free video streaming over the network with minimal resource constraint._
  * _Added exclusive internal wrapper around VideoGear API for this API._
  * _Implemented complete server-client handling and options to use variable protocols/patterns for this API._
  * _Implemented support for  all four ZeroMQ messaging patterns: i.e `zmq.PAIR`, `zmq.REQ/zmq.REP`, `zmq.PUB/zmq.SUB`, and `zmq.PUSH/zmq.PULL`._
  * _Implemented initial support for `tcp` and `ipc` protocols._

* **Asynchronous Enhancements** 
  * _Added `asyncio` package to vidgear for handling asynchronous network APIs._
  * _Various Performance enhancements for these Asyncio APIs for achieving concurrency within a single thread._

* ***Added new highly-precise Threaded FPS class for accurate VidGear benchmarking with `time.perf_counter` python module and [many more...](changelog.md)***


&nbsp;

&nbsp;


# Documentation

The complete documentation for all VidGear APIs and  functions can be found in the link below:

* [**Wiki Documentation - English**][wiki]


&nbsp;

&nbsp;


# Installation

## Prerequisites:

Before installing VidGear, you must verify that the following dependencies are met:

* ### Supported Systems:

  VidGear is tested and supported on the following systems with [**Python 3.6+**](#supported-python-legacies) and [**pip**][pip] already installed:

  * Any Linux distro released in 2016 or later
  * Windows 7 or later
  * macOS 10.12.6 (Sierra) or later

* ### Supported Python legacies:

  * [Python 3.6+][drop35] are only supported legacies for installing Vidgear v0.1.7 and above.


* ### Pip Dependencies:

  When [installing VidGear with pip](#option-1-pypi-installrecommended), you need to install following dependencies manually:


  * **OpenCV:** Must Require OpenCV(3.0+) python binaries installed for its core functions. For installation, you can either follow these complete online tutorials for [Windows][OpenCV-windows], [Linux][OpenCV-linux] and [Raspberry Pi][OpenCV-pi], or, just install it directly via pip:

      ```sh
        $ pip install -U opencv-python       # or install `opencv-contrib-python` similarly
      ```


  * **FFmpeg:** Must Require FFmpeg for its video compression and encoding compatibilities in [WriteGear](#writegear) API. 

    :star2: Follow this [**FFmpeg wiki page**][ffmpeg-wiki] for its installation. :star2:


  * **Picamera:** Must Required if you're using Raspberry Pi Camera Modules(_such as OmniVision OV5647 Camera Module_) with its [PiGear](#pigear) API. You can easily install it via pip:

      ```sh
        $ pip install picamera
      ``` 
    _:bulb: Also, make sure to [enable Raspberry Pi hardware-specific settings][picamera-setting] prior to using this library._


  * **Uvloop:** Only Required if you're using its [NetGear_Async](#netgear_async) API on UNIX machines for maximum performance. You can easily install it via pip:

      _:warning: Uvloop is [**NOT** yet supported on Windows Systems][uvloop-ns]._

      ```sh
        $ pip install uvloop
      ```

&nbsp;

## Available Installation Options:

### Option 1: PyPI Install(recommended)

> Best option for **quickly** getting VidGear installed.

***:warning: See [Pip Dependencies](#pip-dependencies) before installing!***

```sh
  # Installing stable release
  $ pip install vidgear

  # Installing stable release with Asyncio support
  $ pip install vidgear[asyncio]
```


### Option 2: Release Download

> Best option if you want a **compressed archive**.

VidGear is available for download as wheel(`.whl`) package in our [release][release] section, and can be installed with `pip` as follows:

***:warning: See [Pip Dependencies](#pip-dependencies) before installing!***

```sh
  # directly installs the wheel
  $ pip install vidgear-{downloaded version}-py3-none-any.whl
```


### Option 3: Build from source

> Best option for trying **latest patches(_maybe experimental_), Pull Requests**, or **contributing** to development.

You can easily clone the repository's latest [`testing`](https://github.com/abhiTronix/vidgear/tree/testing) branch, and thereby install it as follows:

```sh
  $ git clone https://github.com/abhiTronix/vidgear.git
  $ cd vidgear
  $ git checkout testing
  $ pip install .[asyncio]           # installs all required dependencies including asyncio
```


&nbsp;

&nbsp;


# Testing, Formatting & Linting

### Requirements:

  Testing VidGear require some *additional dependencies & dataset* that can be downloaded manually as follows:

  * **Install additional python libraries:**
    ```sh
      $ pip install --upgrade six
      $ pip install --upgrade flake8
      $ pip install --upgrade black
      $ pip install --upgrade pytest
      $ pip install --upgrade pytest-asyncio
    ```
  
  * **Download Test Dataset:** 

    To perform tests, additional *test dataset* is required, which can be downloaded *(to your temp dir)* by running [*bash script*][bs_script_dataset] as follows:

    ```sh
      $ chmod +x scripts/bash/prepare_dataset.sh
      $ .scripts/bash/prepare_dataset.sh               #for Windows, use `sh scripts/bash/prepare_dataset.sh`
    ```

### Running Tests: 

* **Pytest:** Then, tests can be run with [`pytest`][pytest](*in VidGear's root folder*) as follows:

  ```sh
    $ pytest -sv                                   #-sv for verbose output.
  ```

### Formatting & Linting: 

For formatting and linting, following tools are used:

* **Flake8:** You must run [`flake8`][flake8] linting for checking the code base against the coding style (PEP8), programming errors and other cyclomatic complexity:

  ```sh
    $ flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
  ```

* **Black:**  Vidgear follows [`black`][black] formatting to make code review faster by producing the smallest diffs possible. You must run it with sensible defaults as follows:

  ```sh
    $ black {source_file_or_directory}
  ```

&nbsp;

&nbsp;
 

# Contributions & Support

Contributions are welcome! Please see our **[Contribution Guidelines](contributing.md)** for more details.

### Support

Love using VidGear? Consider supporting the project to fund new features and improvements

[![ko-fi][kofi-badge]][kofi]

### Contributors

<a href="https://github.com/abhiTronix/vidgear/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=abhiTronix/vidgear" />
</a>


&nbsp;

&nbsp;


# Community Channel

We're on [**Gitter :star2:**][gitter]! Please join us.


&nbsp;

&nbsp;



# Citing

Here is a Bibtex entry you can use to cite this project in a publication:


```latex
@misc{vidgear,
    Title = {vidgear},
    Author = {Abhishek Thakur},
    howpublished = {\url{https://github.com/abhiTronix/vidgear}}   
  }
```

&nbsp;

&nbsp;


# Copyright

**Copyright © abhiTronix 2019**

This library is licensed under the **[Apache 2.0 License][license]**.




<!--
Badges
-->

[appveyor]:https://img.shields.io/appveyor/ci/abhitronix/vidgear.svg?style=for-the-badge&logo=appveyor
[codecov]:https://img.shields.io/codecov/c/github/abhiTronix/vidgear/testing?style=for-the-badge&logo=codecov
[travis-cli]:https://img.shields.io/travis/abhiTronix/vidgear.svg?style=for-the-badge&logo=travis
[prs-badge]:https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAABC0lEQVRYhdWVPQoCMRCFX6HY2ghaiZUXsLW0EDyBrbWtN/EUHsHTWFnYyCL4gxibVZZlZzKTnWz0QZpk5r0vIdkF/kBPAMOKeddE+CQPKoc5Yt5cTjBMdQSwDQToWgBJAn3jmhqgltapAV6E6b5U17MGGAUaUj07TficMfIBZDV6vxowBm1BP9WbSQE4o5h9IjPJmy73TEPDDxVmoZdQrQ5jRhly9Q8tgMUXkIIWn0oG4GYQfAXQzz1PGoCiQndM7b4RgJay/h7zBLT3hASgoKjamQJMreKf0gfuAGyYtXEIAKcL/Dss15iq6ohXghozLYiAMxPuACwtIT4yeQUxAaLrZwAoqGRKGk7qDSYTfYQ8LuYnAAAAAElFTkSuQmCC
[twitter-badge]:https://img.shields.io/badge/Tweet-Now-blue.svg?style=for-the-badge&logo=twitter
[pypi-badge]:https://img.shields.io/pypi/v/vidgear.svg?style=for-the-badge&logo=pypi
[gitter-bagde]:https://img.shields.io/badge/Chat-Gitter-yellow.svg?style=for-the-badge&logo=gitter
[Coffee-badge]:https://abhitronix.github.io/img/vidgear/orange_img.png
[kofi-badge]:https://www.ko-fi.com/img/githubbutton_sm.svg
[kofi]: https://ko-fi.com/W7W8WTYO

<!--
Internal URLs
-->

[release]:https://github.com/abhiTronix/vidgear/releases/latest
[pypi]:https://pypi.org/project/vidgear/
[gitter]:https://gitter.im/vidgear/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
[twitter-intent]:https://twitter.com/intent/tweet?url=https%3A%2F%2Fgithub.com%2FabhiTronix%2Fvidgear&via%20%40abhi_una12&text=VidGear%20-%20A%20simple%2C%20powerful%2C%20flexible%20%26%20threaded%20Python%20Video%20Processing%20Library&hashtags=vidgear%20%23multithreaded%20%23python%20%23video-processing%20%23github
[coffee]:https://www.buymeacoffee.com/2twOXFvlA
[license]:https://github.com/abhiTronix/vidgear/blob/master/LICENSE
[travis]:https://travis-ci.org/abhiTronix/vidgear
[app]:https://ci.appveyor.com/project/abhiTronix/vidgear
[code]:https://codecov.io/gh/abhiTronix/vidgear

[test-4k]:https://github.com/abhiTronix/vidgear/blob/e0843720202b0921d1c26e2ce5b11fadefbec892/vidgear/tests/benchmark_tests/test_benchmark_playback.py#L65
[bs_script_dataset]:https://github.com/abhiTronix/vidgear/blob/testing/scripts/bash/prepare_dataset.sh

[wiki]:https://github.com/abhiTronix/vidgear/wiki
[faq]:https://github.com/abhiTronix/vidgear/wiki/FAQ-&-Troubleshooting#frequently-asked-questions
[wiki-vidgear-purpose]:https://github.com/abhiTronix/vidgear/wiki/Project-Motivation#why-is-vidgear-a-thing
[ultrafast-wiki]:https://github.com/abhiTronix/vidgear/wiki/FAQ-&-Troubleshooting#2-vidgear-is-ultrafast-but-how
[compression-mode-ex]:https://github.com/abhiTronix/vidgear/wiki/Compression-Mode:-FFmpeg#1-writegear-bare-minimum-examplecompression-mode
[live-audio-wiki]:https://github.com/abhiTronix/vidgear/wiki/Working-with-Audio#a-live-audio-input-to-writegear-class
[ffmpeg-wiki]:https://github.com/abhiTronix/vidgear/wiki/FFmpeg-Installation
[youtube-wiki]:https://github.com/abhiTronix/vidgear/wiki/CamGear#2-camgear-api-with-live-youtube-piplineing-using-video-url
[TQM-wiki]:https://github.com/abhiTronix/vidgear/wiki/Threaded-Queue-Mode
[camgear-wiki]:https://github.com/abhiTronix/vidgear/wiki/CamGear#camgear-api
[stablizer-wiki]:https://github.com/abhiTronix/vidgear/wiki/Stabilizer-Class
[stablizer-wiki-ex]:https://github.com/abhiTronix/vidgear/wiki/Real-time-Video-Stabilization#real-time-video-stabilization
[videogear-wiki]:https://github.com/abhiTronix/vidgear/wiki/VideoGear#videogear-api
[pigear-wiki]:https://github.com/abhiTronix/vidgear/wiki/PiGear#pigear-api
[cm-writegear-wiki]:https://github.com/abhiTronix/vidgear/wiki/Compression-Mode:-FFmpeg
[ncm-writegear-wiki]:https://github.com/abhiTronix/vidgear/wiki/Non-Compression-Mode:-OpenCV
[screengear-wiki]:https://github.com/abhiTronix/vidgear/wiki/ScreenGear#screengear-api
[writegear-wiki]:https://github.com/abhiTronix/vidgear/wiki/WriteGear#writegear-api
[netgear-wiki]:https://github.com/abhiTronix/vidgear/wiki/NetGear#netgear-api
[webgear-wiki]:https://github.com/abhiTronix/vidgear/wiki/WebGear#webgear-api
[drop35]:https://github.com/abhiTronix/vidgear/issues/99
[custom-command-wiki]:https://github.com/abhiTronix/vidgear/wiki/Custom-FFmpeg-Commands-in-WriteGear-API#custom-ffmpeg-commands-in-writegear-api
[advanced-webgear-wiki]:https://github.com/abhiTronix/vidgear/wiki/Advanced-WebGear-API-Usage
[netgear_bidata_wiki]:https://github.com/abhiTronix/vidgear/wiki/Bidirectional-Mode:-Bidirectional-Data-Transfer-in-NetGear-API#bi-directional-mode-bidirectional-data-transfer-in-netgear-api
[netgear_compression_wiki]:https://github.com/abhiTronix/vidgear/wiki/Compression-in-NetGear-API#frame-encodingdecoding-compression-capabilities-for-netgear-api
[netgear_security_wiki]:https://github.com/abhiTronix/vidgear/wiki/Secure-Mode:-Authentication-&-Data-Encryption-in-NetGear-API#secure-mode-authentication--data-encryption-in-netgear-api
[netgear_multi_wiki]:https://github.com/abhiTronix/vidgear/wiki/Multi-Server-Mode-for-NetGear-API#multi-server-mode-for-netgear-api
[netgear-exm]: https://github.com/abhiTronix/vidgear/wiki/NetGear#modes-of-operation
[stabilize_webgear_wiki]:https://github.com/abhiTronix/vidgear/wiki/Advanced-WebGear-API-Usage#d2-using-webgear-api-with-real-time-video-stabilization-enabled
[flic]:https://github.com/abhiTronix/vidgear/wiki/CamGear#1-bare-minimum-example
[netgear_Async-cs]: https://github.com/abhiTronix/vidgear/wiki/NetGear_Async#2-use-netgear_async-with-custom-server-source-using-opencv

<!--
External URLs
-->
[asyncio-zmq]:https://pyzmq.readthedocs.io/en/latest/api/zmq.asyncio.html
[uvloop]: https://github.com/MagicStack/uvloop
[uvloop-ns]: https://github.com/MagicStack/uvloop/issues/14
[ffmpeg]:https://www.ffmpeg.org/
[flake8]: https://flake8.pycqa.org/en/latest/
[black]: https://github.com/psf/black
[pytest]:https://docs.pytest.org/en/latest/
[opencv-writer]:https://docs.opencv.org/master/dd/d9e/classcv_1_1VideoWriter.html#ad59c61d8881ba2b2da22cff5487465b5
[OpenCV-windows]:https://www.learnopencv.com/install-opencv3-on-windows/
[OpenCV-linux]:https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/
[OpenCV-pi]:https://www.pyimagesearch.com/2018/09/26/install-opencv-4-on-your-raspberry-pi/
[starlette]:https://www.starlette.io/
[uvicorn]:http://www.uvicorn.org/
[daphne]:https://github.com/django/daphne/
[hypercorn]:https://pgjones.gitlab.io/hypercorn/
[prs]:http://makeapullrequest.com
[opencv]:https://github.com/opencv/opencv
[picamera]:https://github.com/waveform80/picamera
[pafy]:https://github.com/mps-youtube/pafy
[pyzmq]:https://github.com/zeromq/pyzmq
[zmq]:https://zeromq.org/
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
[zmq-pull-push]: https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/pushpull.html#push-pull
[picamera-setting]:https://picamera.readthedocs.io/en/release-1.13/quickstart.html