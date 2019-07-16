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

[Releases][release]&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Gears](#gears)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Wiki Documentation][wiki]&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Installation](#installation)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[License][license]

[![PyPi version][pypi-badge]][pypi] [![PRs Welcome][prs-badge]][prs] [![Build Status][travis-cli]][travis] [![Build Status][appveyor]][app] [![Say Thank you][Thank-you]][thanks] [![Twitter][twitter-badge]][twitter-intent] 

[![Buy Me A Coffee][Coffee-badge]][coffee]

</div>

&nbsp;

VidGear is a powerful python Video Processing library built with multi-threaded [**Gears**]()(_a.k.a APIs_) each with unique set of trailblazing features. These APIs provides a easy-to-use, highly extensible, and multi-threaded wrapper around many underlying state-of-the-art python libraries such as *[OpenCV ➶](https://github.com/opencv/opencv), [FFmpeg ➶](https://ffmpeg.org/), [picamera ➶](https://github.com/waveform80/picamera), [pafy ➶](https://github.com/mps-youtube/pafy), [pyzmq ➶](https://github.com/zeromq/pyzmq) and [python-mss ➶](https://github.com/BoboTiG/python-mss)* 

&nbsp;

The following **functional block diagram** clearly depicts the functioning of VidGear library:

<p align="center">
  <img src="https://abhitronix.github.io/img/vidgear/vidgear_function2-01.svg" alt="@Vidgear Functional Block Diagram" />
</p>

&nbsp;

## Table of Contents

[**TL;DR**](#tldr)

[**New Release : VidGear 0.1.5**](#new-release-vidgear-015)

[**Installation Options**](#installation)
  * [**Dependencies**](#dependencies)
  * [**1 - PyPI Install**](#option-1-pypi-install)
  * [**2 - Release Archive Download**](#option-2-release-archive-download)
  * [**3 - Clone Repo**](#option-3-clone-the-repo)

[**Gears: Features, Functioning and Usage**](#gear)
  * [**CamGear**]()
  * [**PiGear**]()
  * [**VideoGear**]()
  * [**ScreenGear**]()
  * [**WriteGear**]()
  * [**NetGear**]()

[**For Developers**]()
  * [**0 - Prerequisites**]()
  * [**1 - Testing**]()
  * [**2 - Contributing**]()

[**Project Motivation**]()

[**Contact Developers**]()

**Additional Info**
  * [**Supported Python legacies**]()
  * [**Changelog**]()
  * [**License**]()

&nbsp;

## TL;DR
   *VidGear is a ultrafast, compact, flexible and easy-to-adapt complete Video Processing Python Library.*

   Built with simplicity in mind, VidGear lets programmers and software developers to easily integrate and perform complex Video Processing tasks in their existing or new applications, without going through various underlying python library's documentations and using just few lines of code. Beneficial for both, if you're new to Programming with Python language or pro at it. 

   For more advanced information see the [Wiki Documentation ➶][wiki].

&nbsp;

## New Release : VidGear 0.1.5
  * Released new ScreenGear, supports Live ScreenCasting.
  * Released new NetGear, aids real-time frame transfer through messaging(ZmQ) over network.
  * Released new Stabilizer Class, for minimum latency Video Stabilization with OpenCV.
  * Added Option to use VidGear Classes standalone.
  * Added Option to use VideoGear as internal wrapper around Stabilizer Class.
  * Implemented exclusive Threaded Queue Mode(_a.k.a Blocking Mode_) for fast, synchronized, error-free multi-threading.
  * Added New dependencies: `mss`, `pyzmq` and rejected redundant ones.
  * Added Travis CLI bug workaround, Replaced `opencv-contrib-python` with OpenCV built from scratch as dependency.
  * Several performance enhancements and bugs exterminated.
  * Revamped VidGear Docs and [many more...]()

&nbsp;

## Installation

### Dependencies

To use VidGear in your python application, you must have check the following dependencies before you install VidGear :

* Must support [these Python legacies](#python) and [pip](https://pip.pypa.io/en/stable/installing/) already installed.


* **`OpenCV:`** VidGear must require OpenCV(3.0+) python enabled binaries to be installed on your machine for its core functions. For its installation, you can follow these online tutorials for [linux][OpenCV-linux] and [raspberry pi][OpenCV-pi], otherwise, install it via pip:

    ```sh
      pip install opencv-python
    ```

* **`FFmpeg:`** VidGear must requires FFmpeg for its powerful video compression and encoding capabilities. Follow this [WiKi Page]() for installation.

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

> Best option for **automatically installing required dependencies**(_except FFmpeg_), for **latest patches**(_maybe experimental_), or **contributing** to development.

You can clone this repository's `testing` branch for development and thereby can install as follows:
```sh
 git clone https://github.com/abhiTronix/vidgear.git
 cd vidgear
 git checkout testing
 pip install .
```


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
[twitter-intent]:https://twitter.com/intent/tweet?url=https%3A%2F%2Fgithub.com%2FabhiTronix%2Fvidgear&via%20%40abhi_una12&text=Vidgear%20-%20simple%2C%20powerful%2C%20flexible%20%26%20threaded%20Python%20Video%20Processing%20Library&hashtags=vidgear%20multithreaded%20python%20video-processing
[license]:https://github.com/abhiTronix/vidgear/blob/master/LICENSE

<!--
External URLs
-->
[OpenCV-linux]:https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/
[OpenCV-pi]:https://www.pyimagesearch.com/2018/09/26/install-opencv-4-on-your-raspberry-pi/