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

# StreamGear API 


<p align="center">
  <img src="../../../assets/images/streamgear_flow.png" alt="StreamGear Flow Diagram"/>
  <br>
  <sub><i>StreamGear API's generalized workflow</i></sub>
</p>


## Overview

> StreamGear automates transcoding workflow for generating _Ultra-Low Latency, High-Quality, Dynamic & Adaptive Streaming Formats (such as MPEG-DASH)_ in just few lines of python code. 

StreamGear provides a standalone, highly extensible and flexible wrapper around [**FFmpeg**](https://ffmpeg.org/) - a leading multimedia framework, for generating chunked-encoded media segments of the content.

SteamGear API ***automatically transcodes source videos/audio files & real-time frames, and breaks them into a sequence of multiple smaller chunks/segments (typically 2-4 seconds in length) at different quality levels (i.e. different bitrates or spatial resolutions)***. It also creates a Manifest file _(MPD in-case of DASH)_ that describes these segment information _(timing, URL, media characteristics like video resolution and bit rates)_, and is provided to the client prior to the streaming session. Thereby, segments are served on a web server and can be downloaded through HTTP standard compliant GET requests. This makes it possible to stream videos at different quality levels, and to switch in the middle of a video from one quality level to another one – if bandwidth permits – on a per segment basis.


SteamGear currently only supports [**MPEG-DASH**](https://www.encoding.com/mpeg-dash/) _(Dynamic Adaptive Streaming over HTTP, ISO/IEC 23009-1)_ , but other adaptive streaming technologies such as Apple HLS, Microsoft Smooth Streaming, will be added soon. Also, Multiple DRM support is yet to be implemented.

&thinsp;

!!! danger "Important"
	
	* StreamGear **MUST** requires FFmpeg executables for its core operations. Follow these dedicated [Platform specific Installation Instructions ➶](../ffmpeg_install/) for its installation.

	* :warning: StreamGear API will throw **RuntimeError**, if it fails to detect valid FFmpeg executables on your system.

	* It is advised to enable logging _([`logging=True`](../params/#logging))_ on the first run for easily identifying any runtime errors.

&thinsp; 

## Mode of Operations

StreamGear works in two independent modes for transcoding which serves different purposes. These modes are as follows:

### A. Single-Source Mode

In this mode, StreamGear transcodes entire video/audio file _(as opposed to frames by frame)_ into a sequence of multiple smaller chunks/segments for streaming. This mode works exceptionally well, when you're transcoding lossless long-duration videos(with audio) for streaming and required no extra efforts or interruptions. But on the downside, the provided source cannot be changed or manipulated before sending onto FFmpeg Pipeline for processing.  This mode can be easily activated by assigning suitable video path as input to [`-video_source`](../params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter, during StreamGear initialization. ***Learn more about this mode [here ➶](../usage/#a-single-source-mode)***

### B. Real-time Frames Mode 

When no valid input is received on [`-video_source`](../params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter, StreamGear API activates this mode where it directly transcodes video-frames _(as opposed to a entire file)_, into a sequence of multiple smaller chunks/segments for streaming. In this mode, StreamGear supports real-time [`numpy.ndarray`](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) frames, and process them over FFmpeg Trancoding pipeline. But on the downside, audio has to added manually _(as separate source)_ for streams. ***Learn more about this mode [here ➶](../usage/#b-real-time-frames-mode)***


&thinsp; 

## Importing

You can import StreamGear API in your program as follows:

```python
from vidgear.gears import StreamGear
```

&thinsp; 

## Watch Demo

Watch StreamGear transcoded MPEG-DASH Stream:

<div id="player" align="middle" ></div>
<p align="middle"><sub><i>Powered by <a href="https://github.com/clappr/clappr" title="clappr">clappr</a> & <a href="https://github.com/google/shaka-player" title="shaka-player">shaka-player</a></i></sub></p>

!!! note  "This video assets _(MPD and segments)_ are hosted on [GitHub Repository](https://github.com/abhiTronix/streamgear_chunks) and served with [raw.githack.com](https://raw.githack.com)" 

&thinsp;

## Recommended Stream Players

### GUI Players

- [x] **[MPV Player](https://mpv.io/):** _(recommended)_ MPV is a free, open source, and cross-platform media player. It supports a wide variety of media file formats, audio and video codecs, and subtitle types. 
- [x] **[VLC Player](https://www.videolan.org/vlc/releases/3.0.0.html):** VLC is a free and open source cross-platform multimedia player and framework that plays most multimedia files as well as DVDs, Audio CDs, VCDs, and various streaming protocols.
- [x] **[Parole](https://docs.xfce.org/apps/parole/start):** _(UNIX only)_  Parole is a modern simple media player based on the GStreamer framework for Unix and Unix-like operating systems. 

### Command-Line Players

- [x] **[MP4Client](https://github.com/gpac/gpac/wiki/MP4Client-Intro):** [GPAC](https://gpac.wp.imt.fr/home/) provides a highly configurable multimedia player called MP4Client. GPAC itself is an open source multimedia framework developed for research and academic purposes, and used in many media production chains.
- [x] **[ffplay](https://ffmpeg.org/ffplay.html):** FFplay is a very simple and portable media player using the FFmpeg libraries and the SDL library. It is mostly used as a testbed for the various FFmpeg APIs. 

### Online Players

!!! tip "To run Online players locally, you'll need a HTTP server. For creating one yourself, See [this well-curated list  ➶](https://gist.github.com/abhiTronix/7d2798bc9bc62e9e8f1e88fb601d7e7b)"

- [x] **[Clapper](https://github.com/clappr/clappr):** Clappr is an extensible media player for the web.
- [x] **[Shaka Player](https://github.com/google/shaka-player):** Shaka Player is an open-source JavaScript library for playing adaptive media in a browser.
- [x] **[MediaElementPlayer](https://github.com/mediaelement/mediaelement):** MediaElementPlayer is a complete HTML/CSS audio/video player.

&thinsp;

## Usage Examples

<div class="zoom">
<a href="../usage/">See here 🚀</a>
</div>

## Parameters

<div class="zoom">
<a href="../params/">See here 🚀</a>
</div>

## Reference

<div class="zoom">
<a href="../../../bonus/reference/streamgear/">See here 🚀</a>
</div>


## FAQs

<div class="zoom">
<a href="../../../help/streamgear_faqs/">See here 🚀</a>
</div>

&thinsp;