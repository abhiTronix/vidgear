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

# StreamGear API 


<figure>
  <img src="../../../assets/images/streamgear_flow.webp" loading="lazy" alt="StreamGear Flow Diagram" />
  <figcaption>StreamGear API's generalized workflow</figcaption>
</figure>


## Overview

> StreamGear automates transcoding workflow for generating _Ultra-Low Latency, High-Quality, Dynamic & Adaptive Streaming Formats (such as MPEG-DASH and Apple HLS)_ in just few lines of python code. 

StreamGear provides a standalone, highly extensible, and flexible wrapper around [**FFmpeg**](https://ffmpeg.org/) multimedia framework for generating chunked-encoded media segments of the content.

SteamGear is an out-of-the-box solution for transcoding source videos/audio files & real-time video frames and breaking them into a sequence of multiple smaller chunks/segments of suitable lengths. These segments make it possible to stream videos at different quality levels _(different bitrates or spatial resolutions)_ and can be switched in the middle of a video from one quality level to another – if bandwidth permits – on a per-segment basis. A user can serve these segments on a web server that makes it easier to download them through HTTP standard-compliant GET requests.

SteamGear currently supports [**MPEG-DASH**](https://www.encoding.com/mpeg-dash/) _(Dynamic Adaptive Streaming over HTTP, ISO/IEC 23009-1)_  and [**Apple HLS**](https://developer.apple.com/documentation/http_live_streaming) _(HTTP Live Streaming)_. 

SteamGear also creates a Manifest file _(such as MPD in-case of DASH)_ or a Master Playlist _(such as M3U8 in-case of Apple HLS)_ besides segments that describe these segment information _(timing, URL, media characteristics like video resolution and adaptive bit rates)_ and is provided to the client before the streaming session.

!!! alert "For streaming with older traditional protocols such as RTMP, RTSP/RTP you could use [WriteGear](../../writegear/introduction/) API instead."

&thinsp;

??? new "New in v0.2.2" 

    Apple HLS support was added in `v0.2.2`.


!!! danger "Important"
	
	* StreamGear **MUST** requires FFmpeg executables for its core operations. Follow these dedicated [Platform specific Installation Instructions ➶](../ffmpeg_install/) for its installation.

	* :warning: StreamGear API will throw **RuntimeError**, if it fails to detect valid FFmpeg executable on your system.

	* It is advised to enable logging _([`logging=True`](../params/#logging))_ on the first run for easily identifying any runtime errors.

!!! tip "Useful Links"
    
    - Checkout [this detailed blogpost](https://ottverse.com/mpeg-dash-video-streaming-the-complete-guide/) on how MPEG-DASH works.
    - Checkout [this detailed blogpost](https://ottverse.com/hls-http-live-streaming-how-does-it-work/) on how HLS works.
    - Checkout [this detailed blogpost](https://ottverse.com/hls-http-live-streaming-how-does-it-work/) for HLS vs. MPEG-DASH comparison.


&thinsp; 

## Mode of Operations

StreamGear primarily operates in following independent modes for transcoding:


??? warning "Real-time Frames Mode is NOT Live-Streaming."

    Rather, you can enable live-streaming in Real-time Frames Mode by using the exclusive [`-livestream`](../params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter in StreamGear API. Checkout [this usage example](../rtfm/usage/#bare-minimum-usage-with-live-streaming) for more information.


- [**Single-Source Mode**](../ssm/overview): In this mode, StreamGear **transcodes entire video file** _(as opposed to frame-by-frame)_ into a sequence of multiple smaller chunks/segments for streaming. This mode works exceptionally well when you're transcoding long-duration lossless videos(with audio) for streaming that required no interruptions. But on the downside, the provided source cannot be flexibly manipulated or transformed before sending onto FFmpeg Pipeline for processing. 

- [**Real-time Frames Mode**](../rtfm/overview): In this mode, StreamGear directly **transcodes frame-by-frame** _(as opposed to a entire video file)_, into a sequence of multiple smaller chunks/segments for streaming. This mode works exceptionally well when you desire to flexibility manipulate or transform [`numpy.ndarray`](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) frames in real-time before sending them onto FFmpeg Pipeline for processing. But on the downside, audio has to added manually _(as separate source)_ for streams. 

&thinsp; 

## Importing

You can import StreamGear API in your program as follows:

```python
from vidgear.gears import StreamGear
```

&thinsp; 


## Watch Demo

=== "Watch MPEG-DASH Stream"

    Watch StreamGear transcoded MPEG-DASH Stream:

    <div class="container">
      <div class="video">
        <div class="embed-responsive embed-responsive-16by9">
          <div id="player_dash" class="embed-responsive-item"></div>
        </div>
      </div>
    </div>
    <p align="middle">Powered by <a href="https://github.com/clappr/clappr" title="clappr">clappr</a> & <a href="https://github.com/google/shaka-player" title="shaka-player">shaka-player</a></p>

    !!! info  "This video assets _(Manifest and segments)_ are hosted on [GitHub Repository](https://github.com/abhiTronix/vidgear-docs-additionals) and served with [raw.githack.com](https://raw.githack.com)" 

    !!! quote "Video Credits: [**"Tears of Steel"** - Project Mango Teaser](https://mango.blender.org/download/)"

=== "Watch APPLE HLS Stream"

    Watch StreamGear transcoded APPLE HLS Stream:

    <div class="container">
      <div class="video">
        <div class="embed-responsive embed-responsive-16by9">
          <div id="player_hls" class="embed-responsive-item"></div>
        </div>
      </div>
    </div>
    <p align="middle">Powered by <a href="https://github.com/clappr/clappr" title="clappr">clappr</a> & <a href="https://github.com/clappr/hlsjs-playback" title="hlsjs-playback">HlsjsPlayback</a></p>

    !!! info  "This video assets _(Playlist and segments)_ are hosted on [GitHub Repository](https://github.com/abhiTronix/vidgear-docs-additionals) and served with [raw.githack.com](https://raw.githack.com)" 

    !!! quote "Video Credits: [**"Sintel"** - Project Durian Teaser](https://durian.blender.org/download/)"

&thinsp;

## Recommended Players

=== "GUI Players"
    - [x] **[MPV Player](https://mpv.io/):** _(recommended)_ MPV is a free, open source, and cross-platform media player. It supports a wide variety of media file formats, audio and video codecs, and subtitle types. 
    - [x] **[VLC Player](https://www.videolan.org/vlc/releases/3.0.0.html):** VLC is a free and open source cross-platform multimedia player and framework that plays most multimedia files as well as DVDs, Audio CDs, VCDs, and various streaming protocols.
    - [x] **[Parole](https://docs.xfce.org/apps/parole/start):** _(UNIX only)_  Parole is a modern simple media player based on the GStreamer framework for Unix and Unix-like operating systems. 

=== "Command-Line Players"
    - [x] **[MP4Client](https://github.com/gpac/gpac/wiki/MP4Client-Intro):** [GPAC](https://gpac.wp.imt.fr/home/) provides a highly configurable multimedia player called MP4Client. GPAC itself is an open source multimedia framework developed for research and academic purposes, and used in many media production chains.
    - [x] **[ffplay](https://ffmpeg.org/ffplay.html):** FFplay is a very simple and portable media player using the FFmpeg libraries and the SDL library. It is mostly used as a testbed for the various FFmpeg APIs. 

=== "Online Players"
    !!! alert "To run Online players locally, you'll need a HTTP server. For creating one yourself, See [this well-curated list  ➶](https://gist.github.com/abhiTronix/7d2798bc9bc62e9e8f1e88fb601d7e7b)"

    - [x] **[Clapper](https://github.com/clappr/clappr):** Clappr is an extensible media player for the web.
    - [x] **[Shaka Player](https://github.com/google/shaka-player):** Shaka Player is an open-source JavaScript library for playing adaptive media in a browser.
    - [x] **[MediaElementPlayer](https://github.com/mediaelement/mediaelement):** MediaElementPlayer is a complete HTML/CSS audio/video player.
    - [x] **[Native MPEG-Dash + HLS Playback](https://chrome.google.com/webstore/detail/native-mpeg-dash-%20-hls-pl/cjfbmleiaobegagekpmlhmaadepdeedn?hl=en)(Chrome Extension):** Allow the browser to play HLS (m3u8) or MPEG-Dash (mpd) video urls 'natively' on chrome browsers.

&thinsp;

## Parameters

<div>
<a href="../params/">See here 🚀</a>
</div>

## References

<div>
<a href="../../../bonus/reference/streamgear/">See here 🚀</a>
</div>


## FAQs

<div>
<a href="../../../help/streamgear_faqs/">See here 🚀</a>
</div>

&thinsp;