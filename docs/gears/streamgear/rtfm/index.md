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

# StreamGear API: Real-time Frames Mode :material-camera-burst:


<figure>
  <img src="../../../assets/images/streamgear_real.webp" loading="lazy" alt="Real-time Frames Mode Flow Diagram"/>
  <figcaption>Real-time Frames Mode generalized workflow</figcaption>
</figure>


## Overview

When no valid input is received on [`-video_source`](../../params/#a-exclusive-parameters) attribute of [`stream_params`](../../params/#supported-parameters) dictionary parameter, StreamGear API activates this mode where it directly transcodes real-time [`numpy.ndarray`](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) video-frames _(as opposed to a entire video file)_ into a sequence of multiple smaller chunks/segments for adaptive streaming. 

This mode works exceptionally well when you desire to flexibility manipulate or transform video-frames in real-time before sending them onto FFmpeg Pipeline for processing. But on the downside, StreamGear **DOES NOT** automatically maps video-source's audio to generated streams with this mode. You need to manually assign separate audio-source through [`-audio`](../../params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter.

SteamGear supports both [**MPEG-DASH**](https://www.encoding.com/mpeg-dash/) _(Dynamic Adaptive Streaming over HTTP, ISO/IEC 23009-1)_  and [**Apple HLS**](https://developer.apple.com/documentation/http_live_streaming) _(HTTP Live Streaming)_ with this mode.

For this mode, StreamGear API provides exclusive [`stream()`](../../../../bonus/reference/streamgear/#vidgear.gears.streamgear.StreamGear.stream) method for directly trancoding video-frames into streamable chunks. 

&emsp;

??? new "New in v0.2.2" 

    Apple HLS support was added in `v0.2.2`.


!!! alert "Real-time Frames Mode itself is NOT Live-Streaming :material-video-wireless-outline:"
    To enable live-streaming in Real-time Frames Mode, use the exclusive [`-livestream`](../params/#a-exclusive-parameters) attribute of the `stream_params` dictionary parameter in the StreamGear API. Checkout [this usage example ➶](../usage/#bare-minimum-usage-with-live-streaming) for more information.


!!! danger "Please Remember :material-police-badge-outline:"

    * Using [`transcode_source()`](../../../../bonus/reference/streamgear/#vidgear.gears.streamgear.StreamGear.transcode_source) function instead of [`stream()`](../../../../bonus/reference/streamgear/#vidgear.gears.streamgear.StreamGear.stream) in Real-time Frames Mode will immediately result in **`RuntimeError`**!

    * **NEVER** assign anything to [`-video_source`](../../params/#a-exclusive-parameters) attribute of [`stream_params`](../../params/#supported-parameters) dictionary parameter, otherwise [Single-Source Mode](../#a-single-source-mode) get activated, and as a result, using [`stream()`](../../../../bonus/reference/streamgear/#vidgear.gears.streamgear.StreamGear.stream) function will throw **`RuntimeError`**!

    * Input framerate defaults to `25.0` fps if [`-input_framerate`](../../params/#a-exclusive-parameters) attribute value not defined. 


&thinsp;

## Usage Examples

<div>
<a href="../usage/">See here 🚀</a>
</div>

!!! example "After going through StreamGear Usage Examples, Checkout more of its advanced configurations [here ➶](../../../help/streamgear_ex/)"


## Parameters

<div>
<a href="../../params/">See here 🚀</a>
</div>

## References

<div>
<a href="../../../../bonus/reference/streamgear/">See here 🚀</a>
</div>


## FAQs

<div>
<a href="../../../../help/streamgear_faqs/">See here 🚀</a>
</div>

&thinsp;