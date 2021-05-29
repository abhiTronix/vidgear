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

# StreamGear API: Real-time Frames Mode


<figure>
  <img src="../../../../assets/images/streamgear_real.webp" loading="lazy" alt="Real-time Frames Mode Flow Diagram"/>
  <figcaption>Real-time Frames Mode generalized workflow</figcaption>
</figure>


## Overview

When no valid input is received on [`-video_source`](../../params/#a-exclusive-parameters) attribute of [`stream_params`](../../params/#supported-parameters) dictionary parameter, StreamGear API activates this mode where it directly transcodes real-time [`numpy.ndarray`](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) video-frames _(as opposed to a entire file)_ into a sequence of multiple smaller chunks/segments for streaming. 

In this mode, StreamGear **DOES NOT** automatically maps video-source audio to generated streams. You need to manually assign separate audio-source through [`-audio`](../../params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter.

This mode provide [`stream()`](../../../../bonus/reference/streamgear/#vidgear.gears.streamgear.StreamGear.stream) function for directly trancoding video-frames into streamable chunks over the FFmpeg pipeline. 


!!! warning 

    * Using [`transcode_source()`](../../../../bonus/reference/streamgear/#vidgear.gears.streamgear.StreamGear.transcode_source) function instead of [`stream()`](../../../../bonus/reference/streamgear/#vidgear.gears.streamgear.StreamGear.stream) in Real-time Frames Mode will instantly result in **`RuntimeError`**!

    * **NEVER** assign anything to [`-video_source`](../../params/#a-exclusive-parameters) attribute of [`stream_params`](../../params/#supported-parameters) dictionary parameter, otherwise [Single-Source Mode](../#a-single-source-mode) may get activated, and as a result, using [`stream()`](../../../../bonus/reference/streamgear/#vidgear.gears.streamgear.StreamGear.stream) function will throw **`RuntimeError`**!

    * You **MUST** use [`-input_framerate`](../../params/#a-exclusive-parameters) attribute to set exact value of input framerate when using external audio in this mode, otherwise audio delay will occur in output streams.

    * Input framerate defaults to `25.0` fps if [`-input_framerate`](../../params/#a-exclusive-parameters) attribute value not defined. 


&thinsp;

## Usage Examples

<div>
<a href="../usage/">See here 🚀</a>
</div>

&thinsp;