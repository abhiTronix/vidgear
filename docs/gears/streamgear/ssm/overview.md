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

# StreamGear API: Single-Source Mode

<figure>
  <img src="../../../../assets/images/streamgear_flow.webp" loading="lazy" alt="StreamGear Flow Diagram" />
  <figcaption>StreamGear API's generalized workflow</figcaption>
</figure>


## Overview

In this mode, StreamGear transcodes entire video/audio file _(as opposed to frames by frame)_ into a sequence of multiple smaller chunks/segments for streaming. This mode works exceptionally well, when you're transcoding lossless long-duration videos(with audio) for streaming and required no extra efforts or interruptions. But on the downside, the provided source cannot be changed or manipulated before sending onto FFmpeg Pipeline for processing.

SteamGear supports both [**MPEG-DASH**](https://www.encoding.com/mpeg-dash/) _(Dynamic Adaptive Streaming over HTTP, ISO/IEC 23009-1)_  and [**Apple HLS**](https://developer.apple.com/documentation/http_live_streaming) _(HTTP Live Streaming)_ with this mode.

This mode provide [`transcode_source()`](../../../../bonus/reference/streamgear/#vidgear.gears.streamgear.StreamGear.transcode_source) function to process audio-video files into streamable chunks.

This mode can be easily activated by assigning suitable video path as input to [`-video_source`](../../params/#a-exclusive-parameters) attribute of [`stream_params`](../../params/#stream_params) dictionary parameter, during StreamGear initialization.

&emsp;

!!! new "New in v0.2.2" 

    Apple HLS support was added in `v0.2.2`.


!!! warning 

    * Using [`stream()`](../../../../bonus/reference/streamgear/#vidgear.gears.streamgear.StreamGear.stream) function instead of [`transcode_source()`](../../../../bonus/reference/streamgear/#vidgear.gears.streamgear.StreamGear.transcode_source) in Single-Source Mode will instantly result in **`RuntimeError`**!
    * Any invalid value to the [`-video_source`](../../params/#a-exclusive-parameters) attribute will result in **`AssertionError`**! 

&thinsp;

## Usage Examples

<div>
<a href="../usage/">See here 🚀</a>
</div>

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