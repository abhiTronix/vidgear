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

# StreamGear FAQs

&thinsp;

## What is StreamGear API and what does it do?

**Answer:** StreamGear automates transcoding workflow for generating _Ultra-Low Latency, High-Quality, Dynamic & Adaptive Streaming Formats (such as MPEG-DASH)_ in just few lines of python code. _For more info. see [StreamGear doc ➶](../../gears/streamgear/introduction/)_

&thinsp;

## How to get started with StreamGear API?

**Answer:** See [StreamGear doc ➶](../../gears/streamgear/introduction/). Still in doubt, then ask us on [Gitter ➶](https://gitter.im/vidgear/community) Community channel.

&thinsp;

## What is `.mpd` file created with StreamGear?

**Answer:** SteamGear also creates a Manifest file _(such as MPD in-case of DASH)_ besides segments that describe these segment information _(timing, URL, media characteristics like video resolution and bit rates)_ and is provided to the client before the streaming session.

&thinsp;

## How to play Streaming Assets created with StreamGear API?

**Answer:** You can easily feed Manifest file(`.mpd`) to DASH Supported Players Input but sure encoded chunks are present along with it. See this list of [recommended players ➶](../../gears/streamgear/introduction/#recommended-stream-players)

&thinsp;

## What Adaptive Streaming Formats are supported yet?

**Answer:** SteamGear currently only supports [**MPEG-DASH**](https://www.encoding.com/mpeg-dash/) _(Dynamic Adaptive Streaming over HTTP, ISO/IEC 23009-1)_ , but other adaptive streaming technologies such as Apple HLS, Microsoft Smooth Streaming, will be added soon.

&thinsp;

## Is DRM Encryption supported in StreamGear API?

**Answer:** No, DRM Encryption is **NOT** supported yet.

&thinsp;

## How to create additional streams in StreamGear API?

**Answer:** [See this example ➶](../../gears/streamgear/ssm/usage/#usage-with-additional-streams)

&thinsp;


## How to use StreamGear API with OpenCV?

**Answer:** [See this example ➶](../../gears/streamgear/rtfm/usage/#bare-minimum-usage-with-opencv)

&thinsp;

## How to use StreamGear API with real-time frames?

**Answer:** See [Real-time Frames Mode ➶](../../gears/streamgear/rtfm/overview)

&thinsp;

## Is Real-time Frames Mode only used for Live-Streaming?

**Answer:** Real-time Frame Modes and Live-Streaming are completely different terms and not directly related. 

- **Real-time Frame Mode** is one of [primary mode](../../gears/streamgear/introduction/#mode-of-operations) for directly transcoding real-time [`numpy.ndarray`](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) video-frames _(as opposed to a entire file)_ into a sequence of multiple smaller chunks/segments for streaming. 

- **Live-Streaming** is feature of StreamGear's primary modes that activates behaviour where chunks will contain information for few new frames only and forgets all previous ones for low latency streaming. It can be activated for any primary mode using exclusive [`-livestream`](../../gears/streamgear/params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter.


## How to use Hardware/GPU encoder for StreamGear trancoding?

**Answer:** [See this example ➶](../../gears/streamgear/rtfm/usage/#usage-with-hardware-video-encoder)

&thinsp;