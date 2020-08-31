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

# StreamGear FAQs

&thinsp;

## What is StreamGear API and what does it do?

**Answer:** SteamGear API ***automatically transcodes source videos/audio files & real-time frames, and breaks them into a sequence of multiple smaller chunks/segments (typically 2-4 seconds in length) at different quality levels (i.e. different bitrates or spatial resolutions)***. It also creates a Manifest file _(MPD in-case of DASH)_ that describes these segment information _(timing, URL, media characteristics like video resolution and bit rates)_, and is provided to the client prior to the streaming session. Thereby, segments are served on a web server and can be downloaded through HTTP standard compliant GET requests. This makes it possible to stream videos at different quality levels, and to switch in the middle of a video from one quality level to another one – if bandwidth permits – on a per segment basis. _For more info. see [StreamGear doc ➶](../../gears/streamgear/overview/)_

&thinsp;

## How to get started with StreamGear API?

**Answer:** See [StreamGear doc ➶](../../gears/streamgear/overview/). Still in doubt, then ask us on [Gitter ➶](https://gitter.im/vidgear/community) Community channel.

&thinsp;

## What is MPD file created with StreamGear?

**Answer:** The MPD _(Media Presentation Description)_ is an XML file that represents the different qualities of the media content and the individual segments of each quality with HTTP Uniform Resource Locators (URLs). Each MPD could contain one or more Periods. Each of those Periods contains media components such as video components e.g., different view angles or with different codecs, audio components for different languages , subtitle or caption components, etc. Those components have certain characteristics like the bitrate, frame rate, audio-channels, etc. which do not change during one Period. The client is able to adapt according to the available bitrates, resolutions, codecs, etc. that are available in a given Period. 

&thinsp;

## How to play Streaming Assets created with StreamGear API?

**Answer:** You can easily feed Manifest file(`.mpd`) to DASH Supported Players Input. See this list of [recommended players ➶](../../gears/streamgear/overview/#recommended-stream-players)

&thinsp;

##What Adaptive Streaming Formats are supported yet?

**Answer:** SteamGear currently only supports [**MPEG-DASH**](https://www.encoding.com/mpeg-dash/) _(Dynamic Adaptive Streaming over HTTP, ISO/IEC 23009-1)_ , but other adaptive streaming technologies such as Apple HLS, Microsoft Smooth Streaming, will be added soon.

&thinsp;

## Is DRM Encryption supported in StreamGear API?

**Answer:** No, DRM Encryption is not supported yet.

&thinsp;

## How to create additional streams in StreamGear API?

**Answer:** [See this example ➶](../../gears/streamgear/usage/#a2-usage-with-additional-streams)

&thinsp;

## How to use StreamGear API with real-time frames?

**Answer:** See [Real-time Frames Mode ➶](../../gears/streamgear/usage/#b-real-time-frames-mode)

&thinsp;

## How to use StreamGear API with OpenCV?

**Answer:** [See this example ➶](../../gears/streamgear/usage/#b4-bare-minimum-usage-with-opencv)

&thinsp;

## How to use Hardware/GPU encoder for StreamGear trancoding?

**Answer:** [See this example ➶](../../gears/streamgear/usage/#b7-usage-with-hardware-video-encoder)

&thinsp;