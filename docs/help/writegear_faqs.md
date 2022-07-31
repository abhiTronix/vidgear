
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

# WriteGear FAQs

&nbsp;

## What is WriteGear API and what does it do?

**Answer:** WriteGear handles various powerful Writer Tools that provide us the freedom to do almost anything imagine with multimedia files. _For more info. see [WriteGear doc ➶](../../gears/writegear/introduction/)_

&nbsp;

## I'm only familiar with OpenCV, how to get started with WriteGear API?

**Answer:** First, see [Switching from OpenCV](../../switch_from_cv/#switching-the-videowriter-api), then go through [WriteGear doc](../../gears/writegear/introduction/). Still in doubt, then ask us on [Gitter ➶](https://gitter.im/vidgear/community) Community channel.

&nbsp;

## Why WriteGear is throwing `ValueError`?

**Answer:** WriteGear will exit with `ValueError` if you feed frames of different dimensions or channels.

&nbsp;

## How to install and configure FFmpeg correctly for WriteGear on my machine?

**Answer:** Follow these [Installation Instructions ➶](../../gears/writegear/compression/advanced/ffmpeg_install/) for its installation.

&nbsp;

## Can I use WriteGear directly with OpenCV?

**Answer:** Yes,

* For Compression Mode: See [this usage example ➶](../../gears/writegear/compression/usage/#using-compression-mode-with-opencv).
* For  Non-Compression Mode: See [this usage example ➶](../../gears/writegear/non_compression/usage/#using-non-compression-mode-with-opencv)

&nbsp;

## What FFmpeg's encoders and parameters are supported by WriteGear in compression mode?

**Answer:** See [Supported Parameters ➶](../../gears/writegear/compression/params/#supported-parameters) and [Supported encoders ➶](../../gears/writegear/compression/params/#supported-encoders)

&nbsp;

## What OpenCV's FOURCC and parameters are supported by WriteGear in non-compression mode?

**Answer:** See [Supported Parameters ➶](../../gears/writegear/non_compression/params/#supported-parameters) and [Supported FOURCC ➶](../../gears/writegear/non_compression/params/#supported-fourcc-codecs).

&nbsp;

## Why this FOURCC is not working for me?

**Answer:** Remember not all the FOURCC and Video extensions are compatible and supported by OpenCV VideoWriter Class. You’ll need to try different combinations of FourCC and file extensions. Furthermore, OpenCV does not return any helpful error messages regarding this problem, so it’s pretty much based on _trial and error_.

&nbsp;

## Can I pass my custom FFmpeg commands directly in WriteGear API?

**Answer:** Yes, See [Custom FFmpeg Commands in WriteGear API ➶](../../gears/writegear/compression/advanced/cciw/).

&nbsp;

## How to use specific Hardware Encoder in WriteGear?

**Answer:** See [this usage example ➶](../../gears/writegear/compression/usage/#using-compression-mode-with-hardware-encoders)

&nbsp;


## How to add live audio to WriteGear?

**Answer:** See [this doc ➶](../../gears/writegear/compression/usage/#using-compression-mode-with-live-audio-input)

&nbsp;

## How to separate and merge audio from/to video?

**Answer:** See [these usage examples ➶](../../gears/writegear/compression/advanced/cciw/#usage-examples)

&nbsp;

## Can I live stream to Twitch with WriteGear API?

**Answer:** Yes, See [this usage example ➶](../../gears/writegear/compression/usage/#using-compression-mode-for-live-streaming)

&nbsp;

## Is YouTube-Live Streaming possible with WriteGear?

**Answer:** Yes, See [this bonus example ➶](../writegear_ex/#using-writegears-compression-mode-for-youtube-live-streaming).

&nbsp;

## How to Live-Streaming using RTSP/RTP protocol with WriteGear?

**Answer:** See [this bonus example ➶](../writegear_ex/#using-writegears-compression-mode-for-rtsprtp-live-streaming).

&nbsp;


## How to create MP4 segments from a video stream with WriteGear?

**Answer:** See [this bonus example ➶](../writegear_ex/#using-writegears-compression-mode-creating-mp4-segments-from-a-video-stream).

&nbsp;


## How add external audio file input to video frames?

**Answer:** See [this bonus example ➶](../writegear_ex/#using-writegears-compression-mode-to-add-external-audio-file-input-to-video-frames).

&nbsp;

## Why this FFmpeg parameter is not working for me in compression mode?

**Answer:** If some FFmpeg parameter doesn't work for you, then [tell us on Gitter ➶](https://gitter.im/vidgear/community), and if that doesn't help, then finally [report an issue ➶](../../contribution/issue/)

&nbsp;

## Why WriteGear is switching to Non-compression Mode, even if it is not enable?

**Answer:** In case WriteGear API fails to detect valid FFmpeg executables on your system _(even if Compression Mode is enabled)_, it will automatically fallback to Non-Compression Mode. Follow [Installation Instructions ➶](../../gears/writegear/compression/advanced/ffmpeg_install/) for FFmpeg installation.

&nbsp;