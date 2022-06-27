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

# WriteGear API: Compression Mode

<figure>
  <img src="../../../../assets/images/writegear_cm.png" loading="lazy" alt="Compression Mode" class="center-small shadow"/>
  <figcaption>WriteGear API's Compression Mode generalized workflow</figcaption>
</figure>

## Overview

When [`compression_mode`](../params/#compression_mode) parameter is enabled (.i.e `compression_mode = True`), WriteGear API provides a complete, flexible & robust wrapper around [**FFmpeg**](https://ffmpeg.org/) to encode lossless & compressed multimedia files.

This mode can process real-time video frames into a lossless compressed format with any suitable setting video/audio properties such as _bitrate, codec, framerate, resolution, subtitles, and much more_ in just a few easy lines of code. It can also perform complex tasks such as [Live-Streaming](../usage/#using-compression-mode-for-live-streaming) _(such as for Twitch)_, multiplexing video with audio in real-time (see this [usage example](../usage/#using-compression-mode-with-live-audio-input)) while handling all errors robustly.


&nbsp; 


!!! danger "Important Information"

	* WriteGear **MUST** requires FFmpeg executables for its Compression capabilities. Follow these dedicated [Installation Instructions ➶](../advanced/ffmpeg_install/) for its installation.

	* In case WriteGear API fails to detect valid FFmpeg executables on your system _(even if Compression Mode is enabled)_, it automatically fallbacks to [Non-Compression Mode](../../non_compression/overview/).

	* It is advised to enable logging([`logging = True`](../params/#logging)) to see the FFmpeg command that is being executed in WriteGear's pipeline. This helps you debug any issues/errors easily and make suitable adjustments accordingly. 


!!! tip "You can speed up the execution time by disabling logging (.i.e [`logging = False`](../params/#logging)) for production use, and by tweaking FFmpeg parameters in [`output_params`](../params/#output_params) values. Look into [FFmpeg docs ➶](https://ffmpeg.org/documentation.html) for such hacks."


&nbsp;


## Custom FFmpeg Commands in WriteGear API

WriteGear API now provides the **[`execute_ffmpeg_cmd`](../../../../bonus/reference/writegear/#vidgear.gears.writegear.WriteGear.execute_ffmpeg_cmd) Function** in Compression Mode, that enables the user to pass any custom CLI commands as an input to its internal FFmpeg Pipeline by formating it as a list. 

This function opens endless possibilities of exploiting any FFmpeg supported parameter within WriteGear, without relying on a third-party library/API to do the same, and while doing that it robustly handles all errors/warnings quietly.

!!! tip "A complete guide on `execute_ffmpeg_cmd` Function can be found [here ➶](../advanced/cciw/)"


&nbsp;

## Usage Examples

<div>
<a href="../usage/">See here 🚀</a>
</div>

!!! examples "After going through WriteGear Usage Examples, Checkout more bonus examples [here ➶](../../../help/writegear_ex/)"


## Parameters

<div>
<a href="../params/">See here 🚀</a>
</div>

&thinsp; 