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

# FFGear API

<figure>
  <img src="../../assets/images/ffgear.png" alt="FFGear Functional Block Diagram" loading="lazy" class="center" />
  <figcaption>FFGear API's generalized workflow</figcaption>
</figure>

## Overview

> FFGear is a **multi-threaded , high-performance :zap:** wrapper around [**DeFFcode's FFdecoder API**](https://abhitronix.github.io/deffcode/latest/reference/ffdecoder/) that compiles and executes an :simple-ffmpeg: FFmpeg pipeline inside a subprocess pipe for generating real-time, low-overhead, lightning-fast decoded video frames in Python.

FFGear API provides **direct, transparent access** to the full FFdecoder feature-set, including:

- [x] **Hardware-Accelerated Decoding** — CUDA/CUVID and other `-hwaccel` backends for GPU-powered decoding :zap:
- [x] **Flexible Pixel Formats** — any FFmpeg-supported `-pix_fmt` (e.g. `bgr24`, `yuv420p`, `gray`), with an optional OpenCV-compatibility patch (`-enforce_cv_patch`) for YUV/NV layouts.
- [x] **Per-Frame Metadata Extraction** — asynchronous `showinfo` filter integration via `-extract_metadata`, yielding `(frame, metadata)` tuples with `frame_num`, `pts_time`, `is_keyframe`, and `frame_type`.
- [x] **Complex Filtergraphs** — live simple (`-vf`) and complex (`-filter_complex`) FFmpeg filter pipelines.
- [x] **Multi-Input Sources** — multiple simultaneous inputs routed via `-map` or `-filter_complex`.
- [x] **Wide Source Support** — USB/Virtual/IP camera feeds, multimedia files, image sequences, desktop screen, and network protocols (`HTTP(s)`, `RTP/RTSP`, etc.).

Internally, FFGear employs the [**Threaded Queue mode**](../../bonus/TQM/) (configurable via `QUEUE_SIZE`, `THREADED_QUEUE_MODE`, `THREAD_TIMEOUT`) for zero-bottleneck asynchronous frame delivery, and maintains the standard OpenCV-Python coding syntax for drop-in integration.

Similar to CamGear, FFGear also supports the `yt_dlp` backend via `stream_mode=True` for seamlessly pipelining live video-frames and metadata from streaming services like YouTube, Dailymotion, Twitch, and [many more ➶](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md#supported-sites).

&thinsp;

!!! warning "FFGear requires the `deffcode` library"

    FFGear API **MUST** have the [`deffcode`][deffcode] library installed, along with a valid FFmpeg executable. Any failure in detection will raise `ImportError`/`RuntimeError` immediately.

    Install via pip:

    ```sh
    pip install deffcode
    ```

    For FFmpeg installation, see [FFmpeg Installation ➶](advanced/ffmpeg_install/)

!!! tip "Helpful Tips"

    * Enable `logging=True` on the first run to easily identify any runtime errors.

    * FFGear follows the same `.start()` → `.read()` → `.stop()` pattern as CamGear — it's a drop-in upgrade for FFmpeg-powered sources.

    * Use `ytv_metadata` class variable to retrieve streaming video metadata when using `stream_mode=True`.

&thinsp;

## Usage Examples

<div>
<a href="usage/">See here 🚀</a>
</div>

!!! example "After going through FFGear Usage Examples, Checkout more of its advanced configurations [here ➶](advanced/)"

## Parameters

<div>
<a href="params/">See here 🚀</a>
</div>

## References

<div>
<a href="../../bonus/reference/ffgear/">See here 🚀</a>
</div>


&thinsp;

[deffcode]:https://github.com/abhiTronix/deffcode
[yt_dlp]:https://github.com/yt-dlp/yt-dlp
[ffmpeg]:https://www.ffmpeg.org/
