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

# FFGear FAQs

&thinsp;

## What is FFGear API and what does it do?

**Answer:** FFGear is a multi-threaded, high-performance wrapper around [DeFFcode's FFdecoder API](https://abhitronix.github.io/deffcode/latest/reference/ffdecoder/) that compiles and executes an FFmpeg pipeline inside a subprocess pipe for generating real-time, low-overhead, lightning-fast decoded video frames in Python. It supports hardware-accelerated decoding, flexible pixel formats, per-frame metadata extraction, complex filtergraphs, and wide source support (cameras, files, network streams, streaming services). _For more info. see [FFGear doc ➶](../../gears/ffgear/)._

&thinsp;

## What are the requirements for FFGear API?

**Answer:** FFGear requires the `deffcode` library and a valid FFmpeg executable on your system. Install `deffcode` via pip:

```sh
pip install deffcode
```

For FFmpeg installation, follow the [dedicated FFmpeg Installation guide ➶](../../gears/ffgear/advanced/ffmpeg_install/).

&thinsp;

## How is FFGear different from CamGear?

**Answer:** Both CamGear and FFGear follow the same `.start()` → `.read()` → `.stop()` coding pattern and support streaming URLs via `yt_dlp`. The key differences are:

- **CamGear** is built on top of OpenCV's `VideoCapture` and is ideal for general-purpose camera/file/network stream reading.
- **FFGear** is built on top of [DeFFcode's FFdecoder](https://abhitronix.github.io/deffcode/latest/), which drives FFmpeg directly via subprocess pipes, giving you hardware-accelerated decoding, any FFmpeg pixel format, live filtergraphs, per-frame metadata, and multi-input pipelines — none of which are available in CamGear.

Use FFGear when you need GPU decoding, advanced FFmpeg pipelines, or YUV pixel formats for throughput-sensitive workloads.

&thinsp;

## I'm only familiar with OpenCV, how do I get started with FFGear?

**Answer:** FFGear maintains the same OpenCV-like coding syntax as CamGear. First, refer to the [Switching from OpenCV](../../switch_from_cv/#switching-the-videocapture-apis) guide, then go through [FFGear documentation ➶](../../gears/ffgear/). If you still have doubts, ask us on [Gitter ➶](https://gitter.im/vidgear/community) Community channel.

&thinsp;

## How to change the pixel format of decoded frames in FFGear?

**Answer:** Use the [`frame_format`](../../gears/ffgear/params/#frame_format) parameter. It accepts any FFmpeg-supported pixel format string (e.g. `"bgr24"`, `"rgb24"`, `"gray"`, `"yuv420p"`). The default is `"bgr24"` for direct OpenCV compatibility. See [Usage Examples ➶](../../gears/ffgear/usage/#using-ffgear-with-different-pixel-formats).

&thinsp;

## How to use a hardware decoder (GPU) in FFGear?

**Answer:** Pass the `-vcodec` FFmpeg parameter via the `options` dictionary to specify a hardware decoder like `h264_cuvid` (NVIDIA CUVID) or use `-hwaccel cuda` via `-ffprefixes`. See the [Hardware-Accelerated Decoding ➶](../../gears/ffgear/advanced/#hardware-accelerated-decoding) advanced examples.

&thinsp;

## What is `-enforce_cv_patch` and when do I need it?

**Answer:** `-enforce_cv_patch` is an exclusive FFdecoder parameter that makes YUV/NV pixel-format frames (e.g. `yuv420p`, `nv12`) directly compatible with OpenCV. When enabled, FFGear automatically converts these frames to BGR colorspace internally — so `read()` returns standard BGR frames ready for `cv2.imshow()` or `cv2.imwrite()`. Enable it whenever you use `frame_format="yuv420p"` or similar YUV formats. See [params ➶](../../gears/ffgear/params/#b-ffdecoder-parameters).

&thinsp;

## What is `-extract_metadata` and what does `read()` return when it's enabled?

**Answer:** `-extract_metadata` enables asynchronous per-frame metadata extraction via FFmpeg's `showinfo` filter. When enabled, FFGear's `read()` returns `(frame, metadata)` tuples instead of bare frames. The `metadata` dict contains `frame_num`, `pts_time`, `is_keyframe`, and `frame_type`. This is useful for keyframe-only AI inference, VFR synchronization, and timestamped pipelines. See [advanced examples ➶](../../gears/ffgear/advanced/#per-frame-metadata-extraction).

!!! warning "`-extract_metadata` is incompatible with `-filter_complex`. Both cannot be active simultaneously."

&thinsp;

## How to apply FFmpeg filters in FFGear?

**Answer:** Pass the `-vf` FFmpeg parameter (for simple filtergraphs) or `-filter_complex` (for multi-input graphs) via the `options` dictionary. See [Usage Examples ➶](../../gears/ffgear/usage/#using-ffgear-with-simple-ffmpeg-filtergraphs) and [Advanced Usage ➶](../../gears/ffgear/advanced/#complex-ffmpeg-filtergraphs).

&thinsp;

## How to open RTSP/RTP network streams with FFGear?

**Answer:** Simply provide the RTSP/RTP URL as the `source` parameter. You can also pass FFmpeg transport parameters via `options` (e.g. `{"-rtsp_transport": "tcp"}`). See [Usage Examples ➶](../../gears/ffgear/usage/#using-ffgear-with-network-streams).

&thinsp;

## How to change quality and parameters of YouTube Streams with FFGear?

**Answer:** FFGear provides exclusive `STREAM_RESOLUTION` _(for specifying stream resolution)_ and `STREAM_PARAMS` _(for specifying underlying `yt_dlp` parameters)_ attributes with its [`options`](../../gears/ffgear/params/#a-ffgear-exclusive-parameters) dictionary parameter. See [this Bonus Example ➶](../ffgear_ex/#using-ffgear-with-variable-yt_dlp-parameters-in-stream-mode).

&thinsp;

## How to loop a video indefinitely with FFGear?

**Answer:** Use the `-stream_loop` FFmpeg prefix via the `-ffprefixes` option. Set it to `-1` for infinite looping:

```python
options = {"-ffprefixes": ["-stream_loop", "-1"]}
FFGear(source="myvideo.mp4", **options).start()
```

See [Usage Examples ➶](../../gears/ffgear/usage/#using-ffgear-with-video-looping).

&thinsp;

## Can I use FFGear with multiple camera devices at the same time?

**Answer:** Yes. Simply create separate FFGear instances for each device. See [this Bonus Example ➶](../ffgear_ex/#synchronizing-two-ffgear-sources).

&thinsp;

## How to use FFGear with WriteGear?

**Answer:** FFGear follows the same read-loop pattern as CamGear, so it integrates directly with WriteGear. See [this Bonus Example ➶](../ffgear_ex/#using-ffgear-with-writegears-compression-mode).

&thinsp;

## How to use FFGear with NetGear for network streaming?

**Answer:** Use FFGear on the server side to decode frames and send them via NetGear. See [this Bonus Example ➶](../ffgear_ex/#using-ffgear-with-netgear-server).

&thinsp;

## How to configure the internal queue in FFGear?

**Answer:** Use `QUEUE_SIZE` (int, default `96`), `THREADED_QUEUE_MODE` (bool, default `True`), and `THREAD_TIMEOUT` (float or None, default `None`) in the `options` dictionary. See [params ➶](../../gears/ffgear/params/#a-ffgear-exclusive-parameters).

&thinsp;

## Can FFGear decode 4K/8K video?

**Answer:** Yes, FFGear can decode up to 4K/8K video if your system hardware and FFmpeg build support it. Hardware-accelerated decoding (e.g. via CUVID or VAAPI) is strongly recommended for high-resolution sources to avoid CPU bottlenecks.

&thinsp;

## Why is FFGear throwing `RuntimeError: Source is invalid or unreadable!`?

**Answer:** FFGear will throw `RuntimeError` if it cannot read even a single frame from the given `source`. Common causes include:
- Invalid or non-existent file path.
- Unsupported codec without the right hardware decoder or FFmpeg build.
- Network stream not available at the given URL.

Enable `logging=True` to get detailed FFmpeg output for diagnosing the issue.

&thinsp;

## Why is FFGear throwing `ImportError` about `deffcode`?

**Answer:** FFGear requires the `deffcode` library. Install it via:

```sh
pip install deffcode
```

&thinsp;

## Why is FFGear throwing warning that Threaded Queue Mode is disabled?

**Answer:** That's normal behavior when `THREADED_QUEUE_MODE=False` is set in `options`. Please read about [Threaded Queue Mode ➶](../../bonus/TQM/) to understand its implications.

&thinsp;
