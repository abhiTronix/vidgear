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

# FFGear API Parameters


&thinsp;


## **`source`**

!!! warning "FFGear API will throw `RuntimeError` if `source` provided is invalid or unreadable."

This parameter defines the source for the input stream. FFGear passes it directly to [FFdecoder API](https://abhitronix.github.io/deffcode/latest/reference/ffdecoder/params/#source).

**Data-Type:** Any

**Default Value:** `None`

Its valid input can be one of the following:

- [x] **Index (*integer* or *string of integer*):** _Valid index of the connected camera device, e.g. `0`, `1`, `"0"`, `"-1"`, etc.:_

    ```python
    FFGear(source=0)
    ```

- [x] **Filepath (*string*):** _Valid path of the video file, for e.g. `"myvideo.mp4"`:_

    ```python
    FFGear(source="myvideo.mp4")
    ```

- [x] **Streaming Service URL (*string*):** _Valid streaming URL when `stream_mode=True` is enabled:_

    ```python
    FFGear(source="https://youtu.be/bvetuLwJIkA", stream_mode=True)
    ```

- [x] **Network Stream Address (*string*):** _Valid `http(s)`, `rtsp`, `rtp`, `rtmp`, etc. network address:_

    ```python
    FFGear(source="rtsp://192.168.1.10:554/stream")
    ```

- [x] **Image Sequence (*string with glob pattern*):** _Valid image sequence glob, e.g. `"frames/%04d.jpg"`:_

    ```python
    FFGear(source="frames/%04d.jpg")
    ```

&nbsp;

## **`stream_mode`**

This parameter enables **Stream Mode** for handling streaming URLs via the `yt_dlp` backend. When `True`, FFGear will interpret the given `source` as a streaming service URL.

**Data-Type:** Boolean

**Default Value:** `False`

**Usage:**

!!! info "Supported Streaming Websites"

    The complete list of all supported Streaming Websites URLs can be found [here ➶](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md#supported-sites)

```python
FFGear(source="https://youtu.be/bvetuLwJIkA", stream_mode=True)
```

!!! example "Its complete usage example is given [here ➶](usage/#using-ffgear-with-streaming-websites)."

&nbsp;

## **`source_demuxer`**

This parameter specifies the FFmpeg demuxer for the source, required when the source type cannot be auto-detected. Common values include `"v4l2"` (Linux), `"dshow"` (Windows), `"avfoundation"` (macOS).

**Data-Type:** String or None

**Default Value:** `None`

**Usage:**

```python
# Linux: use v4l2 demuxer for webcam device
FFGear(source="/dev/video0", source_demuxer="v4l2")
```

!!! tip "Platform-specific demuxers"

    | Platform | Demuxer |
    |:--------:|:--------|
    | :fontawesome-brands-windows: Windows | `dshow` |
    | :material-linux: Linux | `v4l2` (or `video4linux2`) |
    | :material-apple: macOS | `avfoundation` |

&nbsp;

## **`frame_format`**

This parameter specifies the **pixel layout** for decoding frames. It accepts any FFmpeg-supported pixel format string.

**Data-Type:** String

**Default Value:** `"bgr24"`

**Usage:**

!!! tip "Use `ffmpeg -pix_fmts` terminal command to list all FFmpeg-supported pixel formats."

```python
# decode as BGR24 (default, OpenCV-compatible)
FFGear(source="myvideo.mp4", frame_format="bgr24")

# decode as grayscale
FFGear(source="myvideo.mp4", frame_format="gray")

# decode as YUV420p (fastest throughput; requires -enforce_cv_patch for OpenCV compatibility)
FFGear(source="myvideo.mp4", frame_format="yuv420p") # auto-convert YUV420p → OpenCV Compatible in FFGear
```

!!! example "YUV420p + `-enforce_cv_patch` usage example is given [here ➶](advanced/#hardware-accelerated-decoding-with-yuv420p-output)."

&nbsp;

## **`custom_ffmpeg`**

This parameter specifies a custom path to the FFmpeg executable. Useful when FFmpeg is not on `PATH` or you want to use a specific version.

**Data-Type:** String

**Default Value:** `""` _(uses system FFmpeg)_

**Usage:**

```python
FFGear(source="myvideo.mp4", custom_ffmpeg="/opt/ffmpeg/bin/ffmpeg")
```

&nbsp;

## **`logging`**

This parameter enables logging _(if `True`)_, essential for debugging.

**Data-Type:** Boolean

**Default Value:** `False`

**Usage:**

```python
FFGear(source="myvideo.mp4", logging=True)
```

&nbsp;

## **`options`**

This parameter provides the ability to configure both **FFdecoder-specific parameters** (passed directly to FFdecoder API) and **FFGear queue-tuning parameters**.

**Data-Type:** Dictionary

**Default Value:** `{}`

&thinsp;

### A. FFGear-Exclusive Parameters

The following `options` keys are consumed by FFGear itself and are **not** forwarded to FFdecoder:

| Parameter | Type | Default | Description |
|:----------|:----:|:-------:|:------------|
| `THREADED_QUEUE_MODE` | bool | `True` | Enables producer-consumer threaded queue for zero-bottleneck frame delivery. Disable only for single-threaded pipelines. |
| `QUEUE_SIZE` | int | `96` | Maximum number of frames buffered in the queue. Increase for smoother playback on high-latency sources. |
| `THREAD_TIMEOUT` | float or None | `None` | Timeout (in seconds) for queue `get()` and thread `join()` calls. `None` means block indefinitely. |
| `STREAM_RESOLUTION` | str | `"best"` | _(Stream Mode only)_ Desired stream resolution, e.g. `"720p"`, `"1080p"`, `"best"`. |
| `STREAM_PARAMS` | dict | `{}` | _(Stream Mode only)_ Extra `yt_dlp` parameters forwarded to the `YT_backend`. |

**Usage:**

```python
options = {
    "THREADED_QUEUE_MODE": True,
    "QUEUE_SIZE": 128,
    "THREAD_TIMEOUT": 5.0,
}
FFGear(source="myvideo.mp4", **options)
```

&thinsp;

### B. FFdecoder Parameters

All remaining keys in `options` are forwarded **directly** to [FFdecoder's `ffparams` dictionary](https://abhitronix.github.io/deffcode/latest/reference/ffdecoder/params/#ffparams). This gives you full access to FFdecoder's exclusive parameters:

| Parameter | Type | Description |
|:----------|:----:|:------------|
| `-vcodec` | str or None | Force a specific video decoder (e.g. `"h264_cuvid"`). |
| `-hwaccel` | str | Hardware acceleration backend (e.g. `"cuda"`, `"vaapi"`). |
| `-vf` | str | Simple FFmpeg video filtergraph (e.g. `"scale=1280:720,hflip"`). |
| `-filter_complex` | str | Complex FFmpeg filtergraph for multi-input routing. |
| `-enforce_cv_patch` | bool | Enables OpenCV-compatibility patch for YUV/NV pixel formats. |
| `-extract_metadata` | bool | Enables per-frame metadata extraction via `showinfo` filter. `read()` returns `(frame, metadata)` tuples. |
| `-ffprefixes` | list | Additional FFmpeg CLI prefixes (e.g. `["-stream_loop", "-1"]`). |

!!! tip "Full list of all FFdecoder parameters can be found [here ➶](https://abhitronix.github.io/deffcode/latest/reference/ffdecoder/params/)"

**Usage:**

```python
options = {
    "-vcodec": "h264_cuvid",         # hardware decoder
    "-enforce_cv_patch": True,        # OpenCV YUV compatibility
    "-vf": "scale=1280:720",          # scale filter
    "QUEUE_SIZE": 64,                 # FFGear queue size
}
FFGear(source="myvideo.mp4", frame_format="yuv420p", **options)
```

&nbsp;
