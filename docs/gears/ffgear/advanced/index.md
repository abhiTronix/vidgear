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

# FFGear API Advanced Usage

> This page covers FFGear's advanced configurations powered by the full FFdecoder feature-set: hardware acceleration, per-frame metadata extraction, complex filtergraphs, and multi-input pipelines.

&thinsp;

## Hardware-Accelerated Decoding

FFGear exposes FFdecoder's full hardware acceleration support via the `options` dictionary. You can use NVIDIA CUVID, CUDA, VAAPI, or any other `-hwaccel` backend supported by your FFmpeg build.

### NVIDIA CUVID (H.264) Decoding

???+ alert "Example Assumptions"

    **Please note that following recipe explicitly assumes:**

    - You're running :fontawesome-brands-linux: Linux with a [**supported NVIDIA GPU**](https://developer.nvidia.com/nvidia-video-codec-sdk).
    - FFmpeg is compiled with `--enable-cuvid --enable-nvenc` flags.
    - Appropriate NVIDIA drivers are installed.

    These assumptions **MAY/MAY NOT** suit your current setup. Use parameters based on your system only.

    ??? danger "Verifying H.264 CUVID decoder support in FFmpeg"
        ```sh
        $ ffmpeg -hide_banner -decoders | grep cuvid

        V..... h264_cuvid  Nvidia CUVID H264 decoder (codec h264)
        V..... hevc_cuvid  Nvidia CUVID HEVC decoder (codec hevc)
        ...
        ```

In this example, we use Nvidia's `h264_cuvid` decoder with `yuv420p` output and the `-enforce_cv_patch` flag for OpenCV compatibility:

```python linenums="1" hl_lines="7-11"
# import required libraries
from vidgear.gears import FFGear
import cv2

# use H.264 CUVID hardware decoder; enable OpenCV patch for YUV frames
options = {
    "-vcodec": "h264_cuvid",    # NVIDIA CUVID hardware decoder
    "-enforce_cv_patch": True,  # auto-convert YUV420p → BGR in FFGear
}

stream = FFGear(
    source="myvideo.mp4",
    frame_format="yuv420p",
    logging=True,
    **options
).start()

# loop over
while True:

    # read BGR frames (auto-converted from YUV420p)
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # Show output window
    cv2.imshow("Output", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
```

!!! info "More information on Nvidia's CUVID can be found [here ➶](https://developer.nvidia.com/blog/nvidia-ffmpeg-transcoding-guide/)"

&nbsp;

### NVIDIA CUDA hwaccel Decoding

???+ alert "Example Assumptions"

    **Please note that following recipe explicitly assumes:**

    - You're running :fontawesome-brands-linux: Linux with a [**supported NVIDIA GPU**](https://developer.nvidia.com/nvidia-video-codec-sdk).
    - FFmpeg is compiled with `--enable-cuda-nvcc --enable-libnpp --enable-cuvid` flags.
    - Appropriate NVIDIA drivers are installed.

    ??? danger "Verifying NVDEC/CUDA support in FFmpeg"
        ```sh
        $ ffmpeg -hide_banner -pix_fmts | grep cuda
        ..H.. cuda    0    0    0

        $ ffmpeg -hide_banner -filters | egrep "cuda|npp"
        ... scale_cuda   V->V  GPU accelerated video resizer
        ... hwdownload   V->V  Copy from a hardware surface to system memory
        ...
        ```

In this example, we use Nvidia's CUDA internal `hwaccel` decoder to keep frames in GPU memory, applying GPU-side scale and FPS filters before downloading as `nv12`:

```python linenums="1" hl_lines="7-24"
# import required libraries
from vidgear.gears import FFGear
import cv2

# CUDA hwaccel: decode in GPU memory, scale & fps in GPU, download as NV12
options = {
    "-vcodec": None,               # skip source decoder, let FFmpeg choose
    "-enforce_cv_patch": True,     # auto-convert NV12 → BGR in FFGear
    "-ffprefixes": [
        "-vsync", "0",             # prevent duplicate frames
        "-hwaccel", "cuda",        # use CUDA accelerator
        "-hwaccel_output_format", "cuda",  # keep frames in GPU memory
    ],
    "-custom_resolution": "null",  # discard source resolution param
    "-framerate": "null",          # discard source framerate param
    "-vf": (
        "scale_cuda=1280:720,"     # GPU-side scale to 1280x720
        "fps=60.0,"                # GPU-side framerate
        "hwdownload,"              # download to system memory
        "format=nv12"             # convert to NV12 pixel format
    ),
}

stream = FFGear(
    source="myvideo.mp4",
    frame_format="null",           # discard source pixel format
    logging=True,
    **options
).start()

# loop over
while True:

    # read BGR frames (auto-converted from NV12)
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # Show output window
    cv2.imshow("Output", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
```

&nbsp;

## Per-Frame Metadata Extraction

> The `-extract_metadata` option enables FFGear to yield `(frame, metadata)` tuples from `read()`, where `metadata` is parsed from FFmpeg's [`showinfo`](https://ffmpeg.org/ffmpeg-filters.html#showinfo) filter via an asynchronous background daemon — leaving the main frame pipe fully unthrottled.

The `metadata` dict contains the following keys:

| Key | Type | Description |
|:----|:----:|:------------|
| `frame_num` | int | Monotonic frame index as emitted by FFmpeg. |
| `pts_time` | float | Presentation timestamp in seconds. |
| `is_keyframe` | bool | `True` if the frame is a keyframe (I-frame). |
| `frame_type` | str | One of `"I"` (keyframe), `"P"` (predictive), `"B"` (bi-predictive), `"?"` (unknown). |

!!! warning "Incompatible with `-filter_complex`"

    `-extract_metadata` cannot be combined with the `-filter_complex` attribute. If both are supplied, a warning is logged and metadata extraction is silently disabled. A pre-existing `-vf` is fine — `showinfo` is automatically comma-chained onto it.

### Keyframe-Only Decoding for AI Inference

> Many Computer Vision workflows — perceptual hashing, scene-change detection, heavyweight AI-model inference _(YOLO, ResNet, etc.)_ — only care about **Keyframes (I-frames)**. Without `-extract_metadata` you'd decode and run your model on every P/B frame, wasting 98%+ of compute.

!!! success "Depending on the GOP size, this pattern reduces downstream processing time by **10–50×** without skipping any scene-boundary information."

```python linenums="1" hl_lines="7-8 18-19"
# import required libraries
from vidgear.gears import FFGear

# enable per-frame metadata extraction
options = {"-extract_metadata": True}

stream = FFGear(
    source="myvideo.mp4",
    frame_format="bgr24",
    logging=True,
    **options
).start()

# loop over
while True:

    # read returns (frame, metadata) tuple when -extract_metadata is enabled
    output = stream.read()

    # check if output is None (end of stream)
    if output is None:
        break

    frame, meta = output

    # OPTIMIZATION: skip non-keyframes entirely
    if not meta["is_keyframe"]:
        continue

    # run heavy AI model on keyframes only (~1-2 fps worth of data)
    # results = heavy_ai_model.predict(frame)
    print(f"Keyframe #{meta['frame_num']} at {meta['pts_time']:.3f}s")

# safely close video stream
stream.stop()
```

&nbsp;

### Variable-Frame-Rate (VFR) Synchronization

> Most modern video sources — smartphones, screen recordings, webcams — are **Variable-Frame-Rate**. Assuming a constant frame rate will drift out of sync very quickly.

With `meta["pts_time"]` you know the **exact presentation timestamp** of every frame:

```python linenums="1" hl_lines="7-8 22-23"
# import required libraries
from vidgear.gears import FFGear

# enable per-frame metadata extraction
options = {"-extract_metadata": True}

stream = FFGear(
    source="screen_recording.mp4",
    frame_format="bgr24",
    **options
).start()

prev_pts = None

# loop over
while True:

    # read returns (frame, metadata) tuple
    output = stream.read()

    if output is None:
        break

    frame, meta = output

    # exact presentation timestamp in seconds
    pts = meta["pts_time"]

    # compute the real inter-frame delta (not the nominal 1/fps value)
    delta_ms = None if prev_pts is None else (pts - prev_pts) * 1000.0
    prev_pts = pts

    # use real delta for per-frame motion/velocity calculations
    # e.g. velocity = displacement_px / delta_ms
    if delta_ms is not None:
        print(f"Frame #{meta['frame_num']} | PTS: {pts:.3f}s | Δt: {delta_ms:.1f}ms")

# safely close video stream
stream.stop()
```

!!! tip "The same `pts_time` stream is what you need to keep processed frames locked to an audio track when re-muxing downstream."

&nbsp;

## Complex FFmpeg Filtergraphs

For advanced multi-input or multi-output pipelines, FFGear supports `-filter_complex` via the `options` dictionary.

!!! warning "Complex filtergraphs (`-filter_complex`) are incompatible with `-extract_metadata`. Both cannot be active simultaneously."

### Overlay Two Video Sources

In this example, we overlay a watermark video on top of a primary source:

```python linenums="1" hl_lines="7-12"
# import required libraries
from vidgear.gears import FFGear
import cv2

# overlay watermark.mp4 on top of primary source at position (10, 10)
options = {
    "-i": "watermark.mp4",         # second input source
    "-filter_complex": (
        "[0:v][1:v]overlay=10:10"  # overlay second input onto first
    ),
    "-map": "0:a?",                # keep audio from primary (if any)
}

stream = FFGear(
    source="myvideo.mp4",
    frame_format="bgr24",
    logging=True,
    **options
).start()

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # Show output window
    cv2.imshow("Output", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
```

&nbsp;

## Using FFGear with WriteGear (Compression Mode)

FFGear integrates seamlessly with VidGear's WriteGear API in Compression Mode for high-quality re-encoding of decoded frames:

```python linenums="1" hl_lines="4 10-14 34"
# import required libraries
from vidgear.gears import FFGear
from vidgear.gears import WriteGear
import cv2

# open source with FFGear
stream = FFGear(source="myvideo.mp4", frame_format="bgr24", logging=True).start()

# define WriteGear output parameters
output_params = {
    "-vcodec": "libx264",
    "-crf": "23",
    "-preset": "fast",
}

# open WriteGear in Compression Mode
writer = WriteGear(output="output.mp4", logging=True, **output_params)

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # write frame to output
    writer.write(frame)

# safely close writer
writer.close()

# safely close stream
stream.stop()
```

&nbsp;
