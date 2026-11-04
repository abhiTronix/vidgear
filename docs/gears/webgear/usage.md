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

# WebGear API Usage Examples:

## Requirements

### Installation with Asyncio Support


WebGear API is the part of `asyncio` package of VidGear, thereby you need to install VidGear with asyncio support as follows:

  ```sh
  pip install vidgear[asyncio]
  ```


### ASGI Server

You'll also need to install an ASGI Server to run following WebGear usage examples, and by default WebGear ships the state-of-the-art [**`uvicorn`**](http://www.uvicorn.org/) Server. But you can also use other ASGI server such as [`daphne`](https://github.com/django/daphne/), or [`hypercorn`](https://pgjones.gitlab.io/hypercorn/) with it.


&thinsp;


## Bare-Minimum Usage with Default MJPEG Mode :octicons-stack-16: 

By default _([`streaming_mode="mjpeg"`](params/#streaming_mode))_, WebGear only streams JPEG-encoded frames over HTTP at `/video` endpoint, with no FFmpeg dependency. See [Modes of Operation ➶](../#modes-of-operation) for its pros and cons.

!!! tip "MJPEG Performance Enhancing Attributes :fontawesome-solid-fire-flame-curved:"

    In MJPEG Mode, WebGear provides certain performance enhancing attributes for its [`options`](params/#options) dictionary parameter to cope with performance-throttling.

    !!! warning "Following JPEG Performance Enhancing Attributes only work in MJPEG Mode _(and in H.264 Mode's MJPEG fallback stream)_, and have no effect on H.264 encoding, except `frame_size_reduction`."

    * **`frame_size_reduction`**: _(int/float)_ _This attribute controls the size reduction(in percentage) of the frame to be streamed on Server._ Its value has the most significant effect on WebGear performance: More its value, smaller will be frame size and faster will be live streaming. The value defaults to `25`, and must be lower than `90` _(higher values are faster, with max compression and barely visible frame-size)_ and no lower than `0` _(slowest, no compression, Original frame-size)_. Its recommended value is between `40~60`. This attribute downsizes frames in H.264 Mode too. Its usage is as follows:

        ```python
        options={"frame_size_reduction": 50} #frame-size will be reduced by 50%
        ```
     
    * **Various Encoding Parameters:**

        In MJPEG Mode, the input video frames are first encoded into [**Motion JPEG (M-JPEG or MJPEG**)](https://en.wikipedia.org/wiki/Motion_JPEG) compression format, in which each video frame or interlaced field of a digital video sequence is compressed separately as a JPEG image using [`simplejpeg`](https://gitlab.com/jfolz/simplejpeg) library, before sending onto a server. Therefore, WebGear API provides various attributes to have full control over JPEG encoding performance and quality, which are as follows:

        * **`jpeg_compression_quality`**: _(int/float)_ This attribute controls the JPEG quantization factor. Its value varies from `10` to `100` (the higher is the better quality but performance will be lower). Its default value is `90`. Its usage is as follows:

            ```python
            # activate jpeg encoding and set quality 95%
            options = {"jpeg_compression_quality": 95}
            ```

        * **`jpeg_compression_fastdct`**: _(bool)_ This attribute if True, WebGear API uses fastest DCT method that speeds up decoding by 4-5% for a minor loss in quality. Its default value is also `True`, and its usage is as follows:
        
            ```python
            # activate jpeg encoding and enable fast dct
            options = {"jpeg_compression_fastdct": True}
            ```

        * **`jpeg_compression_fastupsample`**: _(bool)_ This attribute if True, WebGear API use fastest color upsampling method. Its default value is `False`, and its usage is as follows:
        
            ```python
            # activate jpeg encoding and enable fast upsampling
            options = {"jpeg_compression_fastupsample": True}
            ```

Let's implement our Bare-Minimum usage example in MJPEG Mode with these [**Performance Enhancing Attributes** ➶](#mjpeg-performance-enhancing-attributes) for speeding up the output, as follows:

!!! tip "For accessing WebGear on different Client Devices on the network, use `"0.0.0.0"` as host value instead of `"localhost"` on Host Machine. More information can be found [here ➶](../../help/webgear_faqs/#is-it-possible-to-stream-on-a-different-device-on-the-network-with-webgear)"


```python linenums="1" hl_lines="7-10 14"
# import required libraries
import uvicorn
from vidgear.gears.asyncio import WebGear

# MJPEG mode with various performance tweaks
options = {
    "frame_size_reduction": 40,
    "jpeg_compression_quality": 80,
    "jpeg_compression_fastdct": True,
    "jpeg_compression_fastupsample": False,
}

# initialize WebGear app
web = WebGear(source="foo.mp4", streaming_mode="mjpeg", logging=True, **options)

# run this app on Uvicorn server at address http://localhost:8000/
uvicorn.run(web(), host="localhost", port=8000)

# close app safely
web.shutdown()
```

which can be accessed on any browser on your machine at http://localhost:8000/.

&nbsp;

## Bare-Minimum Usage with Low-Latency H.264 Streaming :fontawesome-solid-zap:

??? new "New in v0.3.7"
    Low-Latency H.264 Streaming was added in `v0.3.7`.

When [`streaming_mode="h264"`](params/#streaming_mode), WebGear encodes frames with FFmpeg _(`libx264 -preset ultrafast -tune zerolatency`)_ and pushes each complete H.264 frame as a binary WebSocket message at `/ws/video`. WebGear's Default Theme decodes them with the browser's hardware-accelerated **WebCodecs** `VideoDecoder` inside a Web Worker and paints them on a `<canvas>`, while a telemetry HUD shows live FPS, bandwidth and decode latency.

!!! danger "Higher Server CPU usage :fontawesome-solid-fire-burner:"
    Every connected viewer starts its own FFmpeg encoder process in this mode, so server CPU usage grows with the number of viewers. Use [MJPEG Mode](#bare-minimum-usage-with-mjpeg-mode-performance-enhancements) for many simultaneous viewers on a low-power server. See [Modes of Operation ➶](../#modes-of-operation) for more details.

!!! warning "Critical Requirements"
    * **Server:** FFmpeg with `libx264` encoder must be installed _(or set via [`custom_ffmpeg`](params/#custom_ffmpeg))_. On Windows :fontawesome-brands-windows:, WebGear looks for FFmpeg on startup, and may download a static FFmpeg build if none is found.
    * **Browser:** WebCodecs support _(Chrome/Edge 94+, Safari 16.4+, Firefox 130+)_ inside a [Secure Context](https://developer.mozilla.org/en-US/docs/Web/Security/Secure_Contexts), i.e. `https://` or `http://localhost`.

!!! info "Automatic MJPEG Fallback"
    The `/video` MJPEG endpoint stays available, and the Default Theme automatically falls back to it whenever WebCodecs is unavailable _(including plain `http://` on non-localhost hosts)_ or WebSocket stream fails. MJPEG [Performance Enhancing Attributes](#mjpeg-performance-enhancing-attributes) therefore still apply to the fallback stream.

```python linenums="1" hl_lines="7 12"
# import required libraries
import uvicorn
from vidgear.gears.asyncio import WebGear

# H.264 mode, tweak encoder, and keep frames at original size
options = {
    "h264_options": {"-crf": 23},  # optional encoder tweaks
    "frame_size_reduction": 0,
}

# initialize WebGear app
web = WebGear(source="foo.mp4", streaming_mode="h264", logging=True, **options)

# run this app on Uvicorn server at address http://localhost:8000/
uvicorn.run(web(), host="localhost", port=8000)

# close app safely
web.shutdown()
```

which can be accessed on any WebCodecs-capable browser on your machine at http://localhost:8000/.

&nbsp;
