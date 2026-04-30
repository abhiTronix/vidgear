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

# FFGear API Usage Examples

!!! example "After going through following Usage Examples, Checkout more of its advanced configurations [here ➶](advanced/)"

&thinsp;

## Bare-Minimum Usage

Following is the bare-minimum code you need to get started with FFGear API:

```python linenums="1"
# import required libraries
from vidgear.gears import FFGear
import cv2

# open any valid video file with FFGear
stream = FFGear(source="myvideo.mp4").start()

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

## Using FFGear with Camera Devices

FFGear API supports camera devices by index, just like OpenCV. Under the hood it uses platform-specific FFmpeg demuxers to capture the device feed.

??? info "Platform-specific demuxers"

    | Platform | Demuxer |
    |:--------:|:--------|
    | :fontawesome-brands-windows: Windows | `dshow` |
    | :material-linux: Linux | `v4l2` |
    | :material-apple: macOS | `avfoundation` |

```python linenums="1" hl_lines="6-7"
# import required libraries
from vidgear.gears import FFGear
import cv2

# open webcam at index 0 using the appropriate demuxer
# (on Linux: "v4l2", on Windows: "dshow", on macOS: "avfoundation")
stream = FFGear(source=0, source_demuxer="v4l2", logging=True).start()

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

## Using FFGear with Network Streams

FFGear API directly supports any network stream URL that FFmpeg supports, including `HTTP(s)`, `RTSP/RTP`, `RTMP`, and more.

=== "HTTP(s) Stream"

    ```python linenums="1" hl_lines="7"
    # import required libraries
    from vidgear.gears import FFGear
    import cv2

    # open an HTTP stream
    stream = FFGear(
        source="https://example.com/live/stream.mp4",
        logging=True
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

=== "RTSP/RTP Stream"

    !!! alert "This example assumes you already have an RTSP server running at the specified address."

    !!! tip "For creating your own RTSP server locally, see [WriteGear's RTSP/RTP bonus example ➶](../../writegear/compression/usage/#using-compression-mode-with-rtsprtp-live-streaming)"

    ```python linenums="1" hl_lines="7-10"
    # import required libraries
    from vidgear.gears import FFGear
    import cv2

    # force TCP transport for RTSP
    options = {"-rtsp_transport": "tcp"}

    # [WARNING] replace with your actual RTSP address
    stream = FFGear(
        source="rtsp://localhost:8554/mystream",
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

## Using FFGear with Streaming Websites

FFGear internally implements the [`yt_dlp`][yt_dlp] backend for seamlessly pipelining live video-frames and metadata from various streaming services like [YouTube](https://youtube.com/), [Twitch](https://www.twitch.tv/), [Dailymotion](https://www.dailymotion.com), and [many more ➶](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md#supported-sites). Enable it by setting `stream_mode=True`.

??? info "Supported Streaming Websites"

    The complete list of all supported Streaming Websites URLs can be found [here ➶](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md#supported-sites)

??? tip "Accessing Stream Metadata :material-database-eye:"

    FFGear exposes a `ytv_metadata` attribute for accessing the stream's metadata as a JSON-like dict:

    ```python
    from vidgear.gears import FFGear

    stream = FFGear(
        source="https://youtu.be/uCy5OuSQnyA", stream_mode=True, logging=True
    ).start()

    # access video metadata
    video_metadata = stream.ytv_metadata
    print(video_metadata.keys())
    print(video_metadata["title"])
    ```

=== "YouTube :fontawesome-brands-youtube:"

    !!! failure "YouTube Playlists are not supported."

    ```python linenums="1" hl_lines="8-11"
    # import required libraries
    from vidgear.gears import FFGear
    import cv2

    # set desired quality as 720p
    options = {"STREAM_RESOLUTION": "720p"}

    # Add YouTube Video URL as source and enable Stream Mode
    stream = FFGear(
        source="https://youtu.be/uCy5OuSQnyA",
        stream_mode=True,
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

=== "Twitch :fontawesome-brands-twitch:"

    !!! warning "If the Twitch user is offline, FFGear will throw `ValueError`."

    ```python linenums="1" hl_lines="8-11"
    # import required libraries
    from vidgear.gears import FFGear
    import cv2

    # set desired quality as 720p
    options = {"STREAM_RESOLUTION": "720p"}

    # Add Twitch stream URL as source and enable Stream Mode
    stream = FFGear(
        source="https://www.twitch.tv/shroud",
        stream_mode=True,
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

=== "Dailymotion :fontawesome-brands-dailymotion:"

    ```python linenums="1" hl_lines="8-11"
    # import required libraries
    from vidgear.gears import FFGear
    import cv2

    # set desired quality as 720p
    options = {"STREAM_RESOLUTION": "720p"}

    # Add Dailymotion Video URL as source and enable Stream Mode
    stream = FFGear(
        source="https://www.dailymotion.com/video/x2yrnum",
        stream_mode=True,
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

## Using FFGear with Different Pixel Formats

FFGear supports decoding frames in any FFmpeg pixel format via the [`frame_format`](params/#frame_format) parameter.

!!! tip "Use `ffmpeg -pix_fmts` terminal command to list all supported pixel formats."

=== "Grayscale (`gray`)"

    ```python linenums="1" hl_lines="6"
    # import required libraries
    from vidgear.gears import FFGear
    import cv2

    # decode as grayscale
    stream = FFGear(source="myvideo.mp4", frame_format="gray", logging=True).start()

    # loop over
    while True:

        # read grayscale frames
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # Show output window
        cv2.imshow("Grayscale Output", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()
    ```

=== "YUV420p (Performance Mode) :zap:"

    !!! success "Performance Mode — Faster Decoding via YUV420p"

        Ingesting frames as 12-bit **YUV 4:2:0** instead of 24-bit **BGR** halves the bytes moving through the FFmpeg pipe. Enable `-enforce_cv_patch` to auto-convert frames to BGR inside FFGear for seamless OpenCV compatibility.

    ```python linenums="1" hl_lines="6-9"
    # import required libraries
    from vidgear.gears import FFGear
    import cv2

    # enable OpenCV patch for YUV420p frames — auto-converts to BGR in FFGear
    options = {"-enforce_cv_patch": True}

    stream = FFGear(
        source="myvideo.mp4",
        frame_format="yuv420p",
        logging=True,
        **options
    ).start()

    # loop over
    while True:

        # read BGR frames (auto-converted from YUV420p by -enforce_cv_patch)
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

## Using FFGear with Simple FFmpeg Filtergraphs

FFGear supports live simple filtergraph pipelines via the `-vf` FFmpeg parameter through the `options` dictionary.

!!! info "Simple filtergraphs have exactly one input and one output. Use them via the `-vf` parameter."

```python linenums="1" hl_lines="7-11"
# import required libraries
from vidgear.gears import FFGear
import cv2

# define a simple filtergraph: horizontally flip and scale to 1280x720
options = {
    "-vf": "hflip,scale=1280:720"
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

## Using FFGear with Video Looping

FFGear supports video looping via the `-stream_loop` prefix option.

!!! note "Using `-stream_loop 3` will loop the video `4` times in total. Use `-1` for infinite looping."

```python linenums="1" hl_lines="7-9"
# import required libraries
from vidgear.gears import FFGear
import cv2

# define loop 3 times (total 4 playbacks) via ffprefixes
options = {
    "-ffprefixes": ["-stream_loop", "3"]
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

[yt_dlp]:https://github.com/yt-dlp/yt-dlp
