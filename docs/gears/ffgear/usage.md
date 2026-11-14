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

!!! tip "Check out all FFdecoder API's basic recipes [here ➶](https://abhitronix.github.io/deffcode/latest/recipes/basic/) to better understand these usage examples."

!!! example "After going through following Usage Examples, Checkout more of its advanced configurations [here ➶](advanced/)"

!!! warning "FFGear requires the `deffcode` library"

    FFGear API **MUST** have the [`deffcode`][deffcode] library installed, along with a valid FFmpeg executable. Any failure in detection will raise `ImportError`/`RuntimeError` immediately.

    Install via pip:

    ```sh
    pip install deffcode
    ```

    For FFmpeg installation, see [FFmpeg Installation ➶](advanced/ffmpeg_install/)

&thinsp;

## Bare-Minimum Usage

Following is the bare-minimum code you need to get started with FFGear API:

```python linenums="1" hl_lines="2 6 12 32"
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

## Using FFGear with Network Streams

FFGear API directly supports any network stream URL that FFmpeg supports, including `HTTP(s)`, `RTSP/RTP`, `RTMP`, and more.

??? tip "Handling stalled network sources _(New in v0.3.7)_"
    A network source can stall forever _(e.g. a killed UDP sender)_. Use `read(timeout=seconds)` to raise `TimeoutError` instead of waiting indefinitely for the next frame; `stop()` then safely tears down the stalled FFmpeg pipeline. After `stop()`, FFmpeg's exit code is available as `stream.returncode` _(and its stderr tail as `stream.stderr_tail` with the `-capture_stderr` option)_. The source framerate is available as `stream.framerate`.

    ```python
    try:
        frame = stream.read(timeout=5.0)
    except TimeoutError:
        stream.stop()  # source stalled
    ```

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

    !!! tip "For creating your own RTSP server locally, see [WriteGear's RTSP/RTP bonus example ➶](../../help/writegear_ex/#using-writegears-compression-mode-for-rtsprtp-live-streaming)"

    ```python linenums="1" hl_lines="6 10"
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

=== "HLS Stream (specific rendition)"

    > A HLS **master playlist** _(`.m3u8`)_ usually carries several renditions of the same video at different resolutions, which FFmpeg exposes as **embedded video streams of one input**. ==Select the rendition explicitly== with FFdecoder's [`-default_stream_indexes`](../params/#b-ffdecoder-parameters) parameter.

    !!! warning "Behaviour change with `deffcode>=0.3.0` _(New in v0.3.8)_"
        Without a selection, FFGear now decodes the **first** rendition in playlist order _(often the lowest quality)_ and logs a warning, instead of the one FFmpeg picks automatically _(usually the highest resolution)_.

    !!! info "Renditions are counted in playlist order: `(1, 0)` selects the second video stream _(FFmpeg's `0:v:1`)_, and `source_video_streams` metadata gives their number."

    ```python linenums="1" hl_lines="10-14 17 20"
    # import required libraries
    from vidgear.gears import FFGear
    from deffcode import Sourcer
    import cv2

    # [WARNING] replace with your actual HLS master playlist URL
    source = "https://example.com/master.m3u8"

    # list each rendition's resolution
    sourcer = Sourcer(source)
    count = sourcer.probe_stream().retrieve_metadata()["source_video_streams"]
    for index in range(count):
        metadata = sourcer.probe_stream(default_stream_indexes=(index, 0)).retrieve_metadata()
        print(index, metadata["source_video_resolution"])

    # select the second rendition (applies `-map 0:v:1` automatically)
    options = {"-default_stream_indexes": (1, 0)}

    # open the selected rendition
    stream = FFGear(source=source, logging=True, **options).start()

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

FFGear internally implements the [`yt_dlp`][yt_dlp] backend for seamlessly pipelining live video-frames and metadata from various streaming services like [YouTube](https://youtube.com/), [Twitch](https://www.twitch.tv/), [PeerTube](https://peertube.tv/), and [many more ➶](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md#supported-sites). Enable it by setting `stream_mode=True`.

??? info "Supported Streaming Websites"

    The complete list of all supported Streaming Websites URLs can be found [here ➶](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md#supported-sites)

??? tip "Accessing Stream Metadata :material-database-eye:"

    FFGear exposes a `ytv_metadata` attribute for accessing the stream's metadata as a JSON-like dict:

    ```python
    from vidgear.gears import FFGear

    stream = FFGear(
        source="https://youtu.be/QDia3e12czc", stream_mode=True, logging=True
    ).start()

    # access video metadata
    video_metadata = stream.ytv_metadata
    print(video_metadata.keys())
    print(video_metadata["title"])
    ```

=== "YouTube :fontawesome-brands-youtube:"

    ???+ warning "AV1 Dependency in FFmpeg"

        Most of the time, YouTube defaults to the AV1 video format. Therefore, you need an FFmpeg build that includes `libdav1d` for software AV1 decoding, which requires it to be compiled using the flag: `--enable-libdav1d`.

        **Check if you already have it:** Run this command in your terminal or command prompt:
        
        ```bash
        ffmpeg -decoders | grep dav1d
        ```

        If you see `V....D libdav1d`, the library is installed and ready to use.


    !!! failure "YouTube Playlists are not supported."

    ```python linenums="1" hl_lines="6 10"
    # import required libraries
    from vidgear.gears import FFGear
    import cv2

    # set desired quality as 720p and video decoder as `libdav1d`
    options = {"STREAM_RESOLUTION": "720p", "-vcodec": "libdav1d"}

    # Add YouTube Video URL as source and enable Stream Mode
    stream = FFGear(
        source="https://youtu.be/QDia3e12czc", stream_mode=True, logging=True, **options
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

    ```python linenums="1" hl_lines="6 10-11"
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

=== "PeerTube :simple-peertube:"

    ```python linenums="1" hl_lines="6 10-11"
    # import required libraries
    from vidgear.gears import FFGear
    import cv2

    # set desired quality as 480p
    options = {"STREAM_RESOLUTION": "480p"}

    # Add PeerTube stream URL as source and enable Stream Mode
    stream = FFGear(
        source="https://peertube.tv/w/q4GM7HcfUnqeBAj3urTUCv",
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

=== "YUV420p (Performance Mode) :material-speedometer:"

    !!! success "Performance Mode :zap: — Faster Decoding via YUV420p"

        Ingesting frames as 12-bit **YUV 4:2:0** instead of 24-bit **BGR** halves the bytes moving through the FFmpeg pipe. Enable `-enforce_cv_patch` to auto-convert `yuv420p`, `nv12` and `nv21` frames to BGR inside FFGear for seamless OpenCV compatibility. Other YUV pixel formats _(4:2:2, 4:4:4, 10-bit)_ are returned as stacked planes.

    ```python linenums="1" hl_lines="6 11"
    # import required libraries
    from vidgear.gears import FFGear
    import cv2

    # IMPORTANT: enable OpenCV patch for YUV420p frames
    options = {"-enforce_cv_patch": True}

    # stream YUV420p frames
    stream = FFGear(
        source="myvideo.mp4",
        frame_format="yuv420p",
        logging=True,
        **options
    ).start()

    # loop over
    while True:
        # read OpenCV compatible YUV420p frames
        frame = stream.read()

        # check for frame if NoneType
        if frame is None:
            break

        # {do something with frame here}

        # show output window
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

=== "Grayscale"

    ```python linenums="1" hl_lines="6"
    # import required libraries
    from vidgear.gears import FFGear
    import cv2

    # stream grayscale frames
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

=== "Grayscale via YUV (fastest) :material-invoice-fast-outline:"

    !!! success "Fastest :zap: RAW-to-Grayscale via `-extract_luma`"

        Every YUV/NV bytestream stores the **Luma (Y) plane** uncompressed at the top of each frame. The exclusive [`-extract_luma`](params/#b-exclusive-parameters) boolean attribute makes FFGear slice that Y-plane directly and hand back a 2D `(H, W)` grayscale ndarray — **no colorspace conversion in FFmpeg, no `cv2.cvtColor` in Python**. This is strictly faster than `frame_format="gray"`, which still asks FFmpeg to do a `yuv→gray` conversion on every frame.

        Combined with the reduced pipe-bytes of YUV 4:2:0 ingest, this is the fastest :fontawesome-solid-tachometer-alt-fast: grayscale pipeline the API can produce.

    ```python linenums="1" hl_lines="6 11"
    # import required libraries
    from vidgear.gears import FFGear
    import cv2

    # enable direct Luma (Y-plane) extraction
    options = {"-extract_luma": True}

    # stream Grayscale via YUV frames
    stream = FFGear(
        source="myvideo.mp4",
        frame_format="yuv420p",
        logging=True,
        **options
    ).start()

    # loop over
    while True:
        # read grayscale frames
        frame = stream.read()

        # check for frame if NoneType
        if frame is None:
            break

        # {do something with Luma (Y-plane) extracted grayscale frame here}

        # NOTE: If you do not need previewing frames, comment following lines
        # show output window
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

## Using FFGear with Camera Devices (Indexes)

??? info "Enumerating all Camera Devices with Indexes"

    Using DeFFcode's [Sourcer API](https://abhitronix.github.io/deffcode/latest/reference/sourcer/), you can easily use its [`enumerate_devices`](https://abhitronix.github.io/deffcode/latest/reference/sourcer/#deffcode.Sourcer.enumerate_devices) property object to enumerate all probed Camera Devices _(connected to your system)_ as **dictionary object** with device indexes as keys and device names as their respective values:

    ```python
    # import the necessary packages
    from deffcode import Sourcer
    import json

    # stream
    sourcer = Sourcer("0").probe_stream()

    # enumerate probed devices as Dictionary object(`dict`)
    print(sourcer.enumerate_devices)

    # enumerate probed devices as JSON string(`json.dump`)
    print(json.dumps(sourcer.enumerate_devices,indent=2))
    ```

    ???+ abstract "After running above python code, the resultant Terminal Output will look something as following on :fontawesome-brands-windows:Windows machine:"

        === "As Dictionary object"

            ```python
            {0: 'Integrated Camera', 1: 'USB2.0 Camera', 2: 'DroidCam Source'}
            ```
        
        === "As JSON string"

            ```json
            {
            "0": "Integrated Camera",
            "1": "USB2.0 Camera",
            "2": "DroidCam Source"
            }
            ```

> FFGear API supports camera devices by index, just like OpenCV. Under the hood it uses platform-specific FFmpeg demuxers to capture the device feed by specifying its matching index value either as **integer** or **string of integer** type to its `source` parameter.

??? danger "Requirement for Index based Camera Device Capturing in FFGear API"

    - [x] **MUST have appropriate FFmpeg binaries, Drivers, and Softwares installed:**
        
        Internally, FFGear APIs achieves Index based Camera Device Capturing by employing some specific FFmpeg demuxers on different platforms(OSes). These platform specific demuxers are as follows:

        | Platform(OS) | Demuxer |
        |:------------:|:-------|
        | :fontawesome-brands-windows: Windows OS|[`dshow`](https://trac.ffmpeg.org/wiki/DirectShow) _(or DirectShow)_ |
        | :material-linux: Linux OS | [`video4linux2`](https://trac.ffmpeg.org/wiki/Capture/Webcam#Linux) _(or its alias `v4l2`)_ |
        | :material-apple: Mac OS | [`avfoundation`](https://ffmpeg.org/ffmpeg-devices.html#avfoundation) |

        **:warning: Important:** Kindly make sure your FFmpeg binaries support these platform specific demuxers as well as system have the appropriate video drivers and related softwares installed.
    - [x] The [`source`](params/#source) parameter value **MUST be exactly the probed Camera Device index** _(Use DeFFcode's [Sourcer API](https://abhitronix.github.io/deffcode/latest/reference/sourcer/) `enumerate_devices` to list them)_.
    - [x] The [`source_demuxer`](params/#source_demuxer) parameter value  **MUST be either `None`_(also means empty)_ or `"auto"`**. 

In this example we stream **BGR24** video frames from **Integrated Camera at index `0`** on a :fontawesome-brands-windows: Windows Machine:

??? tip "Important Facts related to Camera Device Indexing"
    - [x] **Camera Device indexes are 0-indexed**. So the first device is at `0`, second is at `1`, so on. So if the there are `n` devices, the last device is at `n-1`.
    - [x] **Camera Device indexes can be of either integer** _(e.g. `0`,`1`, etc.)_ or **string of integer** _(e.g. `"0"`,`"1"`, etc.)_ **type**.
    - [x] **Camera Device indexes can be negative** _(e.g. `-1`,`-2`, etc.)_, this means you can also start indexing from the end.
        * For example, If there are three devices: 
            ```python
            {0: 'Integrated Camera', 1: 'USB2.0 Camera', 2: 'DroidCam Source'}
            ```
        * Then, You can specify Positive Indexes and its Equivalent Negative Indexes as follows:

            | Positive Indexes | Equivalent Negative Indexes |
            |:------------:|:-------:|
            | `#!python FFGear(source="0").formulate()`| `#!python FFGear(source="-3").formulate()` |
            | `#!python FFGear(source="1").formulate()`| `#!python FFGear(source="-2").formulate()` |
            | `#!python FFGear(source="2").formulate()`| `#!python FFGear(source="-1").formulate()` |

    !!! warning "Out of Index Camera Device index values will raise `ValueError` in FFGear API"

```python linenums="1" hl_lines="7"
# import required libraries
from vidgear.gears import FFGear
import cv2

# stream with "0" index source for BGR24 output
stream = FFGear(
    source="0",
    frame_format="bgr24",
    logging=True,
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

## Using FFGear with Simple FFmpeg Filtergraphs

FFGear supports live simple filtergraph pipelines via the `-vf` FFmpeg parameter through the `options` dictionary.

!!! info "Simple filtergraphs have exactly one input and one output. Use them via the `-vf` parameter."

```python linenums="1" hl_lines="8"
# import required libraries
from vidgear.gears import FFGear
import cv2

# define the Video Filter definition
# horizontally flip and scale to half its original size
options = {
    "-vf": "hflip,scale=w=iw/2:h=ih/2"
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

## Using FFGear with Multi-Stream Sources

> Some sources carry **several embedded video streams** in a single input _(such as an `.mkv` with two video tracks, or an HLS playlist with several renditions)_. By default, FFGear decodes the **first** video stream _(skipping cover art)_ and logs a warning, instead of the one [FFmpeg would pick automatically](https://ffmpeg.org/ffmpeg.html#Automatic-stream-selection) _(`default` disposition, otherwise the highest resolution)_.

In this example we will decode only the second video stream of a source with FFdecoder's [`-default_stream_indexes`](../params/#b-ffdecoder-parameters) parameter. To keep it runnable anywhere, the source is a [`lavfi`](https://ffmpeg.org/ffmpeg-devices.html#lavfi) graph that produces **one input with two video streams** _(red `320x240` and blue `160x120`)_. A file path such as `foo.mkv` works the same way.

!!! new "New in v0.3.8"
    Stream selection requires `deffcode>=0.3.0`. Sources whose first video stream is cover art _(attached picture)_ now decode their real video stream too, instead of raising `RuntimeError`.

!!! info "Stream indexes are **per type**: `1` means the second **video** stream _(same as FFmpeg's `0:v:1` specifier)_, not the absolute `Stream #0:1`. So audio streams placed before the video ones don't shift it."

!!! tip "A plain `-map 0:v:1`, or a `-filter_complex` using only `[0:v:1]` _(e.g. `[0:v:1]hflip`)_, selects the stream too. See [DeFFcode's recipe ➶](https://abhitronix.github.io/deffcode/latest/recipes/basic/decode-video-files/#decoding-a-specific-video-stream-from-a-multi-stream-file) for more details."

```python linenums="1" hl_lines="6 9 12-17 20-21"
# import required libraries
from vidgear.gears import FFGear
import cv2

# one input with two embedded video streams (or use e.g. "foo.mkv")
source = "color=c=red:s=320x240:r=25:d=5[out0];color=c=blue:s=160x120:r=30:d=5[out1]"

# select second video stream (`-map 0:v:1` is applied automatically)
options = {"-default_stream_indexes": (1, 0)}

# open the selected stream
stream = FFGear(
    source=source,
    source_demuxer="lavfi",
    logging=True,
    **options
).start()

# FFdecoder metadata is available as a dict
print(stream.metadata["source_video_streams"])  # 2
print(stream.metadata["source_video_resolution"])  # [160, 120]

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

## Using FFGear with Anamorphic, HDR and Rotated Videos

> Some sources store **non-square pixels** _(anamorphic DVD, HDV or broadcast video, e.g. `1440x1080` shown as `1920x1080`)_ or **HDR** video _(PQ/HLG transfer)_. FFGear decodes their pixels as stored, so frames look **squeezed** or **washed out**, and FFdecoder logs a warning about each _(even with `logging=False`)_.

In this example we will check a video file _(say `foo.mkv`)_ for both, using DeFFcode's Sourcer API `source_video_sar` and `source_video_color_transfer` metadata, and add the matching FFmpeg filters through the `-vf` parameter:

!!! new "New in v0.3.8"
    These metadata keys and warnings require `deffcode>=0.3.0`. The same metadata is also available as a dict from FFGear's `metadata` attribute after initialization _(retained after `stop()`)_.

!!! warning "Tone-mapping uses the `zscale` filter, so it needs FFmpeg built with `libzimg` _(included in most static builds)_."

!!! info "Color metadata is `\"\"` when the source doesn't tag it, and some FFmpeg versions report `unknown` _(also `\"\"`)_ for tags other versions print."

??? tip "Portrait / rotated videos"
    Sources flagged as rotated by ±90° _(e.g. phone portrait videos)_ are decoded **upright**, with width and height swapped _(a `1280x720` source rotated by 90° gives `(1280, 720, 3)` frames)_. To get frames in their coded orientation instead, disable FFmpeg's auto-rotation with `-noautorotate` _(not `-autorotate 0`)_:

    ```python
    stream = FFGear(source="portrait.mp4", **{"-ffprefixes": ["-noautorotate"]}).start()
    ```

```python linenums="1" hl_lines="7 11-20 23"
# import required libraries
from vidgear.gears import FFGear
from deffcode import Sourcer
import cv2

# probe source metadata
metadata = Sourcer("foo.mkv").probe_stream().retrieve_metadata()

filters = []
# tone-map HDR (PQ or HLG) to SDR BT.709
if metadata["source_video_color_transfer"] in ("smpte2084", "arib-std-b67"):
    filters.append(
        "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
        "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv"
    )
# stretch non-square pixels to display aspect ratio (e.g. 1440x1080 to 1920x1080)
if metadata["source_video_sar"] != [1, 1]:
    filters.append("scale=iw*sar:ih,setsar=1")
options = {"-vf": ",".join(filters)} if filters else {}

# open stream with the matching filters
stream = FFGear(source="foo.mkv", logging=True, **options).start()

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

## Using FFGear with Sequence of images

FFGear supports **Image Sequences** such as Sequential(`img%03d.png`) and Glob pattern(`*.png`) in real-time.

??? tip "Extracting Image Sequences from a video"
    
    **You can use following FFmpeg command to extract sequences of images from a video file `foo.mp4`:**
    
    ```sh
    $ ffmpeg -i foo.mp4 /path/to/image-%03d.png
    ```

    The default framerate is `25` fps, therefore this command will extract `25 images/sec` from the video file, and save them as sequences of images _(starting from `image-000.png`, `image-001.png`, `image-002.png` up to `image-999.png`)_. 

    !!! info "If there are more than `1000` frames then the last image will be overwritten with the remaining frames leaving only the last frame."

    The default images width and height is same as the video.

=== "Sequential"

    ??? question "How to start with specific number image?"
        You can use `-start_number` FFmpeg parameter if you want to start with specific number image:

        ```python
        # define `-start_number` such as `5`
        options = {"-ffprefixes":["-start_number", "5"]}

        # initialize and formulate the stream with define parameters
        stream = FFGear(source="/absolute/path/to/image-%03d.png", logging=True, **options).formulate()
        ```
    !!! warning "Ensure `source` points to an absolute image sequence **folder** path"

        Furthermore, Use the exact filename pattern (for example, `image-%03d.png`) that was used when extracting the images. An incorrect or relative pattern may cause the FFGear API to raise a `RuntimeError`.

    ```python linenums="1" hl_lines="6"
    # import required libraries
    from vidgear.gears import FFGear
    import cv2

    stream = FFGear(
        source="/absolute/path/to/image-%03d.png",
        frame_format="bgr24",
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

=== "Glob pattern"

    !!! abstract "Bash-style globbing _(`*` represents any number of any characters)_ is useful if your images are sequential but not necessarily in a numerically sequential order."

    !!! warning "Ensure `source` points to an absolute image sequence **folder** path"

    !!! fail "The glob pattern may not be available on :fontawesome-brands-windows: Windows FFmpeg builds."

    ```python linenums="1" hl_lines="7 11"
    # import required libraries
    from vidgear.gears import FFGear
    import cv2

    # define `-pattern_type glob` for accepting glob pattern via ffprefixes
    options = {
        "-ffprefixes":["-pattern_type", "glob"]
    }

    stream = FFGear(
        source="/absolute/path/to/*.png", # Glob pattern
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

In this example we stream **BGR24** video frames from looping video using FFGear API:

=== "Using `-stream_loop` option"

    The `-stream_loop` flag is a high-level FFmpeg command that **repeats the entire input file** a specified number of times before any filtering occurs. The recommended way to use the `-stream_loop` flag with the FFGear API is through the `-ffprefixes` list attribute of the `options` dictionary parameter.
    
    !!! info "Possible `-stream_loop` values are integer values"

        Integer values `> 0` specify the number of loops, `0` means no looping, and `-1` enables infinite looping.

    !!! note "Using `-stream_loop 3` will loop the video `4` times in total."

    ```python linenums="1" hl_lines="7"
    # import required libraries
    from vidgear.gears import FFGear
    import cv2

    # define loop 3 times (total 4 playbacks) via ffprefixes
    options = {
        "-ffprefixes": ["-stream_loop", "3"]
    }

    # stream with suitable source
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

=== "Using `loop` filter"

    The `loop` filter is used within an FFmpeg filtergraph (`-filter_complex`) to **repeat a specific segment** of video frames. In the FFGear API, it can be configured as an attribute of the `options` dictionary parameter.

    !!! danger "This filter places all frames into memory (RAM), so applying [`trim`](https://ffmpeg.org/ffmpeg-filters.html#toc-trim) filter first is strongly recommended. Otherwise, you might run out of memory."

    !!! tip "Using `loop` filter for looping video"

        The filter accepts the following options:
        
        * `loop`: Sets the number of loops for integer values `> 0`. Setting this value to `-1` results in infinite loops. Default is `0` (no loops).
        * `size`: Sets the maximum size in number of frames. Default is `0`.
        * `start`: Sets the first frame of the loop. Default is `0`.

        !!! info "You must often include `setpts` to fix the timestamps, otherwise frames may be dropped or playback may stutter."

    !!! note "Using `loop=3` will loop the video `4` times in total."

    ```python linenums="1" hl_lines="7"
    # import required libraries
    from vidgear.gears import FFGear
    import cv2

    # define loop 4 times, each loop is 15 frames, each loop skips the first 25 frames
    options = {
        "-filter_complex": "loop=loop=3:size=15:start=25,setpts=N/FRAME_RATE/TB" # Or use: `loop=3:15:25`
    }  

    # stream with suitable source
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

## Using FFGear with OpenCV VideoWriter API

FFGear integrates seamlessly with OpenCV's [`VideoWriter()`](https://docs.opencv.org/3.4/dd/d9e/classcv_1_1VideoWriter.html#ad59c61d8881ba2b2da22cff5487465b5) class to encode video frames into a multimedia file. However, it lacks fine-grained control over output quality, bitrate, compression, and other advanced parameters—features that are readily available with **WriteGear API's Compression Mode**.

!!! tip "You can use FFGear API's `stream.metadata` property object that dumps source Video's metadata information _(as JSON string)_ to retrieve source framerate and resolution."

!!! info "You could also use the [WriteGear API's Non-compression mode](../../gears/writegear/non_compression/), which provides flexible access to OpenCV's VideoWriter API for encoding video frames without compression. See this document for [switching from VideoWriter API to WriteGear API ➶](../../switch_from_cv/#switching-the-videowriter-api)."

=== "BGR frames"

    By default, OpenCV expects `BGR` format frames in its `cv2.write()` method.

    ```python linenums="1" hl_lines="9 12-14 17 32 49"
    # import the necessary packages
    from vidgear.gears import FFGear
    import cv2

    # stream with BGR24 pixel format output
    stream = FFGear(source="myvideo.mp4", frame_format="bgr24", logging=True).start()

    # retrieve Metadata as dict
    metadata_dict = stream.metadata

    # prepare OpenCV parameters
    FOURCC = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    FRAMERATE = metadata_dict["output_framerate"]
    FRAMESIZE = tuple(metadata_dict["output_frames_resolution"])

    # Define writer with parameters and suitable output filename for e.g. `output_foo.avi`
    writer = cv2.VideoWriter("output_foo.avi", FOURCC, FRAMERATE, FRAMESIZE)

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # writing BGR24 frame to writer
        writer.write(frame)

        # let's also show output window
        cv2.imshow("Output", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close stream
    stream.stop()

    # safely close writer
    writer.release()
    ```

=== "RGB frames"

    Since OpenCV expects `BGR` format frames in its `cv2.write()` method, therefore we need to convert `RGB` frames into `BGR` before encoding as follows:

    ```python linenums="1" hl_lines="9 12-14 17 32 35 52"
    # import the necessary packages
    from vidgear.gears import FFGear
    import cv2

    # stream with RGB24 pixel format output
    stream = FFGear(source="myvideo.mp4", frame_format="rgb24", logging=True).start()

    # retrieve Metadata as dict
    metadata_dict = stream.metadata

    # prepare OpenCV parameters
    FOURCC = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    FRAMERATE = metadata_dict["output_framerate"]
    FRAMESIZE = tuple(metadata_dict["output_frames_resolution"])

    # Define writer with parameters and suitable output filename for e.g. `output_foo.avi`
    writer = cv2.VideoWriter("output_foo.avi", FOURCC, FRAMERATE, FRAMESIZE)

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the RGB frame here}

        # converting RGB24 to BGR24 frame
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # writing BGR24 frame to writer
        writer.write(frame_bgr)

        # let's also show output window
        cv2.imshow("Output", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close stream
    stream.stop()

    # safely close writer
    writer.release()
    ```

=== "GRAYSCALE frames"

    OpenCV also directly consumes `GRAYSCALE` frames in its `cv2.write()` method.

    !!! note "When writing a grayscale video, don't forget to add that flag (`isColor=False`) to the `VideoWriter`."

    ```python linenums="1" hl_lines="9 12-14 17 32 49"
    # import the necessary packages
    from vidgear.gears import FFGear
    import cv2

    # stream with GRAYSCALE pixel format output
    stream = FFGear(source="myvideo.mp4", frame_format="gray", logging=True).start()

    # retrieve Metadata as dict
    metadata_dict = stream.metadata

    # prepare OpenCV parameters
    FOURCC = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    FRAMERATE = metadata_dict["output_framerate"]
    FRAMESIZE = tuple(metadata_dict["output_frames_resolution"])

    # Define writer with parameters and suitable output filename for e.g. `output_foo_gray.avi`
    writer = cv2.VideoWriter("output_foo_gray.avi", FOURCC, FRAMERATE, FRAMESIZE, isColor=False) 

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the GRAYSCALE frame here}

        # writing GRAYSCALE frame to writer
        writer.write(frame)

        # let's also show output window
        cv2.imshow("Output", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close stream
    stream.stop()

    # safely close writer
    writer.release()
    ```

=== "YUV frames"

    > With FFGear API, frames extracted with YUV pixel formats _(`yuv420p`, `nv12`, `nv21` etc.)_ are generally incompatible with OpenCV APIs. But you can make them easily compatible by using exclusive `-enforce_cv_patch` boolean attribute of its `options` dictionary parameter, which auto-converts `yuv420p`, `nv12` and `nv21` frames to BGR.

    Let's encode YUV420p pixel-format frames with OpenCV's `write()` method:

    !!! info "You can also use `nv12` or `nv21` pixel-formats in the similar manner. Other YUV pixel-formats such as `yuv422p`(4:2:2 subsampling), `yuv444p`(4:4:4 subsampling) or 10-bit ones have no OpenCV conversion, and are returned as stacked planes."

    ```python linenums="1" hl_lines="6 17 20-22 25 40"
    # import the necessary packages
    from vidgear.gears import FFGear
    import cv2

    # enable OpenCV patch for YUV420p frames
    options = {"-enforce_cv_patch": True}

    # stream YUV420p frames
    stream = FFGear(
        source="myvideo.mp4",
        frame_format="yuv420p",
        logging=True,
        **options
    ).start()

    # retrieve Metadata as dict
    metadata_dict = stream.metadata

    # prepare OpenCV parameters
    FOURCC = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    FRAMERATE = metadata_dict["output_framerate"]
    FRAMESIZE = tuple(metadata_dict["output_frames_resolution"])

    # Define writer with parameters and suitable output filename for e.g. `output_foo_yuv.avi`
    writer = cv2.VideoWriter("output_foo_yuv.avi", FOURCC, FRAMERATE, FRAMESIZE)

    # loop over
    while True:

        # read OpenCV compatible YUV420 frames
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with frame here}

        # writing BGR frame to writer
        writer.write(frame)

        # let's also show output window
        cv2.imshow("Output", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close stream
    stream.stop()

    # safely close writer
    writer.release()
    ```

&nbsp;

## Using FFGear with WriteGear API (Compression Mode)

???+ danger "High CPU Usage when chaining FFGear with WriteGear"

    When chaining FFGear with WriteGear, both FFmpeg processes _(decoding + encoding)_ run **as fast as your hardware allows** with no artificial pacing between them. This causes the pipeline to max out your CPU to process the video in the shortest time possible, which may be undesirable.

    You can mitigate this in two ways depending on your use case:

    === "Throttle to Real-Time Speed"

        Pass the `-re` flag via FFGear's `-ffprefixes` parameter to force FFmpeg to read the input at its native framerate. This naturally paces the pipeline to real-time speed and **drastically reduces CPU usage**:

        ```python
        # force input to be read at native framerate
        stream = FFGear(source="foo.mp4", frame_format="bgr24", **{"-ffprefixes": ["-re"]})
        ```

    === "Limit FFmpeg Threads"

        Pass `-threads` to both FFGear and WriteGear to cap the number of CPU threads each FFmpeg process may use. This leaves headroom for other system tasks:

        ```python
        # limit decoder to 2 threads
        stream = FFGear(source="foo.mp4", frame_format="bgr24", **{"-threads": 2})

        # limit encoder to 2 threads
        writer = WriteGear(output="output_foo.mp4", **{"-input_framerate": fps, "-threads": 2})
        ```

    !!! tip ":fontawesome-solid-microchip: Hardware Acceleration"
        
        If your machine has a dedicated GPU, you can offload video encoding entirely to the GPU. See [CUDA-NVENC-accelerated Transcoding with WriteGear API (Compression Mode) ➶](advanced/#cuda-nvenc-accelerated-transcoding-with-writegear-api-compression-mode) example for reference, which demonstrates shifting the heavy encoding workload away from the CPU.

FFGear integrates seamlessly with VidGear's WriteGear API in Compression Mode for high-quality re-encoding of decoded frames:

!!! tip "You can use FFGear API's `stream.metadata` property object that dumps source Video's metadata information _(as JSON string)_ to retrieve source framerate."

=== "BGR frames"

    WriteGear API by default expects `BGR` format frames in its `write()` class method.

    ```python linenums="1" hl_lines="3 11 16 31 37"
    # import required libraries
    from vidgear.gears import FFGear
    from vidgear.gears import WriteGear

    # open source with FFGear and BGR frames
    stream = FFGear(source="myvideo.mp4", frame_format="bgr24", logging=True).start()

    # retrieve framerate from source JSON Metadata and pass it as `-input_framerate` 
    # parameter for controlled framerate
    output_params = {
        "-input_framerate": stream.metadata["source_video_framerate"]
    }

    # Define writer with default parameters and suitable
    # output filename for e.g. `output_foo.mp4`
    writer = WriteGear(output="output_foo.mp4", **output_params)

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

    # safely close stream
    stream.stop()

    # safely close writer
    writer.close()
    ```

=== "RGB frames"

    In WriteGear API, you can use [`rgb_mode`](../../bonus/reference/writegear/#vidgear.gears.writegear.WriteGear.write) parameter in  `write()` class method to write `RGB` format frames instead of default `BGR` as follows:

    ```python linenums="1" hl_lines="3 11 16 31 37"
    # import required libraries
    from vidgear.gears import FFGear
    from vidgear.gears import WriteGear

    # open source with FFGear and RGB frames
    stream = FFGear(source="myvideo.mp4", frame_format="rgb24", logging=True).start()

    # retrieve framerate from source JSON Metadata and pass it as `-input_framerate` 
    # parameter for controlled framerate
    output_params = {
        "-input_framerate": stream.metadata["source_video_framerate"]
    }

    # Define writer with default parameters and suitable
    # output filename for e.g. `output_foo_rgb.mp4`
    writer = WriteGear(output="output_foo_rgb.mp4", **output_params)

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # write frame to output
        writer.write(frame, rgb_mode=True)

    # safely close stream
    stream.stop()

    # safely close writer
    writer.close()
    ```

=== "Grayscale frames"

    WriteGear API also directly consumes `GRAYSCALE` format frames in its `write()` class method. 

    ```python linenums="1" hl_lines="3 11 16 31 37"
    # import required libraries
    from vidgear.gears import FFGear
    from vidgear.gears import WriteGear

    # open source with FFGear and GRAYSCALE frames
    stream = FFGear(source="myvideo.mp4", frame_format="gray", logging=True).start()

    # retrieve framerate from source JSON Metadata and pass it as `-input_framerate` 
    # parameter for controlled framerate
    output_params = {
        "-input_framerate": stream.metadata["source_video_framerate"]
    }

    # Define writer with default parameters and suitable
    # output filename for e.g. `output_foo_gray.mp4`
    writer = WriteGear(output="output_foo_gray.mp4", **output_params)

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

    # safely close stream
    stream.stop()

    # safely close writer
    writer.close()
    ```

=== "YUV420p (Performance Mode) :material-speedometer:"

    !!! success "Performance Mode :zap: — Faster Decoding via YUV420p"

        Ingesting frames as 12-bit **YUV 4:2:0** instead of 24-bit **BGR** halves the bytes moving through the FFmpeg pipe. Enable `-enforce_cv_patch` to auto-convert `yuv420p`, `nv12` and `nv21` frames to BGR inside FFGear for seamless OpenCV compatibility. Other YUV pixel formats _(4:2:2, 4:4:4, 10-bit)_ are returned as stacked planes.

    WriteGear API also directly consume `YUV` _(or basically any other supported pixel format)_ frames in its `write()` class method with its `-input_pixfmt` attribute in compression mode. 

    !!! note "You can also use `yuv422p`(4:2:2 subsampling) or `yuv444p`(4:4:4 subsampling) instead for more higher dynamic ranges."

    ```python linenums="1" hl_lines="3 12-13 18 33 39"
    # import required libraries
    from vidgear.gears import FFGear
    from vidgear.gears import WriteGear

    # open source with FFGear stream for YUV420 output
    stream = FFGear(source="myvideo.mp4", frame_format="yuv420p", logging=True).start()

    # retrieve framerate from source JSON Metadata and pass it as 
    # `-input_framerate` parameter for controlled framerate
    # and also add input pixfmt as yuv420p 
    output_params = {
        "-input_framerate": stream.metadata["output_framerate"],
        "-input_pixfmt": "yuv420p"
    }

    # Define writer with default parameters and suitable
    # output filename for e.g. `output_foo_yuv.mp4`
    writer = WriteGear(output="output_foo_yuv.mp4", logging=True, **output_params)

    # loop over
    while True:

        # read OpenCV compatible YUV420p frames
        frame = stream.read()

        # check for frame if NoneType
        if frame is None:
            break

        # {do something with the YUV420p frame here}

        # write frame to output
        writer.write(frame)

    # safely close stream
    stream.stop()

    # safely close writer
    writer.close()
    ```

=== "Grayscale via YUV (fastest) :material-invoice-fast-outline:"

    !!! success "Fastest :zap: RAW-to-Grayscale via `-extract_luma`"

        Every YUV/NV bytestream stores the **Luma (Y) plane** uncompressed at the top of each frame. The exclusive [`-extract_luma`](params/#b-exclusive-parameters) boolean attribute makes FFGear slice that Y-plane directly and hand back a 2D `(H, W)` grayscale ndarray — **no colorspace conversion in FFmpeg, no `cv2.cvtColor` in Python**. This is strictly faster than `frame_format="gray"`, which still asks FFmpeg to do a `yuv→gray` conversion on every frame.

        Combined with the reduced pipe-bytes of YUV 4:2:0 ingest, this is the fastest :fontawesome-solid-tachometer-alt-fast: grayscale pipeline the API can produce.

    Similar to normal `GRAYSCALE` format frames, you can directly consume these frames in WriteGear API’s `write()` class method:

    ```python linenums="1" hl_lines="3 6 11 19 24 38 44"
    # import required libraries
    from vidgear.gears import FFGear
    from vidgear.gears import WriteGear

    # enable direct Luma (Y-plane) extraction
    options = {"-extract_luma": True}

    # stream Grayscale via YUV frames
    stream = FFGear(
        source="myvideo.mp4",
        frame_format="yuv420p",
        logging=True,
        **options
    ).start()

    # retrieve framerate from source JSON Metadata and pass it as `-input_framerate` 
    # parameter for controlled framerate
    output_params = {
        "-input_framerate": stream.metadata["source_video_framerate"]
    }

    # Define writer with default parameters and suitable
    # output filename for e.g. `output_foo_yuv_gray.mp4`
    writer = WriteGear(output="output_foo_yuv_gray.mp4", **output_params)

    # loop over
    while True:
        # read grayscale frames
        frame = stream.read()

        # check for frame if NoneType
        if frame is None:
            break

        # {do something with Luma (Y-plane) extracted grayscale frame here}

        # write frame to output
        writer.write(frame)

    # safely close stream
    stream.stop()

    # safely close writer
    writer.close()
    ```

&thinsp;

[yt_dlp]:https://github.com/yt-dlp/yt-dlp
[deffcode]:https://github.com/abhiTronix/deffcode
