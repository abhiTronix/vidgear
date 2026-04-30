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

# FFGear Bonus Examples

&thinsp;

## Using FFGear with WriteGear's Compression Mode

FFGear integrates seamlessly with WriteGear's Compression Mode for high-quality FFmpeg-powered re-encoding of decoded frames.

```python linenums="1" hl_lines="4 7 10-14 17"
# import required libraries
from vidgear.gears import FFGear
from vidgear.gears import WriteGear
import cv2

# open any valid video source with FFGear
stream = FFGear(source="myvideo.mp4", frame_format="bgr24", logging=True).start()

# define WriteGear Compression Mode parameters
output_params = {
    "-vcodec": "libx264",
    "-crf": "23",
    "-preset": "fast",
}

# define WriteGear writer
writer = WriteGear(output="output.mp4", logging=True, **output_params)

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # write frame to writer
    writer.write(frame)

# safely close video stream
stream.stop()

# safely close writer
writer.close()
```

&thinsp;

## Using FFGear with WriteGear for RTSP/RTP Live-Streaming

You can pipeline FFGear's decoded frames directly into WriteGear for re-streaming to an RTSP server:

???+ tip "Creating your own RTSP Server locally"
    Checkout [**MediaMTX (formerly rtsp-simple-server)**](https://github.com/bluenviron/mediamtx) — a ready-to-use, zero-dependency real-time media server.

!!! warning "This example assumes you already have an RTSP Server running at `rtsp://localhost:8554/mystream`."

!!! danger "Make sure to change RTSP address `rtsp://localhost:8554/mystream` with yours before running!"

```python linenums="1" hl_lines="7 10-11 14-16"
# import required libraries
from vidgear.gears import FFGear
from vidgear.gears import WriteGear
import cv2

# open any valid video source with FFGear
stream = FFGear(source="myvideo.mp4", frame_format="bgr24", logging=True).start()

# define WriteGear RTSP streaming parameters
output_params = {"-f": "rtsp", "-rtsp_transport": "tcp"}

# Define WriteGear writer with RTSP output address
# [WARNING] Change RTSP address with yours!
writer = WriteGear(
    output="rtsp://localhost:8554/mystream", logging=True, **output_params
)

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # write frame to writer
    writer.write(frame)

# safely close video stream
stream.stop()

# safely close writer
writer.close()
```

&thinsp;

## Using FFGear with WriteGear for YouTube-Live Streaming

You can use FFGear with WriteGear to transcode and publish to YouTube Live:

!!! alert "This example assumes you already have a [**YouTube Account with Live-Streaming enabled**](https://support.google.com/youtube/answer/2474026#enable)."

!!! danger "Make sure to replace [_YouTube-Live Stream Key_](https://support.google.com/youtube/answer/2907883) with yours before running!"

```python linenums="1" hl_lines="7 10-19 22-24"
# import required libraries
from vidgear.gears import FFGear
from vidgear.gears import WriteGear
import cv2

# open any valid video source with FFGear
stream = FFGear(source="myvideo.mp4", frame_format="bgr24", logging=True).start()

# define required FFmpeg parameters for YouTube Live
output_params = {
    "-clones": ["-f", "lavfi", "-i", "anullsrc"],  # add silent audio (required by YT)
    "-vcodec": "libx264",
    "-preset": "medium",
    "-b:v": "4500k",
    "-bufsize": "512k",
    "-pix_fmt": "yuv420p",
    "-f": "flv",
}

# [WARNING] Change your YouTube-Live Stream Key here:
YOUTUBE_STREAM_KEY = "xxxx-xxxx-xxxx-xxxx-xxxx"

# Define WriteGear writer with YouTube RTMP address
writer = WriteGear(
    output="rtmp://a.rtmp.youtube.com/live2/{}".format(YOUTUBE_STREAM_KEY),
    logging=True,
    **output_params
)

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # write frame to writer
    writer.write(frame)

# safely close video stream
stream.stop()

# safely close writer
writer.close()
```

&thinsp;

## Using FFGear with NetGear (Server)

You can use FFGear as a high-performance video source on a **NetGear Server** to stream decoded frames over the network:

!!! info "Note down the local IP-address of the Server (required at Client's end). You can follow [this FAQ](../netgear_faqs/#how-to-find-local-ip-address-on-different-os-platforms) for this purpose."

### Server

Open a terminal on the **Server** machine and run:

```python linenums="1" hl_lines="4 7 11-17"
# import required libraries
from vidgear.gears import FFGear
from vidgear.gears import NetGear
import cv2

# open video source with FFGear (e.g. hardware-decoded video file)
stream = FFGear(source="myvideo.mp4", frame_format="bgr24", logging=True).start()

# define NetGear Server at default address and port
# !!! change address '192.168.x.xxx' with your Client's IP address !!!
server = NetGear(
    address="192.168.x.xxx",
    port="5454",
    protocol="tcp",
    pattern=1,
    logging=True,
)

# loop over until KeyBoard Interrupted
while True:

    try:
        # read frames from FFGear stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # send frame to NetGear client
        server.send(frame)

    except KeyboardInterrupt:
        break

# safely close FFGear stream
stream.stop()

# safely close NetGear server
server.close()
```

### Client

Open a terminal on the **Client** machine and run:

```python linenums="1" hl_lines="8-14"
# import required libraries
from vidgear.gears import NetGear
import cv2

# define NetGear Client
# !!! change address '192.168.x.xxx' with your Server's IP address !!!
client = NetGear(
    receive_mode=True,
    address="192.168.x.xxx",
    port="5454",
    protocol="tcp",
    pattern=1,
    logging=True,
)

# loop over
while True:

    # receive frames from network
    frame = client.recv()

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

# safely close NetGear client
client.close()
```

&thinsp;

## Using FFGear with NetGear and WebGear

You can combine FFGear as a high-performance server-side video source with NetGear for network transport and WebGear for browser-based display:

!!! warning "Make sure you use different `port` values for NetGear and WebGear APIs."

!!! info "Note down the local IP-address of the Client system and replace it in the following code."

### Client + WebGear Server

```python linenums="1"
# import necessary libs
import uvicorn, asyncio, cv2
from vidgear.gears import NetGear
from vidgear.gears.asyncio import WebGear
from vidgear.gears.asyncio.helper import reducer

# initialize WebGear app without any source
web = WebGear(logging=True)

# activate JPEG compression for performance
options = {
    "jpeg_compression": True,
    "jpeg_compression_quality": 90,
    "jpeg_compression_fastdct": True,
    "jpeg_compression_fastupsample": True,
}

# create custom frame producer
async def my_frame_producer():
    # Define NetGear Client
    # !!! change '192.168.x.xxx' with your Server's IP address !!!
    client = NetGear(
        receive_mode=True,
        address="192.168.x.xxx",
        port="5454",
        protocol="tcp",
        pattern=1,
        logging=True,
        **options,
    )

    # loop over frames
    while True:
        # receive frames from network
        frame = client.recv()

        # if NoneType
        if frame is None:
            break

        # {do something with the frame here}

        # reduce frame size for better performance (optional)
        frame = await reducer(frame, percentage=30, interpolation=cv2.INTER_AREA)

        # encode as JPEG
        encodedImage = cv2.imencode(".jpg", frame)[1].tobytes()
        yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" + encodedImage + b"\r\n")
        await asyncio.sleep(0)

    # close stream
    client.close()


# assign custom frame producer to WebGear config
web.config["generator"] = my_frame_producer

# run WebGear app on Uvicorn at http://localhost:8000/
uvicorn.run(web(), host="localhost", port=8000)

# close app safely
web.shutdown()
```

!!! success "On successfully running this code, the output stream will be displayed at http://localhost:8000/ in your browser."

### Server (FFGear + NetGear)

Open a terminal on the **Server** machine:

!!! note "Replace `192.168.x.xxx` with the Client's IP address you noted earlier."

```python linenums="1"
# import required libraries
from vidgear.gears import FFGear
from vidgear.gears import NetGear
import cv2

# activate JPEG compression for performance
options = {
    "jpeg_compression": True,
    "jpeg_compression_quality": 90,
    "jpeg_compression_fastdct": True,
    "jpeg_compression_fastupsample": True,
}

# open video source with FFGear
stream = FFGear(source="myvideo.mp4", frame_format="bgr24", logging=True).start()

# Define NetGear Server
# !!! change '192.168.x.xxx' with your Client's IP address !!!
server = NetGear(
    address="192.168.x.xxx",
    port="5454",
    protocol="tcp",
    pattern=1,
    logging=True,
    **options
)

# loop over until KeyBoard Interrupted
while True:

    try:
        # read frames from FFGear
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # send frame to client
        server.send(frame)

    except KeyboardInterrupt:
        break

# safely close FFGear stream
stream.stop()

# safely close NetGear server
server.close()
```

&thinsp;

## Synchronizing Two FFGear Sources

You can run two independent FFGear instances simultaneously for synchronized multi-source processing:

!!! danger "Using the same source file path with more than one FFGear instance may cause [GIL](https://wiki.python.org/moin/GlobalInterpreterLock) contention. Use different sources where possible."

```python linenums="1"
# import required libraries
from vidgear.gears import FFGear
import cv2

# open first video source with FFGear
stream1 = FFGear(source="myvideo1.mp4", frame_format="bgr24", logging=True).start()

# open second video source with FFGear
stream2 = FFGear(source="myvideo2.mp4", frame_format="bgr24", logging=True).start()

# infinite loop
while True:

    # read frames from both streams
    frameA = stream1.read()
    frameB = stream2.read()

    # check if either frame is None
    if frameA is None or frameB is None:
        break

    # {do something with frameA and frameB here}

    # show output windows
    cv2.imshow("Output Stream1", frameA)
    cv2.imshow("Output Stream2", frameB)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # press 'w' to save both frames simultaneously
    if key == ord("w"):
        cv2.imwrite("Image-1.jpg", frameA)
        cv2.imwrite("Image-2.jpg", frameB)

# close output windows
cv2.destroyAllWindows()

# safely close both FFGear streams
stream1.stop()
stream2.stop()
```

&thinsp;

## Using FFGear with Variable `yt_dlp` Parameters in Stream Mode

FFGear provides exclusive `STREAM_RESOLUTION` and `STREAM_PARAMS` attributes with its [`options`](../../gears/ffgear/params/#options) dictionary parameter for fine-grained control over streaming URL resolution and `yt_dlp` behavior.

```python linenums="1" hl_lines="6"
# import required libraries
from vidgear.gears import FFGear
import cv2

# specify stream quality and custom yt_dlp parameters
options = {"STREAM_RESOLUTION": "720p", "STREAM_PARAMS": {"nocheckcertificate": True}}

# Add YouTube Video URL as source and enable Stream Mode
stream = FFGear(
    source="https://youtu.be/bvetuLwJIkA",
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

&thinsp;

## Using FFGear with WriteGear for MP4 Segmentation

You can use FFGear as a source for WriteGear's segment muxer to split a video stream into fixed-duration MP4 segments:

```python linenums="1" hl_lines="7 10-18 21"
# import required libraries
from vidgear.gears import FFGear
from vidgear.gears import WriteGear
import cv2

# open source video with FFGear
stream = FFGear(source="myvideo.mp4", frame_format="bgr24", logging=True).start()

# define WriteGear parameters for MP4 segmentation
output_params = {
    "-c:v": "libx264",
    "-crf": 22,
    "-map": 0,
    "-segment_time": 9,
    "-g": 9,
    "-sc_threshold": 0,
    "-force_key_frames": "expr:gte(t,n_forced*9)",
    "-clones": ["-f", "segment"],
}

# Define WriteGear writer with segment output pattern
writer = WriteGear(output="output%03d.mp4", logging=True, **output_params)

# loop over
while True:

    # read frames from FFGear stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # write frame to writer
    writer.write(frame)

# safely close video stream
stream.stop()

# safely close writer
writer.close()
```

&thinsp;
