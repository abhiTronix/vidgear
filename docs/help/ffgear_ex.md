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

## Using FFGear with WriteGear for YouTube-Live Streaming

You can use FFGear with WriteGear to transcode and publish to YouTube Live:

!!! alert "This example assumes you already have a [**YouTube Account with Live-Streaming enabled**](https://support.google.com/youtube/answer/2474026#enable)."

!!! danger "Make sure to replace [_YouTube-Live Stream Key_](https://support.google.com/youtube/answer/2907883) with yours before running!"

```python linenums="1" hl_lines="11-17 21 25"
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

```python linenums="1" hl_lines="2 7 24 33"
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

```python linenums="1"
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

```python linenums="1" hl_lines="2 15 33 42"
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

## Using FFGear with Live Custom watermark image overlay

<figure markdown>
  ![Big Buck Bunny with watermark](https://abhitronix.github.io/deffcode/latest/assets/gifs/watermark_overlay.gif)
  <figcaption>Big Buck Bunny with custom watermark</figcaption>
</figure>

In this example, we apply a watermark image _(say `watermark.png` with transparent background)_ overlay to the `10` seconds of video file _(say `foo.mp4`)_ using FFmpeg's [`overlay`](https://ffmpeg.org/ffmpeg-filters.html#toc-overlay-1) filter with some additional filtering:

!!! info "You can use FFGear API's [`stream.metadata`](../../reference/ffdecoder/#deffcode.ffdecoder.FFdecoder.metadata) property object that dumps Source Metadata as JSON to retrieve source framerate and frame-size."

!!! alert "Remember to replace `watermark.png` watermark image file-path with yours before using this recipe."

```python linenums="1" hl_lines="8-18 27"
# import the necessary packages
from vidgear.gears import FFGear
from vidgear.gears import WriteGear
import json, cv2

# define the Complex Video Filter with additional `watermark.png` image input
options = {
    "-ffprefixes": ["-t", "10"],  # playback time of 10 seconds
    "-clones": [
        "-i",
        "watermark.png",  # !!! [WARNING] define your `watermark.png` here.
    ],
    "-filter_complex": "[1]format=rgba,"  # change 2nd(image) input format to yuv444p
    + "colorchannelmixer=aa=0.7[logo];"  # apply colorchannelmixer to image for controlling alpha [logo]
    + "[0][logo]overlay=W-w-{pixel}:H-h-{pixel}:format=auto,".format(  # apply overlay to 1st(video) with [logo]
        pixel=5  # at 5 pixels from the bottom right corner of the input video
    )
    + "format=bgr24",  # change output format to `bgr24`
}

# open source with FFGear and BGR frames with given params
stream = FFGear(source="foo.mp4", frame_format="bgr24", logging=True).start()

# retrieve framerate from source JSON Metadata and pass it as `-input_framerate` 
# parameter for controlled framerate
output_params = {
    "-input_framerate": json.loads(stream.stream.metadata)["source_video_framerate"]
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

# safely close writer
writer.close()

# safely close stream
stream.stop()
```

&thinsp;

## Using FFGear to generate Mandelbrot test pattern with vectorscope & waveforms

> The [`mandelbrot`](https://ffmpeg.org/ffmpeg-filters.html#toc-mandelbrot) graph generate a [**Mandelbrot set fractal**](https://en.wikipedia.org/wiki/Mandelbrot_set), that progressively zoom towards a specific point.

<figure markdown>
  ![mandelbrot test pattern](https://abhitronix.github.io/deffcode/latest/assets/gifs/mandelbrot_vectorscope_waveforms.gif){ width="500" }
  <figcaption>Mandelbrot pattern with a Vectorscope & two Waveforms</figcaption>
</figure>


In this example we generate `30` seconds of a **Mandelbrot test pattern** _(`1280x720` frame size & `30` framerate)_ using [`mandelbrot`](https://ffmpeg.org/ffmpeg-filters.html#toc-mandelbrot) graph source with `lavfi` input virtual device with a [vectorscope](https://www.studiobinder.com/blog/what-is-a-vectorscope-definition/) _(plots 2 color component values)_ & two [waveforms](https://ffmpeg.org/ffmpeg-filters.html#toc-waveform) _(plots YUV color component intensity)_ stacked to it: 


```python linenums="1" hl_lines="7-15 21-22"
# import required libraries
from vidgear.gears import FFGear
import cv2

# define parameters
options = {
    "-ffprefixes": ["-t", "20"],  # playback time of 20 seconds
    "-vf": "format=yuv444p," # change input format to yuv444p
    + "split=4[a][b][c][d]," # split input into 4 identical outputs.
    + "[a]waveform[aa],"  # apply waveform on first output
    + "[b][aa]vstack[V],"  # vertical stack 2nd output with waveform [V]
    + "[c]waveform=m=0[cc],"  # apply waveform on 3rd output
    + "[d]vectorscope=color4[dd],"  # apply vectorscope on 4th output
    + "[cc][dd]vstack[V2],"  # vertical stack waveform and vectorscope [V2]
    + "[V][V2]hstack",  # horizontal stack [V] and [V2] vertical stacks
}

# stream with "mandelbrot" source of
# `1280x720` frame size and `30` framerate for BGR24 output
stream = FFGear(
    source="mandelbrot=size=1280x720:rate=30",
    source_demuxer="lavfi",
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

&thinsp;
