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

## Using FFGear for Real-Time AI/ML Video Inference

FFGear can be used for real-time AI/ML video inference with several built-in optimization techniques for improving throughput, reducing latency, and lowering compute usage.

<figure>
  <img src="https://gitlab.com/abhiTronix/Imbakup/-/raw/master/Images/vidgear/ffgear-yolo10n.gif" alt="FFGear Keyframes (I-frames) optimization in action!" loading="lazy" width=85%/>
  <figcaption>FFGear Keyframes (I-frames) optimization in action!</figcaption>
</figure>

Here's an example of using FFGear to optimize [**YOLOv10-Nano** model inference](https://docs.ultralytics.com/models/yolov10/) by processing only Keyframes (I-frames) while skipping all non-keyframes (P/B-frames), reducing unnecessary compute usage and saving the annotated output as an **Optimized GIF** with the WriteGear API:

???+ warning "This example requires the latest `ultralytics` package"

    Install or upgrade the `ultralytics` package with pip:

    ```sh
    pip install -U ultralytics
    ```

    Explore additional installation methods [here ➶](https://docs.ultralytics.com/quickstart/).


```python linenums="1" hl_lines="3-4 7 15-16 20 44-54"
# import required libraries
from vidgear.gears import FFGear
from vidgear.gears import WriteGear
from ultralytics import YOLO

# Initialize YOLOv10-Nano model
model = YOLO("yolov10n.pt")

# Configure FFGear with per-frame metadata extraction enable
options = {"-extract_metadata": True}
stream = FFGear(source="myvideo.mp4", frame_format="bgr24", logging=True, **options).start()

# Add additional required FFmpeg parameters for Optimized GIF
output_params = {
    "-filter_complex": "[0:v] fps=10,scale=640:-1:flags=lanczos,split [a][b];[a] palettegen [p];[b][p] paletteuse",
    "-vcodec": None,  # let FFmpeg auto-select the best video encoder for the output (e.g. GIF)
}

# Define writer with defined parameters
writer = WriteGear(output="output_foo.gif", **output_params)

# loop over
while True:

    # read data from stream
    output = stream.read()

    # check if end of stream
    if output is None:
        break

    # Unpack the frame and its associated metadata
    frame, meta = output

    # --- OPTIMIZATION STEP ---
    # We skip all non-keyframes to save processing power.
    # This ensures the model only runs on the most information-dense frames.
    if meta and not meta["is_keyframe"]:
        continue  # <-- Skips Non-key frames (P, B-frames)

    # Log keyframe details
    print(f"Keyframe #{meta['frame_num']} at {meta['pts_time']:.3f}s")

    # Perform AI Inference on keyframes (I-frames) only
    # Because we skip non-keyframes, this heavy task runs significantly less often.
    results = model(frame)

    # Annotate the frame with detection boxes and labels
    annotated_frame = results[0].plot()

    # {Insert your custom logic here, e.g., displaying/saving frames or triggering an alert}

    # writing Annotated frame to writer
    writer.write(annotated_frame)

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