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


## Performance Enhancements :fire:

WebGear provides certain performance enhancing attributes for its [`options`](../params/#options) dictionary parameter to cope with performance-throttling.

!!! tip "Performance Enhancing Attributes"

    * **`frame_size_reduction`**: _(int/float)_ _This attribute controls the size reduction(in percentage) of the frame to be streamed on Server._ Its value has the most significant effect on WebGear performance: More its value, smaller will be frame size and faster will be live streaming. The value defaults to `20`, and must be no higher than `90` _(fastest, max compression, Barely Visible frame-size)_ and no lower than `0` _(slowest, no compression, Original frame-size)_. Its recommended value is between `40~60`. Its usage is as follows:

        ```python
        options={"frame_size_reduction": 50} #frame-size will be reduced by 50%
        ```
     
    * **Various Encoding Parameters:**

        In WebGear API, the input video frames are first encoded into [**Motion JPEG (M-JPEG or MJPEG**)](https://en.wikipedia.org/wiki/Motion_JPEG) compression format, in which each video frame or interlaced field of a digital video sequence is compressed separately as a JPEG image using [`simplejpeg`](https://gitlab.com/jfolz/simplejpeg) library, before sending onto a server. Therefore, WebGear API provides various attributes to have full control over JPEG encoding performance and quality, which are as follows:

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

&nbsp; 


## Bare-Minimum Usage with Performance Enhancements

Let's implement our Bare-Minimum usage example with these [**Performance Enhancing Attributes** ➶](#performance-enhancements) for speeding up the output.

### Running Programmatically

You can access and run WebGear VideoStreamer Server programmatically in your python script in just a few lines of code, as follows:

!!! tip "For accessing WebGear on different Client Devices on the network, use `"0.0.0.0"` as host value instead of `"localhost"` on Host Machine. More information can be found [here ➶](../../../help/webgear_faqs/#is-it-possible-to-stream-on-a-different-device-on-the-network-with-webgear)"


```python hl_lines="7-10"
# import required libraries
import uvicorn
from vidgear.gears.asyncio import WebGear

# various performance tweaks
options = {
    "frame_size_reduction": 40,
    "jpeg_compression_quality": 80,
    "jpeg_compression_fastdct": True,
    "jpeg_compression_fastupsample": False,
}

# initialize WebGear app
web = WebGear(source="foo.mp4", logging=True, **options)

# run this app on Uvicorn server at address http://localhost:8000/
uvicorn.run(web(), host="localhost", port=8000)

# close app safely
web.shutdown()
```

which can be accessed on any browser on your machine at http://localhost:8000/.


### Running from Terminal

You can also access and run WebGear Server directly from the terminal commandline. The following command will run a WebGear VideoStreamer server at http://localhost:8000/:

!!! danger "Make sure your `PYTHON_PATH` is set to python 3.7+ versions only."

!!! warning "If you're using `--options/-op` flag, then kindly wrap your dictionary value in single `''` quotes."

```sh
python3 -m vidgear.gears.asyncio --source test.avi --logging True --options '{"frame_size_reduction": 50, "jpeg_compression_quality": 80, "jpeg_compression_fastdct": True, "jpeg_compression_fastupsample": False}'
```

which can also be accessed on any browser on the network at http://localhost:8000/.


??? tip "Advanced Usage from Terminal"

    You can run `#!py3 python3 -m vidgear.gears.asyncio -h` help command to see all the advanced settings, as follows:

    ```sh
    usage: python -m vidgear.gears.asyncio [-h] [-m MODE] [-s SOURCE] [-ep ENABLEPICAMERA] [-S STABILIZE]
                [-cn CAMERA_NUM] [-yt stream_mode] [-b BACKEND] [-cs COLORSPACE]
                [-r RESOLUTION] [-f FRAMERATE] [-td TIME_DELAY]
                [-ip IPADDRESS] [-pt PORT] [-l LOGGING] [-op OPTIONS]

    Runs WebGear/WebGear_RTC Video Server through terminal.

    optional arguments:
      -h, --help            show this help message and exit
      -m {mjpeg,webrtc}, --mode {mjpeg,webrtc}
                            Whether to use "MJPEG" or "WebRTC" mode for streaming.
      -s SOURCE, --source SOURCE
                            Path to input source for CamGear API.
      -ep ENABLEPICAMERA, --enablePiCamera ENABLEPICAMERA
                            Sets the flag to access PiGear(if True) or otherwise
                            CamGear API respectively.
      -S STABILIZE, --stabilize STABILIZE
                            Enables/disables real-time video stabilization.
      -cn CAMERA_NUM, --camera_num CAMERA_NUM
                            Sets the camera module index that will be used by
                            PiGear API.
      -yt STREAM_MODE, --stream_mode STREAM_MODE
                            Enables YouTube Mode in CamGear API.
      -b BACKEND, --backend BACKEND
                            Sets the backend of the video source in CamGear API.
      -cs COLORSPACE, --colorspace COLORSPACE
                            Sets the colorspace of the output video stream.
      -r RESOLUTION, --resolution RESOLUTION
                            Sets the resolution (width,height) for camera module
                            in PiGear API.
      -f FRAMERATE, --framerate FRAMERATE
                            Sets the framerate for camera module in PiGear API.
      -td TIME_DELAY, --time_delay TIME_DELAY
                            Sets the time delay(in seconds) before start reading
                            the frames.
      -ip IPADDRESS, --ipaddress IPADDRESS
                            Uvicorn binds the socket to this ipaddress.
      -pt PORT, --port PORT
                            Uvicorn binds the socket to this port.
      -l LOGGING, --logging LOGGING
                            Enables/disables error logging, essential for
                            debugging.
      -op OPTIONS, --options OPTIONS
                            Sets the parameters supported by APIs(whichever being
                            accessed) to the input videostream, But make sure to
                            wrap your dict value in single or double quotes.

    ```

&nbsp; 