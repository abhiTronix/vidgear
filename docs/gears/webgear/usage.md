<!--
===============================================
vidgear library source-code is deployed under the Apache 2.0 License:

Copyright (c) 2019-2020 Abhishek Thakur(@abhiTronix) <abhi.una12@gmail.com>

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

WebGear provides certain performance enhancing attributes for its [`option`](../params/#options) dictionary parameter to cope with performance-throttling.

!!! tip "Performance Enhancing Attributes"

    * **`frame_size_reduction`**: _(int/float)_ _This attribute controls the size reduction(in percentage) of the frame to be streamed on Server._ Its value has the most significant effect on WebGear performance: More its value, smaller will be frame size and faster will be live streaming. The value defaults to `20`, and must be no higher than `90` _(fastest, max compression, Barely Visible frame-size)_ and no lower than `0` _(slowest, no compression, Original frame-size)_. Its recommended value is between `40~60`. Its usage is as follows:

        ```python
        options={"frame_size_reduction": 50} #frame-size will be reduced by 50%
        ```
     
    * **Various Encoding Parameters:**

        In WebGear API, the input video frames are first encoded into [**Motion JPEG (M-JPEG or MJPEG**)](https://en.wikipedia.org/wiki/Motion_JPEG) compression format, in which each video frame or interlaced field of a digital video sequence is compressed separately as a JPEG image, before sending onto a server. Therefore, WebGear API provides various attributes to have full control over JPEG encoding performance and quality, which are as follows:

        *  **`frame_jpeg_quality`**: _(int)_ It controls the JPEG encoder quality. Its value varies from `0` to `100` (the higher is the better quality but performance will be lower). Its default value is `95`. Its usage is as follows:

            ```python
            options={"frame_jpeg_quality": 80} #JPEG will be encoded at 80% quality.
            ```

        * **`frame_jpeg_optimize`**: _(bool)_ It enables various JPEG compression optimizations such as Chroma sub-sampling, Quantization table, etc. These optimizations based on JPEG libs which are used while compiling OpenCV binaries, and recent versions of OpenCV uses [**TurboJPEG library**](https://libjpeg-turbo.org/), which is highly recommended for performance. Its default value is `False`. Its usage is as follows:

            ```python
            options={"frame_jpeg_optimize": True} #JPEG optimizations are enabled.
            ```

        * **`frame_jpeg_progressive`**: _(bool)_ It enables **Progressive** JPEG encoding instead of the **Baseline**.   Progressive Mode, displays an image in such a way that it shows a blurry/low-quality photo in its entirety, and then becomes clearer as the image downloads, whereas in Baseline Mode, an image created using the JPEG compression algorithm that will start to display the image as the data is made available, line by line. Progressive Mode, can drastically improve the performance in WebGear but at the expense of additional CPU load, thereby suitable for powerful systems only. Its default value is `False` meaning baseline mode is in-use. Its usage is as follows:

            ```python
            options={"frame_jpeg_progressive": True} #Progressive JPEG encoding enabled.
            ```

&nbsp; 


## Bare-Minimum Usage with Performance Enhancements

Let's re-implement our previous Bare-Minimum usage example with these [**Performance Enhancing Attributes** ➶](#performance-enhancements) for speeding up the output.

### Running Programmatically

You can access and run WebGear VideoStreamer Server programmatically in your python script in just a few lines of code, as follows:

!!! tip "If you want see output on different machine on the same network, then you need to note down the IP-address of host system. Finally you need to replace this address _(along with selected port)_ on the target machine's browser."

```python
# import required libraries
import uvicorn
from vidgear.gears.asyncio import WebGear

# various performance tweaks
options = {
    "frame_size_reduction": 40,
    "frame_jpeg_quality": 80,
    "frame_jpeg_optimize": True,
    "frame_jpeg_progressive": False,
}

# initialize WebGear app
web = WebGear(source="foo.mp4", logging=True, **options)

# run this app on Uvicorn server at address http://localhost:8000/
uvicorn.run(web(), host="localhost", port=8000)

# close app safely
web.shutdown()
```

which can be accessed on any browser on the network at http://localhost:8000/.


### Running from Terminal

You can also access and run WebGear Server directly from the terminal commandline. The following command will run a WebGear VideoStreamer server at http://localhost:8000/:

!!! danger "Make sure your `PYTHON_PATH` is set to python 3.6+ versions only."

!!! warning "If you're using `--options/-op` flag, then kindly wrap your dictionary value in single `''` quotes."

```sh
python3 -m vidgear.gears.asyncio --source test.avi --logging True --options '{"frame_size_reduction": 50, "frame_jpeg_quality": 80, "frame_jpeg_optimize": True, "frame_jpeg_progressive": False}'
```

which can also be accessed on any browser on the network at http://localhost:8000/.


??? tip "Advanced Usage from Terminal"

    You can run `#!py3 python3 -m vidgear.gears -h` help command to see all the advanced settings, as follows:

    !!! warning "If you're using `--options/-op` flag, then kindly wrap your dictionary value in single `''` quotes as shown in [this example](../advanced/#running-from-terminal)."

    ```sh
    usage: python -m vidgear.gears.asyncio [-h] [-s SOURCE] [-ep ENABLEPICAMERA] [-S STABILIZE]
            [-cn CAMERA_NUM] [-yt stream_mode] [-b BACKEND] [-cs COLORSPACE]
            [-r RESOLUTION] [-f FRAMERATE] [-td TIME_DELAY]
            [-ip IPADDRESS] [-pt PORT] [-l LOGGING] [-op OPTIONS]

    Runs WebGear VideoStreaming Server through terminal.

    optional arguments:
      -h, --help            show this help message and exit
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
      -yt stream_mode, --stream_mode stream_mode
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