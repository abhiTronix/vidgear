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

# WebGear_RTC API Usage Examples:

## Requirements

### Installation with Asyncio Support


WebGear_RTC API is the part of `asyncio` package of VidGear, thereby you need to install VidGear with asyncio support as follows:

  ```sh
  pip install vidgear[asyncio]
  ```

### Aiortc

Must Required with WebGear_RTC API. You can easily install it via pip:

??? failure "Microsoft Visual C++ 14.0 is required."
    
    Installing `aiortc` on windows requires Microsoft Build Tools for Visual C++ libraries installed. You can easily fix this error by installing any **ONE** of these choices:

    !!! info "While the error is calling for VC++ 14.0 - but newer versions of Visual C++ libraries works as well."

      - Microsoft [Build Tools for Visual Studio](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16).
      - Alternative link to Microsoft [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019).
      - Offline installer: [vs_buildtools.exe](https://aka.ms/vs/16/release/vs_buildtools.exe)

    Afterwards, Select: Workloads → Desktop development with C++, then for Individual Components, select only:

      - [x] Windows 10 SDK
      - [x] C++ x64/x86 build tools

    Finally, proceed installing `aiortc` via pip.

```sh
  pip install aiortc
``` 

### ASGI Server

You'll also need to install an ASGI Server to run following WebGear_RTC usage examples, and by default WebGear_RTC ships the state-of-the-art [**`uvicorn`**](http://www.uvicorn.org/) Server. But you can also use other ASGI server such as [`daphne`](https://github.com/django/daphne/), or [`hypercorn`](https://pgjones.gitlab.io/hypercorn/) with it.


&nbsp; 


## Bare-Minimum Usage

Let's implement a Bare-Minimum usage example:

### Running Programmatically :fontawesome-brands-python:

You can access and run WebGear_RTC VideoStreamer Server programmatically in your python script in just a few lines of code, as follows:

!!! tip "For accessing WebGear_RTC on different Client Devices on the network, use `"0.0.0.0"` as host value instead of `"localhost"` on Host Machine. More information can be found [here ➶](../../../help/webgear_rtc_faqs/#is-it-possible-to-stream-on-a-different-device-on-the-network-with-webgear_rtc)"

!!! info "We are using `frame_size_reduction` attribute for frame size reduction _(in percentage)_ to be streamed with its [`options`](../params/#options) dictionary parameter to cope with performance-throttling in this example."

```python linenums="1" hl_lines="7"
# import required libraries
import uvicorn
from vidgear.gears.asyncio import WebGear_RTC

# various performance tweaks
options = {
    "frame_size_reduction": 25,
}

# initialize WebGear_RTC app
web = WebGear_RTC(source="foo.mp4", logging=True, **options)

# run this app on Uvicorn server at address http://localhost:8000/
uvicorn.run(web(), host="localhost", port=8000)

# close app safely
web.shutdown()
```

which can be accessed on any browser on your machine at http://localhost:8000/.


### Running from Terminal :fontawesome-solid-terminal:

You can also access and run WebGear_RTC Server directly from the terminal commandline. The following command will run a WebGear_RTC VideoStreamer server at http://localhost:8000/:

!!! danger "Make sure your `PYTHON_PATH` is set to python 3.7+ versions only."

!!! warning "If you're using `--options/-op` flag, then kindly wrap your dictionary value in single `''` quotes."

```sh
python3 -m vidgear.gears.asyncio --mode webrtc --source test.avi --logging --options '{"frame_size_reduction": 50, "frame_jpeg_quality": 80, "frame_jpeg_optimize": True, "frame_jpeg_progressive": False}'
```

which can also be accessed on any browser on the network at http://localhost:8000/.


??? tip "Advanced Usage from Terminal"

    You can run `#!py3 python3 -m vidgear.gears.asyncio -h` help command to see all the advanced settings, as follows:

    ```sh
    usage: python -m vidgear.gears.asyncio [-h] [-m MODE] [-s SOURCE] [-a API] [-S] [-b BACKEND]
                    [-cs COLORSPACE] [-cn CAMERA_NUM] [-r RESOLUTION] [-f FRAMERATE]
                    [-yt] [-sd SOURCE_DEMUXER] [-ff FRAME_FORMAT] [-cf CUSTOM_FFMPEG]
                    [-td TIME_DELAY] [-ip IPADDRESS] [-pt PORT] [-l] [-op OPTIONS]

    Runs WebGear/WebGear_RTC Video Server through terminal.

    options:
    -h, --help            show this help message and exit
    -m {mjpeg,webrtc}, --mode {mjpeg,webrtc}
                            Whether to use "MJPEG" or "WebRTC" mode for streaming.
    -a {camgear,pigear,ffgear}, --api {camgear,pigear,ffgear}
                            Selects the capture backend for VideoGear. Choices: camgear, pigear, ffgear. Default: camgear.
    -ep, --enablePiCamera
                            [DEPRECATED] Use `--api pigear` instead. Sets the flag to access PiGear API.
    -S, --stabilize       Enables real-time video stabilization.
    -s SOURCE, --source SOURCE
                            Path to input source (device index, filepath, URL, or glob pattern).
    -yt, --stream_mode    Enables YouTube/yt_dlp Stream Mode in CamGear/FFGear API.
    -b BACKEND, --backend BACKEND
                            Sets the backend of the video source in CamGear API (e.g. cv2.CAP_DSHOW).
    -sd SOURCE_DEMUXER, --source_demuxer SOURCE_DEMUXER
                            [FFGear only] FFmpeg demuxer for the source (e.g. "v4l2", "dshow"). Default: auto-detect.
    -ff FRAME_FORMAT, --frame_format FRAME_FORMAT
                            [FFGear only] Pixel format for decoded frames (e.g. "bgr24", "gray"). Default: bgr24.
    -cf CUSTOM_FFMPEG, --custom_ffmpeg CUSTOM_FFMPEG
                            [FFGear only] Path to a custom FFmpeg executable. Default: use PATH.
    -cn CAMERA_NUM, --camera_num CAMERA_NUM
                            [PiGear only] Sets the camera module index.
    -r RESOLUTION, --resolution RESOLUTION
                            [PiGear only] Sets the resolution (width,height) for the camera module.
    -f FRAMERATE, --framerate FRAMERATE
                            [PiGear only] Sets the framerate for the camera module.
    -cs COLORSPACE, --colorspace COLORSPACE
                            Sets the colorspace of the output video stream.
    -td TIME_DELAY, --time_delay TIME_DELAY
                            Sets the time delay (in seconds) before start reading the frames.
    -ip IPADDRESS, --ipaddress IPADDRESS
                            Uvicorn binds the socket to this ipaddress.
    -pt PORT, --port PORT
                            Uvicorn binds the socket to this port.
    -l, --logging         Enables/disables error logging, essential for debugging.
    -op OPTIONS, --options OPTIONS
                            Sets the parameters supported by APIs (whichever being accessed) to the input videostream. Wrap your dict value in
                            single or double quotes.
    ```

&nbsp; 