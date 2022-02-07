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

??? error "Microsoft Visual C++ 14.0 is required."
    
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

### Running Programmatically

You can access and run WebGear_RTC VideoStreamer Server programmatically in your python script in just a few lines of code, as follows:

!!! tip "For accessing WebGear_RTC on different Client Devices on the network, use `"0.0.0.0"` as host value instead of `"localhost"` on Host Machine. More information can be found [here ➶](../../../help/webgear_rtc_faqs/#is-it-possible-to-stream-on-a-different-device-on-the-network-with-webgear_rtc)"

!!! info "We are using `frame_size_reduction` attribute for frame size reduction _(in percentage)_ to be streamed with its [`options`](../params/#options) dictionary parameter to cope with performance-throttling in this example."

```python hl_lines="7"
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


### Running from Terminal

You can also access and run WebGear_RTC Server directly from the terminal commandline. The following command will run a WebGear_RTC VideoStreamer server at http://localhost:8000/:

!!! danger "Make sure your `PYTHON_PATH` is set to python 3.7+ versions only."

!!! warning "If you're using `--options/-op` flag, then kindly wrap your dictionary value in single `''` quotes."

```sh
python3 -m vidgear.gears.asyncio --mode webrtc --source test.avi --logging True --options '{"frame_size_reduction": 50, "frame_jpeg_quality": 80, "frame_jpeg_optimize": True, "frame_jpeg_progressive": False}'
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