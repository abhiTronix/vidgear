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


&nbsp; 


## Bare-Minimum Usage

Following is the bare-minimum code you need to get started with WebGear API:

!!! warning "If you experience any performance-throttling/lag while running this bare-minimum example, then Kindly see advanced [Performance Tweaks ➶](/gears/webgear/advanced/#performance-enhancements)."

### Running Programmatically

You can access and run WebGear VideoStreamer Server programmatically in your python script in just a few lines of code, as follows:

```python
# import libs
import uvicorn
from vidgear.gears.asyncio import WebGear

# initialize WebGear app  
web=WebGear(source="test.mp4")

# run this app on Uvicorn server
uvicorn.run(web(), host='0.0.0.0', port=8000)

# close app safely
web.shutdown()
```
That's all. Now, just run that, and a live video stream can be accessed on any browser at http://0.0.0.0:8000/ address.


### Running from Terminal

You can access and run WebGear Server directly from the terminal commandline as follows:

!!! danger "Make sure your `PYTHON_PATH` is set to python 3.6+ versions only."

The following command will run a WebGear VideoStreamer server at http://0.0.0.0:8000/:

```sh
python3 -m vidgear.gears.asyncio --source test.avi 

```
which can be accessed on any browser on the network.

??? tip "Advanced Usage from Terminal"

    You can run `#!py3 python3 -m vidgear.gears -h` help command to see all the advanced settings, as follows:

    !!! warning "If you're using `--options/-op` flag, then kindly wrap your dictionary value in single `''` quotes as shown in [this example](/gears/webgear/advanced/#running-from-terminal)."

    ```sh
    usage: python -m vidgear.gears.asyncio [-h] [-s SOURCE] [-ep ENABLEPICAMERA] [-S STABILIZE]
            [-cn CAMERA_NUM] [-yt Y_TUBE] [-b BACKEND] [-cs COLORSPACE]
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
      -yt Y_TUBE, --y_tube Y_TUBE
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
