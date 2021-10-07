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

# ScreenGear Examples

&nbsp;

## Using ScreenGear with NetGear and WriteGear

The complete usage example is as follows: 

??? new "New in v0.2.2" 
    This example was added in `v0.2.2`.

### Client + WriteGear

Open a terminal on Client System _(where you want to save the input frames received from the Server)_ and execute the following python code: 

!!! info "Note down the IP-address of this system(required at Server's end) by executing the command: `hostname -I` and also replace it in the following code."

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python
# import required libraries
from vidgear.gears import NetGear
from vidgear.gears import WriteGear
import cv2

# define various tweak flags
options = {"flag": 0, "copy": False, "track": False}

# Define Netgear Client at given IP address and define parameters 
# !!! change following IP address '192.168.x.xxx' with yours !!!
client = NetGear(
    address="192.168.x.xxx",
    port="5454",
    protocol="tcp",
    pattern=1,
    receive_mode=True,
    logging=True,
    **options
)

# Define writer with default parameters and suitable output filename for e.g. `Output.mp4`
writer = WriteGear(output_filename="Output.mp4")

# loop over
while True:

    # receive frames from network
    frame = client.recv()

    # check for received frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # write frame to writer
    writer.write(frame)

# close output window
cv2.destroyAllWindows()

# safely close client
client.close()

# safely close writer
writer.close()
```

### Server + ScreenGear

Now, Open the terminal on another Server System _(with a montior/display attached to it)_, and execute the following python code: 

!!! info "Replace the IP address in the following code with Client's IP address you noted earlier."

!!! tip "You can terminate stream on both side anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import NetGear

# define dimensions of screen w.r.t to given monitor to be captured
options = {"top": 40, "left": 0, "width": 100, "height": 100}

# open stream with defined parameters
stream = ScreenGear(logging=True, **options).start()

# define various netgear tweak flags
options = {"flag": 0, "copy": False, "track": False}

# Define Netgear server at given IP address and define parameters 
# !!! change following IP address '192.168.x.xxx' with client's IP address !!!
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
        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # send frame to server
        server.send(frame)

    except KeyboardInterrupt:
        break

# safely close video stream
stream.stop()

# safely close server
server.close()
```

&nbsp;

## Using ScreenGear with WebGear_RTC

The complete usage example is as follows: 

??? new "New in v0.2.3" 
    This example was added in `v0.2.3`.

```python
# import necessary libs
import uvicorn, asyncio, cv2
from av import VideoFrame
from aiortc import VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from vidgear.gears import ScreenGear
from vidgear.gears.asyncio import WebGear_RTC
from vidgear.gears.asyncio.helper import reducer

# initialize WebGear_RTC app without any source
web = WebGear_RTC(logging=True)

# create your own Bare-Minimum Custom Media Server with ScreenGear source
class Custom_RTCScreenGearServer(VideoStreamTrack):
    """
    Custom Media Server using OpenCV, an inherit-class
    to aiortc's VideoStreamTrack.
    """

    def __init__(self, monitor=None, logging=False):

        # don't forget this line!
        super().__init__()

        # initialize global params
        self.stream = ScreenGear(monitor=monitor, logging=logging).start()

    async def recv(self):
        """
        A coroutine function that yields `av.frame.Frame`.
        """
        # don't forget this function!!!

        # get next timestamp
        pts, time_base = await self.next_timestamp()

        # read video frame
        frame = self.stream.read()

        # if NoneType
        if frame is None:
            return MediaStreamError

        # reducer frames size if you want more performance otherwise comment this line
        frame = await reducer(frame, percentage=20)  # reduce frame by 20%

        # contruct `av.frame.Frame` from `numpy.nd.array`
        if frame.shape[-1] == 4:
            # hack: handles `mss` backend that outputs BGRA frames
            av_frame = VideoFrame.from_ndarray(frame, format="bgra")
        else:
            av_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        av_frame.pts = pts
        av_frame.time_base = time_base

        # return `av.frame.Frame`
        return av_frame

    def terminate(self):
        """
        Gracefully terminates VideoGear stream
        """
        # don't forget this function!!!

        # terminate
        if not (self.stream is None):
            self.stream.stop()
            self.stream = None


# assign your custom media server to config with ScreenGear source
web.config["server"] = Custom_RTCScreenGearServer(logging=True)

# run this app on Uvicorn server at address http://localhost:8000/
uvicorn.run(web(), host="localhost", port=8000)

# close app safely
web.shutdown()
```

&nbsp; 
