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

# NetGear Examples

&nbsp;

## Using NetGear with WebGear

The complete usage example is as follows: 

??? new "New in v0.2.2" 
    This example was added in `v0.2.2`.

### Client + WebGear Server

Open a terminal on Client System where you want to display the input frames _(and setup WebGear server)_ received from the Server and execute the following python code:

!!! danger "After running this code, Make sure to open Browser immediately otherwise NetGear will soon exit with `RuntimeError`. You can also try setting [`max_retries`](../../gears/netgear/params/#options) and [`request_timeout`](../../gears/netgear/params/#options) like attributes to a higher value to avoid this."

!!! warning "Make sure you use different `port` value for NetGear and WebGear API."

!!! alert "High CPU utilization may occur on Client's end. User discretion is advised."

!!! info "Note down the local IP-address of this system (required at Server's end) and also replace it in the following code. You can follow [this FAQ](../netgear_faqs/#how-to-find-local-ip-address-on-different-os-platforms) for this purpose."

```python
# import necessary libs
import uvicorn, asyncio, cv2
from vidgear.gears import NetGear
from vidgear.gears.asyncio import WebGear
from vidgear.gears.asyncio.helper import reducer

# initialize WebGear app without any source
web = WebGear(logging=True)


# activate jpeg encoding and specify other related parameters
options = {
    "jpeg_compression": True,
    "jpeg_compression_quality": 90,
    "jpeg_compression_fastdct": True,
    "jpeg_compression_fastupsample": True,
}

# create your own custom frame producer
async def my_frame_producer():
    # initialize global params
    # Define NetGear Client at given IP address and define parameters
    # !!! change following IP address '192.168.x.xxx' with yours !!!
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

        # do something with your OpenCV frame here

        # reducer frames size if you want more performance otherwise comment this line
        frame = await reducer(
            frame, percentage=30, interpolation=cv2.INTER_AREA
        )  # reduce frame by 30%

        # handle JPEG encoding
        encodedImage = cv2.imencode(".jpg", frame)[1].tobytes()
        # yield frame in byte format
        yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" + encodedImage + b"\r\n")
        await asyncio.sleep(0)
    # close stream
    client.close()


# add your custom frame producer to config with adequate IP address
web.config["generator"] = my_frame_producer

# run this app on Uvicorn server at address http://localhost:8000/
uvicorn.run(web(), host="localhost", port=8000)

# close app safely
web.shutdown()
```

!!! success "On successfully running this code, the output stream will be displayed at address http://localhost:8000/ in your Client's Browser."


### Server

Now, Open the terminal on another Server System _(with a webcam connected to it at index 0)_, and execute the following python code:

!!! note "Replace the IP address in the following code with Client's IP address you noted earlier."

```python
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import NetGear
import cv2

# activate jpeg encoding and specify other related parameters
options = {
    "jpeg_compression": True,
    "jpeg_compression_quality": 90,
    "jpeg_compression_fastdct": True,
    "jpeg_compression_fastupsample": True,
}

# Open live video stream on webcam at first index(i.e. 0) device
stream = VideoGear(source=0).start()

# Define NetGear server at given IP address and define parameters 
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

        # check for frame if None-type
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

## Using NetGear with WebGear_RTC

The complete usage example is as follows: 

??? new "New in v0.2.4" 
    This example was added in `v0.2.4`.

### Client + WebGear_RTC Server

Open a terminal on Client System where you want to display the input frames _(and setup WebGear_RTC server)_ received from the Server and execute the following python code:

!!! danger "After running this code, Make sure to open Browser immediately otherwise NetGear will soon exit with `RuntimeError`. You can also try setting [`max_retries`](../../gears/netgear/params/#options) and [`request_timeout`](../../gears/netgear/params/#options) like attributes to a higher value to avoid this."

!!! warning "Make sure you use different `port` value for NetGear and WebGear_RTC API."

!!! alert "High CPU utilization may occur on Client's end. User discretion is advised."

!!! info "Note down the local IP-address of this system(required at Server's end) and also replace it in the following code. You can follow [this FAQ](../netgear_faqs/#how-to-find-local-ip-address-on-different-os-platforms) for this purpose."

!!! fail "For VideoCapture APIs you also need to implement `start()` in addition to `read()` and `stop()` methods in your Custom Streaming Class as shown in following example, otherwise WebGear_RTC will fail to work!"

```python hl_lines="8-79 92-101"
# import necessary libs
import uvicorn, cv2
from vidgear.gears import NetGear
from vidgear.gears.helper import reducer
from vidgear.gears.asyncio import WebGear_RTC

# create your own custom streaming class
class Custom_Stream_Class:
    """
    Custom Streaming using NetGear Receiver
    """

    def __init__(
        self,
        address=None,
        port="5454",
        protocol="tcp",
        pattern=1,
        logging=True,
        **options,
    ):
        # initialize global params
        # Define NetGear Client at given IP address and define parameters
        self.client = NetGear(
            receive_mode=True,
            address=address,
            port=port,
            protocol=protocol,
            pattern=pattern,
            logging=logging,
            **options
        )
        self.running = False

    def start(self):
        
        # don't forget this function!!!
        # This function is specific to VideoCapture APIs only

        if not self.source is None:
            self.source.start()

    def read(self):

        # don't forget this function!!!

        # check if source was initialized or not
        if self.source is None:
            return None
        # check if we're still running
        if self.running:
            # receive frames from network
            frame = self.client.recv()
            # check if frame is available
            if not (frame is None):

                # do something with your OpenCV frame here

                # reducer frames size if you want more performance otherwise comment this line
                frame = reducer(frame, percentage=20)  # reduce frame by 20%

                # return our gray frame
                return frame
            else:
                # signal we're not running now
                self.running = False
        # return None-type
        return None

    def stop(self):

        # don't forget this function!!!

        # flag that we're not running
        self.running = False
        # close stream
        if not (self.client is None):
            self.client.close()
            self.client = None


# activate jpeg encoding and specify NetGear related parameters
options = {
    "jpeg_compression": True,
    "jpeg_compression_quality": 90,
    "jpeg_compression_fastdct": True,
    "jpeg_compression_fastupsample": True,
}

# assign your Custom Streaming Class with adequate NetGear parameters
# to `custom_stream` attribute in options parameter of WebGear_RTC.
options = {
    "custom_stream": Custom_Stream_Class(
        address="192.168.x.xxx",
        port="5454",
        protocol="tcp",
        pattern=1,
        logging=True,
        **options
    )
}

# initialize WebGear_RTC app without any source
web = WebGear_RTC(logging=True, **options)

# run this app on Uvicorn server at address http://localhost:8000/
uvicorn.run(web(), host="localhost", port=8000)

# close app safely
web.shutdown()
```

!!! success "On successfully running this code, the output stream will be displayed at address http://localhost:8000/ in your Client's Browser."

### Server

Now, Open the terminal on another Server System _(with a webcam connected to it at index 0)_, and execute the following python code:

!!! note "Replace the IP address in the following code with Client's IP address you noted earlier."

```python
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import NetGear
import cv2

# activate jpeg encoding and specify other related parameters
options = {
    "jpeg_compression": True,
    "jpeg_compression_quality": 90,
    "jpeg_compression_fastdct": True,
    "jpeg_compression_fastupsample": True,
}

# Open live video stream on webcam at first index(i.e. 0) device
stream = VideoGear(source=0).start()

# Define NetGear server at given IP address and define parameters 
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
