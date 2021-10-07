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

# NetGear_Async Examples

&nbsp;

## Using NetGear_Async with WebGear

The complete usage example is as follows: 

??? new "New in v0.2.2" 
    This example was added in `v0.2.2`.

### Client + WebGear Server

Open a terminal on Client System where you want to display the input frames _(and setup WebGear server)_ received from the Server and execute the following python code:

!!! danger "After running this code, Make sure to open Browser immediately otherwise NetGear_Async will soon exit with `TimeoutError`. You can also try setting [`timeout`](../../gears/netgear_async/params/#timeout) parameter to a higher value to extend this timeout."

!!! warning "Make sure you use different `port` value for NetGear_Async and WebGear API."

!!! alert "High CPU utilization may occur on Client's end. User discretion is advised."

!!! note "Note down the IP-address of this system _(required at Server's end)_ by executing the  `hostname -I` command and also replace it in the following code.""

```python
# import libraries
from vidgear.gears.asyncio import NetGear_Async
from vidgear.gears.asyncio import WebGear
from vidgear.gears.asyncio.helper import reducer
import uvicorn, asyncio, cv2

# Define NetGear_Async Client at given IP address and define parameters
# !!! change following IP address '192.168.x.xxx' with yours !!!
client = NetGear_Async(
    receive_mode=True,
    pattern=1,
    logging=True,
).launch()

# create your own custom frame producer
async def my_frame_producer():

    # loop over Client's Asynchronous Frame Generator
    async for frame in client.recv_generator():

        # {do something with received frames here}

        # reducer frames size if you want more performance otherwise comment this line
        frame = await reducer(
            frame, percentage=30, interpolation=cv2.INTER_AREA
        )  # reduce frame by 30%

        # handle JPEG encoding
        encodedImage = cv2.imencode(".jpg", frame)[1].tobytes()
        # yield frame in byte format
        yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" + encodedImage + b"\r\n")
        await asyncio.sleep(0)


if __name__ == "__main__":
    # Set event loop to client's
    asyncio.set_event_loop(client.loop)

    # initialize WebGear app without any source
    web = WebGear(logging=True)

    # add your custom frame producer to config with adequate IP address
    web.config["generator"] = my_frame_producer

    # run this app on Uvicorn server at address http://localhost:8000/
    uvicorn.run(web(), host="localhost", port=8000)

    # safely close client
    client.close()

    # close app safely
    web.shutdown()
```

!!! success "On successfully running this code, the output stream will be displayed at address http://localhost:8000/ in your Client's Browser."

### Server

Now, Open the terminal on another Server System _(with a webcam connected to it at index 0)_, and execute the following python code:

!!! note "Replace the IP address in the following code with Client's IP address you noted earlier."

```python
# import library
from vidgear.gears.asyncio import NetGear_Async
import cv2, asyncio

# initialize Server without any source
server = NetGear_Async(
    source=None,
    address="192.168.x.xxx",
    port="5454",
    protocol="tcp",
    pattern=1,
    logging=True,
)

# Create a async frame generator as custom source
async def my_frame_generator():

    # !!! define your own video source here !!!
    # Open any video stream such as live webcam
    # video stream on first index(i.e. 0) device
    stream = cv2.VideoCapture(0)

    # loop over stream until its terminated
    while True:

        # read frames
        (grabbed, frame) = stream.read()

        # check if frame empty
        if not grabbed:
            break

        # do something with the frame to be sent here

        # yield frame
        yield frame
        # sleep for sometime
        await asyncio.sleep(0)

    # close stream
    stream.release()


if __name__ == "__main__":
    # set event loop
    asyncio.set_event_loop(server.loop)
    # Add your custom source generator to Server configuration
    server.config["generator"] = my_frame_generator()
    # Launch the Server
    server.launch()
    try:
        # run your main function task until it is complete
        server.loop.run_until_complete(server.task)
    except (KeyboardInterrupt, SystemExit):
        # wait for interrupts
        pass
    finally:
        # finally close the server
        server.close()
```

&nbsp;
