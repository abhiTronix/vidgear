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

# Bidirectional Mode for NetGear_Async API 

<h3 align="center">
  <img src="../../../../assets/images/bidir_async.png" alt="Bidirectional Mode" loading="lazy" class="center"/>
  <figcaption>NetGear_Async's Bidirectional Mode</figcaption>
</h3>

## Overview

??? new "New in v0.2.2" 
    This document was added in `v0.2.2`.

Bidirectional Mode enables seamless support for Bidirectional data transmission between Client and Sender along with video-frames through its synchronous messaging patterns such as `zmq.PAIR` (ZMQ Pair Pattern) & `zmq.REQ/zmq.REP` (ZMQ Request/Reply Pattern) in NetGear_Async API.

In Bidirectional Mode, we utilizes the NetGear_Async API's [`transceive_data`](../../../../bonus/reference/netgear_async/#vidgear.gears.asyncio.netgear_async.NetGear_Async.transceive_data) method for transmitting data _(at Client's end)_ and receiving data _(in Server's end)_  all while transferring frames in real-time. 

This mode can be easily activated in NetGear_Async through `bidirectional_mode` attribute of its [`options`](../../params/#options) dictionary parameter during initialization.

&nbsp;


!!! danger "Important"

    * In Bidirectional Mode, `zmq.PAIR`(ZMQ Pair) & `zmq.REQ/zmq.REP`(ZMQ Request/Reply) are **ONLY** Supported messaging patterns. Accessing this mode with any other messaging pattern, will result in `ValueError`.

    * Bidirectional Mode ==only works with [**User-defined Custom Source**](../../usage/#using-netgear_async-with-a-custom-sourceopencv) on Server end==. Otherwise, NetGear_Async API will throw `ValueError`.

    * Bidirectional Mode enables you to send data of **ANY**[^1] Data-type along with frame bidirectionally.

    * NetGear_Async API will throw `RuntimeError` if Bidirectional Mode is disabled at Server end or Client end but not both.

    * Bidirectional Mode may lead to additional **LATENCY** depending upon the size of data being transfer bidirectionally. User discretion is advised!


&nbsp;

&nbsp;

## Exclusive Method and Parameter

To send data bidirectionally, NetGear_Async API provides following exclusive method and parameter:

!!! alert "`transceive_data` only works when Bidirectional Mode is enabled."

* [`transceive_data`](../../../../bonus/reference/NetGear_Async/#vidgear.gears.asyncio.netgear_async.NetGear_Async.transceive_data): It's a bidirectional mode exclusive method to transmit data _(in Receive mode)_ and receive data _(in Send mode)_, all while transferring frames in real-time. 

    * `data`: In `transceive_data` method, this parameter enables user to inputs data _(of **ANY**[^1] datatype)_ for sending back to Server at Client's end. 

&nbsp;

&nbsp;


## Usage Examples


!!! warning "For Bidirectional Mode, NetGear_Async must need [User-defined Custom Source](../../usage/#using-netgear_async-with-a-custom-sourceopencv) at its Server end otherwise it will throw ValueError."


### Bare-Minimum Usage with OpenCV

Following is the bare-minimum code you need to get started with Bidirectional Mode over Custom Source Server built using OpenCV and NetGear_Async API:

#### Server End

Open your favorite terminal and execute the following python code:

!!! tip "You can terminate both sides anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="6 33 40 53"
# import library
from vidgear.gears.asyncio import NetGear_Async
import cv2, asyncio

# activate Bidirectional mode
options = {"bidirectional_mode": True}

# initialize Server without any source
server = NetGear_Async(source=None, logging=True, **options)

# Create a async frame generator as custom source
async def my_frame_generator():

    # !!! define your own video source here !!!
    # Open any valid video stream(for e.g `foo.mp4` file)
    stream = cv2.VideoCapture("foo.mp4")

    # loop over stream until its terminated
    while True:
        # read frames
        (grabbed, frame) = stream.read()

        # check for empty frame
        if not grabbed:
            break

        # {do something with the frame to be sent here}

        # prepare data to be sent(a simple text in our case)
        target_data = "Hello, I am a Server."

        # receive data from Client
        recv_data = await server.transceive_data()

        # print data just received from Client
        if not (recv_data is None):
            print(recv_data)

        # send our frame & data
        yield (target_data, frame) # (1)

        # sleep for sometime
        await asyncio.sleep(0)

    # safely close video stream
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

1.  :warning: Everything except [numpy.ndarray](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) datatype data is accepted in `target_data`.


#### Client End

Then open another terminal on the same system and execute the following python code and see the output:

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="6 15 31"
# import libraries
from vidgear.gears.asyncio import NetGear_Async
import cv2, asyncio

# activate Bidirectional mode
options = {"bidirectional_mode": True}

# define and launch Client with `receive_mode=True`
client = NetGear_Async(receive_mode=True, logging=True, **options).launch()


# Create a async function where you want to show/manipulate your received frames
async def main():
    # loop over Client's Asynchronous Frame Generator
    async for (data, frame) in client.recv_generator():

        # do something with receive data from server
        if not (data is None):
            # let's print it
            print(data)

        # {do something with received frames here}

        # Show output window(comment these lines if not required)
        cv2.imshow("Output Frame", frame)
        cv2.waitKey(1) & 0xFF

        # prepare data to be sent
        target_data = "Hi, I am a Client here."
        # send our data to server
        await client.transceive_data(data=target_data)

        # await before continuing
        await asyncio.sleep(0)


if __name__ == "__main__":
    # Set event loop to client's
    asyncio.set_event_loop(client.loop)
    try:
        # run your main function task until it is complete
        client.loop.run_until_complete(main())
    except (KeyboardInterrupt, SystemExit):
        # wait for interrupts
        pass

    # close all output window
    cv2.destroyAllWindows()

    # safely close client
    client.close()
```

&nbsp; 

&nbsp;


### Using Bidirectional Mode with Variable Parameters


#### Client's End

Open a terminal on Client System _(where you want to display the input frames received from the Server)_ and execute the following python code: 

!!! info "Note down the local IP-address of this system(required at Server's end) and also replace it in the following code. You can follow [this FAQ](../../../../help/netgear_faqs/#how-to-find-local-ip-address-on-different-os-platforms) for this purpose."

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="11-17"
# import libraries
from vidgear.gears.asyncio import NetGear_Async
import cv2, asyncio

# activate Bidirectional mode
options = {"bidirectional_mode": True}

# Define NetGear_Async Client at given IP address and define parameters 
# !!! change following IP address '192.168.x.xxx' with yours !!!
client = NetGear_Async(
    address="192.168.x.xxx",
    port="5454",
    protocol="tcp",
    pattern=1,
    receive_mode=True,
    logging=True,
    **options
)

# Create a async function where you want to show/manipulate your received frames
async def main():
    # loop over Client's Asynchronous Frame Generator
    async for (data, frame) in client.recv_generator():

        # do something with receive data from server
        if not (data is None):
            # let's print it
            print(data)

        # {do something with received frames here}

        # Show output window(comment these lines if not required)
        cv2.imshow("Output Frame", frame)
        cv2.waitKey(1) & 0xFF

        # prepare data to be sent
        target_data = "Hi, I am a Client here."
        # send our data to server
        await client.transceive_data(data=target_data)

        # await before continuing
        await asyncio.sleep(0)


if __name__ == "__main__":
    # Set event loop to client's
    asyncio.set_event_loop(client.loop)
    try:
        # run your main function task until it is complete
        client.loop.run_until_complete(main())
    except (KeyboardInterrupt, SystemExit):
        # wait for interrupts
        pass

    # close all output window
    cv2.destroyAllWindows()
    
    # safely close client
    client.close()
```

&nbsp;

#### Server End

Now, Open the terminal on another Server System _(a Raspberry Pi with Camera Module)_, and execute the following python code: 

!!! info "Replace the IP address in the following code with Client's IP address you noted earlier."

!!! tip "You can terminate stream on both side anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="12-18"
# import library
from vidgear.gears.asyncio import NetGear_Async
from vidgear.gears import VideoGear
import cv2, asyncio

# activate Bidirectional mode
options = {"bidirectional_mode": True}

# initialize Server without any source at given IP address and define parameters 
# !!! change following IP address '192.168.x.xxx' with client's IP address !!!
server = NetGear_Async(
    source=None,
    address="192.168.x.xxx",
    port="5454",
    protocol="tcp",
    pattern=1,
    logging=True,
    **options
)

# Create a async frame generator as custom source
async def my_frame_generator():

    # !!! define your own video source here !!!
    # Open any video stream such as live webcam
    # video stream on first index(i.e. 0) device
    # add various Picamera tweak parameters to dictionary
    options = {
        "hflip": True,
        "exposure_mode": "auto",
        "iso": 800,
        "exposure_compensation": 15,
        "awb_mode": "horizon",
        "sensor_mode": 0,
    }

    # open pi video stream with defined parameters
    stream = PiGear(resolution=(640, 480), framerate=60, logging=True, **options).start()

    # loop over stream until its terminated
    while True:
        # read frames
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame to be sent here}

        # prepare data to be sent(a simple text in our case)
        target_data = "Hello, I am a Server."

        # receive data from Client
        recv_data = await server.transceive_data()

        # print data just received from Client
        if not (recv_data is None):
            print(recv_data)

        # send our frame & data
        yield (target_data, frame) # (1)

        # sleep for sometime
        await asyncio.sleep(0)
        
    # safely close video stream
    stream.stop()


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

1.  :warning: Everything except [numpy.ndarray](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) datatype data is accepted in `target_data`.

&nbsp; 

&nbsp;


### Using Bidirectional Mode for Video-Frames Transfer


In this example we are going to implement a bare-minimum example, where we will be sending video-frames _(3-Dimensional numpy arrays)_ of the same Video bidirectionally at the same time, for testing the real-time performance and synchronization between the Server and the Client using this(Bidirectional) Mode. 

!!! tip "This feature is great for building applications like Real-Time Video Chat."

!!! info "We're also using [`reducer()`](../../../../bonus/reference/helper/#vidgear.gears.helper.reducer--reducer) method for reducing frame-size on-the-go for additional performance."

!!! warning "Remember, Sending large HQ video-frames may required more network bandwidth and packet size which may lead to video latency!"

#### Server End

Open your favorite terminal and execute the following python code:

!!! tip "You can terminate both side anytime by pressing ++ctrl+"C"++ on your keyboard!"

!!! alert "Server end can only send [numpy.ndarray](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) datatype as frame but not as data."

```python hl_lines="8 33-48 54 67"
# import library
from vidgear.gears.asyncio import NetGear_Async
from vidgear.gears.asyncio.helper import reducer
import cv2, asyncio
import numpy as np

# activate Bidirectional mode
options = {"bidirectional_mode": True}

# Define NetGear Server without any source and with defined parameters
server = NetGear_Async(source=None, pattern=1, logging=True, **options)

# Create a async frame generator as custom source
async def my_frame_generator():
    # !!! define your own video source here !!!
    # Open any valid video stream(for e.g `foo.mp4` file)
    stream = cv2.VideoCapture("foo.mp4")
    # loop over stream until its terminated
    while True:

        # read frames
        (grabbed, frame) = stream.read()

        # check for empty frame
        if not grabbed:
            break

        # reducer frames size if you want more performance, otherwise comment this line
        frame = await reducer(frame, percentage=30)  # reduce frame by 30%

        # {do something with the frame to be sent here}

        # send frame & data and also receive data from Client
        recv_data = await server.transceive_data()

        # receive data from Client
        if not (recv_data is None):
            # check data is a numpy frame
            if isinstance(recv_data, np.ndarray):

                # {do something with received numpy frame here}

                # Let's show it on output window
                cv2.imshow("Received Frame", recv_data)
                cv2.waitKey(1) & 0xFF
            else:
                # otherwise just print data
                print(recv_data)

        # prepare data to be sent(a simple text in our case)
        target_data = "Hello, I am a Server."

        # send our frame & data to client
        yield (target_data, frame) # (1)

        # sleep for sometime
        await asyncio.sleep(0)

    # safely close video stream
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


1.  :warning: Everything except [numpy.ndarray](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) datatype data is accepted in `target_data`.

&nbsp;

#### Client End

Then open another terminal on the same system and execute the following python code and see the output:

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="7 18 34-43"
# import libraries
from vidgear.gears.asyncio import NetGear_Async
from vidgear.gears.asyncio.helper import reducer
import cv2, asyncio

# activate Bidirectional mode
options = {"bidirectional_mode": True}

# define and launch Client with `receive_mode=True`
client = NetGear_Async(pattern=1, receive_mode=True, logging=True, **options).launch()

# Create a async function where you want to show/manipulate your received frames
async def main():
    # !!! define your own video source here !!!
    # again open the same video stream for comparison
    stream = cv2.VideoCapture("foo.mp4")
    # loop over Client's Asynchronous Frame Generator
    async for (server_data, frame) in client.recv_generator():

        # check for server data
        if not (server_data is None):

            # {do something with the server data here}

            # lets print extracted server data
            print(server_data)

        # {do something with received frames here}

        # Show output window
        cv2.imshow("Output Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # read frame target data from stream to be sent to server
        (grabbed, target_data) = stream.read()
        # check for frame
        if grabbed:
            # reducer frames size if you want more performance, otherwise comment this line
            target_data = await reducer(
                target_data, percentage=30
            )  # reduce frame by 30%
            # send our frame data
            await client.transceive_data(data=target_data)

        # await before continuing
        await asyncio.sleep(0)

    # safely close video stream
    stream.release()


if __name__ == "__main__":
    # Set event loop to client's
    asyncio.set_event_loop(client.loop)
    try:
        # run your main function task until it is complete
        client.loop.run_until_complete(main())
    except (KeyboardInterrupt, SystemExit):
        # wait for interrupts
        pass
    # close all output window
    cv2.destroyAllWindows()
    # safely close client
    client.close()
```

&nbsp;

[^1]: 
    
    !!! warning "Additional data of [numpy.ndarray](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) datatype is **ONLY SUPPORTED** at Client's end with [`transceive_data`](../../../../bonus/reference/NetGear_Async/#vidgear.gears.asyncio.netgear_async.NetGear_Async.transceive_data) method using its `data` parameter. Whereas Server end can only send [numpy.ndarray](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) datatype as frame but not as data."


&nbsp;