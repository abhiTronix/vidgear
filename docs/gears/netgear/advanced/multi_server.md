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

# Multi-Servers Mode for NetGear API 

<figure>
  <img src="../../../../assets/images/multi_server.png" alt="NetGear's Multi-Servers Mode" loading="lazy"/>
  <figcaption>NetGear's Multi-Servers Mode</figcaption>
</figure>

## Overview

In Multi-Servers Mode, NetGear API robustly handles Multiple Servers at once, thereby providing seamless access to frames and unidirectional data transfer across multiple Publishers/Servers in the network at the same time. Each new server connects to a single client can be identified by its unique port address on the network. 

The supported patterns for this mode are Publish/Subscribe (`zmq.PUB/zmq.SUB`) and Request/Reply(`zmq.REQ/zmq.REP`) and can be easily activated in NetGear API through `multiserver_mode` attribute of its [`options`](../../params/#options) dictionary parameter during initialization.


&nbsp;


!!! danger "Important Information regarding Multi-Servers Mode"

    * A unique PORT address **MUST** be assigned to each Server on the network using its [`port`](../../params/#port) parameter.
    
    * A list/tuple of PORT addresses of all unique Servers **MUST** be assigned at Client's end using its [`port`](../../params/#port) parameter for a successful connection.

    * Patterns `1` _(i.e. Request/Reply `zmq.REQ/zmq.REP`)_ and `2` _(i.e. Publish/Subscribe `zmq.PUB/zmq.SUB`)_ are the only supported values for this Mode. Therefore, calling any other pattern value with is mode will result in `ValueError`.

    * Multi-Servers and Multi-Clients exclusive modes **CANNOT** be enabled simultaneously, Otherwise NetGear API will throw `ValueError`.

    * The [`address`](../../params/#address) parameter value of each Server **MUST** exactly match the Client. 

&nbsp;

## Key Features

- [x] Enables Multiple Server(s) connection with a single Client.

- [x] Ability to [send any additional data](../../advanced/multi_server/#using-multi-servers-mode-with-custom-data-transfer) of any[^1] datatype along with frames in real-time.

- [x] Number of Servers can be extended to several numbers depending upon your system's hardware limit.

- [x] Employs powerful **Publish/Subscribe & Request/Reply** messaging patterns.

- [x] Each new Server on the network can be identified at Client's end by their unique port addresses.

- [x] NetGear API actively tracks the state of each connected Server.

- [x] If all the connected servers on the network get disconnected, the client itself automatically exits to save resources.

&nbsp;

## Usage Examples


!!! alert "Example Assumptions"

    * For sake of simplicity, in these examples we will use only two unique Servers, but, the number of these Servers can be extended to several numbers depending upon your system hardware limits.

    * All of Servers will be transferring frames to a single Client system at the same time, which will be displaying received frames as a live montage _(multiple frames concatenated together)_.

    * For building Frames Montage at Client's end, We are going to use `imutils` python library function to build montages, by concatenating  together frames received from different servers. Therefore, Kindly install this library with `pip install imutils` terminal command.


&nbsp;


### Bare-Minimum Usage

In this example, we will capturing live video-frames on two independent sources _(a.k.a Servers)_, each with a webcam connected to it. Afterwards, these frames will be sent over the network to a single system _(a.k.a Client)_ using this Multi-Servers Mode in NetGear API in real time, and will be displayed as a live montage.


!!! tip "This example is useful for building applications like Real-Time Security System with multiple cameras."


#### Client's End

Open a terminal on Client System _(where you want to display the input frames received from Multiple Servers)_ and execute the following python code: 

!!! info "Important Notes"

    * Note down the local IP-address of this system(required at all Server(s) end) and also replace it in the following code. You can follow [this FAQ](../../../../help/netgear_faqs/#how-to-find-local-ip-address-on-different-os-platforms) for this purpose.
    * Also, assign the tuple/list of port address of all Servers you are going to connect to this system. 

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="7 14 40-52"
# import required libraries
from vidgear.gears import NetGear
from imutils import build_montages # (1)
import cv2

# activate multiserver_mode
options = {"multiserver_mode": True}

# Define NetGear Client at given IP address and assign list/tuple 
# of all unique Server((5566,5567) in our case) and other parameters
# !!! change following IP address '192.168.x.xxx' with yours !!!
client = NetGear(
    address="192.168.x.x",
    port=(5566, 5567),
    protocol="tcp",
    pattern=1,
    receive_mode=True,
    **options
)

# Define received frame dictionary
frame_dict = {}

# loop over until Keyboard Interrupted
while True:

    try:
        # receive data from network
        data = client.recv()

        # check if data received isn't None
        if data is None:
            break

        # extract unique port address and its respective frame
        unique_address, frame = data

        # {do something with the extracted frame here}

        # get extracted frame's shape
        (h, w) = frame.shape[:2]

        # update the extracted frame in the received frame dictionary
        frame_dict[unique_address] = frame

        # build a montage using data dictionary
        montages = build_montages(frame_dict.values(), (w, h), (2, 1))

        # display the montage(s) on the screen
        for (i, montage) in enumerate(montages):

            cv2.imshow("Montage Footage {}".format(i), montage)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    except KeyboardInterrupt:
        break

# close output window
cv2.destroyAllWindows()

# safely close client
client.close()
```

1.  For building Frames Montage you'll need `imutils` python library. Install it with `pip install imutils` command.

&nbsp;

#### Server-1's End


Now, Open the terminal on another Server System _(with a webcam connected to it at index `0`)_, and let's called it Server-1. Now execute the following python code: 

!!! info "Replace the IP address in the following code with Client's IP address you noted earlier and also assign a unique port address _(required by Client to identify this system)_."

!!! tip "You can terminate stream anytime by pressing ++ctrl+"C"++ on your keyboard!"


```python hl_lines="9 14"
# import libraries
from vidgear.gears import NetGear
from vidgear.gears import CamGear

# Open suitable video stream (webcam on first index in our case)
stream = CamGear(source=0).start()

# activate multiserver_mode
options = {"multiserver_mode": True}

# Define NetGear Server at Client's IP address and assign a unique port address and other parameters
# !!! change following IP address '192.168.x.xxx' with yours !!!
server = NetGear(
    address="192.168.x.x", port="5566", protocol="tcp", pattern=1, **options
)

# loop over until Keyboard Interrupted
while True:

    try:
        # read frames from stream
        frame = stream.read()

        # check for frame if not None-type
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

#### Server-2's End

Finally, Open the terminal on another Server System _(also with a webcam connected to it at index `0`)_, and let's called it Server-2. Now execute the following python code: 

!!! info "Replace the IP address in the following code with Client's IP address you noted earlier and also assign a unique port address _(required by Client to identify this system)_."

!!! tip "You can terminate stream anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="9 14"
# import libraries
from vidgear.gears import NetGear
from vidgear.gears import CamGear

# Open suitable video stream (webcam on first index in our case)
stream = CamGear(source=0).start()

# activate multiserver_mode
options = {"multiserver_mode": True}

# Define NetGear Server at Client's IP address and assign a unique port address and other parameters
# !!! change following IP address '192.168.x.xxx' with yours !!!
server = NetGear(
    address="192.168.x.x", port="5567", protocol="tcp", pattern=1, **options
)

# loop over until Keyboard Interrupted
while True:

    try:
        # read frames from stream
        frame = stream.read()

        # check for frame if not None-type
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

&nbsp;

### Bare-Minimum Usage with OpenCV

In this example, we will be re-implementing previous bare-minimum example with OpenCV and NetGear API.


#### Client's End

Open a terminal on Client System _(where you want to display the input frames received from Mutiple Servers)_ and execute the following python code: 

!!! info "Important Notes"

    * Note down the local IP-address of this system(required at all Server(s) end) and also replace it in the following code. You can follow [this FAQ](../../../../help/netgear_faqs/#how-to-find-local-ip-address-on-different-os-platforms) for this purpose.
    * Also, assign the tuple/list of port address of all Servers you are going to connect to this system. 

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python
# import required libraries
from vidgear.gears import NetGear
from imutils import build_montages # (1)
import cv2

# activate multiserver_mode
options = {"multiserver_mode": True}

# Define NetGear Client at given IP address and assign list/tuple of all 
# unique Server((5566,5567) in our case) and other parameters
# !!! change following IP address '192.168.x.xxx' with yours !!!
client = NetGear(
    address="192.168.x.x",
    port=(5566, 5567),
    protocol="tcp",
    pattern=2,
    receive_mode=True,
    **options
)

# Define received frame dictionary
frame_dict = {}

# loop over until Keyboard Interrupted
while True:

    try:
        # receive data from network
        data = client.recv()

        # check if data received isn't None
        if data is None:
            break

        # extract unique port address and its respective frame
        unique_address, frame = data

        # {do something with the extracted frame here}

        # get extracted frame's shape
        (h, w) = frame.shape[:2]

        # update the extracted frame in the received frame dictionary
        frame_dict[unique_address] = frame

        # build a montage using data dictionary
        montages = build_montages(frame_dict.values(), (w, h), (2, 1))

        # display the montage(s) on the screen
        for (i, montage) in enumerate(montages):

            cv2.imshow("Montage Footage {}".format(i), montage)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    except KeyboardInterrupt:
        break

# close output window
cv2.destroyAllWindows()

# safely close client
client.close()
```

1.  For building Frames Montage you'll need `imutils` python library. Install it with `pip install imutils` command.

&nbsp;

#### Server-1's End

Now, Open the terminal on another Server System _(with a webcam connected to it at index `0`)_, and let's called it Server-1. Now execute the following python code: 

!!! info "Replace the IP address in the following code with Client's IP address you noted earlier and also assign a unique port address _(required by Client to identify this system)_."

!!! tip "You can terminate stream anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python
# import libraries
from vidgear.gears import NetGear
import cv2

# Open suitable video stream (webcam on first index in our case)
stream = cv2.VideoCapture(0)

# activate multiserver_mode
options = {"multiserver_mode": True}

# Define NetGear Server at Client's IP address and assign a unique port address and other parameter
# !!! change following IP address '192.168.x.xxx' with yours !!!
server = NetGear(
    address="192.168.x.x", port="5566", protocol="tcp", pattern=2, **options
)

# loop over until Keyboard Interrupted
while True:

    try:
        # read frames from stream
        (grabbed, frame) = stream.read()

        # check for frame if not grabbed
        if not grabbed:
            break

        # {do something with the frame here}

        # send frame to server
        server.send(frame)

    except KeyboardInterrupt:
        break

# safely close video stream
stream.release()

# safely close server
server.close()
```

&nbsp;

#### Server-2's End

Finally, Open the terminal on another Server System _(also with a webcam connected to it at index `0`)_, and let's called it Server-2. Now execute the following python code: 

!!! info "Replace the IP address in the following code with Client's IP address you noted earlier and also assign a unique port address _(required by Client to identify this system)_."

!!! tip "You can terminate stream anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python
# import libraries
from vidgear.gears import NetGear
import cv2

# Open suitable video stream (webcam on first index in our case)
stream = cv2.VideoCapture(0)

# activate multiserver_mode
options = {"multiserver_mode": True}

# Define NetGear Server at Client's IP address and assign a unique port address and other parameters
# !!! change following IP address '192.168.x.xxx' with yours !!!
server = NetGear(
    address="192.168.x.x", port="5567", protocol="tcp", pattern=2, **options
)

# loop over until Keyboard Interrupted
while True:
    
    try:
        # read frames from stream
        (grabbed, frame) = stream.read()

        # check for frame if not grabbed
        if not grabbed:
            break

        # {do something with the frame here}

        # send frame to server
        server.send(frame)

    except KeyboardInterrupt:
        break

# safely close video stream
stream.release()

# safely close server
server.close()
```

&nbsp;

&nbsp;


### Using Multi-Servers Mode for Unidirectional Custom Data Transfer

!!! abstract
    With Multi-Servers Mode, you can send additional data of *any datatype*[^1] along with frame with frame in real-time, from all connected Server(s) to a single Client unidirectionally.

    !!! warning "But [`numpy.ndarray`](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) data-type is **NOT** supported as data."


In this example, We will be transferring video-frames and data _(a Text String, for the sake of simplicity)_ from two Servers _(consisting of a Raspberry Pi with Camera Module & a Laptop with webcam)_ to a single Client over the network in real-time. The received video-frames at Client's end will displayed as a live montage, whereas the received data will be printed to the terminal.

#### Client's End

Open a terminal on Client System _(where you want to display the input frames received from Mutiple Servers)_ and execute the following python code: 

!!! info "Important Notes"

    * Note down the local IP-address of this system(required at all Server(s) end) and also replace it in the following code. You can follow [this FAQ](../../../../help/netgear_faqs/#how-to-find-local-ip-address-on-different-os-platforms) for this purpose.
    * Also, assign the tuple/list of port address of all Servers you are going to connect to this system. 

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="38-61"
# import required libraries
from vidgear.gears import NetGear
from imutils import build_montages # (1)
import cv2

# activate multiserver_mode
options = {"multiserver_mode": True}

# Define NetGear Client at given IP address and assign list/tuple of all unique Server((5577,5578) in our case) and other parameters
# !!! change following IP address '192.168.x.xxx' with yours !!!
client = NetGear(
    address="192.168.x.x",
    port=(5577, 5578),
    protocol="tcp",
    pattern=1,
    receive_mode=True,
    logging=True,
    **options
)  
# Define received frame dictionary
frame_dict = {}

# loop over until Keyboard Interrupted
while True:

    try:
        # receive data from network
        data = client.recv()

        # check if data received isn't None
        if data is None:
            break

        # extract unique port address and its respective frame and received data
        unique_address, extracted_data, frame = data

        # {do something with the extracted frame and data here}
        # let's display extracted data on our extracted frame
        cv2.putText(
            frame,
            extracted_data,
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # get extracted frame's shape
        (h, w) = frame.shape[:2]

        # update the extracted frame in the frame dictionary
        frame_dict[unique_address] = frame

        # build a montage using data dictionary
        montages = build_montages(frame_dict.values(), (w, h), (2, 1))

        # display the montage(s) on the screen
        for (i, montage) in enumerate(montages):

            cv2.imshow("Montage Footage {}".format(i), montage)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    except KeyboardInterrupt:
        break

# close output window
cv2.destroyAllWindows()

# safely close client
client.close()
```

1.  For building Frames Montage you'll need `imutils` python library. Install it with `pip install imutils` command.


&nbsp;


#### Server-1's End

Now, Open the terminal on another Server System _(with a webcam connected to it at index `0`)_, and let's called it Server-1. Now execute the following python code: 

!!! info "Replace the IP address in the following code with Client's IP address you noted earlier and also assign a unique port address _(required by Client to identify this system)_."

!!! tip "You can terminate stream anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="40"
# import libraries
from vidgear.gears import NetGear
from vidgear.gears import VideoGear
import cv2

# Open suitable video stream (webcam on first index in our case)
stream = VideoGear(source=0).start()

# activate multiserver_mode
options = {"multiserver_mode": True}

# Define NetGear Server at Client's IP address and assign a unique port address and other parameters
# !!! change following IP address '192.168.x.xxx' with yours !!!
server = NetGear(
    address="192.168.x.x",
    port="5577",
    protocol="tcp",
    pattern=1,
    logging=True,
    **options
)

# loop over until Keyboard Interrupted
while True:

    try:
        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with frame and data(to be sent) here}

        # let's prepare a text string as data
        target_data = "I'm Server-1 at Port: 5577"

        # send frame and data through server
        server.send(frame, message=target_data) # (1)

    except KeyboardInterrupt:
        break

# safely close video stream
stream.stop()

# safely close server
server.close()
```

1.  :warning: Everything except [numpy.ndarray](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) datatype data is accepted as `target_data` in `message` parameter.


&nbsp;

#### Server-2's End

Finally, Open the terminal on another Server System _(this time a Raspberry Pi with Camera Module connected to it)_, and let's called it Server-2. Now execute the following python code: 

!!! info "Replace the IP address in the following code with Client's IP address you noted earlier and also assign a unique port address _(required by Client to identify this system)_."

!!! tip "You can terminate stream anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="50"
# import libraries
from vidgear.gears import NetGear
from vidgear.gears import PiGear
import cv2

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

# activate multiserver_mode
options = {"multiserver_mode": True}

# Define NetGear Server at Client's IP address and assign a unique port address and other parameters
# !!! change following IP address '192.168.x.xxx' with yours !!!
server = NetGear(
    address="192.168.1.xxx",
    port="5578",
    protocol="tcp",
    pattern=1,
    logging=True,
    **options
)

# loop over until Keyboard Interrupted
while True:

    try:
        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with frame and data(to be sent) here}

        # let's prepare a text string as data
        text = "I'm Server-2 at Port: 5578"

        # send frame and data through server
        server.send(frame, message=text)

    except KeyboardInterrupt:
        break

# safely close video stream.
stream.stop()

# safely close server
server.close()
```

&nbsp;

&nbsp;


### Using Multi-Servers Mode with Bidirectional Mode

!!! abstract
    Multi-Servers Mode now also compatible with [Bidirectional Mode](../../advanced/bidirectional_mode/), which lets you send additional data of ***any datatype***[^1]  along with frame in real-time bidirectionally between a single Client and all connected Server(s).

!!! warning "Important Information"
    * Bidirectional data transfer **ONLY** works with pattern `1` _(i.e. Request/Reply `zmq.REQ/zmq.REP`)_, and **NOT** with pattern `2` _(i.e. Publish/Subscribe `zmq.PUB/zmq.SUB`)_
    * Additional data of [numpy.ndarray](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) data-type is **NOT SUPPORTED** at Server(s) with their [`message`](../../../../bonus/reference/netgear/#vidgear.gears.netgear.NetGear.send) parameter.
    * Bidirectional Mode may lead to additional **LATENCY** depending upon the size of data being transfer bidirectionally. User discretion is advised!

??? new "New in v0.2.5" 
    This example was added in `v0.2.5`.

In this example, We will be transferring video-frames and data _(a Text String, for the sake of simplicity)_ from two Servers _(consisting of a Raspberry Pi with Camera Module & a Laptop with webcam)_ to a single Client, and at same time sending back data _(a Text String, for the sake of simplicity)_ to them over the network all in real-time. The received video-frames at Client's end will displayed as a live montage, whereas the received data will be printed to the terminal.

#### Client's End

Open a terminal on Client System _(where you want to display the input frames received from Mutiple Servers)_ and execute the following python code: 

!!! info "Important Notes"

    * Note down the local IP-address of this system(required at all Server(s) end) and also replace it in the following code. You can follow [this FAQ](../../../../help/netgear_faqs/#how-to-find-local-ip-address-on-different-os-platforms) for this purpose.
    * Also, assign the tuple/list of port address of all Servers you are going to connect to this system. 

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="7 27-31 38 41-64"
# import required libraries
from vidgear.gears import NetGear
from imutils import build_montages # (1)
import cv2

# activate both multiserver and bidirectional modes
options = {"multiserver_mode": True, "bidirectional_mode": True}

# Define NetGear Client at given IP address and assign list/tuple of all unique Server((5577,5578) in our case) and other parameters
# !!! change following IP address '192.168.x.xxx' with yours !!!
client = NetGear(
    address="192.168.x.x",
    port=(5577, 5578),
    protocol="tcp",
    pattern=1,
    receive_mode=True,
    logging=True,
    **options
)  
# Define received frame dictionary
frame_dict = {}

# loop over until Keyboard Interrupted
while True:

    try:
        # prepare data to be sent
        target_data = "Hi, I am a Client here."

        # receive data from server(s) and also send our data
        data = client.recv(return_data=target_data)

        # check if data received isn't None
        if data is None:
            break

        # extract unique port address and its respective frame and received data
        unique_address, extracted_data, frame = recv_data

        # {do something with the extracted frame and data here}
        # let's display extracted data on our extracted frame
        cv2.putText(
            frame,
            extracted_data,
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # get extracted frame's shape
        (h, w) = frame.shape[:2]

        # update the extracted frame in the frame dictionary
        frame_dict[unique_address] = frame

        # build a montage using data dictionary
        montages = build_montages(frame_dict.values(), (w, h), (2, 1))

        # display the montage(s) on the screen
        for (i, montage) in enumerate(montages):

            cv2.imshow("Montage Footage {}".format(i), montage)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    except KeyboardInterrupt:
        break

# close output window
cv2.destroyAllWindows()

# safely close client
client.close()
```

1.  For building Frames Montage you'll need `imutils` python library. Install it with `pip install imutils` command.

&nbsp;


#### Server-1's End

Now, Open the terminal on another Server System _(with a webcam connected to it at index `0`)_, and let's called it Server-1. Now execute the following python code: 

!!! info "Replace the IP address in the following code with Client's IP address you noted earlier and also assign a unique port address _(required by Client to identify this system)_."

!!! tip "You can terminate stream anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="10 36-44"
# import libraries
from vidgear.gears import NetGear
from vidgear.gears import VideoGear
import cv2

# Open suitable video stream (webcam on first index in our case)
stream = VideoGear(source=0).start()

# activate both multiserver and bidirectional modes
options = {"multiserver_mode": True, "bidirectional_mode": True}

# Define NetGear Server at Client's IP address and assign a unique port address and other parameters
# !!! change following IP address '192.168.x.xxx' with yours !!!
server = NetGear(
    address="192.168.x.x",
    port="5577",
    protocol="tcp",
    pattern=1,
    logging=True,
    **options
)

# loop over until Keyboard Interrupted
while True:

    try:
        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with frame and data(to be sent) here}

        # let's prepare a text string as data
        target_data = "I'm Server-1 at Port: 5577"

        # send frame & data and also receive data from Client
        recv_data = server.send(frame, message=target_data) # (1)

        # print data just received from Client
        if not (recv_data is None):
            print(recv_data)

    except KeyboardInterrupt:
        break

# safely close video stream
stream.stop()

# safely close server
server.close()
```

1.  :warning: Everything except [numpy.ndarray](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) datatype data is accepted as `target_data` in `message` parameter.

&nbsp;

#### Server-2's End

Finally, Open the terminal on another Server System _(this time a Raspberry Pi with Camera Module connected to it)_, and let's called it Server-2. Now execute the following python code: 

!!! info "Replace the IP address in the following code with Client's IP address you noted earlier and also assign a unique port address _(required by Client to identify this system)_."

!!! tip "You can terminate stream anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="20 46-54"
# import libraries
from vidgear.gears import NetGear
from vidgear.gears import PiGear
import cv2

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

# activate both multiserver and bidirectional modes
options = {"multiserver_mode": True, "bidirectional_mode": True}

# Define NetGear Server at Client's IP address and assign a unique port address and other parameters
# !!! change following IP address '192.168.x.xxx' with yours !!!
server = NetGear(
    address="192.168.1.xxx",
    port="5578",
    protocol="tcp",
    pattern=1,
    logging=True,
    **options
)

# loop over until Keyboard Interrupted
while True:

    try:
        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with frame and data(to be sent) here}

        # let's prepare a text string as data
        target_data = "I'm Server-2 at Port: 5578"

        # send frame & data and also receive data from Client
        recv_data = server.send(frame, message=target_data) # (1)

        # print data just received from Client
        if not (recv_data is None):
            print(recv_data)

    except KeyboardInterrupt:
        break

# safely close video stream.
stream.stop()

# safely close server
server.close()
```

1.  :warning: Everything except [numpy.ndarray](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) datatype data is accepted as `target_data` in `message` parameter.

&nbsp;

[^1]: 
    
    !!! warning "Additional data of [numpy.ndarray](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) data-type is **NOT SUPPORTED** at Server(s) with their [`message`](../../../../bonus/reference/netgear/#vidgear.gears.netgear.NetGear.send) parameter."
