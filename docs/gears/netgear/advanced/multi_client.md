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

# Multi-Clients Mode for NetGear API 

<figure>
  <img src="../../../../assets/images/multi_client.png" alt="NetGear's Multi-Clients Mode" loading="lazy"/>
  <figcaption>NetGear's Multi-Clients Mode</figcaption>
</figure>

## Overview

In Multi-Clients Mode, NetGear robustly handles Multiple Clients at once thereby able to broadcast frames and data across multiple Clients/Consumers in the network at same time. This mode works contrary to [Multi-Servers Mode](../multi_server/) such that every new Client that connects to single Server can be identified by its unique port address on the network. 

The supported patterns for this mode are Publish/Subscribe (`zmq.PUB/zmq.SUB`) and Request/Reply(`zmq.REQ/zmq.REP`) and can be easily activated in NetGear API through `multiclient_mode` attribute of its [`options`](../../params/#options) dictionary parameter during initialization.


!!! alert "Multi-Clients Mode is best for broadcasting **Meta-Data with Video-frames** to specific limited number of clients in real time. But if you're looking to scale broadcast to a very large pool of clients, then see our [WebGear](../../../webgear/overview/) or [WebGear_RTC](../../../webgear_rtc/overview/) APIs."

&nbsp;


!!! danger "Important Information regarding Multi-Clients Mode"

    * A unique PORT address **MUST** be assigned to each Client on the network using its [`port`](../../params/#port) parameter.
    
    * A list/tuple of PORT addresses of all unique Clients **MUST** be assigned at Server's end using its [`port`](../../params/#port) parameter for a successful connection.

    * Patterns `1` _(i.e. Request/Reply `zmq.REQ/zmq.REP`)_ and `2` _(i.e. Publish/Subscribe `zmq.PUB/zmq.SUB`)_ are the only supported pattern values for this Mode. Therefore, calling any other pattern value with is mode will result in `ValueError`.

    * Multi-Clients and Multi-Servers exclusive modes **CANNOT** be enabled simultaneously, Otherwise NetGear API will throw `ValueError`.

    * The [`address`](../../params/#address) parameter value of each Client **MUST** exactly match the Server. 

&nbsp;

## Features of Multi-Clients Mode

- [x] Enables Multiple Client(s) connection with a single Server.

- [x] Ability to [send any additional data](../../advanced/multi_client/#using-multi-clients-mode-with-custom-data-transfer) of any datatype along with frames in real-time.

- [x] Number of Clients can be extended to several numbers depending upon your system's hardware limit.

- [x] Employs powerful **Publish/Subscribe & Request/Reply** messaging patterns.

- [x] Each new Client on the network can be identified at Server's end by their unique port addresses.

- [x] NetGear API actively tracks the state of each connected Client.

- [x] If the server gets disconnected, all the clients will automatically exit to save resources.


&nbsp;

## Usage Examples


!!! alert "Important"

    * ==Frame/Data transmission will **NOT START** until all given Client(s) are connected to the Server.==

    * For sake of simplicity, in these examples we will use only two unique Clients, but the number of these Clients can be extended to **SEVERAL** numbers depending upon your Network bandwidth and System Capabilities.


&nbsp;


### Bare-Minimum Usage

In this example, we will capturing live video-frames from a source _(a.k.a Server)_ with a webcam connected to it. Afterwards, those captured frame will be sent over the network to two independent system _(a.k.a Clients)_ using this Multi-Clients Mode in NetGear API. Finally, both Clients will be displaying received frames in Output Windows in real time.

!!! tip "This example is useful for building applications like Real-Time Video Broadcasting to multiple clients in local network."

#### Server's End

Now, Open the terminal on a Server System _(with a webcam connected to it at index `0`)_. Now execute the following python code: 

!!! info "Important Notes"

    * Note down the local IP-address of this system(required at all Client(s) end) and also replace it in the following code. You can follow [this FAQ](../../../../help/netgear_faqs/#how-to-find-local-ip-address-on-different-os-platforms) for this purpose.
    * Also, assign the tuple/list of port address of all Client you are going to connect to this system. 

!!! warning "Frame/Data transmission will **NOT START** untill all given Client(s) are connected to this Server."

!!! tip "You can terminate streaming anytime by pressing ++ctrl+"C"++ on your keyboard!"


```python hl_lines="9 16 39-52"
# import required libraries
from vidgear.gears import NetGear
from vidgear.gears import CamGear

# Open suitable video stream (webcam on first index in our case)
stream = CamGear(source=0).start()

# activate multiclient_mode mode
options = {"multiclient_mode": True}

# Define NetGear Client at given IP address and assign list/tuple of
# all unique Server((5577,5578) in our case) and other parameters
# !!! change following IP address '192.168.x.xxx' with yours !!!
server = NetGear(
    address="192.168.x.x",
    port=(5567, 5577),
    protocol="tcp",
    pattern=1,
    logging=True,
    **options
)

# Define received data dictionary
data_dict = {}

# loop over until KeyBoard Interrupted
while True:

    try:
        # read frames from stream
        frame = stream.read()

        # check for frame if not None-type
        if frame is None:
            break

        # {do something with the frame here}

        # send frame and also receive data from Client(s)
        recv_data = server.send(frame)

        # check if valid data received
        if not (recv_data is None):
            # extract unique port address and its respective data
            unique_address, data = recv_data
            # update the extracted data in the data dictionary
            data_dict[unique_address] = data

        if data_dict:
            # print data just received from Client(s)
            for key, value in data_dict.items():
                print("Client at port {} said: {}".format(key, value))

    except KeyboardInterrupt:
        break

# safely close video stream
stream.stop()
# safely close server
server.close()
```

&nbsp;

#### Client-1's End

Now, Open a terminal on another Client System _(where you want to display the input frames received from Server)_, let's name it Client-1. Execute the following python code: 

!!! info "Replace the IP address in the following code with Server's IP address you noted earlier and also assign a unique port address _(required by Server to identify this system)_."

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="6 11-17"
# import required libraries
from vidgear.gears import NetGear
import cv2

# activate Multi-Clients mode
options = {"multiclient_mode": True}

# Define NetGear Client at Server's IP address and assign a unique port address and other parameters
# !!! change following IP address '192.168.x.xxx' with yours !!!
client = NetGear(
    address="192.168.x.x",
    port="5567",
    protocol="tcp",
    pattern=1,
    receive_mode=True,
    logging=True,
    **options
) 

# loop over
while True:
    # receive data from server
    frame = client.recv()

    # check for frame if None
    if frame is None:
        break

    # {do something with frame here}

    # Show output window
    cv2.imshow("Client 5567 Output", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close client
client.close()
```

&nbsp;


#### Client-2's End

Finally, Open a terminal on another Client System _(where you want to display the input frames received from Server)_, let's name it Client-2. Execute the following python code: 

!!! info "Replace the IP address in the following code with Server's IP address you noted earlier and also assign a unique port address _(required by Server to identify this system)_."

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="6  11-17"
# import required libraries
from vidgear.gears import NetGear
import cv2

# activate Multi-Clients mode
options = {"multiclient_mode": True}

# Define NetGear Client at Server's IP address and assign a unique port address and other parameters
 # !!! change following IP address '192.168.x.xxx' with yours !!!
client = NetGear(
    address="192.168.x.x",
    port="5577",
    protocol="tcp",
    pattern=1,
    receive_mode=True,
    logging=True,
    **options
)

# loop over
while True:

    # receive data from server
    frame = client.recv()

    # check for frame if None
    if frame is None:
        break

    # {do something with frame here}

    # Show output window
    cv2.imshow("Client 5577 Output", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close client
client.close()
```

&nbsp;

&nbsp;

### Bare-Minimum Usage with OpenCV

In this example, we will be re-implementing previous bare-minimum example with OpenCV and NetGear API.


#### Server's End

Now, Open the terminal on a Server System _(with a webcam connected to it at index `0`)_. Now execute the following python code: 

!!! info "Important Notes"

    * Note down the local IP-address of this system(required at all Client(s) end) and also replace it in the following code. You can follow [this FAQ](../../../../help/netgear_faqs/#how-to-find-local-ip-address-on-different-os-platforms) for this purpose.
    * Also, assign the tuple/list of port address of all Client you are going to connect to this system. 

!!! warning "Frame/Data transmission will **NOT START** untill all given Client(s) are connected to this Server."

!!! tip "You can terminate streaming anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python
# import required libraries
from vidgear.gears import NetGear
import cv2

# Open suitable video stream (webcam on first index in our case)
stream = cv2.VideoCapture(0)

# activate multiclient_mode mode
options = {"multiclient_mode": True}

# Define NetGear Client at given IP address and assign list/tuple of all unique Server((5577,5578) in our case) and other parameters
# !!! change following IP address '192.168.x.xxx' with yours !!!
server = NetGear(
    address="192.168.x.x",
    port=(5567, 5577),
    protocol="tcp",
    pattern=2,
    logging=True,
    **options
)

# Define received data dictionary
data_dict = {}

# loop over until KeyBoard Interrupted
while True:

    try:
        # read frames from stream
        (grabbed, frame) = stream.read()

        # check for frame if not grabbed
        if not grabbed:
            break

        # {do something with the frame here}

        # send frame and also receive data from Client(s)
        recv_data = server.send(frame)

        # check if valid data received
        if not (recv_data is None):
            # extract unique port address and its respective data
            unique_address, data = recv_data
            # update the extracted data in the data dictionary
            data_dict[unique_address] = data

        if data_dict:
            # print data just received from Client(s)
            for key, value in data_dict.items():
                print("Client at port {} said: {}".format(key, value))

    except KeyboardInterrupt:
        break

# safely close video stream
stream.release()
# safely close server
server.close()
```

&nbsp;

#### Client-1's End

Now, Open a terminal on another Client System _(where you want to display the input frames received from Server)_, let's name it Client-1. Execute the following python code: 

!!! info "Replace the IP address in the following code with Server's IP address you noted earlier and also assign a unique port address _(required by Server to identify this system)_."

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python
# import required libraries
from vidgear.gears import NetGear
import cv2

# activate Multi-Clients mode
options = {"multiclient_mode": True}

# Define NetGear Client at Server's IP address and assign a unique port address and other parameters
# !!! change following IP address '192.168.x.xxx' with yours !!!
client = NetGear(
    address="192.168.x.x",
    port="5567",
    protocol="tcp",
    pattern=2,
    receive_mode=True,
    logging=True,
    **options
) 

# loop over
while True:
    # receive data from server
    frame = client.recv()

    # check for frame if None
    if frame is None:
        break

    # {do something with frame here}

    # Show output window
    cv2.imshow("Client 5567 Output", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close client
client.close()
```

&nbsp;

#### Client-2's End

Finally, Open a terminal on another Client System _(also, where you want to display the input frames received from Server)_, let's name it Client-2. Execute the following python code: 

!!! info "Replace the IP address in the following code with Server's IP address you noted earlier and also assign a unique port address _(required by Server to identify this system)_."

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python
# import required libraries
from vidgear.gears import NetGear
import cv2

# activate Multi-Clients mode
options = {"multiclient_mode": True}

# Define NetGear Client at Server's IP address and assign a unique port address and other parameters
# !!! change following IP address '192.168.x.xxx' with yours !!!
client = NetGear(
    address="192.168.x.x",
    port="5577",
    protocol="tcp",
    pattern=2,
    receive_mode=True,
    logging=True,
    **options
) 

# loop over
while True:
    # receive data from server
    frame = client.recv()

    # check for frame if None
    if frame is None:
        break

    # {do something with frame here}

    # Show output window
    cv2.imshow("Client 5577 Output", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close client
client.close()
```

&nbsp;

&nbsp;


### Using Multi-Clients Mode for Unidirectional Custom Data Transfer


!!! abstract

    With Multi-Clients Mode, you can also send additional data of any data-type _(such as list, tuple, string, int, ndarray etc.)_ along with frame, from all connected Clients(s) back to  a Server unidirectionally.

    !!! warning "In Multi-Clients Mode, unidirectional data transfer **ONLY** works with pattern `1` _(i.e. Request/Reply `zmq.REQ/zmq.REP`)_, and **NOT** with pattern `2` _(i.e. Publish/Subscribe `zmq.PUB/zmq.SUB`)_!"

In this example, We will be transferring video-frames from a single Server _(consisting of  Raspberry Pi with Camera Module)_ over the network to two independent Client for displaying them in real-time. At the same time, we will be sending data _(a Text String, for the sake of simplicity)_ from both the Client(s) back to our Server, which will be printed onto the terminal.

#### Server's End

Now, Open the terminal on a Server System _(with a webcam connected to it at index `0`)_. Now execute the following python code: 

!!! info "Important Notes"

    * Note down the local IP-address of this system(required at all Client(s) end) and also replace it in the following code. You can follow [this FAQ](../../../../help/netgear_faqs/#how-to-find-local-ip-address-on-different-os-platforms) for this purpose.
    * Also, assign the tuple/list of port address of all Client you are going to connect to this system. 

!!! warning "Frame/Data transmission will **NOT START** untill all given Client(s) are connected to this Server."

!!! tip "You can terminate streaming anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="47-60"
# import required libraries
from vidgear.gears import PiGear
from vidgear.gears import NetGear

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

# activate multiclient_mode mode
options = {"multiclient_mode": True}

# Define NetGear Client at given IP address and assign list/tuple of all unique Server((5577,5578) in our case) and other parameters
server = NetGear(
    address="192.168.x.x",
    port=(5577, 5578),
    protocol="tcp",
    pattern=1,
    logging=True,
    **options
)  # !!! change following IP address '192.168.x.xxx' with yours !!!

# Define received data dictionary
data_dict = {}

# loop over until KeyBoard Interrupted
while True:

    try:
        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # send frame and also receive data from Client(s)
        recv_data = server.send(frame)

        # check if valid data received
        if not (recv_data is None):
            # extract unique port address and its respective data
            unique_address, data = recv_data
            # update the extracted data in the data dictionary
            data_dict[unique_address] = data

        if data_dict:
            # print data just received from Client(s)
            for key, value in data_dict.items():
                print("Client at port {} said: {}".format(key, value))

    except KeyboardInterrupt:
        break

# safely close video stream
stream.stop()

# safely close server
server.close()
```

&nbsp;


#### Client-1's End

Now, Open a terminal on another Client System _(where you want to display the input frames received from Server)_, let's name it Client-1. Execute the following python code: 

!!! info "Replace the IP address in the following code with Server's IP address you noted earlier and also assign a unique port address _(required by Server to identify this system)_."

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="27"
# import required libraries
from vidgear.gears import NetGear
import cv2

# activate Multi-Clients mode
options = {"multiclient_mode": True}

# Define NetGear Client at Server's IP address and assign a unique port address and other parameters
# !!! change following IP address '192.168.x.xxx' with yours !!!
client = NetGear(
    address="192.168.x.x",
    port="5577",
    protocol="tcp",
    pattern=1,
    receive_mode=True,
    logging=True,
    **options
)

# loop over
while True:

    # prepare data to be sent
    target_data = "Hi, I am 5577 Client here."

    # receive data from server and also send our data
    frame = client.recv(return_data=target_data)

    # check for frame if None
    if frame is None:
        break

    # {do something with frame here}

    # Show output window
    cv2.imshow("Client 5577 Output", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close client
client.close()
```

&nbsp;

#### Client-2's End

Finally, Open a terminal on another Client System _(also, where you want to display the input frames received from Server)_, let's name it Client-2. Execute the following python code: 

!!! info "Replace the IP address in the following code with Server's IP address you noted earlier and also assign a unique port address _(required by Server to identify this system)_."

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"


```python hl_lines="27"
# import required libraries
from vidgear.gears import NetGear
import cv2

# activate Multi-Clients mode
options = {"multiclient_mode": True}

# Define NetGear Client at Server's IP address and assign a unique port address and other parameters
# !!! change following IP address '192.168.x.xxx' with yours !!!
client = NetGear(
    address="192.168.x.x",
    port="5578",
    protocol="tcp",
    pattern=1,
    receive_mode=True,
    logging=True,
    **options
) 

# loop over
while True:

    # prepare data to be sent
    target_data = "Hi, I am 5578 Client here."

    # receive data from server and also send our data
    frame = client.recv(return_data=target_data)

    # check for frame if None
    if frame is None:
        break

    # {do something with frame here}

    # Show output window
    cv2.imshow("Client 5578 Output", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close client
client.close()
```

&nbsp;

&nbsp;


### Using Multi-Clients Mode with Bidirectional Mode

!!! abstract
    Multi-Clients Mode now also compatible with [Bidirectional Mode](../../advanced/bidirectional_mode/), which lets you send additional data of ***any datatype***[^1]  along with frame in real-time bidirectionally between a single Server and all connected Client(s).

!!! warning "Important Information"
    * Bidirectional data transfer **ONLY** works with pattern `1` _(i.e. Request/Reply `zmq.REQ/zmq.REP`)_, and **NOT** with pattern `2` _(i.e. Publish/Subscribe `zmq.PUB/zmq.SUB`)_
    * Additional data of [numpy.ndarray](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) data-type is **NOT SUPPORTED** at Server's end with its [`message`](../../../../bonus/reference/netgear/#vidgear.gears.netgear.NetGear.send) parameter.
    * Bidirectional Mode may lead to additional **LATENCY** depending upon the size of data being transfer bidirectionally. User discretion is advised!

??? new "New in v0.2.5" 
    This example was added in `v0.2.5`.

In this example, We will be transferring video-frames and data _(a Text String, for the sake of simplicity)_ from a single Server _(In this case, Raspberry Pi with Camera Module)_ over the network to two independent Clients for displaying them both in real-time. At the same time, we will be sending data _(a Text String, for the sake of simplicity)_ back from both the Client(s) to our Server, which will be printed onto the terminal.

#### Server's End

Now, Open the terminal on a Server System _(with a webcam connected to it at index `0`)_. Now execute the following python code: 

!!! info "Important Notes"

    * Note down the local IP-address of this system(required at all Client(s) end) and also replace it in the following code. You can follow [this FAQ](../../../../help/netgear_faqs/#how-to-find-local-ip-address-on-different-os-platforms) for this purpose.
    * Also, assign the tuple/list of port address of all Client you are going to connect to this system. 

!!! alert "Frame/Data transmission will **NOT START** untill all given Client(s) are connected to this Server."

!!! tip "You can terminate streaming anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="19 48-64"
# import required libraries
from vidgear.gears import PiGear
from vidgear.gears import NetGear

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

# activate both multiclient and bidirectional modes
options = {"multiclient_mode": True, "bidirectional_mode": True}

# Define NetGear Client at given IP address and assign list/tuple of 
# all unique Server((5577,5578) in our case) and other parameters
server = NetGear(
    address="192.168.x.x",
    port=(5577, 5578),
    protocol="tcp",
    pattern=1,
    logging=True,
    **options
)  # !!! change following IP address '192.168.x.xxx' with yours !!!

# Define received data dictionary
data_dict = {}

# loop over until KeyBoard Interrupted
while True:

    try:
        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # prepare data to be sent(a simple text in our case)
        target_data = "Hello, I am a Server."

        # send frame & data and also receive data from Client(s)
        recv_data = server.send(frame, message=target_data) # (1)

        # check if valid data received
        if not (recv_data is None):
            # extract unique port address and its respective data
            unique_address, data = recv_data
            # update the extracted data in the data dictionary
            data_dict[unique_address] = data

        if data_dict:
            # print data just received from Client(s)
            for key, value in data_dict.items():
                print("Client at port {} said: {}".format(key, value))

    except KeyboardInterrupt:
        break

# safely close video stream
stream.stop()

# safely close server
server.close()
```

1.  :warning: Everything except [numpy.ndarray](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) datatype data is accepted as `target_data` in `message` parameter.


&nbsp;


#### Client-1's End

Now, Open a terminal on another Client System _(where you want to display the input frames received from Server)_, let's name it Client-1. Execute the following python code: 

!!! info "Replace the IP address in the following code with Server's IP address you noted earlier and also assign a unique port address _(required by Server to identify this system)_."

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="6 23-34 42-44"
# import required libraries
from vidgear.gears import NetGear
import cv2

# activate both multiclient and bidirectional modes
options = {"multiclient_mode": True, "bidirectional_mode": True}

# Define NetGear Client at Server's IP address and assign a unique port address and other parameters
# !!! change following IP address '192.168.x.xxx' with yours !!!
client = NetGear(
    address="192.168.x.x",
    port="5577",
    protocol="tcp",
    pattern=1,
    receive_mode=True,
    logging=True,
    **options
)

# loop over
while True:

    # prepare data to be sent
    target_data = "Hi, I am 5577 Client here."

    # receive data from server and also send our data
    data = client.recv(return_data=target_data)

    # check for data if None
    if data is None:
        break

    # extract server_data & frame from data
    server_data, frame = data

    # again check for frame if None
    if frame is None:
        break

    # {do something with the extracted frame and data here}

    # lets print extracted server data
    if not (server_data is None):
        print(server_data)

    # Show output window
    cv2.imshow("Client 5577 Output", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close client
client.close()
```

&nbsp;

#### Client-2's End

Finally, Open a terminal on another Client System _(also, where you want to display the input frames received from Server)_, let's name it Client-2. Execute the following python code: 

!!! info "Replace the IP address in the following code with Server's IP address you noted earlier and also assign a unique port address _(required by Server to identify this system)_."

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"


```python hl_lines="6 23-34 42-44"
# import required libraries
from vidgear.gears import NetGear
import cv2

# activate both multiclient and bidirectional modes
options = {"multiclient_mode": True, "bidirectional_mode": True}

# Define NetGear Client at Server's IP address and assign a unique port address and other parameters
# !!! change following IP address '192.168.x.xxx' with yours !!!
client = NetGear(
    address="192.168.x.x",
    port="5578",
    protocol="tcp",
    pattern=1,
    receive_mode=True,
    logging=True,
    **options
) 

# loop over
while True:

    # prepare data to be sent
    target_data = "Hi, I am 5578 Client here."

    # receive data from server and also send our data
    data = client.recv(return_data=target_data)

    # check for data if None
    if data is None:
        break

    # extract server_data & frame from data
    server_data, frame = data

    # again check for frame if None
    if frame is None:
        break

    # {do something with the extracted frame and data here}

    # lets print extracted server data
    if not (server_data is None):
        print(server_data)

    # Show output window
    cv2.imshow("Client 5578 Output", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close client
client.close()
```

&nbsp;

[^1]: 
    
    !!! warning "Additional data of [numpy.ndarray](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) data-type is **NOT SUPPORTED** at Server's end with its [`message`](../../../../bonus/reference/netgear/#vidgear.gears.netgear.NetGear.send) parameter."