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

# Advanced Usage: Bidirectional Mode for NetGear API 

<h3 align="center">
  <img src="../../../../assets/images/bidir.png" alt="Bi-directional Mode" width="85%"/>
</h3>

## Overview

Bi-directional Mode enables seamless support for Bidirectional data transmission between Client/Consumer and Sender/Publisher along with Video frames, through its synchronous messaging patterns such as `zmq.PAIR` (ZMQ Pair Pattern) & `zmq.REQ/zmq.REP` (ZMQ Request/Reply Pattern).


In Bi-directional Mode, We can utilize its [`message`](../../../../bonus/reference/netgear/#vidgear.gears.netgear.NetGear.send) parameter of `send()` method at Server's end for sending data to Client, and [`return_data`](../../../../bonus/reference/netgear/#vidgear.gears.netgear.NetGear.recv) parameter of `recv()` method at Client end to return data back to Server, all while transferring frames in real-time. This mode can be easily activated in NetGear API through `bidirectional_mode` attribute of its [`option`](../../params/#options) dictionary parameter, during initialization.

&nbsp;


!!! warning "Important Information"

    * In Bi-directional Mode, `zmq.PAIR`(ZMQ Pair) & `zmq.REQ/zmq.REP`(ZMQ Request/Reply) are **ONLY** Supported messaging patterns. Accessing this mode with any other messaging pattern, will result in `ValueError`.

    * Bidirectional Mode is **NOT** compatibile with [Multi-Servers mode](../../advanced/multi_server/) and [Multi-Clients mode](../../advanced/multi_client/) exclusive modes. Thereby, if Bidirectional mode is enabled with any of these modes, it will be **DISABLED** automatically.

    * Bi-directional Mode may lead to additional **LATENCY** depending upon the bytes of the data being transfer bidirectionally. User discretion is advised!

    * With Bidirectional Mode, you can also send data of **ANY**[^1] Data-type along with frame bidirectionally.


&nbsp;

&nbsp;

## Features

- [x] Enables easy-to-use seamless bi-directional data transmission between two systems.

- [x] Supports `zmq.PAIR` & `zmq.REQ/zmq.REP` messaging patterns.

- [x] Support for sending data of almost any[^1] datatype.

- [x] Auto-enables reconnection if Server or Client disconnects prematurely.


&nbsp;

&nbsp;


## Method Parameters

To send data bidirectionally, NetGear API provides two exclusive parameters for its methods:

* [`message`](../../../../bonus/reference/netgear/#vidgear.gears.netgear.NetGear.send): It enables us to send data to Client, directly through `send()` method at Server's end. 

* [`return_data`](../../../../bonus/reference/netgear/#vidgear.gears.netgear.NetGear.recv): It enables us to send data back to Server, directly through `recv()` method at Client's end.


&nbsp;

&nbsp;


## Usage Examples


### Bare-Minimum Usage

Following is the bare-minimum code you need to get started with Bidirectional Mode in NetGear API:

#### Server End

Open your favorite terminal and execute the following python code:

!!! tip "You can terminate both sides anytime by pressing **`[Ctrl+C]/[⌘+C]`** on your keyboard!"

```python
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import NetGear

# open any valid video stream(for e.g `test.mp4` file)
stream = VideoGear(source='test.mp4').start()

# activate Bidirectional mode
options = {'bidirectional_mode': True} 

#Define NetGear Server with defined parameters
server = NetGear(logging = True, **options) 

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
    target_data = 'Hello, I am a Server.'

    # send frame & data and also receive data from Client
    recv_data = server.send(frame, message = target_data)

    # print data just received from Client
    if not(recv_data is None): 
      print(recv_data)
  
  except KeyboardInterrupt:
    break

# safely close video stream
stream.stop()

# safely close server
server.close()
```

#### Client End

Then open another terminal on the same system and execute the following python code and see the output:

!!! tip "You can terminate client anytime by pressing **`[Ctrl+C]/[⌘+C]`** on your keyboard!"

```python
# import required libraries
from vidgear.gears import NetGear
import cv2

# activate Bidirectional mode
options = {'bidirectional_mode': True} 

#define NetGear Client with `receive_mode = True` and defined parameter
client = NetGear(receive_mode = True, logging = True, **options)

# loop over
while True:

    #prepare data to be sent
    target_data = "Hi, I am a Client here."

    # receive data from server and also send our data
    data = client.recv(return_data = target_data)

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
    if not(server_data is None): 
        print(server_data)

    # Show output window
    cv2.imshow("Output Frame", frame)

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


### Using Bidirectional Mode with Variable Parameters


#### Client's End

Open a terminal on Client System _(where you want to display the input frames received from the Server)_ and execute the following python code: 

!!! info "Note down the IP-address of this system(required at Server's end) by executing the command: `hostname -I` and also replace it in the following code."

!!! tip "You can terminate client anytime by pressing **`[Ctrl+C]/[⌘+C]`** on your keyboard!"

```python
# import required libraries
from vidgear.gears import NetGear
import cv2

# activate Bidirectional mode
options = {'bidirectional_mode': True} 

# Define NetGear Client at given IP address and define parameters (!!! change following IP address '192.168.x.xxx' with yours !!!)
client = NetGear(address = '192.168.x.xxx', port = '5454', protocol = 'tcp',  pattern = 1, receive_mode = True, logging = True, **options)

# loop over
while True:

    #prepare data to be sent
    target_data = "Hi, I am a Client here."

    # receive data from server and also send our data
    data = client.recv(return_data = target_data)

    # check for data if None
    if data is None:
        break

    # extract server_data & frame from data
    server_data, frame = data

    # again check for frame if None
    if frame is None:
        break

    # {do something with the extracted frame and data here}

    # lets print recieved server data
    if not(server_data is None): 
        print(server_data)

    # Show output window
    cv2.imshow("Output Frame", frame)

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

#### Server End

Now, Open the terminal on another Server System _(a Raspberry Pi with Camera Module)_, and execute the following python code: 

!!! info "Replace the IP address in the following code with Client's IP address you noted earlier."

!!! tip "You can terminate stream on both side anytime by pressing **`[Ctrl+C]/[⌘+C]`** on your keyboard!"

```python
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import PiGear

# add various Picamera tweak parameters to dictionary
options = {"hflip": True, "exposure_mode": "auto", "iso": 800, "exposure_compensation": 15, "awb_mode": "horizon", "sensor_mode": 0}

# open pi video stream with defined parameters
stream = PiGear(resolution=(640, 480), framerate=60, logging=True, **options).start() 

# activate Bidirectional mode
options = {'bidirectional_mode': True} 

# Define NetGear server at given IP address and define parameters (!!! change following IP address '192.168.x.xxx' with client's IP address !!!)
server = NetGear(address = '192.168.x.xxx', port = '5454', protocol = 'tcp',  pattern = 1, logging = True, **options) #

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
    target_data = 'Hello, I am a Server.'

    # send frame & data and also receive data from Client
    recv_data = server.send(frame, message = target_data)

    # print data just received from Client
    if not(recv_data is None): 
      print(recv_data)
  
  except KeyboardInterrupt:
    break

# safely close video stream
stream.stop()

# safely close server
server.close()
```

&nbsp; 

&nbsp;


### Using Bidirectional Mode for Video-Frames Transfer


In this example we are going to implement a bare-minimum example, where we will be sending video-frames _(3-Dimensional numpy arrays)_ of the same Video bidirectionally at the same time, for testing the real-time performance and synchronization between the Server and the Client using this(Bidirectional) Mode. 

!!! tip "This feature is great for building applications like Real-Time Video Chat."

!!! info "We're using VidGear's real-time _Frame-Size Reducer_(`reducer`) method for reducing frame-size on-the-go for additional performance."

!!! warning "Remember, sending large HQ video-frames may required more network bandwidth and packet size, which may add to video latency!"

#### Server End

Open your favorite terminal and execute the following python code:

!!! tip "You can terminate both side anytime by pressing **`[Ctrl+C]/[⌘+C]`** on your keyboard!"

```python
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import NetGear
from vidgear.gears.helper import reducer
import numpy as np
import cv2

# open any valid video stream(for e.g `test.mp4` file)
stream = VideoGear(source='test.mp4').start()

# activate Bidirectional mode
options = {'bidirectional_mode': True} 

#Define NetGear Server with defined parameters
server = NetGear(pattern = 1, logging = True, **options) 

# loop over until KeyBoard Interrupted
while True:

  try: 

     # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # reducer frames size if you want more performance, otherwise comment this line
    frame = reducer(frame, percentage = 30) #reduce frame by 30%

    # {do something with the frame here}

    # prepare data to be sent(a simple text in our case)
    target_data = 'Hello, I am a Server.'

    # send frame & data and also receive data from Client
    recv_data = server.send(frame, message = target_data)

    # check data just received from Client is of numpy datatype
    if not(recv_data is None) and isinstance(recv_data, np.ndarray): 

      # {do something with received numpy array here}

      # Let's show it on output window
      cv2.imshow("Received Frame", recv_data)
      key = cv2.waitKey(1) & 0xFF
  
  except KeyboardInterrupt:
    break

# safely close video stream
stream.stop()

# safely close server
server.close()
```

&nbsp;

#### Client End

Then open another terminal on the same system and execute the following python code and see the output:

!!! tip "You can terminate client anytime by pressing **`[Ctrl+C]/[⌘+C]`** on your keyboard!"

```python
# import required libraries
from vidgear.gears import NetGear
from vidgear.gears import VideoGear
from vidgear.gears.helper import reducer
import cv2

# activate Bidirectional mode
options = {'bidirectional_mode': True} 

# again open the same video stream
stream = VideoGear(source='test.mp4').start()

#define NetGear Client with `receive_mode = True` and defined parameter
client = NetGear(receive_mode = True, pattern = 1, logging = True, **options)

# loop over
while True:

     # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # reducer frames size if you want more performance, otherwise comment this line
    frame = reducer(frame, percentage = 30) #reduce frame by 30%

    # receive data from server and also send our data
    data = client.recv(return_data = frame)

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
    if not(server_data is None): 
        print(server_data)

    # Show output window
    cv2.imshow("Output Frame", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()

# safely close client
client.close()
```

&nbsp;


## Using Bidirectional Mode for Video-Frames Transfer with Frame Compression :fire:


See complete usage example [here ➶](../../advanced/compression/#using-bidirectional-mode-for-video-frames-transfer-with-frame-compression)


&nbsp;


[^1]: 
    
    !!! warning "Additional data of [numpy.ndarray](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) data-type is **ONLY SUPPORTED** at Client's end with its [`return_data`](../../../../bonus/reference/netgear/#vidgear.gears.netgear.NetGear.recv) parameter."


&nbsp;