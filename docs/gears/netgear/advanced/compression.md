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

# Frame Compression for NetGear API 


<figure>
  <img src="../../../../assets/images/compression.webp" alt="Frame Compression" loading="lazy" title="NetGear's Frame Compression" width="75%" class="center" />
</figure>


## Overview

NetGear API enables real-time **JPEG Frame Compression** capabilities for optimizing performance significantly while sending frames over the network. 

For enabling Frame Compression, NetGear uses powerful [`simplejpeg`](https://gitlab.com/jfolz/simplejpeg) library at its backend, which is based on recent versions of [libjpeg-turbo](https://libjpeg-turbo.org/) JPEG image codec, to accelerate baseline JPEG compression and decompression on all modern systems. NetGear API employs its exposed `decode_jpeg` and `encode_jpeg` methods to encode video-frames to [JFIF](https://en.wikipedia.org/wiki/JPEG_File_Interchange_Format) format  before sending it at Server, and cleverly decode it at the Client(s) all in real-time, thereby leveraging performance at cost of minor loss in frame quality.

Frame Compression is enabled by default in NetGear, and can be easily controlled through `jpeg_compression_quality`, `jpeg_compression_fastdct`, `jpeg_compression_fastupsample` like attributes of its [`options`](../../params/#options) dictionary parameter during initialization.

&nbsp;


!!! info "Useful Information about Frame Compression"
    
    * Frame Compression is enabled by default in NetGear along with fast dct and compression-quality at 90%.
    * Exclusive [`jpeg_compression`](#supported-attributes) attribute can also be used to disable Frame Compression.
    * Frame Compression can leverage performance up-to 5-10% with exclusive [performance attributes](#performance-attributes).
    * Frame Compression is compatible with all messaging pattern and modes.

!!! warning "Frame Compression is primarily controlled by Server end. That means, if Frame Compression is enabled at Server, then Client(s) will automatically enforce the Frame Compression with defined performance attributes. Otherwise if it is disabled, then Client(s) disables it too." 


&nbsp;

## Exclusive Attributes

For implementing Frame Compression, NetGear API currently provide following exclusive attribute for its [`options`](../../params/#options) dictionary parameter to leverage performance with Frame Compression:

* `jpeg_compression`:  _(bool/str)_ This internal attribute is used to activate/deactivate JPEG Frame Compression as well as to specify incoming frames colorspace with compression. Its usage is as follows:

    - [x] **For activating JPEG Frame Compression _(Boolean)_:**
    
        !!! alert "In this case, colorspace will default to `BGR`."

        !!! note "You can set `jpeg_compression` value to `False` at Server end to completely disable Frame Compression."

        ```python
        # enable jpeg encoding
        options = {"jpeg_compression": True}
        ```

    - [x] **For specifying Input frames colorspace _(String)_:**

        !!! alert "In this case, JPEG Frame Compression is activated automatically."

        !!! info "Supported colorspace values are `RGB`, `BGR`, `RGBX`, `BGRX`, `XBGR`, `XRGB`, `GRAY`, `RGBA`, `BGRA`, `ABGR`, `ARGB`, `CMYK`. More information can be found [here ➶](https://gitlab.com/jfolz/simplejpeg)"
    
        ```python
        # Specify incoming frames are `grayscale`
        options = {"jpeg_compression": "GRAY"}
        ```

* ### Performance Attributes :zap:

    * `jpeg_compression_quality`: _(int/float)_ This attribute controls the JPEG quantization factor. Its value varies from `10` to `100` (the higher is the better quality but performance will be lower). Its default value is `90`. Its usage is as follows:

        ```python
        # activate jpeg encoding and set quality 95%
        options = {"jpeg_compression": True, "jpeg_compression_quality": 95}
        ```

    * `jpeg_compression_fastdct`: _(bool)_ This attribute if True, NetGear API uses fastest DCT method that speeds up decoding by 4-5% for a minor loss in quality. Its default value is also `True`, and its usage is as follows:
    
        ```python
        # activate jpeg encoding and enable fast dct
        options = {"jpeg_compression": True, "jpeg_compression_fastdct": True}
        ```

    * `jpeg_compression_fastupsample`: _(bool)_ This attribute if True, NetGear API use fastest color upsampling method. Its default value is `False`, and its usage is as follows:
    
        ```python
        # activate jpeg encoding and enable fast upsampling
        options = {"jpeg_compression": True, "jpeg_compression_fastupsample": True}
        ```

&nbsp;

&nbsp;

## Usage Examples


### Bare-Minimum Usage

Following is the bare-minimum code you need to get started with Frame Compression in NetGear API:

#### Server End

Open your favorite terminal and execute the following python code:

!!! tip "You can terminate both sides anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="11-14"
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import NetGear
import cv2

# open any valid video stream(for e.g `test.mp4` file)
stream = VideoGear(source="test.mp4").start()

# activate jpeg encoding and specify other related parameters
options = {
    "jpeg_compression": True,
    "jpeg_compression_quality": 90,
    "jpeg_compression_fastdct": True,
    "jpeg_compression_fastupsample": True,
}

# Define NetGear Server with defined parameters
server = NetGear(pattern=1, logging=True, **options)

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

#### Client End

Then open another terminal on the same system and execute the following python code and see the output:

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

!!! note "If compression is enabled at Server, then Client will automatically enforce Frame Compression with its performance attributes."

```python
# import required libraries
from vidgear.gears import NetGear
import cv2

# define NetGear Client with `receive_mode = True` and defined parameter
client = NetGear(receive_mode=True, pattern=1, logging=True)

# loop over
while True:

    # receive frames from network
    frame = client.recv()

    # check for received frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

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


### Bare-Minimum Usage with Variable Colorspace

Frame Compression also supports specify incoming frames colorspace with compression. In following bare-minimum code, we will be sending [**GRAY**](https://en.wikipedia.org/wiki/Grayscale) frames from Server to Client:

??? new "New in v0.2.2" 
    This example was added in `v0.2.2`.

!!! example "This example works in conjunction with [Source ColorSpace manipulation for VideoCapture Gears ➶](../../../../bonus/colorspace_manipulation/#source-colorspace-manipulation)"

!!! info "Supported colorspace values are `RGB`, `BGR`, `RGBX`, `BGRX`, `XBGR`, `XRGB`, `GRAY`, `RGBA`, `BGRA`, `ABGR`, `ARGB`, `CMYK`. More information can be found [here ➶](https://gitlab.com/jfolz/simplejpeg)"

#### Server End

Open your favorite terminal and execute the following python code:

!!! tip "You can terminate both sides anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="7 11"
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import NetGear
import cv2

# open any valid video stream(for e.g `test.mp4` file) and change its colorspace to grayscale
stream = VideoGear(source="test.mp4", colorspace="COLOR_BGR2GRAY").start()

# activate jpeg encoding and specify other related parameters
options = {
    "jpeg_compression": "GRAY", # set grayscale
    "jpeg_compression_quality": 90,
    "jpeg_compression_fastdct": True,
    "jpeg_compression_fastupsample": True,
}

# Define NetGear Server with defined parameters
server = NetGear(pattern=1, logging=True, **options)

# loop over until KeyBoard Interrupted
while True:

    try:
        # read grayscale frames from stream
        frame = stream.read()

        # check for frame if None-type
        if frame is None:
            break

        # {do something with the frame here}

        # send grayscale frame to server
        server.send(frame)

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

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

!!! note "If compression is enabled at Server, then Client will automatically enforce Frame Compression with its performance attributes."

!!! info "Client's end also automatically enforces Server's colorspace, there's no need to define it again."

```python
# import required libraries
from vidgear.gears import NetGear
import cv2

# define NetGear Client with `receive_mode = True` and defined parameter
client = NetGear(receive_mode=True, pattern=1, logging=True)

# loop over
while True:

    # receive grayscale frames from network
    frame = client.recv()

    # check for received frame if Nonetype
    if frame is None:
        break

    # {do something with the grayscale frame here}

    # Show output window
    cv2.imshow("Output Grayscale Frame", frame)

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

### Using Frame Compression with Variable Parameters


#### Client's End

Open a terminal on Client System _(where you want to display the input frames received from the Server)_ and execute the following python code: 

!!! info "Note down the local IP-address of this system(required at Server's end) and also replace it in the following code. You can follow [this FAQ](../../../../help/netgear_faqs/#how-to-find-local-ip-address-on-different-os-platforms) for this purpose."

!!! note "If compression is enabled at Server, then Client will automatically enforce Frame Compression with its performance attributes."

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="9-15"
# import required libraries
from vidgear.gears import NetGear
import cv2


# Define NetGear Client at given IP address and define parameters 
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

#  loop over
while True:

    # receive frames from network
    frame = client.recv()

    # check for received frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

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

Now, Open the terminal on another Server System _(with a webcam connected to it at index `0`)_, and execute the following python code: 

!!! info "Replace the IP address in the following code with Client's IP address you noted earlier."

!!! tip "You can terminate stream on both side anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="20-25"
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

&nbsp; 

### Using Bidirectional Mode for Video-Frames Transfer with Frame Compression :fire:


NetGear now supports ==Dual Frame Compression== for transferring video-frames with its exclusive Bidirectional Mode for achieving unmatchable performance bidirectionally. You can easily enable Frame Compression with its performance attributes at both ends to boost performance bidirectionally.

In this example we are going to implement a bare-minimum example, where we will be sending video-frames _(3-Dimensional numpy arrays)_ of the same Video bidirectionally at the same time for testing the real-time performance and synchronization between the Server and Client using [**Bidirectional Mode**](../bidirectional_mode). Furthermore, we're going to use optimal Dual Frame Compression Setting for Sending and Receiving frames at both Server and Client end.

!!! example "This example is great for building applications like **Real-time Video Chat System**."

!!! note "This Dual Frame Compression feature also available for [Multi-Clients](../../advanced/multi_client/) Mode."

!!! info "We're also using [`reducer()`](../../../../bonus/reference/helper/#vidgear.gears.helper.reducer--reducer) Helper method for reducing frame-size on-the-go for additional performance."

!!! success "Remember to define Frame Compression's [performance attributes](#performance-attributes) both on Server and Client ends in Dual Frame Compression to boost performance bidirectionally!"


#### Server End

Open your favorite terminal and execute the following python code:

!!! tip "You can terminate both side anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="12-16 42"
# import required libraries
from vidgear.gears import NetGear
from vidgear.gears.helper import reducer
import numpy as np
import cv2

# open any valid video stream(for e.g `test.mp4` file)
stream = cv2.VideoCapture("test.mp4")

# activate Bidirectional mode and Frame Compression
options = {
    "bidirectional_mode": True,
    "jpeg_compression": True,
    "jpeg_compression_quality": 95,
    "jpeg_compression_fastdct": True,
    "jpeg_compression_fastupsample": True,
}

# Define NetGear Server with defined parameters
server = NetGear(pattern=1, logging=True, **options)

# loop over until KeyBoard Interrupted
while True:

    try:
        # read frames from stream
        (grabbed, frame) = stream.read()

        # check for frame if not grabbed
        if not grabbed:
            break

        # reducer frames size if you want even more performance, otherwise comment this line
        frame = reducer(frame, percentage=20)  # reduce frame by 20%

        # {do something with the frame here}

        # prepare data to be sent(a simple text in our case)
        target_data = "Hello, I am a Server."

        # send frame & data and also receive data from Client
        recv_data = server.send(frame, message=target_data) # (1)

        # check data just received from Client is of numpy datatype
        if not (recv_data is None) and isinstance(recv_data, np.ndarray):

            # {do something with received numpy array here}

            # Let's show it on output window
            cv2.imshow("Received Frame", recv_data)
            key = cv2.waitKey(1) & 0xFF

    except KeyboardInterrupt:
        break

# safely close video stream
stream.release()

# safely close server
server.close()
```

1.  :warning: Everything except [numpy.ndarray](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html#numpy-ndarray) datatype data is accepted as `target_data` in `message` parameter.

&nbsp;

#### Client End

Then open another terminal on the same system and execute the following python code and see the output:

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="8-12 35"
# import required libraries
from vidgear.gears import NetGear
from vidgear.gears.helper import reducer
import cv2

# activate Bidirectional mode and Frame Compression
options = {
    "bidirectional_mode": True,
    "jpeg_compression": True,
    "jpeg_compression_quality": 95,
    "jpeg_compression_fastdct": True,
    "jpeg_compression_fastupsample": True,
}

# again open the same video stream
stream = cv2.VideoCapture("test.mp4")

# define NetGear Client with `receive_mode = True` and defined parameter
client = NetGear(receive_mode=True, pattern=1, logging=True, **options)

# loop over
while True:

    # read frames from stream
    (grabbed, frame) = stream.read()

    # check for frame if not grabbed
    if not grabbed:
        break

    # reducer frames size if you want even more performance, otherwise comment this line
    frame = reducer(frame, percentage=20)  # reduce frame by 20%

    # receive data from server and also send our data
    data = client.recv(return_data=frame)

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
    cv2.imshow("Output Frame", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.release()

# safely close client
client.close()
```

&nbsp; 