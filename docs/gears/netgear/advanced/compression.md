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

# Advanced Usage: Frame Compression for NetGear API 


<p align="center">
  <img src="../../../../assets/images/compression.png" alt="Frame Compression" width="75%" />
</p>


## Overview

NetGear API supports real-time Frame Compression _(Encoding/Decoding capabilities)_, for optimizing performance while sending frames over the network. This compression works by encoding the frame before sending it at Server's end, and thereby, smartly decoding it on the Client's end, all in real-time.

In Frame Compression, NetGear API utilizes OpenCV's [imencode](https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga26a67788faa58ade337f8d28ba0eb19e) & [imdecode](https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga26a67788faa58ade337f8d28ba0eb19e) methods in conjunction with its flexible APIs at Server and Client end respectively. Furthermore, this aid us to achieve better control over the compression of the frame being sent over the network, and thereby helps in optimizing the performance, but only at the cost of quality. 

Frame Compression can be easily activated in NetGear API through `compression_format` & `compression_param` attributes of its [`option`](../../params/#options) dictionary parameter, during initialization.

&nbsp;


!!! warning "Important Information"

    * Frame Compression only supports three `JPG` or `JPEG`, `PNG` & `BMP` encoding formats as of now.

    * Any Incorrect/Invalid encoding format will **DISABLE** this Frame Compression!

    * Incorrect Format-specific parameters through `compression_param` attribute are skipped automatically.


&nbsp;

## Features


- [x] Enables real-time Frame Compression for further optimizing performance.

- [x] Client End intelligently decodes frame only w.r.t the encoding used at Server End.

- [x] Encoding and decoding supports all Format-specific flags.

- [x] Support for `JPG`, `PNG` & `BMP` encoding formats.

- [x] Compatible with any messaging pattern and exclusive Multi-Server mode.


&nbsp;

## Supported Attributes

For implementing Frame Compression, NetGear API currently provide following attribute for its [`option`](../../params/#options) dictionary parameter:

* `compression_format` (_string_): This attribute activates compression with selected encoding format. Its possible valid values are: `'.jpg'`/`'.jpeg'` or `'.png'` or `'.bmp'`, and its usage is as follows:

    !!! warning "Any Incorrect/Invalid encoding format value on `compression_format` attribute will **DISABLE** this Frame Compression!"

    !!! info "Even if you assign different `compression_format` value at Client's end, NetGear will auto-select the Server's encoding format instead."
    
    ```python
    options = {'compression_format': '.jpg'} #activates jpeg encoding
    ```

* `compression_param`: This attribute allow us to pass different compression-format specific encoding/decoding flags. Its possible value are as follows:

    * **Encoding**: Assigning Encoding Parameters _(list)_ at Server end only:

        ```python
        options = {'compression_format': '.jpg', 'compression_param':[cv2.IMWRITE_JPEG_QUALITY, 80]} # activate jpeg encoding optimizations and compression quality 80
        ```

        !!! tip "All supported Encoding(`Imwrite`) Flags can be found [here ➶](https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga292d81be8d76901bff7988d18d2b42ac)"

    * **Decoding**: Assigning Decoding flag _(integer)_ at Client end only:

        ```python
        options = {'compression_param':cv2.IMREAD_UNCHANGED} # decode image as is with alpha channel
        ```

        !!! tip "All supported Decoding(`Imread`) Flags can be found [here ➶](https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga61d9b0126a3e57d9277ac48327799c80)"

&nbsp;

&nbsp;

## Usage Examples


### Bare-Minimum Usage

Following is the bare-minimum code you need to get started with Frame Compression in NetGear API:

#### Server End

Open your favorite terminal and execute the following python code:

!!! tip "You can terminate both sides anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import NetGear
import cv2

# open any valid video stream(for e.g `test.mp4` file)
stream = VideoGear(source='test.mp4').start()

# activate jpeg encoding and specify other related parameters
options = {'compression_format': '.jpg', 'compression_param':[cv2.IMWRITE_JPEG_QUALITY, 50]} 

#Define NetGear Server with defined parameters
server = NetGear(pattern = 1, logging = True, **options) 

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

```python
# import required libraries
from vidgear.gears import NetGear
import cv2

# define decode image as 3 channel BGR color image
options = {'compression_format': '.jpg', 'compression_param':cv2.IMREAD_COLOR}

#define NetGear Client with `receive_mode = True` and defined parameter
client = NetGear(receive_mode = True, pattern = 1, logging = True, **options)

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


### Using Frame Compression with Variable Parameters


#### Client's End

Open a terminal on Client System _(where you want to display the input frames received from the Server)_ and execute the following python code: 

!!! info "Note down the IP-address of this system(required at Server's end) by executing the command: `hostname -I` and also replace it in the following code."

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python
# import required libraries
from vidgear.gears import NetGear
import cv2

# activate jpeg encoding and specify other related parameters
options = {'compression_format': '.jpg', 'compression_param':[cv2.IMWRITE_JPEG_QUALITY, 80, cv2.IMWRITE_JPEG_PROGRESSIVE, True, cv2.IMWRITE_JPEG_OPTIMIZE, True,]} 

# Define NetGear Client at given IP address and define parameters (!!! change following IP address '192.168.x.xxx' with yours !!!)
client = NetGear(address = '192.168.x.xxx', port = '5454', protocol = 'tcp',  pattern = 1, receive_mode = True, logging = True, **options)

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

```python
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import NetGear
import cv2

# define decode image as 3 channel BGR color image
options = {'compression_format': '.jpg', 'compression_param':cv2.IMREAD_COLOR}

# Open live video stream on webcam at first index(i.e. 0) device
stream = VideoGear(source=0).start()

# Define NetGear server at given IP address and define parameters (!!! change following IP address '192.168.x.xxx' with client's IP address !!!)
server = NetGear(address = '192.168.x.xxx', port = '5454', protocol = 'tcp',  pattern = 1, logging = True, **options)

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

## Using Bidirectional Mode for Video-Frames Transfer with Frame Compression :fire:


NetGear now supports ==Dual Frame Compression== for transferring video-frames with its exclusive Bidirectional Mode for achieving unmatchable performance bidirectionally. You can easily pass the required encoding/decoding parameters by formatting them as `tuple` for altering Frame Compression while sending/receiving video-frames at both ends.


In this example we are going to implement a bare-minimum example, where we will be sending video-frames _(3-Dimensional numpy arrays)_ of the same Video bidirectionally at the same time, for testing the real-time performance and synchronization between the Server and the Client using Bidirectional Mode. Furthermore, we're going to use optimal Dual Frame Compression Setting for Sending and Receiving frames at both Server and Client end.

!!! tip "This feature is great for building applications like Real-Time Video Chat."

!!! note "This Dual Frame Compression feature also available for [Multi-Clients](../../advanced/multi_client/) Mode at Client(s) end only."

!!! info "We're also using VidGear's real-time _Frame-Size Reducer_(`reducer`) method for reducing frame-size on-the-go for additional performance."

!!! warning "Remember, sending large HQ video-frames may required more network bandwidth and packet size, which may add to video latency!"


#### Server End

Open your favorite terminal and execute the following python code:

!!! tip "You can terminate both side anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import NetGear
from vidgear.gears.helper import reducer
import numpy as np
import cv2

# open any valid video stream(for e.g `test.mp4` file)
stream = VideoGear(source='test.mp4').start()

# activate Bidirectional mode and Dual Frame Compression
options = {'bidirectional_mode': True, 'compression_format': '.jpg', 'compression_param': (cv2.IMREAD_COLOR, [cv2.IMWRITE_JPEG_QUALITY, 60, cv2.IMWRITE_JPEG_PROGRESSIVE, False, cv2.IMWRITE_JPEG_OPTIMIZE, True,])} 


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

    # reducer frames size if you want even more performance, otherwise comment this line
    frame = reducer(frame, percentage = 20) #reduce frame by 20%

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

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python
# import required libraries
from vidgear.gears import NetGear
from vidgear.gears import VideoGear
from vidgear.gears.helper import reducer
import cv2

# activate Bidirectional mode and Dual Frame Compression
options = {'bidirectional_mode': True, 'compression_format': '.jpg', 'compression_param': (cv2.IMREAD_COLOR, [cv2.IMWRITE_JPEG_QUALITY, 60, cv2.IMWRITE_JPEG_PROGRESSIVE, False, cv2.IMWRITE_JPEG_OPTIMIZE, True,])} 

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

    # reducer frames size if you want even more performance, otherwise comment this line
    frame = reducer(frame, percentage = 20) #reduce frame by 20%

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