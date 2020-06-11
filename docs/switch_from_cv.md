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

# Switching from OpenCV

Switching OpenCV with VidGear APIs is usually a fairly painless process, and will just require changing a few lines in your python script. 

!!! quote "This document is intended to software developers who want to migrate their code from OpenCV to VidGear APIs."

!!! warning "Prior knowledge of Python and OpenCV is recommended as they won't be covered in this guide. Proficiency with OpenCV is a must in order understand this document."

!!! tip "If your getting started with OpenCV-Python, then see [here ➶](/help/general_faqs/#im-new-to-python-programming-or-its-usage-in-computer-vision-how-to-use-vidgear-in-my-projects)"


&nbsp; 


## Switching VideoCapture APIs 

VidGear introduces many new state-of-the-art features, multi-device support, and performance upgrade for its [VideoCapture Gears](/gears/#a-videocapture-gears) comparing to [OpenCV's VideoCapture Class](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#a57c0e81e83e60f36c83027dc2a188e80), while maintaining the same standard OpenCV-Python _(Python API for OpenCV)_ coding syntax.  


!!! note "CamGear API share the same syntax as other [VideoCapture Gears](/gears/#a-videocapture-gears), thereby you can easily switch to any of those Gear in a similar manner."

_Let's compare a bare-minimum python code for extracting frames out of webcam/USB camera (connected at index 0), between OpenCV's VideoCapture Class and VidGear's [CamGear](/gears/camgear/overview/) VideoCapture API side-by-side:_


```python tab="OpenCV VideoCapture Class"
# import required libraries
import cv2

# Open suitable video stream, such as webcam on first index(i.e. 0)
stream = cv2.VideoCapture(0) 

# loop over
while True:

    # read frames from stream
    (grabbed, frame) = stream.read()

    # check for frame if not grabbed
    if not grabbed:
      break


    # {do something with the frame here}


    # Show output window
    cv2.imshow("Output", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.release()
```

```python tab="VidGear's CamGear API"
# import required libraries
from vidgear.gears import CamGear
import cv2

# Open suitable video stream, such as webcam on first index(i.e. 0)
stream = CamGear(source=0).start() 

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break


    # {do something with the frame here}


    # Show output window
    cv2.imshow("Output", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
```

and both syntax almost looks the same, easy, isn't it?


### Changes

Let's breakdown a few noteworthy changes in syntax of VidGear's CamGear API as compared to standard OpenCV's VideoCapture Class:

| Task | OpenCV VideoCapture Class | VidGear's CamGear API |
| :----------: | :--------------------: | :---------------------: |
| Initiating | `#!python stream = cv2.VideoCapture(0)` | `#!python stream = CamGear(source=0).start()` |
| Reading frames | `#!python (grabbed, frame) = stream.read()` | `#!python frame = stream.read()` |
| Checking empty frame | `#!python if not grabbed:` | `#!python if frame is None:` |
| Terminating | `#!python stream.release()` | `#!python stream.stop()` |


&nbsp; 


## Switching VideoWriter API

VidGear with its [WriteGear](/gears/writegear/introduction/) API, provide a complete, flexible & robust wrapper around [FFmpeg](https://www.ffmpeg.org/) - a leading multimedia framework, processes real-time video frames into a lossless compressed format with any suitable specification and many more, as compared to standard [OpenCV's VideoWriter Class](https://docs.opencv.org/3.4/dd/d9e/classcv_1_1VideoWriter.html#ad59c61d8881ba2b2da22cff5487465b5), while maintaining the same standard OpenCV-Python coding syntax.

!!! info "WriteGear API also provides backend for OpenCV's VideoWriter Class. More information [here ➶](/gears/writegear/non_compression/overview/)"

_Let's extend previous bare-minimum python code to save extracted frames to disk as a valid file, with OpenCV's VideoWriter Class and VidGear's WriteGear API(with FFmpeg backend), compared side-to-side:_

```python tab="OpenCV VideoWriter Class"
# import required libraries
import cv2

# Open suitable video stream, such as webcam on first index(i.e. 0)
stream = cv2.VideoCapture(0) 

# Define the codec and create VideoWriter object with suitable output filename for e.g. `Output.avi`
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
writer = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480)) 

# loop over
while True:

    # read frames from stream
    (grabbed, frame) = stream.read()

    # check for frame if not grabbed
    if not grabbed:
      break


    # {do something with the frame here}


    # write frame to writer
    writer.write(frame)


    # Show output window
    cv2.imshow("Output", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.release()

# safely close writer
writer.release() 
```

```python tab="VidGear's WriteGear API"
# import required libraries
from vidgear.gears import CamGear
from vidgear.gears import WriteGear
import cv2

# Open suitable video stream, such as webcam on first index(i.e. 0)
stream = CamGear(source=0).start() 

# Define WriteGear Object with suitable output filename for e.g. `Output.mp4`
writer = WriteGear(output_filename = 'Output.mp4') 

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if None-type
    if frame is None:
        break


    # {do something with the frame here}


    # write frame to writer
    writer.write(frame)

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

# safely close writer
writer.close()
```

Noticed WriteGear's coding syntax looks similar but less complex?


### Changes

Let's breakdown a few noteworthy changes in syntax of VidGear's WriteGear API as compared to standard OpenCV's VideoWriter Class:

| Task | OpenCV VideoWriter Class | VidGear's WriteGear API |
| :----------: | :--------------------: | :---------------------: |
| Initiating | <code>fourcc = cv2.VideoWriter_fourcc(*'XVID') <br> writer = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))</code> | `#!python writer = WriteGear(output_filename='Output.mp4')` |
| Writing frames | `#!python writer.write(frame)` | `#!python writer.write(frame)` |
| Terminating | `#!python writer.release()` | `#!python writer.close()` |

&nbsp; 