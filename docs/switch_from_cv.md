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

<figure>
  <img src="../assets/images/cv2vidgear.png" loading="lazy" alt="Switching from OpenCV" class="shadow" />
</figure>

&emsp; 

# Switching from OpenCV Library

Switching OpenCV with VidGear APIs is fairly painless process, and will just require changing a few lines in your python script. 

!!! abstract "This document is intended to software developers who want to migrate their python code from OpenCV Library to VidGear APIs."

!!! warning "Prior knowledge of Python or OpenCV won't be covered in this guide. Proficiency with [**OpenCV-Python**](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) _(Python API for OpenCV)_ is a must in order understand this document."

!!! tip "If you're just getting started with OpenCV-Python programming, then refer this [FAQ ➶](../help/general_faqs/#im-new-to-python-programming-or-its-usage-in-opencv-library-how-to-use-vidgear-in-my-projects)"

&nbsp; 

## Why VidGear is better than OpenCV?

!!! info "Learn more about OpenCV [here ➶](https://software.intel.com/content/www/us/en/develop/articles/what-is-opencv.html)"

VidGear employs OpenCV at its backend and enhances its existing capabilities even further by introducing many new state-of-the-art functionalities such as:

- [x] Accelerated [Multi-Threaded](../bonus/TQM/#what-does-threaded-queue-mode-exactly-do) Performance.
- [x] Out-of-the-box support for OpenCV APIs.
- [x] Real-time [Stabilization](../gears/stabilizer/overview/) ready.
- [x] Lossless hardware enabled video [encoding](../gears/writegear/compression/usage/#using-compression-mode-with-hardware-encoders) and [transcoding](../gears/streamgear/rtfm/usage/#usage-with-hardware-video-encoder).
- [x] Inherited multi-backend support for various video sources and devices.
- [x] Screen-casting, Multi-bitrate network-streaming, and [way much more ➶](../gears)

Vidgear offers all this at once while maintaining the same standard OpenCV-Python _(Python API for OpenCV)_ coding syntax for all of its APIs, thereby making it even easier to implement complex real-time OpenCV applications in python code without changing things much.

&nbsp; 

## Switching the VideoCapture APIs 

Let's compare a bare-minimum python code for extracting frames out of any Webcam/USB-camera _(connected at index 0)_, between OpenCV's [VideoCapture Class](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#a57c0e81e83e60f36c83027dc2a188e80) and VidGear's [CamGear](../gears/camgear/overview/) VideoCapture API side-by-side:

!!! tip "CamGear API share the same syntax as other [VideoCapture APIs](../gears/#a-videocapture-gears), thereby you can easily switch to any of those APIs in a similar manner."

=== "OpenCV VideoCapture Class"

    ```python hl_lines="5 11 14-15 33"
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
    
=== "VidGear's CamGear API"

    ```python hl_lines="6 12 15-16 34"
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

&thinsp; 

### Differences

Let's breakdown a few noteworthy difference in both syntaxes:

| Task | OpenCV VideoCapture Class | VidGear's CamGear API |
| :----------: | :--------------------: | :---------------------: |
| Initiating | `#!python stream = cv2.VideoCapture(0)` | `#!python stream = CamGear(source=0).start()` |
| Reading frames | `#!python (grabbed, frame) = stream.read()` | `#!python frame = stream.read()` |
| Checking empty frame | `#!python if not grabbed:` | `#!python if frame is None:` |
| Terminating | `#!python stream.release()` | `#!python stream.stop()` |


!!! success "Now checkout other [VideoCapture Gears ➶](../gears/#a-videocapture-gears)"


&nbsp; 

&nbsp; 

## Switching the VideoWriter API

Let's extend previous bare-minimum python code and save those extracted frames to disk as a valid file, with [OpenCV's VideoWriter Class](https://docs.opencv.org/3.4/dd/d9e/classcv_1_1VideoWriter.html#ad59c61d8881ba2b2da22cff5487465b5) and VidGear's [WriteGear](../gears/writegear/introduction/) _(with FFmpeg backend)_, compared side-to-side:

!!! info "WriteGear API also provides backend for OpenCV's VideoWriter Class. More information [here ➶](../gears/writegear/non_compression/overview/)"

=== "OpenCV VideoWriter Class"

    ```python hl_lines="9-10 27 45"
    # import required libraries
    import cv2

    # Open suitable video stream, such as webcam on first index(i.e. 0)
    stream = cv2.VideoCapture(0) 

    # Define the codec and create VideoWriter object with suitable output 
    # filename for e.g. `Output.avi`
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

=== "VidGear's WriteGear API"

    ```python hl_lines="10 27 44"
    # import required libraries
    from vidgear.gears import CamGear
    from vidgear.gears import WriteGear
    import cv2

    # Open suitable video stream, such as webcam on first index(i.e. 0)
    stream = CamGear(source=0).start() 

    # Define WriteGear Object with suitable output filename for e.g. `Output.mp4`
    writer = WriteGear(output = 'Output.mp4') 

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

&thinsp;

### Differences

Let's breakdown a few noteworthy difference in both syntaxes:

| Task | OpenCV VideoWriter Class | VidGear's WriteGear API |
| :----------: | :--------------------: | :---------------------: |
| Initiating | `#!python writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))` | `#!python writer = WriteGear(output='Output.mp4')` |
| Writing frames | `#!python writer.write(frame)` | `#!python writer.write(frame)` |
| Terminating | `#!python writer.release()` | `#!python writer.close()` |

!!! success "Now checkout more about WriteGear API [here ➶](../gears/writegear/introduction/)"

&thinsp; 