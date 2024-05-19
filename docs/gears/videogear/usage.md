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

# VideoGear API Usage Examples:

!!! example "After going through following Usage Examples, Checkout more of its advanced configurations [here ➶](../../../help/videogear_ex/)"

&thinsp;


## Bare-Minimum Usage with CamGear backend

!!! abstract "VideoGear by default provides direct internal access to [CamGear API](../../camgear/overview/)."

Following is the bare-minimum code you need to access CamGear API with VideoGear:

```python linenums="1"
# import required libraries
from vidgear.gears import VideoGear
import cv2


# open any valid video stream(for e.g `myvideo.avi` file)
stream = VideoGear(source="myvideo.avi").start()

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
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

# safely close video stream
stream.stop()

```

&nbsp; 

## Bare-Minimum Usage with PiGear backend

!!! abstract "VideoGear contains a special [`enablePiCamera`](../params/#enablepicamera) flag that when `True` provides internal access to [PiGear API](../../pigear/overview/)."

Following is the bare-minimum code you need to access PiGear API with VideoGear:

??? info "Under the hood, PiGear API _(version `0.3.3` onwards)_ prioritizes the new [`picamera2`](https://github.com/raspberrypi/picamera2) API backend."

    However, PiGear API seamlessly switches to the legacy [`picamera`](https://picamera.readthedocs.io/en/release-1.13/index.html) backend, if the `picamera2` library is unavailable or not installed.
    
    !!! tip "It is advised to enable logging(`logging=True`) to see which backend is being used."

    !!! note "You could also enforce the legacy picamera API backend in PiGear by using the [`enforce_legacy_picamera`](../params) user-defined optional parameter boolean attribute."

!!! warning "Make sure to [complete Raspberry Pi Camera Hardware-specific settings](https://www.raspberrypi.com/documentation/accessories/camera.html#installing-a-raspberry-pi-camera) prior using this API, otherwise nothing will work."

```python linenums="1" hl_lines="6"
# import required libraries
from vidgear.gears import VideoGear
import cv2

# enable enablePiCamera boolean flag to access PiGear API backend
stream = VideoGear(enablePiCamera=True).start()

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
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

# safely close video stream
stream.stop()
```

&nbsp;

## Using VideoGear with Video Stabilizer backend

!!! abstract "VideoGear API provides a special internal wrapper around VidGear's Exclusive [**Video Stabilizer**](../../stabilizer/overview/) class and provides easy way of activating stabilization for various video-streams _(real-time or not)_ with its [`stabilize`](../params/#stabilize) boolean parameter during initialization."

The usage example is as follows:

!!! tip "For a more detailed information on Video-Stabilizer Class, Read [here ➶](../../stabilizer/overview/)"

!!! warning "The stabilizer might be slower for High-Quality/Resolution videos-frames."

```python linenums="1" hl_lines="7"
# import required libraries
from vidgear.gears import VideoGear
import numpy as np
import cv2

# open any valid video stream with stabilization enabled(`stabilize = True`)
stream_stab = VideoGear(source="test.mp4", stabilize=True).start()

# loop over
while True:

    # read stabilized frames
    frame_stab = stream_stab.read()

    # check for stabilized frame if None-type
    if frame_stab is None:
        break

    # {do something with the frame here}

    # Show output window
    cv2.imshow("Stabilized Output", frame_stab)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close streams
stream_stab.stop()
```

&nbsp;

## Advanced VideoGear usage with CamGear Backend

!!! abstract "VideoGear provides internal access to both CamGear and PiGear APIs, and thereby all additional parameters of [PiGear API](../params/#parameters-with-pigear-backend) or [CamGear API](../params/#parameters-with-camgear-backend) are also easily accessible within VideoGear API."

The usage example of VideoGear API with Variable Camera Properties is as follows:

!!! info "This example demonstrates how to use the VideoGear API in a similar manner to the CamGear's [example](../../camgear/usage/#using-camgear-with-variable-camera-properties) for controlling variable source properties. Any [CamGear usage example](../../camgear/usage/) can be implemented using the VideoGear API in a similar way."

!!! tip "All the supported Source Tweak Parameters can be found [here ➶](../../camgear/advanced/source_params/#source-tweak-parameters-for-camgear-api)"

```python linenums="1" hl_lines="15"
# import required libraries
from vidgear.gears import VideoGear
import cv2


# define suitable tweak parameters for your stream.
options = {
    "CAP_PROP_FRAME_WIDTH": 320, # resolution 320x240
    "CAP_PROP_FRAME_HEIGHT": 240,
    "CAP_PROP_FPS": 60, # framerate 60fps
}

# To open live video stream on webcam at first index(i.e. 0) 
# device and apply source tweak parameters
stream = VideoGear(source=0, logging=True, **options).start()

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

&nbsp; 

## Advanced VideoGear usage with PiGear Backend

!!! abstract "VideoGear provides internal access to both CamGear and PiGear APIs, and thereby all additional parameters of [PiGear API](../params/#parameters-with-pigear-backend) or [CamGear API](../params/#parameters-with-camgear-backend) are also easily accessible within VideoGear API."

The usage example of VideoGear API with Variable Camera Properties is as follows:

!!! info "This example demonstrates how to use the VideoGear API in a similar manner to the PiGear's [example](../../pigear/usage/#using-pigear-with-variable-camera-properties) for using variable camera properties. Any [PiGear usage example](../../pigear/usage/) can be implemented using the VideoGear API in a similar way."

!!! new "Backend PiGear API now fully supports the newer [`picamera2`](https://github.com/raspberrypi/picamera2) python library under the hood for Raspberry Pi :fontawesome-brands-raspberry-pi: camera modules. Follow this [guide ➶](../../../installation/pip_install/#picamera2) for its installation."

!!! warning "Make sure to [complete Raspberry Pi Camera Hardware-specific settings](https://www.raspberrypi.com/documentation/accessories/camera.html#installing-a-raspberry-pi-camera) prior using this backend, otherwise nothing will work."


=== "New Picamera2 backend"

    ```python linenums="1" hl_lines="3 9-13"
    # import required libraries
    from vidgear.gears import VideoGear
    from libcamera import Transform
    import cv2

    # formulate various Picamera2 API 
    # configurational parameters
    options = {
        "queue": True,
        "buffer_count": 4,
        "controls": {"Brightness": 0.5, "ExposureValue": 2.0},
        "transform": Transform(hflip=1),
        "auto_align_output_config": True,  # auto-align camera configuration
    }

    # open pi video stream with defined parameters
    stream = VideoGear(enablePiCamera=True, resolution=(640, 480), framerate=60, logging=True, **options).start()

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
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

    # safely close video stream
    stream.stop()
    ```
    
=== "Legacy Picamera backend"

    ??? info "Under the hood, Backend PiGear API _(version `0.3.3` onwards)_ prioritizes the new [`picamera2`](https://github.com/raspberrypi/picamera2) API backend."

        However, the API seamlessly switches to the legacy [`picamera`](https://picamera.readthedocs.io/en/release-1.13/index.html) backend, if the `picamera2` library is unavailable or not installed.
        
        !!! tip "It is advised to enable logging(`logging=True`) to see which backend is being used."

        !!! note "You could also enforce the legacy picamera API backend in PiGear by using the [`enforce_legacy_picamera`](../params) user-defined optional parameter boolean attribute."

    ```python linenums="1" hl_lines="8-13"
    # import required libraries
    from vidgear.gears import VideoGear
    import cv2

    # formulate various Picamera API 
    # configurational parameters
    options = {
        "hflip": True,
        "exposure_mode": "auto",
        "iso": 800,
        "exposure_compensation": 15,
        "awb_mode": "horizon",
        "sensor_mode": 0,
    }

    # open pi video stream with defined parameters
    stream = VideoGear(enablePiCamera=True, resolution=(640, 480), framerate=60, logging=True, **options).start()

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
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

    # safely close video stream
    stream.stop()
    ```

&nbsp; 


## Using VideoGear with Colorspace Manipulation

VideoGear API also supports **Colorspace Manipulation** but **NOT Direct** like other VideoCapture Gears. 

!!! failure "Important: `color_space` global variable is NOT Supported in VideoGear API"

    * `color_space` global variable is **NOT Supported** in VideoGear API, calling it will result in `AttribueError`. More details can be found [here ➶](../../../bonus/colorspace_manipulation/#source-colorspace-manipulation)
    * Any incorrect or None-type value on [`colorspace`](../params/#colorspace) parameter will be skipped automatically.


In following example code, we will convert source colorspace to [**HSV**](https://en.wikipedia.org/wiki/HSL_and_HSV) on initialization:


```python linenums="1" hl_lines="6"
# import required libraries
from vidgear.gears import VideoGear
import cv2

# Open any source of your choice, like Webcam first index(i.e. 0) and change its colorspace to `HSV`
stream = VideoGear(source=0, colorspace="COLOR_BGR2HSV", logging=True).start()

# loop over
while True:

    # read HSV frames
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the HSV frame here}

    # Show output window
    cv2.imshow("Output Frame", frame)

    # check for key if pressed
    key = cv2.waitKey(1) & 0xFF

    # check for 'q' key is pressed
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
```

&nbsp;

## Bonus Examples

!!! example "Checkout more advanced VideoGear examples with unusual configuration [here ➶](../../../help/videogear_ex/)"

&nbsp;