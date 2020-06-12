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

# VideoGear API Usage Examples:


## Bare-Minimum Usage with CamGear backend

Following is the bare-minimum code you need to access CamGear API with VideoGear:

```python
# import required libraries
from vidgear.gears import VideoGear
import cv2


# open any valid video stream(for e.g `myvideo.avi` file), and disable enablePiCamera flag (default is also `False`).
stream = VideoGear(enablePiCamera=False, source='myvideo.avi').start() 

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

Following is the bare-minimum code you need to access PiGear API with VideoGear:

```python
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

VideoGear API provides a special internal wrapper around VidGear's exclusive [**Video Stabilizer**](../../stabilizer/overview/) class, and the stabilization can be activated with its [`stabilize`](../params/#stabilize) boolean parameter during initialization. Thereby, it enables easy stabilization for various video-streams _(real-time or not)_  with minimum effort and using just fewer lines of code. The complete usage example is as follows:


!!! tip "For a more detailed information on Video Stabilization, read [here ➶](../../stabilizer/overview/)"


```python
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

    # read un-stabilized frame
    frame_org = stream_org.read()

    # concatenate both frames
    output_frame = np.concatenate((frame_org, frame_stab), axis=1)

    # put text over concatenated frame
    cv2.putText(
        output_frame, "Before", (10, output_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 255, 0), 2,
    )
    cv2.putText(
        output_frame, "After", (output_frame.shape[1] // 2 + 10, output_frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 255, 0), 2,
    )

    # Show output window
    cv2.imshow("Stabilized Comparison", output_frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close both video streams
stream_org.stop()
stream_stab.stop()
```

&nbsp;

## Using VideoGear with Variable PiCamera Properties


VideoGear contains special [`enablePiCamera`](../params/#enablepicamera) flag that provide internal access to both CamGear and PiGear APIs, and thereby only one of them can be accessed at a given instance. Therefore, the additional parameters of VideoGear API are also based on API _([PiGear API](../params/#parameters-with-pigear-backend) or [CamGear API](../params/#parameters-with-camgear-backend))_ being accessed. The complete usage example of VideoGear API with Variable PiCamera Properties is as follows:

!!! info "This example is basically a VideoGear API implementation of this [PiGear usage example](../../pigear/usage/#using-pigear-with-variable-camera-properties). Thereby, any [CamGear](../../camgear/usage/) or [PiGear](../../pigear/usage/) usage examples can be implemented with VideoGear API in the similar manner."


```python
# import required libraries
from vidgear.gears import VideoGear
import cv2

# add various Picamera tweak parameters to dictionary
options = {"hflip": True, "exposure_mode": "auto", "iso": 800, "exposure_compensation": 15, "awb_mode": "horizon", "sensor_mode": 0}

# activate enablePiCamera and open pi video stream with defined parameters
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

VideoGear API also supports **Colorspace Manipulation** but not direct. 

!!! danger "Important"

    * `color_space` global variable is **NOT Supported** in VideoGear API, calling it will result in `AttribueError`. More details can be found [here ➶](../../../bonus/colorspace_manipulation/#using-color_space-global-variable)

    * Any incorrect or None-type value, will immediately revert the colorspace to default i.e. `BGR`.


In following example code, we will convert source colorspace to [**HSV**](https://en.wikipedia.org/wiki/HSL_and_HSV) on initialization:


```python
# import required libraries
from vidgear.gears import VideoGear
import cv2

# Open any source of your choice, like Webcam first index(i.e. 0) and change its colorspace to `HSV`
stream = VideoGear(source=0, colorspace = 'COLOR_BGR2HSV', logging=True).start()

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