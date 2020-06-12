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

# CamGear API Usage Examples:


## Bare-Minimum Usage

Following is the bare-minimum code you need to get started with CamGear API:

```python
# import required libraries
from vidgear.gears import CamGear
import cv2


# open any valid video stream(for e.g `myvideo.avi` file)
stream = CamGear(source='myvideo.avi').start() 

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

## Using Camgear with Youtube Videos

CamGear API provides complete support for **Live + Normal YouTube Video frames pipelining**. You just have to provide the desired YouTube Video's URL to its `source` parameter and enable `y_tube` parameter. The complete usage example is as follows:

!!! bug "Bug in `opencv-python` library"

    Due to a recent [bug](https://github.com/skvark/opencv-python/issues/204), if you're using `pip` installed [`opencv-python`](https://pypi.org/project/opencv-python/), instead of [**official OpenCV Binaries**](https://github.com/opencv/opencv), then you must need to install the latest `opencv-python` or `opencv-contrib-python` binaries _([v4.1.1.26 or above](https://github.com/skvark/opencv-python/releases/latest))_ on your machine, as follows:

    ```sh
    pip install -U opencv-python       #or install opencv-contrib-python similarly
    ```

```python
# import required libraries
from vidgear.gears import CamGear
import cv2


# Add YouTube Video URL as input source (for e.g https://youtu.be/dQw4w9WgXcQ) and enable `y_Tube`
stream = CamGear(source='https://youtu.be/dQw4w9WgXcQ', y_tube=True, logging=True).start() 

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

## Using CamGear with Variable Camera Properties

CamGear API also flexibly support various **Source Tweak Parameters** available within [OpenCV's VideoCapture API](https://docs.opencv.org/master/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d). These parameters can be easily applied to source stream in CamGear API through its `options` dictionary parameter by formatting them as its attributes. The complete usage example is as follows:


!!! tip "All the supported Source Tweak Parameters can be found [here ➶](../source_params/#source-tweak-parameters-for-camgear-api)"


```python
# import required libraries
from vidgear.gears import CamGear
import cv2


# define suitable tweak parameters for your stream.
options = {"CAP_PROP_FRAME_WIDTH":320, "CAP_PROP_FRAME_HEIGHT":240, "CAP_PROP_FPS":60}

# To open live video stream on webcam at first index(i.e. 0) device and apply source tweak parameters
stream = CamGear(source=0, logging=True, **options).start() 

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

## Using Camgear with Direct Colorspace Manipulation

CamGear API also supports **Direct Colorspace Manipulation**, which is ideal for changing source colorspace on the run. 

!!! info "A more detailed  information on colorspace manipulation can be found [here ➶](../../../bonus/TQM/#threaded-queue-mode)"

In following example code, we will start with [**HSV**](https://en.wikipedia.org/wiki/HSL_and_HSV) as source colorspace, and then we will switch to [**GRAY**](https://en.wikipedia.org/wiki/Grayscale)  colorspace when `w` key is pressed, and then [**LAB**](https://en.wikipedia.org/wiki/CIELAB_color_space) colorspace when `e` key is pressed, finally default colorspace _(i.e. **BGR**)_ when `s` key is pressed. Also, quit when `q` key is pressed:


!!! warning "Any incorrect or None-type value, will immediately revert the colorspace to default i.e. `BGR`."


```python
# import required libraries
from vidgear.gears import CamGear
import cv2

# Open any source of your choice, like Webcam first index(i.e. 0) and change its colorspace to `HSV`
stream = CamGear(source=0, colorspace = 'COLOR_BGR2HSV', logging=True).start()

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

    # check if 'w' key is pressed
    if key == ord("w"):
        #directly change colorspace at any instant
        stream.color_space = cv2.COLOR_BGR2GRAY #Now colorspace is GRAY
      
    # check for 'e' key is pressed
    if key == ord("e"):
        stream.color_space = cv2.COLOR_BGR2LAB  #Now colorspace is CieLAB
   
    # check for 's' key is pressed
    if key == ord("s"):
         stream.color_space = None #Now colorspace is default(ie BGR)

    # check for 'q' key is pressed
    if key == ord("q"):
      break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
```

&nbsp;