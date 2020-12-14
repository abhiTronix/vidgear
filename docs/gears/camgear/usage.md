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

&thinsp;

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

CamGear API provides direct support for **Live + Normal YouTube Video frames pipelining**. All you have to do is to provide the desired YouTube Video's URL to its `source` parameter and enable the `y_tube` parameter. The complete usage example is as follows:

!!! danger "Update Requirements"

    If you're using `pip` installed [`opencv-python`](https://pypi.org/project/opencv-python/), then you must need to install the latest `opencv-python` or `opencv-contrib-python` binaries on your machine, along with latest `youtube-dl` and `pafy`. You can do it as follows:

    ```sh
    pip install -U opencv-python       #or install opencv-contrib-python similarly
    pip install -U youtube-dl pafy
    ```

??? bug "Bug in Live YouTube Stream"

    Due to a [**bug**](https://github.com/abhiTronix/vidgear/issues/133#issuecomment-638263225) with OpenCV's FFmpeg, some Live Youtube-Stream URLs playback freezes after a few seconds, and CamGear API exit with an error: `/io/opencv/modules/videoio/src/cap_images.cpp:235: error: (-5:Bad argument) CAP_IMAGES: error, expected '0?[1-9][du]' pattern`. This bug occurs with only some live YouTube streams _(not all)_ and hasn't been fixed yet. ***The only workaround for this bug is suggested [here  ➶](https://github.com/abhiTronix/vidgear/issues/133#issuecomment-638567443)***



```python
# import required libraries
from vidgear.gears import CamGear
import cv2


# Add YouTube Video URL as input source (for e.g https://youtu.be/bvetuLwJIkA) and enable `y_tube = True`
stream = CamGear(source='https://youtu.be/bvetuLwJIkA', y_tube=True, logging=True).start() 

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

CamGear API also flexibly support various **Source Tweak Parameters** available within [OpenCV's VideoCapture API](https://docs.opencv.org/master/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d). These tweak parameters can be used to manipulate input source Camera-Device properties _(such as its brightness, saturation, size, iso, gain etc.)_ seemlessly, and can be easily applied in CamGear API through its `options` dictionary parameter by formatting them as its attributes. The complete usage example is as follows:


!!! tip "All the supported Source Tweak Parameters can be found [here ➶](../advanced/source_params/#source-tweak-parameters-for-camgear-api)"


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

!!! info "A more detailed  information on colorspace manipulation can be found [here ➶](../../../bonus/colorspace_manipulation/)"

In following example code, we will start with [**HSV**](https://en.wikipedia.org/wiki/HSL_and_HSV) as source colorspace, and then we will switch to [**GRAY**](https://en.wikipedia.org/wiki/Grayscale)  colorspace when `w` key is pressed, and then [**LAB**](https://en.wikipedia.org/wiki/CIELAB_color_space) colorspace when `e` key is pressed, finally default colorspace _(i.e. **BGR**)_ when `s` key is pressed. Also, quit when `q` key is pressed:


!!! fail "Any incorrect or None-type value, will immediately revert the colorspace to default i.e. `BGR`."


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