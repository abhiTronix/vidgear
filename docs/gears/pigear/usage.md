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

# PiGear API Usage Examples:


!!! warning "Make sure to [enable Raspberry Pi hardware-specific settings](https://picamera.readthedocs.io/en/release-1.13/quickstart.html) prior using this API, otherwise nothing will work."

!!! experiment "After going through following Usage Examples, Checkout more of its advanced configurations [here ➶](../../../help/pigear_ex/)"



&thinsp;


## Bare-Minimum Usage

Following is the bare-minimum code you need to get started with PiGear API:

```python
# import required libraries
from vidgear.gears import PiGear
import cv2


# open pi video stream with default parameters
stream = PiGear().start()

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

## Using PiGear with Variable Camera Module Properties

PiGear supports almost every parameter available within [**Picamera library**](https://picamera.readthedocs.io/en/release-1.13/api_camera.html). These parameters can be easily applied to the source stream in PiGear API through its [`options`](../params/#options) dictionary parameter by formatting them as its attributes. The complete usage example is as follows:


!!! tip "All supported parameters are listed in [PiCamera Docs ➶](https://picamera.readthedocs.io/en/release-1.13/api_camera.html)"


```python hl_lines="7-12"
# import required libraries
from vidgear.gears import PiGear
import cv2

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

## Using PiGear with Direct Colorspace Manipulation

PiGear API also supports **Direct Colorspace Manipulation**, which is ideal for changing source colorspace on the run. 

!!! info "A more detailed  information on colorspace manipulation can be found [here ➶](../../../bonus/colorspace_manipulation/)"

In following example code, we will start with [**HSV**](https://en.wikipedia.org/wiki/HSL_and_HSV) as source colorspace, and then we will switch to [**GRAY**](https://en.wikipedia.org/wiki/Grayscale)  colorspace when `w` key is pressed, and then [**LAB**](https://en.wikipedia.org/wiki/CIELAB_color_space) colorspace when `e` key is pressed, finally default colorspace _(i.e. **BGR**)_ when `s` key is pressed. Also, quit when `q` key is pressed:


!!! warning "Any incorrect or None-Type value will immediately revert the colorspace to default _(i.e. `BGR`)_."


```python hl_lines="20 47 51 55"
# import required libraries
from vidgear.gears import PiGear
import cv2


# add various Picamera tweak parameters to dictionary
options = {
    "hflip": True,
    "exposure_mode": "auto",
    "iso": 800,
    "exposure_compensation": 15,
    "awb_mode": "horizon",
    "sensor_mode": 0,
}

# open pi video stream with defined parameters and change colorspace to `HSV`
stream = PiGear(
    resolution=(640, 480),
    framerate=60,
    colorspace="COLOR_BGR2HSV",
    logging=True,
    **options
).start()


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
        # directly change colorspace at any instant
        stream.color_space = cv2.COLOR_BGR2GRAY  # Now colorspace is GRAY

    # check for 'e' key is pressed
    if key == ord("e"):
        stream.color_space = cv2.COLOR_BGR2LAB  # Now colorspace is CieLAB

    # check for 's' key is pressed
    if key == ord("s"):
        stream.color_space = None  # Now colorspace is default(ie BGR)

    # check for 'q' key is pressed
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
```

&nbsp;

## Using PiGear with WriteGear API

PiGear can be easily used with WriteGear API directly without any compatibility issues. The suitable example is as follows:

```python
# import required libraries
from vidgear.gears import PiGear
from vidgear.gears import WriteGear
import cv2

# add various Picamera tweak parameters to dictionary
options = {
    "hflip": True,
    "exposure_mode": "auto",
    "iso": 800,
    "exposure_compensation": 15,
    "awb_mode": "horizon",
    "sensor_mode": 0,
}

# define suitable (Codec,CRF,preset) FFmpeg parameters for writer
output_params = {"-vcodec": "libx264", "-crf": 0, "-preset": "fast"}

# open pi video stream with defined parameters
stream = PiGear(resolution=(640, 480), framerate=60, logging=True, **options).start()

# Define writer with defined parameters and suitable output filename for e.g. `Output.mp4`
writer = WriteGear(output="Output.mp4", logging=True, **output_params)

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}
    # lets convert frame to gray for this example
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # write gray frame to writer
    writer.write(gray)

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

&nbsp; 