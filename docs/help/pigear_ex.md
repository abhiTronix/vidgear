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

# PiGear Examples

&nbsp;

## Setting variable `picamera` parameters for Camera Module at runtime

You can use `stream` global parameter in PiGear to feed any [`picamera`](https://picamera.readthedocs.io/en/release-1.10/api_camera.html) parameters at runtime. 

In this example we will set initial Camera Module's `brightness` value `80`, and will change it `50` when **`z` key** is pressed at runtime:

```python hl_lines="35"
# import required libraries
from vidgear.gears import PiGear
import cv2

# initial parameters
options = {"brightness": 80} # set brightness to 80

# open pi video stream with default parameters
stream = PiGear(logging=True, **options).start() 

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
    # check for 'z' key if pressed
    if key == ord("z"):
        # change brightness to 50
        stream.stream.brightness = 50

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
``` 

&nbsp;