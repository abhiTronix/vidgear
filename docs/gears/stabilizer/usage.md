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

# Stabilizer Class Usage Examples:

&thinsp;

!!! fail "The stabilizer may not perform well against High-frequency jitter in video. Use at your own risk!"

!!! warning "The stabilizer might be slower :snail: for High-Quality/Resolution :material-high-definition-box: videos-frames."

!!! tip "It is advised to enable logging on the first run for easily identifying any runtime errors."

!!! experiment "After going through Stabilizer Class Usage Examples, Checkout more of its advanced configurations [here ➶](../../../help/stabilizer_ex/)"


&thinsp;

&thinsp;

## Bare-Minimum Usage with VideoCapture Gears

Following is the bare-minimum code you need to get started with Stabilizer Class and various VideoCapture Gears:

!!! tip "You can use any VideoCapture Gear instead of CamGear in the similar manner, as shown in this usage example." 

```python
# import required libraries
from vidgear.gears.stabilizer import Stabilizer
from vidgear.gears import CamGear
import cv2

# To open live video stream on webcam at first index(i.e. 0) device
stream = CamGear(source=0).start()

# initiate stabilizer object with default parameters
stab = Stabilizer()

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # send current frame to stabilizer for processing
    stabilized_frame = stab.stabilize(frame)

    # wait for stabilizer which still be initializing
    if stabilized_frame is None:
        continue

    # {do something with the stabilized frame here}

    # Show output window
    cv2.imshow("Output Stabilized Frame", stabilized_frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# clear stabilizer resources
stab.clean()

# safely close video stream
stream.stop()
```

&nbsp; 

## Bare-Minimum Usage with OpenCV

The VidGear's stabilizer class can also work standalone easily with any Computer Vision library such as OpenCV itself. Following is the bare-minimum code you need to get started with Stabilizer Class and OpenCV:

```python
# import required libraries
from vidgear.gears.stabilizer import Stabilizer
import cv2

# Open suitable video stream, such as webcam on first index(i.e. 0)
stream = cv2.VideoCapture(0)

# initiate stabilizer object with default parameters
stab = Stabilizer()

# loop over
while True:

    # read frames from stream
    (grabbed, frame) = stream.read()

    # check for frame if not grabbed
    if not grabbed:
        break

    # send current frame to stabilizer for processing
    stabilized_frame = stab.stabilize(frame)

    # wait for stabilizer which still be initializing
    if stabilized_frame is None:
        continue

    # {do something with the stabilized frame here}

    # Show output window
    cv2.imshow("Stabilized Frame", stabilized_frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# clear stabilizer resources
stab.clean()

# safely close video stream
stream.release()
```

&nbsp;

## Using Stabilizer with Variable Parameters

Stabilizer class provide certain [parameters](../params/) which you can use to tweak its internal properties. The complete usage example is as follows:

```python hl_lines="10"
# import required libraries
from vidgear.gears.stabilizer import Stabilizer
from vidgear.gears import CamGear
import cv2

# To open live video stream on webcam at first index(i.e. 0) device
stream = CamGear(source=0).start()

# initiate stabilizer object with defined parameters
stab = Stabilizer(smoothing_radius=30, crop_n_zoom=True, border_size=5, logging=True)

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # send current frame to stabilizer for processing
    stabilized_frame = stab.stabilize(frame)

    # wait for stabilizer which still be initializing
    if stabilized_frame is None:
        continue

    # {do something with the stabilized frame here}

    # Show output window
    cv2.imshow("Output Stabilized Frame", stabilized_frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# clear stabilizer resources
stab.clean()

# safely close video stream
stream.stop()
```

&nbsp;


## Using Stabilizer with WriteGear

VideoGear's stabilizer can be used in conjunction with WriteGear API directly without any compatibility issues. The complete usage example is as follows:

!!! tip "You can also add live audio input to WriteGear pipeline. See this [bonus example](../../../help)"

```python
# import required libraries
from vidgear.gears.stabilizer import Stabilizer
from vidgear.gears import CamGear
from vidgear.gears import WriteGear
import cv2

# Open suitable video stream
stream = CamGear(source="unstabilized_stream.mp4").start()

# initiate stabilizer object with default parameters
stab = Stabilizer()

# Define writer with default parameters and suitable output filename for e.g. `Output.mp4`
writer = WriteGear(output="Output.mp4")

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if not None-type
    if frame is None:
        break

    # send current frame to stabilizer for processing
    stabilized_frame = stab.stabilize(frame)

    # wait for stabilizer which still be initializing
    if stabilized_frame is None:
        continue

    # {do something with the stabilized frame here}

    # write stabilized frame to writer
    writer.write(stabilized_frame)

    # Show output window
    cv2.imshow("Stabilized Frame", stabilized_frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# clear stabilizer resources
stab.clean()

# safely close video stream
stream.stop()

# safely close writer
writer.close()
```


&nbsp;

## Using VideoGear with Stabilizer backend

[VideoGear API](../../videogear/overview/) provides a special internal wrapper around Stabilizer class that enables easy stabilization for various video-streams _(real-time or not)_  with minimum effort and writing way fewer lines of code.

!!! example "The complete usage example can be found [here ➶](../../videogear/usage/#using-videogear-with-video-stabilizer-backend)"

&nbsp;