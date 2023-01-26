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

# CamGear Examples

&nbsp;

## Synchronizing Two Sources in CamGear

In this example both streams and corresponding frames will be processed synchronously i.e. with no delay:

!!! danger "Using same source with more than one instances of CamGear can lead to [Global Interpreter Lock (GIL)](https://wiki.python.org/moin/GlobalInterpreterLock#:~:text=In%20CPython%2C%20the%20global%20interpreter,conditions%20and%20ensures%20thread%20safety.&text=The%20GIL%20can%20degrade%20performance%20even%20when%20it%20is%20not%20a%20bottleneck.) that degrades performance even when it is not a bottleneck."

```python
# import required libraries
from vidgear.gears import CamGear
import cv2
import time

# define and start the stream on first source ( For e.g #0 index device)
stream1 = CamGear(source=0, logging=True).start() 

# define and start the stream on second source ( For e.g #1 index device)
stream2 = CamGear(source=1, logging=True).start() 

# infinite loop
while True:
    
    frameA = stream1.read()
    # read frames from stream1

    frameB = stream2.read()
    # read frames from stream2

    # check if any of two frame is None
    if frameA is None or frameB is None:
        #if True break the infinite loop
        break
    
    # do something with both frameA and frameB here
    cv2.imshow("Output Frame1", frameA)
    cv2.imshow("Output Frame2", frameB)
    # Show output window of stream1 and stream 2 separately

    key = cv2.waitKey(1) & 0xFF
    # check for 'q' key-press
    if key == ord("q"):
        #if 'q' key-pressed break out
        break

    if key == ord("w"):
        #if 'w' key-pressed save both frameA and frameB at same time
        cv2.imwrite("Image-1.jpg", frameA)
        cv2.imwrite("Image-2.jpg", frameB)
        #break   #uncomment this line to break out after taking images

cv2.destroyAllWindows()
# close output window

# safely close both video streams
stream1.stop()
stream2.stop()
```

&nbsp;

## Using variable `yt_dlp` parameters in CamGear

CamGear provides exclusive attributes `STREAM_RESOLUTION` _(for specifying stream resolution)_ & `STREAM_PARAMS` _(for specifying underlying API(e.g. `yt_dlp`) parameters)_ with its [`options`](../../gears/camgear/params/#options) dictionary parameter. 

The complete usage example is as follows: 

!!! tip "More information on `STREAM_RESOLUTION` & `STREAM_PARAMS` attributes can be found [here ➶](../../gears/camgear/advanced/source_params/#exclusive-camgear-parameters)"

```python hl_lines="6"
# import required libraries
from vidgear.gears import CamGear
import cv2

# specify attributes
options = {"STREAM_RESOLUTION": "720p", "STREAM_PARAMS": {"nocheckcertificate": True}}

# Add YouTube Video URL as input source (for e.g https://youtu.be/bvetuLwJIkA)
# and enable Stream Mode (`stream_mode = True`)
stream = CamGear(
    source="https://youtu.be/bvetuLwJIkA", stream_mode=True, logging=True, **options
).start()

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


## Using CamGear for capturing RTSP/RTMP URLs

You can open any network stream _(such as RTSP/RTMP)_ just by providing its URL directly to CamGear's [`source`](../../gears/camgear/params/#source) parameter. 

Here's a high-level wrapper code around CamGear API to enable auto-reconnection during capturing: 

??? new "New in v0.2.2" 
    This example was added in `v0.2.2`.

??? tip "Enforcing UDP stream"
    
    You can easily enforce UDP for RTSP streams inplace of default TCP, by putting following lines of code on the top of your existing code:

    ```python 
    # import required libraries
    import os

    # enforce UDP
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
    ```

    Finally, use [`backend`](../../gears/camgear/params/#backend) parameter value as `backend=cv2.CAP_FFMPEG` in CamGear.


```python
from vidgear.gears import CamGear
import cv2
import datetime
import time


class Reconnecting_CamGear:
    def __init__(self, cam_address, reset_attempts=50, reset_delay=5):
        self.cam_address = cam_address
        self.reset_attempts = reset_attempts
        self.reset_delay = reset_delay
        self.source = CamGear(source=self.cam_address).start()
        self.running = True

    def read(self):
        if self.source is None:
            return None
        if self.running and self.reset_attempts > 0:
            frame = self.source.read()
            if frame is None:
                self.source.stop()
                self.reset_attempts -= 1
                print(
                    "Re-connection Attempt-{} occured at time:{}".format(
                        str(self.reset_attempts),
                        datetime.datetime.now().strftime("%m-%d-%Y %I:%M:%S%p"),
                    )
                )
                time.sleep(self.reset_delay)
                self.source = CamGear(source=self.cam_address).start()
                # return previous frame
                return self.frame
            else:
                self.frame = frame
                return frame
        else:
            return None

    def stop(self):
        self.running = False
        self.reset_attempts = 0
        self.frame = None
        if not self.source is None:
            self.source.stop()


if __name__ == "__main__":
    # open any valid video stream
    stream = Reconnecting_CamGear(
        cam_address="rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov",
        reset_attempts=20,
        reset_delay=5,
    )

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if None-type
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