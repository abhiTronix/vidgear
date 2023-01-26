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

# VideoGear Examples

&nbsp;

## Using VideoGear with ROS(Robot Operating System) 

We will be using [`cv_bridge`](http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython) to convert OpenCV frames to ROS image messages and vice-versa. 

In this example, we'll create a node that convert OpenCV frames into ROS image messages, and then publishes them over ROS.

??? new "New in v0.2.2" 
    This example was added in `v0.2.2`.

!!! note "This example is vidgear implementation of this [wiki example](http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython)." 

```python
# import roslib
import roslib

roslib.load_manifest("my_package")

# import other required libraries
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from vidgear.gears import VideoGear

# custom publisher class
class image_publisher:
    def __init__(self, source=0, logging=False):
        # create CV bridge
        self.bridge = CvBridge()
        # define publisher topic
        self.image_pub = rospy.Publisher("image_topic_pub", Image)
        # open stream with given parameters
        self.stream = VideoGear(source=source, logging=logging).start()
        # define publisher topic
        rospy.Subscriber("image_topic_sub", Image, self.callback)

    def callback(self, data):

        # {do something with received ROS node data here}

        # read frames
        frame = self.stream.read()
        # check for frame if None-type
        if not (frame is None):

            # {do something with the frame here}

            # publish our frame
            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
            except CvBridgeError as e:
                # catch any errors
                print(e)

    def close(self):
        # stop stream
        self.stream.stop()


def main(args):
    # !!! define your own video source here !!!
    # Open any video stream such as live webcam
    # video stream on first index(i.e. 0) device

    # define publisher
    ic = image_publisher(source=0, logging=True)
    # initiate ROS node on publisher
    rospy.init_node("image_publisher", anonymous=True)
    try:
        # run node
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        # close publisher
        ic.close()


if __name__ == "__main__":
    main(sys.argv)
```

&nbsp;

## Using VideoGear for capturing RTSP/RTMP URLs

Here's a high-level wrapper code around VideoGear API to enable auto-reconnection during capturing, plus stabilization is enabled _(`stabilize=True`)_ in order to stabilize captured frames on-the-go: 

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

    Finally, use [`backend`](../../gears/videogear/params/#backend) parameter value as `backend=cv2.CAP_FFMPEG` in VideoGear.


```python
from vidgear.gears import VideoGear
import cv2
import datetime
import time


class Reconnecting_VideoGear:
    def __init__(self, cam_address, stabilize=False, reset_attempts=50, reset_delay=5):
        self.cam_address = cam_address
        self.stabilize = stabilize
        self.reset_attempts = reset_attempts
        self.reset_delay = reset_delay
        self.source = VideoGear(
            source=self.cam_address, stabilize=self.stabilize
        ).start()
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
                self.source = VideoGear(
                    source=self.cam_address, stabilize=self.stabilize
                ).start()
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
    stream = Reconnecting_VideoGear(
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


## Using VideoGear for Real-time Stabilization with Audio Encoding

In this example code, we will be directly merging the audio from a Video-File _(to be stabilized)_ with its processed stabilized frames into a compressed video output in real time:

??? new "New in v0.2.4" 
    This example was added in `v0.2.4`.

!!! danger "Make sure this input video-file _(to be stabilized)_ contains valid audio source, otherwise you could encounter multiple errors or no output at all."

!!! warning "You **MUST** use `-input_framerate` attribute to set exact value of input framerate when using external audio in Real-time Frames mode, otherwise audio delay will occur in output streams."

!!! alert "Use `-disable_force_termination` flag when video duration is too short(<60sec), otherwise WriteGear will not produce any valid output."

```python
# import required libraries
from vidgear.gears import WriteGear
from vidgear.gears import VideoGear
import cv2

# Give suitable video file path to be stabilized
unstabilized_videofile = "test.mp4"

# open any valid video path with stabilization enabled(`stabilize = True`)
stream_stab = VideoGear(source=unstabilized_videofile, stabilize=True, logging=True).start()

# define required FFmpeg optimizing parameters for your writer
output_params = {
    "-i": unstabilized_videofile,
    "-c:a": "aac",
    "-input_framerate": stream_stab.framerate,
    "-clones": ["-shortest"],
    # !!! Uncomment following line if video duration is too short(<60sec). !!!
    #"-disable_force_termination": True,
}

# Define writer with defined parameters and suitable output filename for e.g. `Output.mp4
writer = WriteGear(output="Output.mp4", logging=True, **output_params)

# loop over
while True:

    # read frames from stream
    frame_stab = stream_stab.read()

    # check for frame if not grabbed
    if frame_stab is None:
        break

    # {do something with the stabilized frame here}

    # write stabilized frame to writer
    writer.write(frame_stab)

# safely close streams
stream_stab.stop()

# safely close writer
writer.close()
```

&nbsp;