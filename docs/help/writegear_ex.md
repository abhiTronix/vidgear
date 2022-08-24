
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

# WriteGear Examples

&nbsp;

## Using WriteGear's Compression Mode for RTSP/RTP Live-Streaming

In Compression Mode, you can use WriteGear for livestreaming with traditional protocols such as RTSP/RTP. The example to achieve that is as follows:   

??? new "New in v0.2.6" 
    This example was added in `v0.2.6`.

!!! alert "This example assume you already have a RTSP Server running at specified RTSP address with format *`rtsp://[RTSP_ADDRESS]:[RTSP_PORT]/[RTSP_PATH]`* for publishing video frames."

??? tip "Creating your own RTSP Server locally"
    If you want to create your RTSP Server locally, then checkout [**rtsp-simple-server**](https://github.com/aler9/rtsp-simple-server) - a ready-to-use and zero-dependency server and proxy that allows users to publish, read and proxy live video and audio streams through various protocols such as RTSP, RTMP etc.
    
!!! danger "Make sure to change RTSP address `rtsp://localhost:8554/mystream` with yours in following code before running!"


```python hl_lines="10 15"
# import required libraries
import cv2
from vidgear.gears import CamGear
from vidgear.gears import WriteGear

# open any valid video stream(for e.g `foo.mp4` file)
stream = CamGear(source="foo.mp4").start()

# define required FFmpeg parameters for your writer
output_params = {"-f": "rtsp", "-rtsp_transport": "tcp"}

# Define writer with defined parameters and RTSP address
# [WARNING] Change your RTSP address `rtsp://localhost:8554/mystream` with yours!
writer = WriteGear(
    output="rtsp://localhost:8554/mystream", logging=True, **output_params
)

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # write frame to writer
    writer.write(frame)

# safely close video stream
stream.stop()

# safely close writer
writer.close()
```

&nbsp;


## Using WriteGear's Compression Mode for YouTube-Live Streaming

In Compression Mode, you can also use WriteGear for Youtube-Livestreaming. The example is as follows:   

??? new "New in v0.2.1" 
    This example was added in `v0.2.1`.

!!! alert "This example assume you already have a [**YouTube Account with Live-Streaming enabled**](https://support.google.com/youtube/answer/2474026#enable) for publishing video."

!!! danger "Make sure to change [_YouTube-Live Stream Key_](https://support.google.com/youtube/answer/2907883#zippy=%2Cstart-live-streaming-now) with yours in following code before running!"

=== "Without Audio"

    ```python hl_lines="11-17 21 25"
    # import required libraries
    from vidgear.gears import CamGear
    from vidgear.gears import WriteGear
    import cv2

    # define and open video source
    stream = CamGear(source="/home/foo/foo.mp4", logging=True).start()

    # define required FFmpeg parameters for your writer
    output_params = {
        "-clones": ["-f", "lavfi", "-i", "anullsrc"],
        "-vcodec": "libx264",
        "-preset": "medium",
        "-b:v": "4500k",
        "-bufsize": "512k",
        "-pix_fmt": "yuv420p",
        "-f": "flv",
    }

    # [WARNING] Change your YouTube-Live Stream Key here:
    YOUTUBE_STREAM_KEY = "xxxx-xxxx-xxxx-xxxx-xxxx"

    # Define writer with defined parameters
    writer = WriteGear(
        output="rtmp://a.rtmp.youtube.com/live2/{}".format(YOUTUBE_STREAM_KEY),
        logging=True,
        **output_params
    )

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # write frame to writer
        writer.write(frame)

    # safely close video stream
    stream.stop()

    # safely close writer
    writer.close()
    ```

=== "With Audio"

    !!! warning "This code assume given input video source contains valid audio stream."

    ```python hl_lines="7 15-24 28 32"
    # import required libraries
    from vidgear.gears import CamGear
    from vidgear.gears import WriteGear
    import cv2

    # define video source(with audio) here
    VIDEO_SOURCE = "/home/foo/foo.mp4"

    # Open stream
    stream = CamGear(source=VIDEO_SOURCE, logging=True).start()

    # define required FFmpeg parameters for your writer
    # [NOTE]: Added VIDEO_SOURCE as audio-source
    output_params = {
        "-i": VIDEO_SOURCE,
        "-acodec": "aac",
        "-ar": 44100,
        "-b:a": 712000,
        "-vcodec": "libx264",
        "-preset": "medium",
        "-b:v": "4500k",
        "-bufsize": "512k",
        "-pix_fmt": "yuv420p",
        "-f": "flv",
    }

    # [WARNING] Change your YouTube-Live Stream Key here:
    YOUTUBE_STREAM_KEY = "xxxx-xxxx-xxxx-xxxx-xxxx"

    # Define writer with defined parameters
    writer = WriteGear(
        output="rtmp://a.rtmp.youtube.com/live2/{}".format(YOUTUBE_STREAM_KEY),
        logging=True,
        **output_params
    )

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # write frame to writer
        writer.write(frame)

    # safely close video stream
    stream.stop()

    # safely close writer
    writer.close()
    ```

&nbsp;


## Using WriteGear's Compression Mode with v4l2loopback Virtual Cameras

With WriteGear's Compression Mode, you can directly feed video-frames to [`v4l2loopback`](https://github.com/umlaeute/v4l2loopback) generated Virtual Camera devices on Linux Machines. The complete usage example is as follows:

??? new "New in v0.3.0" 
    This example was added in `v0.3.0`.

???+ danger "Example Assumptions"

    * ==You're running are a Linux machine.==
    * ==WriteGear API's backend FFmpeg binaries are compiled with `v4l2/v4l2loopback` demuxer support.==
    * You already have `v4l2loopback` Virtual Camera device running at address: `/dev/video0`

??? tip "Creating your own Virtual Camera device with `v4l2loopback` module."
    
    To install and create a v4l2loopback virtual camera device on Linux Mint OS/Ubuntu _(may slightly differ for other distros)_, run following two terminal commands:

    ```sh
    $ sudo apt-get install v4l2loopback-dkms v4l2loopback-utils linux-modules-extra-$(uname -r)

    $ sudo modprobe v4l2loopback devices=1 video_nr=0 exclusive_caps=1 card_label='VCamera'
    ```

    !!! note "For further information on parameters used, checkout [v4l2loopback](https://github.com/umlaeute/v4l2loopback#v4l2loopback---a-kernel-module-to-create-v4l2-loopback-devices) docs"

    Finally, You can check the loopback device you just created by listing contents of `/sys/devices/virtual/video4linux` directory with terminal command:

    ```sh
    $ sudo ls -1 /sys/devices/virtual/video4linux

    video0 
    ```

    Now you can use `/dev/video0` Virtual Camera device path in WriteGear API.

??? fail "v4l2: open /dev/videoX: Permission denied"

    If you got this error, then you must add your username to the `video` group by running following commands:
    ```sh
    $ sudo adduser $(whoami) video
    $ sudo usermod -a -G video $(whoami)
    ```
    Afterwards, restart your computer to finialize these changes.

    **Note:** If the problem still persists, then try to run your python script as superuser with `sudo` command.


??? warning "Default `libx264` encoder is incompatible with `v4l2loopback` module."

    Kindly use other encoders such as `libxvid`, `mpeg4` etc.



```python hl_lines="12-15 19"
# import required libraries
from vidgear.gears import CamGear
from vidgear.gears import WriteGear
import cv2

# open any valid video stream(for e.g `foo.mp4` file)
stream = CamGear(source="foo.mp4").start()

# define required FFmpeg parameters for your writer
# also retrieve framerate from CamGear Stream and pass it as `-input_framerate` parameter
output_params = {
    "-input_framerate": stream.framerate,
    "-vcodec": "libxvid",
    "-f": "v4l2",
    "-pix_fmt": "yuv420p",
}

# Define writer with "/dev/video0" as source and user-defined parameters 
writer = WriteGear(output="/dev/video0", logging=True, **output_params)

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

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()

# safely close writer
writer.close()
```

!!! success "The data sent to the v4l2loopback device `/dev/video0` in this example with WriteGear API, can then be read by any v4l2-capable application _(such as OpenCV, VLC, ffplay etc.)_"

&nbsp; 


## Using WriteGear's Compression Mode for creating MP4 segments

In Compression Mode, you can also use WriteGear for creating MP4 segments from almost any video source. The example is as follows:   

??? new "New in v0.2.1" 
    This example was added in `v0.2.1`.

```python hl_lines="13-20 24"
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import WriteGear
import cv2

# Open any video source `foo.mp4`
stream = VideoGear(
    source="foo.mp4", logging=True
).start()

# define required FFmpeg optimizing parameters for your writer
output_params = {
    "-c:v": "libx264",
    "-crf": 22,
    "-map": 0,
    "-segment_time": 9,
    "-g": 9,
    "-sc_threshold": 0,
    "-force_key_frames": "expr:gte(t,n_forced*9)",
    "-clones": ["-f", "segment"],
}

# Define writer with defined parameters
writer = WriteGear(output="output%03d.mp4", logging=True, **output_params)

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
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

&nbsp;


## Using WriteGear's Compression Mode to add external audio file input to video frames

You can also use WriteGear for merging external audio with live video-source:  

??? new "New in v0.2.1" 
    This example was added in `v0.2.1`.

!!! failure "Make sure this `-i` audio-source it compatible with provided video-source, otherwise you could encounter multiple errors or no output at all."

```python hl_lines="11-12"
# import required libraries
from vidgear.gears import CamGear
from vidgear.gears import WriteGear
import cv2

# open any valid video stream(for e.g `foo_video.mp4` file)
stream = CamGear(source="foo_video.mp4").start()

# add various parameters, along with custom audio
stream_params = {
    "-input_framerate": stream.framerate,  # controlled framerate for audio-video sync !!! don't forget this line !!!
    "-i": "foo_audio.aac",  # assigns input audio-source: "foo_audio.aac"
}

# Define writer with defined parameters
writer = WriteGear(output="Output.mp4", logging=True, **stream_params)

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
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


&nbsp;


## Using WriteGear's Compression Mode for generating Timely Accurate Video

If you need timely accurate video with exactly same speed as real-time input, then you need to use FFmpeg directly through its `execute_ffmpeg_cmd` method: 

??? new "New in v0.2.4" 
    This example was added in `v0.2.4`.

In this example we are capturing video from desktop screen in a Timely Accurate manner.

=== "Windows"

    ```python hl_lines="8-17"
    # import required libraries
    from vidgear.gears import WriteGear

    # Define writer with defined parameters and with some dummy name
    writer = WriteGear(output="Output.mp4", logging=True)

    # format FFmpeg command to generate time accurate video
    ffmpeg_command = [
        "-y",
        "-f",
        "gdigrab",
        "-framerate",
        "30",
        "-i",
        "desktop",
        "Output.mkv",
    ]  # `-y` parameter is to overwrite outputfile if exists

    # execute FFmpeg command
    writer.execute_ffmpeg_cmd(ffmpeg_command)

    # safely close writer
    writer.close()
    ```

=== "Linux"

    ```python hl_lines="8-17"
    # import required libraries
    from vidgear.gears import WriteGear

    # Define writer with defined parameters and with some dummy name
    writer = WriteGear(output="Output.mp4", logging=True)

    # format FFmpeg command to generate time accurate video
    ffmpeg_command = [
        "-y",
        "-f",
        "x11grab",
        "-framerate",
        "30",
        "-i",
        "default",
        "Output.mkv",
    ]  # `-y` parameter is to overwrite outputfile if exists

    # execute FFmpeg command
    writer.execute_ffmpeg_cmd(ffmpeg_command)

    # safely close writer
    writer.close()
    ```

=== "macOS"

    ```python hl_lines="8-17"
    # import required libraries
    from vidgear.gears import WriteGear

    # Define writer with defined parameters and with some dummy name
    writer = WriteGear(output="Output.mp4", logging=True)

    # format FFmpeg command to generate time accurate video
    ffmpeg_command = [
        "-y",
        "-f",
        "avfoundation",
        "-framerate",
        "30",
        "-i",
        "default",
        "Output.mkv",
    ]  # `-y` parameter is to overwrite outputfile if exists

    # execute FFmpeg command
    writer.execute_ffmpeg_cmd(ffmpeg_command)

    # safely close writer
    writer.close()
    ```


&nbsp;


## Using WriteGear with ROS(Robot Operating System) 

We will be using [`cv_bridge`](http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython) to convert OpenCV frames to ROS image messages and vice-versa. 

In this example, we'll create a node that listens to a ROS image message topic, converts the received images messages into OpenCV frames, draws a circle on it, and then process these frames into a lossless compressed file format in real-time.

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
from vidgear.gears import WriteGear

# custom publisher class
class image_subscriber:
    def __init__(self, output="Output.mp4"):
        # create CV bridge
        self.bridge = CvBridge()
        # define publisher topic
        self.image_pub = rospy.Subscriber("image_topic_sub", Image, self.callback)
        # Define writer with default parameters
        self.writer = WriteGear(output=output)

    def callback(self, data):
        # convert received data to frame
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # check if frame is valid
        if cv_image:

            # {do something with the frame here}
            # let's add a circle
            (rows, cols, channels) = cv_image.shape
            if cols > 60 and rows > 60:
                cv2.circle(cv_image, (50, 50), 10, 255)

            # write frame to writer
            self.writer.write(cv_image)

        def close(self):
            # safely close video stream
            self.writer.close()


def main(args):
    # define publisher with suitable output filename
    # such as `Output.mp4` for saving output
    ic = image_subscriber(output="Output.mp4")
    # initiate ROS node on publisher
    rospy.init_node("image_subscriber", anonymous=True)
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