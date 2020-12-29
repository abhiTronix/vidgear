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

# CamGear FAQs

&nbsp;

## What is CamGear API and what does it do?

**Answer:** CamGear supports a diverse range of video streams which can handle/control video stream almost any IP/USB Cameras, multimedia video file format (upto 4k tested), any network stream URL such as http(s), rtp, rstp, rtmp, mms, etc. In addition to this, it also supports live Gstreamer's RAW pipelines and YouTube video/livestreams URLs. _For more info. see [CamGear doc ➶](../../gears/camgear/overview/)._

&nbsp;

## I'm only familiar with OpenCV, how to get started with CamGear API?

**Answer:** First, see [Switching from OpenCV](../../switch_from_cv/#switching-videocapture-apis), then go through [CamGear doc](../../gears/camgear/overview/). Still in doubt, then ask us on [Gitter ➶](https://gitter.im/vidgear/community) Community channel.

&nbsp;


## How to change OpenCV source backend in CamGear API?

**Answer:** See [its Parameters ➶](../../gears/camgear/params/). Its, `backend`(int) parameter sets the backend of the source. Its value can be for e.g. `backend = cv2.CAP_DSHOW` in case of Direct Show.

&nbsp;

## How to get framerate of the source in CamGear API?

**Answer:** CamGear's `framerate` global variable can be used to retrieve framerate of the input video stream.  See [this example ➶](../../gears/writegear/compression/usage/#using-compression-mode-with-controlled-framerate).

&nbsp;

## How to compile OpenCV with GStreamer support?

**Answer:** For compiling OpenCV with GSstreamer(`>=v1.0.0`) support, checkout this [tutorial](https://web.archive.org/web/20201225140454/https://medium.com/@galaktyk01/how-to-build-opencv-with-gstreamer-b11668fa09c) for Linux and Windows OSes, and **for MacOS do as follows:**

**Step-1:** First Brew install GStreamer:

```sh
brew update
brew install gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-libav
```

**Step-2:** Then, Follow [this tutorial ➶](https://www.learnopencv.com/install-opencv-4-on-macos/)


&nbsp;


## How to change quality and parameters of YouTube Streams with CamGear?

CamGear provides exclusive attributes `STREAM_RESOLUTION` _(for specifying stream resolution)_ & `STREAM_PARAMS` _(for specifying underlying API(e.g. `youtube-dl`) parameters)_ with its [`option`](../../gears/camgear/params/#options) dictionary parameter. The complete usage example is as follows: 

!!! tip "More information on `STREAM_RESOLUTION` & `STREAM_PARAMS` attributes can be found [here ➶](../../gears/camgear/advanced/source_params/#exclusive-camgear-parameters)"

```python
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


## How to open RSTP network streams with CamGear?

You can open any local network stream _(such as RTSP)_ just by providing its URL directly to CamGear's [`source`](../params/#source) parameter. The complete usage example is as follows: 

```python
# import required libraries
from vidgear.gears import CamGear
import cv2

# open valid network video-stream
stream = CamGear(source="rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov").start()

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

## How to set Camera Settings with CamGear?

**Answer:** See [this usage example ➶](../../gears/camgear/usage/#using-camgear-with-variable-camera-properties).

&nbsp;

## Can I play 4K video with CamGear API?

**Answer:** Yes, you can if your System Hardware supports it. It proven by our [playback benchmarking test](https://github.com/abhiTronix/vidgear/blob/master/vidgear/tests/benchmark_tests/test_benchmark_playback.py).

&nbsp;

## How to synchronize between two cameras?

**Answer:** See [this issue comment ➶](https://github.com/abhiTronix/vidgear/issues/1#issuecomment-473943037).

&nbsp;

## Can I use GPU to decode the video source?

**Answer:** See [this issue comment ➶](https://github.com/abhiTronix/vidgear/issues/69#issuecomment-551112764).

&nbsp;

## Can I perform Deep Learning task with VidGear?

**Answer:** VidGear is a powerful Video Processing library _(similar to OpenCV, FFmpeg, etc.)_ that can read, write, process, send & receive a sequence of video-frames from/to various devices in way easy, flexible, and faster manner. So for Deep Learning or Machine Learning tasks, you have to use a third-party library with VidGear.  Being said that, VidGear's high-performance APIs definitely will leverage the overall performance if you're processing video/audio streams in your application along with Deep Learning tasks.

&nbsp;

## Why CamGear is throwing warning that Threaded Queue Mode is disabled?

**Answer:** That's a normal behavior. Please read about [Threaded Queue Mode ➶](../../bonus/TQM/)

&nbsp;