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

# CamGear FAQs

&thinsp;

## What is CamGear API and what does it do?

**Answer:** CamGear supports a diverse range of video streams which can handle/control video stream almost any IP/USB Cameras, multimedia video file format (upto 4k tested), any network stream URL such as http(s), rtp, rtsp, rtmp, mms, etc. In addition to this, it also supports live Gstreamer's RAW pipelines and YouTube video/livestreams URLs. _For more info. see [CamGear doc ➶](../../gears/camgear/)._

&thinsp;

## I'm only familiar with OpenCV, how to get started with CamGear API?

**Answer:** **Answer:** First, refer to the [Switching from OpenCV](../../switch_from_cv/#switching-the-videocapture-apis) guide, then go through [CamGear documentation ➶](../../gears/camgear/). If you still have doubts, ask us on [Gitter ➶](https://gitter.im/vidgear/community) Community channel.

&thinsp;


## How to change OpenCV source backend in CamGear API?

**Answer:** See [its Parameters ➶](../../gears/camgear/params/). Its, `backend`(int) parameter sets the backend of the source. Its value can be for e.g. `backend = cv2.CAP_DSHOW` in case of Direct Show.

&thinsp;

## How to get framerate of the source in CamGear API?

**Answer:** CamGear's `framerate` global variable can be used to retrieve framerate of the input video stream.  See [this example ➶](../../gears/writegear/compression/usage/#using-compression-mode-with-controlled-framerate).

&thinsp;

## How to compile OpenCV with GStreamer support?

**Answer:** For compiling OpenCV with GSstreamer(`>=v1.0.0`) support:

=== ":material-linux: Linux"

    - [x] **Compile manually:** **Follow [this tutorial ➶](https://medium.com/@galaktyk01/how-to-build-opencv-with-gstreamer-b11668fa09c)**

    - [x] **Compile using Pip:** **Follow [this GitHub issue ➶](https://github.com/opencv/opencv-python/issues/530)**

=== ":fontawesome-brands-windows: Windows"

    - [x] **Compile manually:** **Follow [this tutorial ➶](https://medium.com/@galaktyk01/how-to-build-opencv-with-gstreamer-b11668fa09c)**

=== ":material-apple: MacOS"
    
    - [x] **Compile manually:** **Follow [this tutorial ➶](https://medium.com/testinium-tech/how-to-install-opencv-with-java-and-gstreamer-support-on-macos-c3c7b28d2864)**

&thinsp;


## How to change quality and parameters of YouTube Streams with CamGear?

**Answer:** CamGear provides exclusive attributes `STREAM_RESOLUTION` _(for specifying stream resolution)_ & `STREAM_PARAMS` _(for specifying underlying API(e.g. `yt_dlp`) parameters)_ with its [`options`](../../gears/camgear/params/#options) dictionary parameter. See [this Bonus Example ➶](../camgear_ex/#using-variable-yt_dlp-parameters-in-camgear).


&thinsp;


## How to open RTSP network streams with CamGear?

**Answer:** You can open any local network stream _(such as RTSP)_ just by providing its URL directly to CamGear's [`source`](../../gears/camgear/params/#source) parameter. See [this Bonus Example ➶](../camgear_ex/#using-camgear-for-capturing-rtsprtmp-urls).

&thinsp;

## How to set Camera Settings with CamGear?

**Answer:** See [this usage example ➶](../../gears/camgear/usage/#using-camgear-with-variable-camera-properties).

&thinsp;

## Can I play 4K/8k video with CamGear API?

**Answer:** Yes, you can if your System Hardware supports it.

&thinsp;

## How to synchronize between two cameras?

**Answer:** See [this Bonus Example ➶](../camgear_ex/#synchronizing-two-sources-in-camgear).

&thinsp;

## Can I use GPU to decode the video source?

**Answer:** See [this issue comment ➶](https://github.com/abhiTronix/vidgear/issues/69#issuecomment-551112764).

&thinsp;

## Why CamGear is throwing warning that Threaded Queue Mode is disabled?

**Answer:** That's a normal behavior. Please read about [Threaded Queue Mode ➶](../../bonus/TQM/)

&thinsp;