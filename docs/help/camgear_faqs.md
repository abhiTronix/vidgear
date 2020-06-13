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

**Answer:** CamGear supports a diverse range of video streams which can handle/control video stream almost any IP/USB Cameras, multimedia video file format (upto 4k tested), any network stream URL such as http(s), rtp, rstp, rtmp, mms, etc. In addition to this, it also supports live Gstreamer's RAW pipelines and YouTube video/livestreams URLs. _For more info. see [CamGear doc➶](../../gears/camgear/overview/)._

&nbsp;

## I'm only familiar with OpenCV, how to get started with CamGear API?

**Answer:** First see [Switching from OpenCV](../../switch_from_cv/#switching-videocapture-apis), then go through [CamGear doc](../../gears/camgear/overview/). Still in doubt, then ask us on [Gitter ➶](https://gitter.im/vidgear/community) Community channel.

&nbsp;

## Why CamGear is throwing `RuntimeError`?

**Answer:** CamGear API will throw `RuntimeError` if source provided is Invalid. Recheck the `source` parameter value!.

&nbsp;

## How to change OpenCV source backend in CamGear API?

**Answer:** See [its Parameters ➶](../../gears/camgear/params/). Its, `backend`(int) parameter sets the backend of the source. Its value can be for e.g. `backend = cv2.CAP_DSHOW` in case of Direct Show.

&nbsp;

## How to get framerate of the source?

**Answer:** CamGear's `framerate` global variable can be used to retrieve framerate of the input video stream.  See [this example ➶](../../gears/writegear/compression/usage/#using-compression-mode-with-controlled-framerate).

&nbsp;

## How to open network streams with VidGear?

**Answer:** Just give your stream URL directly to CamGear's [`source`](../../gears/camgear/params/#source) parameter.

&nbsp;

## How to open Gstreamer pipeline in vidgear?

**Answer:** CamGear API supports GStreamer Pipeline. Note ***that needs your OpenCV to have been built with GStreamer support***, you can check this with `print(cv2.getBuildInformation())` python command and check output if contains something similar as follows:

 ```bash
   Video I/O:
  ...
      GStreamer:                   
        base:                      YES (ver 1.8.3)
        video:                     YES (ver 1.8.3)
        app:                       YES (ver 1.8.3)
 ...
 ```

Also finally, be sure videoconvert outputs into BGR format. For example as follows:

```python
stream = CamGear(source='udpsrc port=5000 ! application/x-rtp,media=video,payload=96,clock-rate=90000,encoding-name=H264, ! rtph264depay ! decodebin ! videoconvert ! video/x-raw, format=BGR ! appsink').start()
```

&nbsp;

## How to set USB camera properties?

**Answer:** See [this usage example ➶](../../gears/camgear/usage/#using-camgear-with-variable-camera-properties).

&nbsp;

## How to play YouTube Live Stream or Video with CamGear API?

**Answer:** See [this usage example ➶](../../gears/camgear/usage/#using-camgear-with-youtube-videos).

&nbsp;

## Can I play 4k video with vidgear?

**Answer:** Yes, you can if your System Hardware supports it. It proven by our [playback benchmarking test](https://github.com/abhiTronix/vidgear/blob/master/vidgear/tests/benchmark_tests/test_benchmark_playback.py).

&nbsp;

## How to synchronize between two cameras?

**Answer:** See [this issue comment ➶](https://github.com/abhiTronix/vidgear/issues/1#issuecomment-473943037).

&nbsp;

## Can I use GPU to decode the video source?

**Answer:** See [this issue comment ➶](https://github.com/abhiTronix/vidgear/issues/69#issuecomment-551112764).

&nbsp;

## Can I perform Deep Learning task with VidGear?

**Answer:** VidGear is a Video Processing library _(similar to OpenCV, FFmpeg etc.)_, so you have to use a third party library with VidGear to deal with Deep Learning operations. But surely VidGear's  high-performance APIs will definitely leverages the overall performance.

&nbsp;

## Why CamGear is throwing warning that Threaded Queue Mode is disabled?

**Answer:** That's a normal behaviour. Please read about [Threaded Queue Mode ➶](../../bonus/TQM/)

&nbsp;