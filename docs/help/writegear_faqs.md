
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

# WriteGear FAQs

&nbsp;

## What is WriteGear API and what does it do?

**Answer:** WriteGear handles various powerful Writer Tools that provide us the freedom to do almost anything imagine with multimedia files. _For more info. see [WriteGear doc ➶](../../gears/writegear/introduction/)_

&nbsp;

## I'm only familiar with OpenCV, how to get started with WriteGear API?

**Answer:** First, see [Switching from OpenCV](../../switch_from_cv/#switching-the-videowriter-api), then go through [WriteGear doc](../../gears/writegear/introduction/). Still in doubt, then ask us on [Gitter ➶](https://gitter.im/vidgear/community) Community channel.

&nbsp;

## Why WriteGear is throwing `ValueError`?

**Answer:** WriteGear will exit with `ValueError` if you feed frames of different dimensions or channels.


&nbsp;


## How to install and configure FFmpeg correctly for WriteGear on my machine?

**Answer:** Follow these [Installation Instructions ➶](../../gears/writegear/compression/advanced/ffmpeg_install/) for its installation.

&nbsp;

## Can I use WriteGear directly with OpenCV?

**Answer:** Yes,

* For Compression Mode: See [this usage example ➶](../../gears/writegear/compression/usage/#using-compression-mode-with-opencv).
* For  Non-Compression Mode: See [this usage example ➶](../../gears/writegear/non_compression/usage/#using-non-compression-mode-with-opencv)

&nbsp;

## What FFmpeg's encoders and parameters are supported by WriteGear in compression mode?

**Answer:** See [Supported Parameters ➶](../../gears/writegear/compression/params/#supported-parameters) and [Supported encoders ➶](../../gears/writegear/compression/params/#supported-encoders)

&nbsp;

## What OpenCV's FOURCC and parameters are supported by WriteGear in non-compression mode?

**Answer:** See [Supported Parameters ➶](../../gears/writegear/non_compression/params/#supported-parameters) and [Supported FOURCC ➶](../../gears/writegear/non_compression/params/#supported-fourcc-codecs).

&nbsp;

## Why this FOURCC is not working for me?

**Answer:** Remember not all the FOURCC and Video extensions are compatible and supported by OpenCV VideoWriter Class. You’ll need to try different combinations of FourCC and file extensions. Furthermore, OpenCV does not return any helpful error messages regarding this problem, so it’s pretty much based on _trial and error_.

&nbsp;

## Can I pass my custom FFmpeg commands directly in WriteGear API?

**Answer:** Yes, See [Custom FFmpeg Commands in WriteGear API ➶](../../gears/writegear/compression/advanced/cciw/).

&nbsp;

## How to use specific Hardware Encoder in WriteGear?

**Answer:** See [this usage example ➶](../../gears/writegear/compression/usage/#using-compression-mode-with-hardware-encoders)

&nbsp;


## How to add live audio to WriteGear?

**Answer:** See [this doc ➶](../../gears/writegear/compression/usage/#using-compression-mode-with-live-audio-input)

&nbsp;

## How to separate and merge audio from/to video?

**Answer:** See [these usage examples ➶](../../gears/writegear/compression/advanced/cciw/#usage-examples)

&nbsp;

## Can I live stream to Twitch with WriteGear API?

**Answer:** Yes, See [this usage example ➶](../../gears/writegear/compression/usage/#using-compression-mode-for-streaming-urls)

&nbsp;

## Is YouTube-Live Streaming possibe with WriteGear?

**Answer:** Yes, See example below:

!!! new "New in v0.2.1" 
    This example was added in `v0.2.1`.

!!! alert "This example assume you already have a [**YouTube Account with Live-Streaming enabled**](https://support.google.com/youtube/answer/2474026#enable) for publishing video."

!!! danger "Make sure to change [_YouTube-Live Stream Key_](https://support.google.com/youtube/answer/2907883#zippy=%2Cstart-live-streaming-now) with yours in following code before running!"

```python
# import required libraries
from vidgear.gears import CamGear
from vidgear.gears import WriteGear
import cv2

# define video source
VIDEO_SOURCE = "/home/foo/foo.mp4"

# Open stream
stream = CamGear(source=VIDEO_SOURCE, logging=True).start()

# define required FFmpeg optimizing parameters for your writer
# [NOTE]: Added VIDEO_SOURCE as audio-source, since YouTube rejects audioless streams!
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

# Define writer with defined parameters and
writer = WriteGear(
    output_filename="rtmp://a.rtmp.youtube.com/live2/{}".format(YOUTUBE_STREAM_KEY),
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


## How to create MP4 segments from a video stream with WriteGear?

**Answer:** See example below:

!!! new "New in v0.2.1" 
    This example was added in `v0.2.1`.

```python
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
writer = WriteGear(output_filename="output%03d.mp4", logging=True, **output_params)

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


## How add external audio file input to video frames?

**Answer:** See example below:

!!! new "New in v0.2.1" 
    This example was added in `v0.2.1`.

!!! failure "Make sure this `-i` audio-source it compatible with provided video-source, otherwise you encounter multiple errors or no output at all."

```python
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
writer = WriteGear(output_filename="Output.mp4", logging=True, **stream_params)

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

## Why this FFmpeg parameter is not working for me in compression mode?

**Answer:** If some FFmpeg parameter doesn't work for you, then [tell us on Gitter ➶](https://gitter.im/vidgear/community), and if that doesn't help, then finally [report an issue ➶](../../contribution/issue/)

&nbsp;

## Why WriteGear is switching to Non-compression Mode, even if it is not enable?

**Answer:** In case WriteGear API fails to detect valid FFmpeg executables on your system _(even if Compression Mode is enabled)_, it will automatically fallback to Non-Compression Mode. Follow [Installation Instructions ➶](../../gears/writegear/compression/advanced/ffmpeg_install/) for FFmpeg installation.

&nbsp;