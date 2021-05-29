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

# StreamGear API Usage Examples: Real-time Frames Mode


!!! warning "Important Information"
    
    * StreamGear **MUST** requires FFmpeg executables for its core operations. Follow these dedicated [Platform specific Installation Instructions ➶](../../ffmpeg_install/) for its installation.

    * StreamGear API will throw **RuntimeError**, if it fails to detect valid FFmpeg executables on your system.

    * By default, when no additional streams are defined, ==StreamGear generates a primary stream of same resolution and framerate[^1] as the input video  _(at the index `0`)_.==

    * Always use `terminate()` function at the very end of the main code.


&thinsp;


## Bare-Minimum Usage

Following is the bare-minimum code you need to get started with StreamGear API in Real-time Frames Mode:

!!! note "We are using [CamGear](../../../camgear/overview/) in this Bare-Minimum example, but any [VideoCapture Gear](../../../#a-videocapture-gears) will work in the similar manner."

```python
# import required libraries
from vidgear.gears import CamGear
from vidgear.gears import StreamGear
import cv2

# open any valid video stream(for e.g `foo1.mp4` file)
stream = CamGear(source='foo1.mp4').start() 

# describe a suitable manifest-file location/name
streamer = StreamGear(output="dash_out.mpd")

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break


    # {do something with the frame here}


    # send frame to streamer
    streamer.stream(frame)

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

# safely close streamer
streamer.terminate()
```

!!! success "After running these bare-minimum commands, StreamGear will produce a Manifest file _(`dash.mpd`)_ with steamable chunks that contains information about a Primary Stream of same resolution and framerate[^1] as input _(without any audio)_."


&thinsp;

## Bare-Minimum Usage with Live-Streaming

You can easily activate ==Low-latency Livestreaming in Real-time Frames Mode==, where chunks will contain information for few new frames only and forgets all previous ones), using exclusive [`-livestream`](../../params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter as follows:

!!! tip "Use `-window_size` & `-extra_window_size` FFmpeg parameters for controlling number of frames to be kept in Chunks. Less these value, less will be latency."

!!! warning "All Chunks will be overwritten in this mode after every few Chunks _(equal to the sum of `-window_size` & `-extra_window_size` values)_, Hence Newer Chunks and Manifest contains NO information of any older video-frames."

!!! note "In this mode, StreamGear **DOES NOT** automatically maps video-source audio to generated streams. You need to manually assign separate audio-source through [`-audio`](../../params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter."

```python
# import required libraries
from vidgear.gears import CamGear
from vidgear.gears import StreamGear
import cv2

# open any valid video stream(from web-camera attached at index `0`)
stream = CamGear(source=0).start()

# enable livestreaming and retrieve framerate from CamGear Stream and
# pass it as `-input_framerate` parameter for controlled framerate
stream_params = {"-input_framerate": stream.framerate, "-livestream": True}

# describe a suitable manifest-file location/name
streamer = StreamGear(output="dash_out.mpd", **stream_params)

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # send frame to streamer
    streamer.stream(frame)

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

# safely close streamer
streamer.terminate()
```

&thinsp;

## Bare-Minimum Usage with RGB Mode

In Real-time Frames Mode, StreamGear API provide [`rgb_mode`](../../../../../bonus/reference/streamgear/#vidgear.gears.streamgear.StreamGear.stream) boolean parameter with its `stream()` function, which if enabled _(i.e. `rgb_mode=True`)_, specifies that incoming frames are of RGB format _(instead of default BGR format)_, thereby also known as ==RGB Mode==. The complete usage example is as follows:

```python
# import required libraries
from vidgear.gears import CamGear
from vidgear.gears import StreamGear
import cv2

# open any valid video stream(for e.g `foo1.mp4` file)
stream = CamGear(source='foo1.mp4').start() 

# describe a suitable manifest-file location/name
streamer = StreamGear(output="dash_out.mpd")

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break


    # {simulating RGB frame for this example}
    frame_rgb = frame[:,:,::-1]


    # send frame to streamer
    streamer.stream(frame_rgb, rgb_mode = True) #activate RGB Mode

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

# safely close streamer
streamer.terminate()
```

&thinsp;

## Bare-Minimum Usage with controlled Input-framerate

In Real-time Frames Mode, StreamGear API provides exclusive [`-input_framerate`](../../params/#a-exclusive-parameters)  attribute for its `stream_params` dictionary parameter, that allow us to set the assumed constant framerate for incoming frames. In this example, we will retrieve framerate from webcam video-stream, and set it as value for `-input_framerate` attribute in StreamGear:

!!! danger "Remember, Input framerate default to `25.0` fps if [`-input_framerate`](../../params/#a-exclusive-parameters) attribute value not defined in Real-time Frames mode."

```python
# import required libraries
from vidgear.gears import CamGear
from vidgear.gears import StreamGear
import cv2

# Open live video stream on webcam at first index(i.e. 0) device
stream = CamGear(source=0).start()

# retrieve framerate from CamGear Stream and pass it as `-input_framerate` value
stream_params = {"-input_framerate":stream.framerate}

# describe a suitable manifest-file location/name and assign params
streamer = StreamGear(output="dash_out.mpd", **stream_params)

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break


    # {do something with the frame here}


    # send frame to streamer
    streamer.stream(frame)

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

# safely close streamer
streamer.terminate()
```

&thinsp;

## Bare-Minimum Usage with OpenCV

You can easily use StreamGear API directly with any other Video Processing library(_For e.g. [OpenCV](https://github.com/opencv/opencv) itself_) in Real-time Frames Mode. The complete usage example is as follows:

!!! tip "This just a bare-minimum example with OpenCV, but any other Real-time Frames Mode feature/example will work in the similar manner."

```python
# import required libraries
from vidgear.gears import StreamGear
import cv2

# Open suitable video stream, such as webcam on first index(i.e. 0)
stream = cv2.VideoCapture(0) 

# describe a suitable manifest-file location/name
streamer = StreamGear(output="dash_out.mpd")

# loop over
while True:

    # read frames from stream
    (grabbed, frame) = stream.read()

    # check for frame if not grabbed
    if not grabbed:
      break

    # {do something with the frame here}
    # lets convert frame to gray for this example
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # send frame to streamer
    streamer.stream(gray)

    # Show output window
    cv2.imshow("Output Gray Frame", gray)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.release()

# safely close streamer
streamer.terminate()
```

&thinsp;

## Usage with Additional Streams

Similar to Single-Source Mode, you can easily generate any number of additional Secondary Streams of variable bitrates or spatial resolutions, using exclusive [`-streams`](../../params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter _(More detailed information can be found [here ➶](../../params/#a-exclusive-parameters))_ in Real-time Frames Mode. The complete example is as follows:

!!! danger "Important `-streams` attribute Information"
    * On top of these additional streams, StreamGear by default, generates a primary stream of same resolution and framerate[^1] as the input, at the index `0`.
    * :warning: Make sure your System/Machine/Server/Network is able to handle these additional streams, discretion is advised! 
    * You **MUST** need to define `-resolution` value for your stream, otherwise stream will be discarded!
    * You only need either of `-video_bitrate` or `-framerate` for defining a valid stream. Since with `-framerate` value defined, video-bitrate is calculated automatically.
    * If you define both `-video_bitrate` and `-framerate` values at the same time, StreamGear will discard the `-framerate` value automatically.

!!! fail "Always use `-stream` attribute to define additional streams safely, any duplicate or incorrect definition can break things!"

```python
# import required libraries
from vidgear.gears import CamGear
from vidgear.gears import StreamGear
import cv2

# Open suitable video stream, such as webcam on first index(i.e. 0)
stream = CamGear(source=0).start() 

# define various streams
stream_params = {
    "-streams": [
        {"-resolution": "1280x720", "-framerate": 30.0},  # Stream2: 1280x720 at 30fps framerate
        {"-resolution": "640x360", "-framerate": 60.0},  # Stream3: 640x360 at 60fps framerate
        {"-resolution": "320x240", "-video_bitrate": "500k"},  # Stream3: 320x240 at 500kbs bitrate
    ],
}

# describe a suitable manifest-file location/name and assign params
streamer = StreamGear(output="dash_out.mpd")

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break


    # {do something with the frame here}


    # send frame to streamer
    streamer.stream(frame)

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

# safely close streamer
streamer.terminate()
```

&thinsp;

## Usage with Audio-Input

In Real-time Frames Mode, if you want to add audio to your streams, you've to use exclusive [`-audio`](../../params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter. You need to input the path of your audio to this attribute as string value, and StreamGear API will automatically validate and map it to all generated streams. The complete example is as follows:

!!! failure "Make sure this `-audio` audio-source it compatible with provided video-source, otherwise you encounter multiple errors or no output at all."

!!! warning "You **MUST** use [`-input_framerate`](../../params/#a-exclusive-parameters) attribute to set exact value of input framerate when using external audio in Real-time Frames mode, otherwise audio delay will occur in output streams."

!!! tip "You can also assign a valid Audio URL as input, rather than filepath. More details can be found [here ➶](../../params/#a-exclusive-parameters)"

```python
# import required libraries
from vidgear.gears import CamGear
from vidgear.gears import StreamGear
import cv2

# open any valid video stream(for e.g `foo1.mp4` file)
stream = CamGear(source='foo1.mp4').start() 

# add various streams, along with custom audio
stream_params = {
    "-streams": [
        {"-resolution": "1920x1080", "-video_bitrate": "4000k"},  # Stream1: 1920x1080 at 4000kbs bitrate
        {"-resolution": "1280x720", "-framerate": 30.0},  # Stream2: 1280x720 at 30fps
        {"-resolution": "640x360", "-framerate": 60.0},  # Stream3: 640x360 at 60fps
    ],
    "-input_framerate": stream.framerate, # controlled framerate for audio-video sync !!! don't forget this line !!!
    "-audio": "/home/foo/foo1.aac" # assigns input audio-source: "/home/foo/foo1.aac"
}

# describe a suitable manifest-file location/name and assign params
streamer = StreamGear(output="dash_out.mpd", **stream_params)

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break


    # {do something with the frame here}


    # send frame to streamer
    streamer.stream(frame)

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

# safely close streamer
streamer.terminate()
```

&thinsp;

## Usage with Hardware Video-Encoder


In Real-time Frames Mode, you can also easily change encoder as per your requirement just by passing `-vcodec` FFmpeg parameter as an attribute in `stream_params` dictionary parameter. In addition to this, you can also specify the additional properties/features/optimizations for your system's GPU similarly. 

In this example, we will be using `h264_vaapi` as our hardware encoder and also optionally be specifying our device hardware's location (i.e. `'-vaapi_device':'/dev/dri/renderD128'`) and other features such as `'-vf':'format=nv12,hwupload'` like properties by formatting them as `option` dictionary parameter's attributes, as follows:

!!! warning "Check VAAPI support"

    **This example is just conveying the idea on how to use FFmpeg's hardware encoders with WriteGear API in Compression mode, which MAY/MAY-NOT suit your system. Kindly use suitable parameters based your supported system and FFmpeg configurations only.**

    To use `h264_vaapi` encoder, remember to check if its available and your FFmpeg compiled with VAAPI support. You can easily do this by executing following one-liner command in your terminal, and observing if output contains something similar as follows:

    ```sh
    ffmpeg  -hide_banner -encoders | grep vaapi 

     V..... h264_vaapi           H.264/AVC (VAAPI) (codec h264)
     V..... hevc_vaapi           H.265/HEVC (VAAPI) (codec hevc)
     V..... mjpeg_vaapi          MJPEG (VAAPI) (codec mjpeg)
     V..... mpeg2_vaapi          MPEG-2 (VAAPI) (codec mpeg2video)
     V..... vp8_vaapi            VP8 (VAAPI) (codec vp8)
    ```


```python
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import StreamGear
import cv2

# Open suitable video stream, such as webcam on first index(i.e. 0)
stream = VideoGear(source=0).start() 

# add various streams with custom Video Encoder and optimizations
stream_params = {
    "-streams": [
        {"-resolution": "1920x1080", "-video_bitrate": "4000k"},  # Stream1: 1920x1080 at 4000kbs bitrate
        {"-resolution": "1280x720", "-framerate": 30.0},  # Stream2: 1280x720 at 30fps
        {"-resolution": "640x360", "-framerate": 60.0},  # Stream3: 640x360 at 60fps
    ],
    "-vcodec": "h264_vaapi", # define custom Video encoder
    "-vaapi_device": "/dev/dri/renderD128", # define device location
    "-vf": "format=nv12,hwupload",  # define video pixformat
}

# describe a suitable manifest-file location/name and assign params
streamer = StreamGear(output="dash_out.mpd", **stream_params)

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break


    # {do something with the frame here}


    # send frame to streamer
    streamer.stream(frame)

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

# safely close streamer
streamer.terminate()
```

&nbsp;

[^1]: 
    :bulb: In Real-time Frames Mode, the Primary Stream's framerate defaults to [`-input_framerate`](../../params/#a-exclusive-parameters) attribute value, if defined, else it will be 25fps.