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

# WriteGear API Usage Examples: Compression Mode


!!! warning "Important Information"

    * WriteGear **MUST** requires FFmpeg executables for its Compression capabilities in Compression Mode. Follow these dedicated [Installation Instructions ➶](../advanced/ffmpeg_install/) for its installation.

    * ==In case WriteGear API fails to detect valid FFmpeg executables on your system _(even if Compression Mode is enabled)_, it automatically fallbacks to [Non-Compression Mode](../../non_compression/overview/).==

    * **DO NOT** feed frames with different dimensions or channels to WriteGear, otherwise WriteGear will exit with `ValueError`.

    * **DO NOT** provide additional video-source with `-i` FFmpeg parameter in [`output_params`](../params/#output_params), otherwise it will interfere with frame you input later, and it will break things!

    * Heavy resolution multimedia files take time to render which can last up to _~.1-to-1 seconds_. Kindly wait till the WriteGear API terminates itself, and **DO NOT** try to kill the process instead.

    * Always use `writer.close()` at the very end of the main code. **NEVER USE IT INBETWEEN CODE** to avoid undesired behavior.



## Bare-Minimum Usage

Following is the bare-minimum code you need to get started with WriteGear API in Compression Mode:

```python
# import required libraries
from vidgear.gears import CamGear
from vidgear.gears import WriteGear
import cv2

# open any valid video stream(for e.g `myvideo.avi` file)
stream = CamGear(source='myvideo.avi').start() 

# Define writer with default parameters and suitable output filename for e.g. `Output.mp4`
writer = WriteGear(output_filename = 'Output.mp4') 

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

## Using Compression Mode in RGB Mode

For Compression Mode, WriteGear API contains [`rgb_mode`](../../../../bonus/reference/writegear/#vidgear.gears.writegear.WriteGear.write) boolean parameter, which if enabled _(i.e. `rgb_mode=True`)_, specifies that incoming frames are of RGB format _(instead of default BGR format)_, thereby also known as ==RGB Mode==. The complete usage example is as follows:

```python
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import WriteGear
import cv2

# Open live video stream on webcam at first index(i.e. 0) device
stream = VideoGear(source=0).start()

# Define writer with default parameters and suitable output filename for e.g. `Output.mp4`
writer = WriteGear(output_filename = 'Output.mp4') 

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break


    # simulating RGB frame for example
    frame_rgb = frame[:,:,::-1]


    # writing RGB frame to writer
    writer.write(frame_rgb, rgb_mode = True) #activate RGB Mode

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

## Using Compression Mode with controlled FrameRate

WriteGear API provides [`-input_framerate`](../params/#supported-parameters)  attribute for its `options` dictionary parameter in Compression Mode, which allow us to control/set the constant framerate of the output video. 

??? tip "Advanced Tip for setting constant framerate"

    If `-input_framerate` attribute doesn't works for you, then define it in conjunction with another `-r` FFmpeg parameter as attribute:

    ```python
    # set output constant framerate to (say 60 fps)
    output_params = {"-input_framerate":60, "-r":60}
    # assign that to WriteGear
    writer = WriteGear(output_filename="out.mp4", logging =True, **output_params)
    ```

    But make sure you ==MUST set value of `-r` and `-input_framerate` parameter less than or equal to your input source framerate.==


In this code we will retrieve framerate from video stream, and set it as `-input_framerate` attribute for `option` parameter in WriteGear API:

```python
# import required libraries
from vidgear.gears import CamGear
from vidgear.gears import WriteGear
import cv2

# Open live video stream on webcam at first index(i.e. 0) device
stream = CamGear(source=0).start()

# retrieve framerate from CamGear Stream and pass it as `-input_framerate` parameter
output_params = {"-input_framerate":stream.framerate}

# Define writer with defined parameters and suitable output filename for e.g. `Output.mp4`
writer = WriteGear(output_filename = 'Output.mp4', **output_params)

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

## Using Compression Mode with Hardware encoders


By default, WriteGear API uses *libx264/libx265 encoders* for encoding its output files in Compression Mode. But you can easily change encoder to your suitable [supported encoder](../params/#supported-encoders) by passing `-vcodec` FFmpeg parameter as an attribute in its [*output_param*](../params/#output_params) dictionary parameter. In addition to this, you can also specify the additional properties/features of your system's GPU easily. 

??? warning "User Discretion Advised"

    This example is just conveying the idea on how to use FFmpeg's hardware encoders with WriteGear API in Compression mode, which **MAY/MAY NOT** suit your system. Kindly use suitable parameters based your supported system and FFmpeg configurations only.


In this example, we will be using `h264_vaapi` as our hardware encoder and also optionally be specifying our device hardware's location (i.e. `'-vaapi_device':'/dev/dri/renderD128'`) and other features such as `'-vf':'format=nv12,hwupload'` like properties by formatting them as `option` dictionary parameter's attributes, as follows:

!!! danger "Check VAAPI support"

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
from vidgear.gears import CamGear
from vidgear.gears import WriteGear
import cv2

# Open live webcam video stream on first index(i.e. 0) device
stream = CamGear(source=0, logging=True).start() 

# define required FFmpeg parameters for your writer
output_params = {'-vcodec': 'h264_vaapi', '-vaapi_device':'/dev/dri/renderD128', '-vf':'format=nv12,hwupload'} 

# Define writer with defined parameters and suitable output filename for e.g. `Output.mp4`
writer = WriteGear(output_filename = 'Output.mp4', **output_params)

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

## Using Compression Mode with OpenCV

You can easily use WriterGear API directly with any Video Processing library(_For e.g OpenCV itself_) in Compression Mode. The complete usage example is as follows:

```python
# import required libraries
from vidgear.gears import WriteGear
import cv2

# define suitable (Codec,CRF,preset) FFmpeg parameters for writer
output_params = {"-vcodec":"libx264", "-crf": 0, "-preset": "fast"}

# Open suitable video stream, such as webcam on first index(i.e. 0)
stream = cv2.VideoCapture(0) 

# Define writer with defined parameters and suitable output filename for e.g. `Output.mp4`
writer = WriteGear(output_filename = 'Output.mp4', logging = True, **output_params)

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

    # write gray frame to writer
    writer.write(gray)

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

# safely close writer
writer.close()
```

&nbsp; 

## Using Compression Mode with VideoCapture Gears

WriteGear API can be used in conjunction with any other Gear effortlessly in Compression Mode. The complete usage example is as follows:

```python
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import WriteGear
import cv2


# define suitable tweak parameters for your stream.
options = {"CAP_PROP_FRAME_WIDTH ":320, "CAP_PROP_FRAME_HEIGHT":240, "CAP_PROP_FPS ":60}

# define suitable (Codec,CRF,preset) FFmpeg parameters for writer
output_params = {"-vcodec":"libx264", "-crf": 0, "-preset": "fast"}

# open live video stream on webcam at first index(i.e. 0) device and apply source tweak parameters
stream = VideoGear(source=0, logging=True, **options).start()

# Define writer with defined parameters and suitable output filename for e.g. `Output.mp4`
writer = WriteGear(output_filename = 'Output.mp4', logging = True, **output_params)


# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break


    # {do something with the frame here}
    # lets convert frame to gray for this example
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # write gray frame to writer
    writer.write(gray)

    # Show output window
    cv2.imshow("Output Gray Frame", gray)

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

## Using Compression Mode with Live Audio Input

In Compression Mode, WriteGear API allows us to exploit almost all FFmpeg supported parameters that you can think of, in its Compression Mode. Hence, processing, encoding, and combining audio with video is pretty much straightforward.

!!! warning "Example Assumptions"

    * You're running are Linux machine.
    * You already have appropriate audio & video drivers and softwares installed on your machine.

!!! danger "Locate your Sound Card"

    Remember to locate your Sound Card before running this example:

    * Note down the Sound Card value using `arecord -L` command on the your Linux terminal. 
    * It may be similar to this  `plughw:CARD=CAMERA,DEV=0`

??? tips

    The useful audio input options for ALSA input are `-ar` (_audio sample rate_) and `-ac` (_audio channels_). Specifying audio sampling rate/frequency will force the audio card to record the audio at that specified rate. Usually the default value is `"44100"` (Hz) but `"48000"`(Hz) works, so chose wisely. Specifying audio channels will force the audio card to record the audio as mono, stereo or even 2.1, and 5.1(_if supported by your audio card_). Usually the default value is `"1"` (mono) for Mic input and `"2"` (stereo) for Line-In input. Kindly go through [FFmpeg docs](https://ffmpeg.org/ffmpeg.html) for more of such options.


In this example code, we will merge the audio from a Audio Source _(for e.g. Webcam inbuilt mic)_ to the frames of a Video Source _(for e.g external webcam)_, and save this data as a compressed video file, all in real time:

```python
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import WriteGear
import cv2

# Open live video stream on webcam at first index(i.e. 0) device
stream = VideoGear(source=0).start()

# change with your webcam soundcard, plus add additional required FFmpeg parameters for your writer
output_params = {'-thread_queue_size': '512', '-f': 'alsa', '-ac': '1', '-ar': '48000', '-i': 'plughw:CARD=CAMERA,DEV=0'}  

# Define writer with defined parameters and suitable output filename for e.g. `Output.mp4
writer = WriteGear(output_filename = 'Output.mp4', logging = True, **output_params)

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