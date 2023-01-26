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

# WriteGear API Usage Examples: Compression Mode


!!! warning "Important Information"

    * WriteGear **MUST** requires FFmpeg executables for its Compression capabilities in Compression Mode. Follow these dedicated [Installation Instructions ➶](../advanced/ffmpeg_install/) for its installation.

    * ==In case WriteGear API fails to detect valid FFmpeg executables on your system _(even if Compression Mode is enabled)_, it automatically fallbacks to [Non-Compression Mode](../../non_compression/overview/).==

    * **DO NOT** feed frames with different dimensions or channels to WriteGear, otherwise WriteGear will exit with `ValueError`.

    * While providing additional av-source with `-i` FFmpeg parameter in `output_params` make sure it don't interfere with WriteGear's frame pipeline otherwise it will break things!

    * Use [`-disable_force_termination`](../params/#supported-parameters) flag when video duration is too short(<60sec), otherwise WriteGear will not produce any valid output.

    * Heavy resolution multimedia files take time to render which can last up to _0.1-1 seconds_. Kindly wait till the WriteGear API terminates itself, and **DO NOT** try to kill the process instead.

    * Always use `writer.close()` at the very end of the main code. **NEVER USE IT INBETWEEN CODE** to avoid undesired behavior.


!!! example "After going through WriteGear Usage Examples, Checkout more bonus examples [here ➶](../../../help/writegear_ex/)"

&thinsp;


## Bare-Minimum Usage

Following is the bare-minimum code you need to get started with WriteGear API in Compression Mode:

```python
# import required libraries
from vidgear.gears import CamGear
from vidgear.gears import WriteGear
import cv2

# open any valid video stream(for e.g `myvideo.avi` file)
stream = CamGear(source="myvideo.avi").start()

# Define writer with default parameters and suitable output filename for e.g. `Output.mp4`
writer = WriteGear(output="Output.mp4")

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

In Compression Mode, WriteGear API contains [`rgb_mode`](../../../../bonus/reference/writegear/#vidgear.gears.writegear.WriteGear.write) boolean parameter for RGB Mode, which when enabled _(i.e. `rgb_mode=True`)_, specifies that incoming frames are of RGB format _(instead of default BGR format)_. This mode makes WriteGear directly compatible with libraries that only supports RGB format. 

The complete usage example is as follows:

```python hl_lines="26"
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import WriteGear
import cv2

# Open live video stream on webcam at first index(i.e. 0) device
stream = VideoGear(source=0).start()

# Define writer with default parameters and suitable output filename for e.g. `Output.mp4`
writer = WriteGear(output="Output.mp4")

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # simulating RGB frame for example
    frame_rgb = frame[:, :, ::-1]

    # writing RGB frame to writer
    writer.write(frame_rgb, rgb_mode=True)  # activate RGB Mode

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
    writer = WriteGear(output="out.mp4", logging =True, **output_params)
    ```

    But make sure you ==MUST set value of `-r` and `-input_framerate` parameter less than or equal to your input source framerate.==


In this code we will retrieve framerate from video stream, and set it as `-input_framerate` attribute for `option` parameter in WriteGear API:

```python hl_lines="10"
# import required libraries
from vidgear.gears import CamGear
from vidgear.gears import WriteGear
import cv2

# Open live video stream on webcam at first index(i.e. 0) device
stream = CamGear(source=0).start()

# retrieve framerate from CamGear Stream and pass it as `-input_framerate` parameter
output_params = {"-input_framerate": stream.framerate}

# Define writer with defined parameters and suitable output filename for e.g. `Output.mp4`
writer = WriteGear(output="Output.mp4", **output_params)

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


## Using Compression Mode for live streaming

In Compression Mode, WriteGear also allows URL strings _(as output)_ for live streaming realtime frames with its [`output`](../params/#output) parameter.  

In this example, we will stream live camera frames directly to Twitch :fontawesome-brands-twitch::

!!! tip "For streaming with traditional protocols such as :material-video-wireless: RTSP/RTP, Checkout this [WriteGear's Bonus Examples ➶](../../../../help/writegear_ex/#using-writegears-compression-mode-for-rtsprtp-live-streaming)."

!!! example ":fontawesome-brands-youtube: YouTube-Live Streaming example code also available in [WriteGear's Bonus Examples ➶](../../../../help/writegear_ex/#using-writegears-compression-mode-for-youtube-live-streaming)"

!!! warning "This example assume you already have a [**Twitch Account**](https://www.twitch.tv/) for publishing video."

!!! alert "Make sure to change [_Twitch Stream Key_](https://www.youtube.com/watch?v=xwOtOfPMIIk) with yours in following code before running!"

```python hl_lines="11-16 20 24"
# import required libraries
from vidgear.gears import CamGear
from vidgear.gears import WriteGear
import cv2

# Open live webcam video stream on first index(i.e. 0) device
stream = CamGear(source=0, logging=True).start()

# define required FFmpeg optimizing parameters for your writer
output_params = {
    "-preset:v": "veryfast",
    "-g": 60,
    "-keyint_min": 60,
    "-sc_threshold": 0,
    "-bufsize": "2500k",
    "-f": "flv",
}

# [WARNING] Change your Twitch Stream Key here:
TWITCH_KEY = "live_XXXXXXXXXX~XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

# Define writer with defined parameters and
writer = WriteGear(
    output="rtmp://live.twitch.tv/app/{}".format(TWITCH_KEY),
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


By default, WriteGear API uses `libx264` encoder for encoding output files in Compression Mode. But you can easily change encoder to your suitable [supported encoder](../params/#supported-encoders) by passing `-vcodec` FFmpeg parameter as an attribute with its [*output_param*](../params/#output_params) dictionary parameter. In addition to this, you can also specify the additional properties/features of your system's GPU :octicons-cpu-16: easily. 

??? warning "User Discretion Advised"

    This example is just conveying the idea on how to use FFmpeg's hardware encoders with WriteGear API in Compression mode, which **MAY/MAY NOT** suit your system. Kindly use suitable parameters based your system hardware settings only.


In this example, we will be using `h264_vaapi` as our hardware encoder and also optionally be specifying our device hardware's location (i.e. `'-vaapi_device':'/dev/dri/renderD128'`) and other features such as `'-vf':'format=nv12,hwupload'`:

??? alert "Remember to check VAAPI support"

    To use `h264_vaapi` encoder, remember to check if its available and your FFmpeg compiled with VAAPI support. You can easily do this by executing following one-liner command in your terminal, and observing if output contains something similar as follows:

    ```sh
    ffmpeg  -hide_banner -encoders | grep vaapi 

     V..... h264_vaapi           H.264/AVC (VAAPI) (codec h264)
     V..... hevc_vaapi           H.265/HEVC (VAAPI) (codec hevc)
     V..... mjpeg_vaapi          MJPEG (VAAPI) (codec mjpeg)
     V..... mpeg2_vaapi          MPEG-2 (VAAPI) (codec mpeg2video)
     V..... vp8_vaapi            VP8 (VAAPI) (codec vp8)
    ```


```python hl_lines="11-13"
# import required libraries
from vidgear.gears import CamGear
from vidgear.gears import WriteGear
import cv2

# Open live webcam video stream on first index(i.e. 0) device
stream = CamGear(source=0, logging=True).start()

# define required FFmpeg parameters for your writer
output_params = {
    "-vcodec": "h264_vaapi",
    "-vaapi_device": "/dev/dri/renderD128",
    "-vf": "format=nv12,hwupload",
}

# Define writer with defined parameters and suitable output filename for e.g. `Output.mp4`
writer = WriteGear(output="Output.mp4", **output_params)

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

```python hl_lines="6"
# import required libraries
from vidgear.gears import WriteGear
import cv2

# define suitable (Codec,CRF,preset) FFmpeg parameters for writer
output_params = {"-vcodec": "libx264", "-crf": 0, "-preset": "fast"}

# Open suitable video stream, such as webcam on first index(i.e. 0)
stream = cv2.VideoCapture(0)

# Define writer with defined parameters and suitable output filename for e.g. `Output.mp4`
writer = WriteGear(output="Output.mp4", logging=True, **output_params)

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

## Using Compression Mode with Live Audio Input

In Compression Mode, WriteGear API allows us to exploit almost all FFmpeg supported parameters that you can think of in its Compression Mode. Hence, combining audio with live video frames is pretty easy. 

In this example code, we will merging the audio from a Audio Device _(for e.g. Webcam inbuilt mic)_ to live frames incoming from the Video Source _(for e.g external webcam)_, and save the output as a compressed video file, all in real time:

!!! alert "Example Assumptions"

    * You're running are Linux machine.
    * You already have appropriate audio driver and software installed on your machine.


??? tip "Identifying and Specifying sound card on different OS platforms"
    
    === ":fontawesome-brands-windows: Windows"

        Windows OS users can use the [dshow](https://trac.ffmpeg.org/wiki/DirectShow) (DirectShow) to list audio input device which is the preferred option for Windows users. You can refer following steps to identify and specify your sound card:

        - [x] **[OPTIONAL] Enable sound card(if disabled):** First enable your Stereo Mix by opening the "Sound" window and select the "Recording" tab, then right click on the window and select "Show Disabled Devices" to toggle the Stereo Mix device visibility. **Follow this [post ➶](https://forums.tomshardware.com/threads/no-sound-through-stereo-mix-realtek-hd-audio.1716182/) for more details.**

        - [x] **Identify Sound Card:** Then, You can locate your soundcard using `dshow` as follows:

            ```sh
            c:\> ffmpeg -list_devices true -f dshow -i dummy
            ffmpeg version N-45279-g6b86dd5... --enable-runtime-cpudetect
              libavutil      51. 74.100 / 51. 74.100
              libavcodec     54. 65.100 / 54. 65.100
              libavformat    54. 31.100 / 54. 31.100
              libavdevice    54.  3.100 / 54.  3.100
              libavfilter     3. 19.102 /  3. 19.102
              libswscale      2.  1.101 /  2.  1.101
              libswresample   0. 16.100 /  0. 16.100
            [dshow @ 03ACF580] DirectShow video devices
            [dshow @ 03ACF580]  "Integrated Camera"
            [dshow @ 03ACF580]  "USB2.0 Camera"
            [dshow @ 03ACF580] DirectShow audio devices
            [dshow @ 03ACF580]  "Microphone (Realtek High Definition Audio)"
            [dshow @ 03ACF580]  "Microphone (USB2.0 Camera)"
            dummy: Immediate exit requested
            ```


        - [x] **Specify Sound Card:** Then, you can specify your located soundcard in WriteGear as follows:

            ```python
            # assign appropriate input audio-source
            output_params = {
                "-f": "dshow", # !!! warning: always keep this line above "-i" parameter !!!
                "-i":"audio=Microphone (USB2.0 Camera)",
                "-thread_queue_size": "512",
                "-ac": "2",
                "-acodec": "aac",
                "-ar": "44100",
            }
            ```

        !!! fail "If audio still doesn't work then [checkout this troubleshooting guide ➶](https://www.maketecheasier.com/fix-microphone-not-working-windows10/) or reach us out on [Gitter ➶](https://gitter.im/vidgear/community) Community channel"


    === ":material-linux: Linux"

        Linux OS users can use the [alsa](https://ffmpeg.org/ffmpeg-all.html#alsa) to list input device to capture live audio input such as from a webcam. You can refer following steps to identify and specify your sound card:

        - [x] **Identify Sound Card:** To get the list of all installed cards on your machine, you can type `arecord -l` or `arecord -L` _(longer output)_.

            ```sh
            arecord -l

            **** List of CAPTURE Hardware Devices ****
            card 0: ICH5 [Intel ICH5], device 0: Intel ICH [Intel ICH5]
              Subdevices: 1/1
              Subdevice #0: subdevice #0
            card 0: ICH5 [Intel ICH5], device 1: Intel ICH - MIC ADC [Intel ICH5 - MIC ADC]
              Subdevices: 1/1
              Subdevice #0: subdevice #0
            card 0: ICH5 [Intel ICH5], device 2: Intel ICH - MIC2 ADC [Intel ICH5 - MIC2 ADC]
              Subdevices: 1/1
              Subdevice #0: subdevice #0
            card 0: ICH5 [Intel ICH5], device 3: Intel ICH - ADC2 [Intel ICH5 - ADC2]
              Subdevices: 1/1
              Subdevice #0: subdevice #0
            card 1: U0x46d0x809 [USB Device 0x46d:0x809], device 0: USB Audio [USB Audio]
              Subdevices: 1/1
              Subdevice #0: subdevice #0
            ```


        - [x] **Specify Sound Card:** Then, you can specify your located soundcard in WriteGear as follows:

            !!! info "The easiest thing to do is to reference sound card directly, namely "card 0" (Intel ICH5) and "card 1" (Microphone on the USB web cam), as `hw:0` or `hw:1`"

            ```python
            # assign appropriate input audio-source
            output_params = {
                "-thread_queue_size": "512",
                "-ac": "2",
                "-ar": "48000",
                "-f": "alsa", # !!! warning: always keep this line above "-i" parameter !!!
                "-i": "hw:1",
            }
            ```

        !!! fail "If audio still doesn't work then reach us out on [Gitter ➶](https://gitter.im/vidgear/community) Community channel"


    === ":material-apple: MacOS"

        MAC OS users can use the [avfoundation](https://ffmpeg.org/ffmpeg-devices.html#avfoundation) to list input devices for grabbing audio from integrated iSight cameras as well as cameras connected via USB or FireWire. You can refer following steps to identify and specify your sound card on MacOS/OSX machines:


        - [x] **Identify Sound Card:** Then, You can locate your soundcard using `avfoundation` as follows:

            ```sh
            ffmpeg -f avfoundation -list_devices true -i ""
            ffmpeg version N-45279-g6b86dd5... --enable-runtime-cpudetect
              libavutil      51. 74.100 / 51. 74.100
              libavcodec     54. 65.100 / 54. 65.100
              libavformat    54. 31.100 / 54. 31.100
              libavdevice    54.  3.100 / 54.  3.100
              libavfilter     3. 19.102 /  3. 19.102
              libswscale      2.  1.101 /  2.  1.101
              libswresample   0. 16.100 /  0. 16.100
            [AVFoundation input device @ 0x7f8e2540ef20] AVFoundation video devices:
            [AVFoundation input device @ 0x7f8e2540ef20] [0] FaceTime HD camera (built-in)
            [AVFoundation input device @ 0x7f8e2540ef20] [1] Capture screen 0
            [AVFoundation input device @ 0x7f8e2540ef20] AVFoundation audio devices:
            [AVFoundation input device @ 0x7f8e2540ef20] [0] Blackmagic Audio
            [AVFoundation input device @ 0x7f8e2540ef20] [1] Built-in Microphone
            ```


        - [x] **Specify Sound Card:** Then, you can specify your located soundcard in WriteGear as follows:

            ```python
            # assign appropriate input audio-source
            output_params = {
                "-thread_queue_size": "512",
                "-ac": "2",
                "-ar": "48000",
                "-f": "avfoundation", # !!! warning: always keep this line above "-audio_device_index" parameter !!!
                "-audio_device_index": "0",
            }
            ```

        !!! fail "If audio still doesn't work then reach us out on [Gitter ➶](https://gitter.im/vidgear/community) Community channel"


!!! danger "Make sure this `-i` audio-source it compatible with provided video-source, otherwise you could encounter multiple errors or no output at all."

!!! warning "You **MUST** use [`-input_framerate`](../params/#supported-parameters) attribute to set exact value of input framerate when using external audio in Real-time Frames mode, otherwise audio delay will occur in output streams."

```python hl_lines="11-15"
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import WriteGear
import cv2

# Open live video stream on webcam at first index(i.e. 0) device
stream = VideoGear(source=0).start()

# change with your webcam soundcard, plus add additional required FFmpeg parameters for your writer
output_params = {
    "-input_framerate": stream.framerate,
    "-thread_queue_size": "512",
    "-ac": "2",
    "-ar": "48000",
    "-f": "alsa", # !!! warning: always keep this line above "-i" parameter !!!
    "-i": "hw:1",
}

# Define writer with defined parameters and suitable output filename for e.g. `Output.mp4
writer = WriteGear(output="Output.mp4", logging=True, **output_params)

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