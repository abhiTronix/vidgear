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

# Stabilizer Class Examples

&nbsp;

## Saving Stabilizer Class output with Live Audio Input

In this example code, we will merging the audio from a Audio Device _(for e.g. Webcam inbuilt mic input)_ with Stabilized frames incoming from the Stabilizer Class _(which is also using same Webcam video input through OpenCV)_, and save the final output as a compressed video file, all in real time:

??? new "New in v0.2.2" 
    This example was added in `v0.2.2`.

!!! alert "Example Assumptions"

    * You're running are Linux machine.
    * You already have appropriate audio driver and software installed on your machine.


??? tip "Identifying and Specifying sound card on different OS platforms"
    
    === "On Windows"

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


        - [x] **Specify Sound Card:** Then, you can specify your located soundcard in StreamGear as follows:

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


    === "On Linux"

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


    === "On MacOS"

        MAC OS users can use the [avfoundation](https://ffmpeg.org/ffmpeg-devices.html#avfoundation) to list input devices for grabbing audio from integrated iSight cameras as well as cameras connected via USB or FireWire. You can refer following steps to identify and specify your sound card on MacOS/OSX machines:


        - [x] **Identify Sound Card:** Then, You can locate your soundcard using `avfoundation` as follows:

            ```sh
            ffmpeg -f qtkit -list_devices true -i ""
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


        - [x] **Specify Sound Card:** Then, you can specify your located soundcard in StreamGear as follows:

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

!!! warning "You **MUST** use [`-input_framerate`](../../gears/writegear/compression/params/#supported-parameters) attribute to set exact value of input framerate when using external audio in Real-time Frames mode, otherwise audio delay will occur in output streams."

```python
# import required libraries
from vidgear.gears import WriteGear
from vidgear.gears.stabilizer import Stabilizer
import cv2

# Open suitable video stream, such as webcam on first index(i.e. 0)
stream = cv2.VideoCapture(0)

# initiate stabilizer object with defined parameters
stab = Stabilizer(smoothing_radius=30, crop_n_zoom=True, border_size=5, logging=True)

# change with your webcam soundcard, plus add additional required FFmpeg parameters for your writer
output_params = {
    "-input_framerate": stream.get(cv2.CAP_PROP_FPS),
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
    (grabbed, frame) = stream.read()

    # check for frame if not grabbed
    if not grabbed:
        break

    # send current frame to stabilizer for processing
    stabilized_frame = stab.stabilize(frame)

    # wait for stabilizer which still be initializing
    if stabilized_frame is None:
        continue

    # {do something with the stabilized frame here}

    # write stabilized frame to writer
    writer.write(stabilized_frame)


# clear stabilizer resources
stab.clean()

# safely close video stream
stream.release()

# safely close writer
writer.close()
```

&nbsp;

## Saving Stabilizer Class output with File Audio Input

In this example code, we will be directly merging the audio from a Video-File _(to be stabilized)_ with its processed stabilized frames into a compressed video output in real time:

??? new "New in v0.2.4" 
    This example was added in `v0.2.4`.

!!! danger "Make sure this input video-file _(to be stabilized)_ contains valid audio source, otherwise you could encounter multiple errors or no output at all."

!!! warning "You **MUST** use [`-input_framerate`](../../gears/writegear/compression/params/#supported-parameters) attribute to set exact value of input framerate when using external audio in Real-time Frames mode, otherwise audio delay will occur in output streams."

!!! alert "Use [`-disable_force_termination`](../../gears/writegear/compression/params/#supported-parameters) flag when video duration is too short(<60sec), otherwise WriteGear will not produce any valid output."

```python
# import required libraries
from vidgear.gears import WriteGear
from vidgear.gears.stabilizer import Stabilizer
import cv2

# Give suitable video file path to be stabilized
unstabilized_videofile = "test.mp4"

# open stream on given path
stream = cv2.VideoCapture(unstabilized_videofile)

# initiate stabilizer object with defined parameters
stab = Stabilizer(smoothing_radius=30, crop_n_zoom=True, border_size=5, logging=True)

# define required FFmpeg optimizing parameters for your writer
output_params = {
    "-i": unstabilized_videofile,
    "-c:a": "aac",
    "-input_framerate": stream.get(cv2.CAP_PROP_FPS),
    "-clones": ["-shortest"],
    # !!! Uncomment following line if video duration is too short(<60sec). !!!
    #"-disable_force_termination": True,
}


# Define writer with defined parameters and suitable output filename for e.g. `Output.mp4
writer = WriteGear(output="Output.mp4", logging=True, **output_params)

# loop over
while True:

    # read frames from stream
    (grabbed, frame) = stream.read()

    # check for frame if not grabbed
    if not grabbed:
        break

    # send current frame to stabilizer for processing
    stabilized_frame = stab.stabilize(frame)

    # wait for stabilizer which still be initializing
    if stabilized_frame is None:
        continue

    # {do something with the stabilized frame here}

    # write stabilized frame to writer
    writer.write(stabilized_frame)


# clear stabilizer resources
stab.clean()

# safely close video stream
stream.release()

# safely close writer
writer.close()
```

&nbsp;