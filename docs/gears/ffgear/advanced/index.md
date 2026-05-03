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

# FFGear API Advanced Usage

> This page covers FFGear's advanced configurations powered by the full FFdecoder feature-sets.

!!! tip "Check out all FFdecoder API's advanced recipes [here ➶](https://abhitronix.github.io/deffcode/latest/recipes/advanced/) to better understand these advanced usage examples."


!!! warning "FFGear requires the `deffcode` library"

    FFGear API **MUST** have the [`deffcode`][deffcode] library installed, along with a valid FFmpeg executable. Any failure in detection will raise `ImportError`/`RuntimeError` immediately.

    Install via pip:

    ```sh
    pip install deffcode
    ```

    For FFmpeg installation, see [FFmpeg Installation ➶](../ffmpeg_install/)


&thinsp;

## Using FFGear with Camera Devices (Custom Demuxer)

???+ alert "Example Assumptions"

    FFmpeg provide set of specific Demuxers on different platforms to read the multimedia streams from a particular type of Video Capture source/device. Please note that following recipe explicitly assumes: 

    - You're running Linux Machine with USB webcam connected to it at node/path `/dev/video0`. 
    - You already have appropriate Linux video drivers and related softwares installed on your machine.
    - You machine uses FFmpeg binaries built with `--enable-libv4l2` flag to support `video4linux2, v4l2` demuxer. BTW, you can list all supported demuxers using the `#!sh ffmpeg --list-demuxers` terminal command.

    These assumptions **MAY/MAY NOT** suit your current setup. Kindly use suitable parameters based your system platform and hardware settings only.

In this example we output **BGR24** video frames from a USB webcam device connected at path `/dev/video0` on a Linux Machine with `video4linux2` _(or simply `v4l2`)_ demuxer:

??? tip "Identifying and Specifying Video Capture Device Name/Path/Index and suitable Demuxer on different OS platforms"

    === ":fontawesome-brands-windows: Windows"

        Windows OS users can use the [dshow](https://trac.ffmpeg.org/wiki/DirectShow) (DirectShow) to list video input device which is the preferred option for Windows users. You can refer following steps to identify and specify your input video device's name:

        - [x] **Identify Video Devices:** You can locate your video device's name _(already connected to your system)_ using `dshow` as follows:

            ```sh
            c:\> ffmpeg.exe -list_devices true -f dshow -i dummy
            
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


        - [x] **Specify Video Device's name:** Then, you can specify and initialize your located Video device's name in FFGear API as follows:

            ```python
            # initialize and formulate the stream with "USB2.0 Camera" source for BGR24 output
            stream = FFGear(source="USB2.0 Camera", source_demuxer="dshow", frame_format="bgr24", logging=True)
            ```

        - [x] **[OPTIONAL] Specify Video Device's index along with name:** If there are multiple Video devices with similar name, then you can use `-video_device_number` parameter to specify the arbitrary index of the particular device. For instance, to open second video device with name `"Camera"` you can do as follows:

            ```python
            # define video_device_number as 1 (numbering start from 0)
            options = {"-ffprefixes":["-video_device_number", "1"]}

            # initialize and formulate the stream with "Camera" source for BGR24 output
            stream = FFGear(source="Camera", source_demuxer="dshow", frame_format="bgr24", logging=True, **options)
            ```

    === ":material-linux: Linux"

        Linux OS users can use the [`video4linux2`](https://trac.ffmpeg.org/wiki/Capture/Webcam#Linux) _(or its alias `v4l2`)_ to list to all capture video devices such as from an USB webcam. You can refer following steps to identify and specify your capture video device's path:

        - [x] **Identify Video Devices:** Linux systems tend to automatically create file device node/path when the device _(e.g. an USB webcam)_ is plugged into the system, and has a name of the kind `'/dev/videoN'`, where `N` is a index associated to the device. To get the list of all available file device node/path on your Linux machine, you can use the `v4l-ctl` command.

            !!! tip "You can use `#!sh sudo apt install v4l-utils` APT command to install `v4l-ctl` tool on Debian-based Linux distros."

            ```sh
            $ v4l2-ctl --list-devices

            USB2.0 PC CAMERA (usb-0000:00:1d.7-1):
                    /dev/video1

            UVC Camera (046d:0819) (usb-0000:00:1d.7-2):
                    /dev/video0
            ```

        - [x] **Specify Video Device's path:** Then, you can specify and initialize your located Video device's path in FFGear API as follows:

            ```python
            # initialize and formulate the stream with "/dev/video0" source for BGR24 output
            stream = FFGear(source="/dev/video0", source_demuxer="v4l2", frame_format="bgr24", logging=True)
            ```

        - [x] **[OPTIONAL] Specify Video Device's additional specifications:** You can also specify additional specifications _(such as pixel format(s), video format(s), framerate, and frame dimensions)_ supported by your Video Device as follows:

            !!! tip "You can use `#!sh ffmpeg -f v4l2 -list_formats all -i /dev/video0` terminal command to list available specifications."

            ```python
            # define video device specifications
            options = {"-ffprefixes":["-framerate", "25", "-video_size", "640x480"]}

            # initialize and formulate the stream with "/dev/video0" source for BGR24 output
            stream = FFGear(source="/dev/video0", source_demuxer="v4l2", frame_format="bgr24", logging=True, **options)
            ```

    === ":material-apple: MacOS"

        MacOS users can use the [AVFoundation](https://ffmpeg.org/ffmpeg-devices.html#avfoundation) to list input devices and is the currently recommended framework by Apple for streamgrabbing on Mac OSX-10.7 (Lion) and later as well as on iOS. You can refer following steps to identify and specify your capture video device's name or index on MacOS/OSX machines:

        !!! note "QTKit is also available for streamgrabbing on Mac OS X 10.4 (Tiger) and later, but has been marked deprecated since OS X 10.7 (Lion) and may not be available on future releases."


        - [x] **Identify Video Devices:** Then, You can locate your Video device's name and index using `avfoundation` as follows:

            ```sh
            $ ffmpeg -f avfoundation -list_devices true -i ""

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


        - [x] **Specify Video Device's name or index:** Then, you can specify and initialize your located Video device in FFGear API using its either the name or the index shown in the device listing:

            === "Using device's index"

                ```python
                # initialize and formulate the stream with `1` index source for BGR24 output
                stream = FFGear(source="1", source_demuxer="avfoundation", frame_format="bgr24", logging=True)
                ```

            === "Using device's name"

                When specifying device's name, abbreviations using just the beginning of the device name are possible. Thus, to capture from a device named "Integrated iSight-camera" just "Integrated" is sufficient:

                ```python
                # initialize and formulate the stream with "Integrated iSight-camera" source for BGR24 output
                stream = FFGear(source="Integrated", source_demuxer="avfoundation", frame_format="bgr24", logging=True)
                ```

        - [x] **[OPTIONAL] Specify Default Video device:** You can also use the default device which is usually the first device in the listing by using "default" as source:

            ```python
            # initialize and formulate the stream with "default" source for BGR24 output
            stream = FFGear(source="default", source_demuxer="avfoundation", frame_format="bgr24", logging=True)
            ```

```python linenums="1" hl_lines="7"
# import required libraries
from vidgear.gears import FFGear
import cv2

# stream with "/dev/video0" source for BGR24 output
stream = FFGear(
    source="/dev/video0", source_demuxer="v4l2", frame_format="bgr24", logging=True
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

## Using FFGear with Desktop (Screen-Capturing)

???+ alert "Example Assumptions"

    Similar to Webcam capturing, FFmpeg provide set of specific Demuxers on different platforms for capturing your desktop _(Screen recording)_. Please note that following recipe explicitly assumes: 

    - You're running Linux Machine with `libxcb` module installed properly on your machine.
    - You machine uses FFmpeg binaries built with `--enable-libxcb` flag to support `x11grab` demuxer. BTW, you can list all supported demuxers using the `#!sh ffmpeg --list-demuxers` terminal command.

    These assumptions **MAY/MAY NOT** suit your current setup. Kindly use suitable parameters based your system platform and hardware settings only.

FFGear API supports screen capture from your entire screen as well as specific regions. Under the hood, it uses platform-specific FFmpeg demuxers for this purpose.

??? tip "Specifying suitable Parameter(s) and Demuxer for Capturing your Desktop on different OS platforms"

    === ":fontawesome-brands-windows: Windows"

        Windows OS users can use the [gdigrab](https://ffmpeg.org/ffmpeg-devices.html#gdigrab) to grab video from the Windows screen. You can refer following steps to specify source for capturing different regions of your display:

        !!! fail "For Windows OS users [`dshow`](https://github.com/rdp/screen-capture-recorder-to-video-windows-free) is also available for grabbing frames from your desktop. But it is highly unreliable and don't works most of the times."

        - [x] **Capturing entire desktop:** For capturing all your displays as one big contiguous display, you can specify source, suitable parameters and demuxers in FFGear API as follows:

            ```python
            # define framerate
            options = {"-framerate": "30"}

            # stream with "desktop" source for BGR24 output
            stream = FFGear(source="desktop", source_demuxer="gdigrab", frame_format="bgr24", logging=True, **options)
            ```

        - [x] **Capturing a region:** If you want to limit capturing to a region, and show the area being grabbed, you can specify source and suitable parameters in FFGear API as follows:

            !!! info "`x_offset` and `y_offset` specify the offsets of the grabbed area with respect to the top-left border of the desktop screen. They default to `0`. "

            ```python
            # define suitable parameters
            options = {
                "-framerate": "30", # input framerate
                "-ffprefixes": [
                    "-offset_x", "10", "-offset_y", "20", # grab at position 10,20
                    "-video_size", "640x480", # frame size
                    "-show_region", "1", # show only region
                ],
            }

            # stream with "desktop" source for BGR24 output
            stream = FFGear(source="desktop", source_demuxer="gdigrab", frame_format="bgr24", logging=True, **options)
            ```

    === ":material-linux: Linux"

        Linux OS users can use the [x11grab](https://ffmpeg.org/ffmpeg-devices.html#x11grab) to capture an X11 display. You can refer following steps to specify source for capturing different regions of your display:

        !!! note "For X11 display, the source input has the syntax: `"display_number.screen_number[+x_offset,y_offset]"`."

        - [x] **Capturing entire desktop:** For capturing all your displays as one big contiguous display, you can specify source, suitable parameters and demuxers in FFGear API as follows:

            ```python
            # define framerate
            options = {"-framerate": "30"}

            # stream with ":0.0" desktop source for BGR24 output
            stream = FFGear(source=":0.0", source_demuxer="x11grab", frame_format="bgr24", logging=True, **options)
            ```

        - [x] **Capturing a region:** If you want to limit capturing to a region, and show the area being grabbed, you can specify source and suitable parameters in FFGear API as follows:

            !!! info "`x_offset` and `y_offset` specify the offsets of the grabbed area with respect to the top-left border of the X11 screen. They default to `0`. "

            ```python
            # define suitable parameters
            options = {
                "-framerate": "30", # input framerate
                "-ffprefixes": [
                    "-video_size", "1024x768", # frame size
                ],
            }

            # stream with ":0.0" desktop source(starting with the upper-left corner at x=10, y=20) 
            # for BGR24 output
            stream = FFGear(source=":0.0+10,20", source_demuxer="x11grab", frame_format="bgr24", logging=True, **options)
            ```

    === ":material-apple: MacOS"

        MacOS users can use the [AVFoundation](https://ffmpeg.org/ffmpeg-devices.html#avfoundation) to list input devices and is the currently recommended framework by Apple for stream grabbing on Mac OSX-10.7 (Lion) and later as well as on iOS. You can refer following steps to identify and specify your capture video device's name or index on MacOS/OSX machines:

        !!! note "QTKit is also available for stream grabbing on Mac OS X 10.4 (Tiger) and later, but has been marked deprecated since OS X 10.7 (Lion) and may not be available on future releases."


        - [x] **Identify Video Devices:**  You can enumerate all the available input devices including screens ready to be captured using `avfoundation` as follows:

            ```sh
            $ ffmpeg -f avfoundation -list_devices true -i ""

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


        - [x] **Capturing entire desktop:** Then, you can specify and initialize your located screens in FFGear API using its index shown:

            ```python
            # stream with `0:` index desktop screen for BGR24 output
            stream = FFGear(source="0:", source_demuxer="avfoundation", frame_format="bgr24", logging=True)
            ```

        - [x] **[OPTIONAL] Capturing mouse:** You can also specify additional specifications to capture the mouse pointer and screen mouse clicks as follows:

            ```python
            # define specifications
            options = {"-ffprefixes":["-capture_cursor", "1", "-capture_mouse_clicks", "0"]}

            # stream with "0:" source for BGR24 output
            stream = FFGear(source="0:", source_demuxer="avfoundation", frame_format="bgr24", logging=True, **options)
            ```

=== "Capturing entire desktop" 

    For capturing all your displays as one big contiguous display in FFGear API:

    ```python linenums="1" hl_lines="6 11-12"
    # import required libraries
    from vidgear.gears import FFGear
    import cv2

    # define framerate
    options = {"-framerate": "30"}

    # stream with ":0.0" desktop source(starting with the upper-left corner at x=10, y=20) 
    # for BGR24 output
    stream = FFGear(
        source=":0.0",
        source_demuxer="x11grab",
        frame_format="bgr24",
        logging=True,
        **options
    ).start()

    # loop over
    while True:

        # read BGR frames (auto-converted from YUV420p)
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


=== "Capturing a region" 

    For limit capturing to a region, and show the area being grabbed:

    !!! info "`x_offset` and `y_offset` specify the offsets of the grabbed area with respect to the top-left border of the X11 screen. They default to `0`. "

    ```python linenums="1" hl_lines="7-10 16-17"
    # import required libraries
    from vidgear.gears import FFGear
    import cv2

    # define suitable parameters
    options = {
        "-framerate": "30", # input framerate
        "-ffprefixes": [
            "-video_size", "1024x768", # frame size
        ],
    }

    # stream with ":0.0" desktop source(starting with the upper-left corner at x=10, y=20) 
    # for BGR24 output
    stream = FFGear(
        source=":0.0+10,20",
        source_demuxer="x11grab",
        frame_format="bgr24",
        logging=True,
        **options
    ).start()

    # loop over
    while True:

        # read BGR frames (auto-converted from YUV420p)
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

## Hardware-Accelerated Decoding

FFGear exposes FFdecoder's full hardware acceleration support via the `options` dictionary. You can use NVIDIA CUVID, CUDA, VAAPI, or any other `-hwaccel` backend supported by your FFmpeg build.

### NVIDIA CUVID Decoding

???+ alert "Example Assumptions"

    **Please note that following recipe explicitly assumes:**

    - You're running :fontawesome-brands-linux: Linux operating system with a [**supported NVIDIA GPU**](https://developer.nvidia.com/nvidia-video-codec-sdk).
    - You're using FFmpeg 4.4 or newer, configured with at least ` --enable-nonfree --enable-cuda-nvcc --enable-libnpp --enable-cuvid --enable-nvenc` configuration flags during compilation. For compilation follow [these instructions ➶](https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/#prerequisites)

    - [x] **Using `h264_cuvid` decoder**: Remember to check if your FFmpeg compiled with H.264 CUVID decoder support by executing following one-liner command in your terminal, and observing if output contains something similar as follows:

        ??? danger "Verifying H.264 CUVID decoder support in FFmpeg"
            ```sh
            $ ffmpeg  -hide_banner -decoders | grep cuvid

            V..... av1_cuvid            Nvidia CUVID AV1 decoder (codec av1)
            V..... h264_cuvid           Nvidia CUVID H264 decoder (codec h264)
            V..... hevc_cuvid           Nvidia CUVID HEVC decoder (codec hevc)
            V..... mjpeg_cuvid          Nvidia CUVID MJPEG decoder (codec mjpeg)
            V..... mpeg1_cuvid          Nvidia CUVID MPEG1VIDEO decoder (codec mpeg1video)
            V..... mpeg2_cuvid          Nvidia CUVID MPEG2VIDEO decoder (codec mpeg2video)
            V..... mpeg4_cuvid          Nvidia CUVID MPEG4 decoder (codec mpeg4)
            V..... vc1_cuvid            Nvidia CUVID VC1 decoder (codec vc1)
            V..... vp8_cuvid            Nvidia CUVID VP8 decoder (codec vp8)
            V..... vp9_cuvid            Nvidia CUVID VP9 decoder (codec vp9)
            ```

            !!! note "You can also use any of above decoder in the similar way, if supported."
            !!! tip "Use `#!sh ffmpeg -decoders` terminal command to lists all FFmpeg supported decoders."

    - You already have appropriate Nvidia video drivers and related softwares installed on your machine.
    - If the stream is not decodable in hardware (for example, it is an unsupported codec or profile) then it will still be decoded in software automatically, but hardware filters won't be applicable.

    These assumptions **MAY/MAY NOT** suit your current setup. Kindly use suitable parameters based your system platform and hardware settings only.

In this example, we use Nvidia's **H.264 CUVID Video decoder(`h264_cuvid`)** in FFGear API to achieve GPU-accelerated hardware video decoding of **YUV420p(`yuv420p`)** frames from a given Video file _(say `foo.mp4`)_, and the [`-enforce_cv_patch`](../params/#b-ffdecoder-parameters) flag for OpenCV compatibility:

!!! info "More information on Nvidia's CUVID can be found [here ➶](https://developer.nvidia.com/blog/nvidia-ffmpeg-transcoding-guide/)"

```python linenums="1" hl_lines="7-8 13"
# import required libraries
from vidgear.gears import FFGear
import cv2

# use H.264 CUVID hardware decoder; enable OpenCV patch for YUV frames
options = {
    "-vcodec": "h264_cuvid",    # NVIDIA CUVID hardware decoder
    "-enforce_cv_patch": True,  # auto-convert YUV420p → BGR in FFGear
}

stream = FFGear(
    source="myvideo.mp4",
    frame_format="yuv420p",
    logging=True,
    **options
).start()

# loop over
while True:

    # read BGR frames (auto-converted from YUV420p)
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

!!! info "More information on Nvidia's CUVID can be found [here ➶](https://developer.nvidia.com/blog/nvidia-ffmpeg-transcoding-guide/)"

&nbsp;

### NVIDIA CUDA Decoding

???+ alert "Example Assumptions"

    **Please note that following recipe explicitly assumes:**

    - You're running :fontawesome-brands-linux: Linux operating system with a [**supported NVIDIA GPU**](https://developer.nvidia.com/nvidia-video-codec-sdk).
    - You're using FFmpeg 4.4 or newer, configured with at least ` --enable-nonfree --enable-cuda-nvcc --enable-libnpp  --enable-cuvid --enable-nvenc` configuration flags during compilation. For compilation follow [these instructions ➶](https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/#prerequisites)

        ??? danger "Verifying NVDEC/CUDA support in FFmpeg"

            To use CUDA Video-decoder(`cuda`), remember to check if your FFmpeg compiled with it by executing following commands in your terminal, and observing if output contains something similar as follows:

            ```sh
            $ ffmpeg  -hide_banner -pix_fmts | grep cuda
            ..H.. cuda                   0              0      0
            
            $ ffmpeg  -hide_banner -filters | egrep "cuda|npp"
            ... bilateral_cuda    V->V       GPU accelerated bilateral filter
            ... chromakey_cuda    V->V       GPU accelerated chromakey filter
            ... colorspace_cuda   V->V       CUDA accelerated video color converter
            ... hwupload_cuda     V->V       Upload a system memory frame to a CUDA device.
            ... overlay_cuda      VV->V      Overlay one video on top of another using CUDA
            ... scale_cuda        V->V       GPU accelerated video resizer
            ... scale_npp         V->V       NVIDIA Performance Primitives video scaling and format conversion
            ... scale2ref_npp     VV->VV     NVIDIA Performance Primitives video scaling and format conversion to the given reference.
            ... sharpen_npp       V->V       NVIDIA Performance Primitives video sharpening filter.
            ... thumbnail_cuda    V->V       Select the most representative frame in a given sequence of consecutive frames.
            ... transpose_npp     V->V       NVIDIA Performance Primitives video transpose
            T.. yadif_cuda        V->V       Deinterlace CUDA frames
            ```

    - You already have appropriate Nvidia video drivers and related softwares installed on your machine.
    - If the stream is not decodable in hardware (for example, it is an unsupported codec or profile) then it will still be decoded in software automatically, but hardware filters won't be applicable.

    These assumptions **MAY/MAY NOT** suit your current setup. Kindly use suitable parameters based your system platform and hardware settings only.

In this example, we use Nvidia's **CUDA Internal hwaccel Video decoder(`cuda`)** in FFGear API to automatically detect best NV-accelerated video codec and keeping video frames in GPU memory _(for applying hardware filters)_, applying GPU-side scale and FPS filters before downloading as **NV12(`nv12`)** pixel-format frames from a given video file _(say `foo.mp4`)_, and the [`-enforce_cv_patch`](../params/#b-ffdecoder-parameters) flag for OpenCV compatibility:

??? warning "`NV12`(for `4:2:0` input) and `NV21`(for `4:4:4` input) are the only supported pixel format. You cannot change pixel format to any other since NV-accelerated video codec supports only them."
    
    NV12 is a biplanar format with a full sized Y plane followed by a single chroma plane with weaved U and V values. NV21 is the same but with weaved V and U values. The 12 in NV12 refers to 12 bits per pixel. NV12 has a half width and half height chroma channel, and therefore is a 420 subsampling. NV16 is 16 bits per pixel, with half width and full height. aka 422. NV24 is 24 bits per pixel with full sized chroma channel. aka 444. Most NV12 functions allow the destination Y pointer to be NULL.

!!! note "More information on Nvidia's GPU Accelerated Decoding can be found [here ➶](https://developer.nvidia.com/blog/nvidia-ffmpeg-transcoding-guide/)"

```python linenums="1" hl_lines="7-21 26"
# import required libraries
from vidgear.gears import FFGear
import cv2

# CUDA hwaccel: decode in GPU memory, scale & fps in GPU, download as NV12
options = {
    "-vcodec": None,               # skip source decoder, let FFmpeg choose
    "-enforce_cv_patch": True,     # auto-convert NV12 → BGR in FFGear
    "-ffprefixes": [
        "-vsync", "0",             # prevent duplicate frames
        "-hwaccel", "cuda",        # use CUDA accelerator
        "-hwaccel_output_format", "cuda",  # keep frames in GPU memory
    ],
    "-custom_resolution": "null",  # discard source resolution param
    "-framerate": "null",          # discard source framerate param
    "-vf": (
        "scale_cuda=1280:720,"     # GPU-side scale to 1280x720
        "fps=60.0,"                # GPU-side framerate
        "hwdownload,"              # download to system memory
        "format=nv12"             # convert to NV12 pixel format
    ),
}

stream = FFGear(
    source="myvideo.mp4",
    frame_format="null",           # discard source pixel format
    logging=True,
    **options
).start()

# loop over
while True:

    # read BGR frames (auto-converted from NV12)
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

## Per-Frame Metadata Extraction

> The `-extract_metadata` option enables FFGear to yield `(frame, metadata)` tuples from `read()`, where `metadata` is parsed from FFmpeg's [`showinfo`](https://ffmpeg.org/ffmpeg-filters.html#showinfo) filter via an asynchronous background daemon — leaving the main frame pipe fully unthrottled.

The `metadata` dict contains the following keys:

| Key | Type | Description |
|:----|:----:|:------------|
| `frame_num` | int | Monotonic frame index as emitted by FFmpeg. |
| `pts_time` | float | Presentation timestamp in seconds. |
| `is_keyframe` | bool | `True` if the frame is a keyframe (I-frame). |
| `frame_type` | str | One of `"I"` (keyframe), `"P"` (predictive), `"B"` (bi-predictive), `"?"` (unknown). |

!!! warning "Incompatible with `-filter_complex`"

    `-extract_metadata` cannot be combined with the `-filter_complex` attribute. If both are supplied, a warning is logged and metadata extraction is silently disabled. A pre-existing `-vf` is fine — `showinfo` is automatically comma-chained onto it.

### Keyframe-Only Decoding for AI Inference

> Many Computer Vision workflows — perceptual hashing, scene-change detection, heavyweight AI-model inference _(YOLO, ResNet, etc.)_ — only care about **Keyframes (I-frames)**. Without `-extract_metadata` you'd decode and run your model on every P/B frame, wasting 98%+ of compute.

!!! success "Depending on the GOP size, this pattern reduces downstream processing time by **10–50×** without skipping any scene-boundary information."

```python linenums="1" hl_lines="5 17-32"
# import required libraries
from vidgear.gears import FFGear

# enable per-frame metadata extraction
options = {"-extract_metadata": True}

stream = FFGear(
    source="myvideo.mp4",
    frame_format="bgr24",
    logging=True,
    **options
).start()

# loop over
while True:

    # read returns (frame, metadata) tuple when -extract_metadata is enabled
    output = stream.read()

    # check if output is None (end of stream)
    if output is None:
        break

    frame, meta = output

    # OPTIMIZATION: skip non-keyframes entirely
    if not meta["is_keyframe"]:
        continue

    # run heavy AI model on keyframes only (~1-2 fps worth of data)
    # results = heavy_ai_model.predict(frame)
    print(f"Keyframe #{meta['frame_num']} at {meta['pts_time']:.3f}s")

# safely close video stream
stream.stop()
```

&nbsp;

### Variable-Frame-Rate (VFR) Synchronization

> Most modern video sources — smartphones, screen recordings, webcams — are **Variable-Frame-Rate**. Assuming a constant frame rate will drift out of sync very quickly.

With `meta["pts_time"]` you know the **exact presentation timestamp** of every frame:

```python linenums="1" hl_lines="5 18-36"
# import required libraries
from vidgear.gears import FFGear

# enable per-frame metadata extraction
options = {"-extract_metadata": True}

stream = FFGear(
    source="screen_recording.mp4",
    frame_format="bgr24",
    **options
).start()

prev_pts = None

# loop over
while True:

    # read returns (frame, metadata) tuple
    output = stream.read()

    if output is None:
        break

    frame, meta = output

    # exact presentation timestamp in seconds
    pts = meta["pts_time"]

    # compute the real inter-frame delta (not the nominal 1/fps value)
    delta_ms = None if prev_pts is None else (pts - prev_pts) * 1000.0
    prev_pts = pts

    # use real delta for per-frame motion/velocity calculations
    # e.g. velocity = displacement_px / delta_ms
    if delta_ms is not None:
        print(f"Frame #{meta['frame_num']} | PTS: {pts:.3f}s | Δt: {delta_ms:.1f}ms")

# safely close video stream
stream.stop()
```

!!! tip "The same `pts_time` stream is what you need to keep processed frames locked to an audio track when re-muxing downstream."

&nbsp;

## Complex FFmpeg Filtergraphs

For advanced multi-input or multi-output pipelines, FFGear supports `-filter_complex` via the `options` dictionary.

!!! warning "Complex filtergraphs (`-filter_complex`) are incompatible with `-extract_metadata`. Both cannot be active simultaneously."

### Overlay Two Video Sources

In this example, we overlay a watermark video on top of a primary source:

```python linenums="1" hl_lines="7-11"
# import required libraries
from vidgear.gears import FFGear
import cv2

# overlay watermark.mp4 on top of primary source at position (10, 10)
options = {
    "-i": "watermark.mp4",         # second input source
    "-filter_complex": (
        "[0:v][1:v]overlay=10:10"  # overlay second input onto first
    ),
    "-map": "0:a?",                # keep audio from primary (if any)
}

stream = FFGear(
    source="myvideo.mp4",
    frame_format="bgr24",
    logging=True,
    **options
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

## Using FFGear with WriteGear API (Compression Mode)

???+ danger "High CPU Usage when chaining FFGear with WriteGear"

    When chaining FFGear with WriteGear, both FFmpeg processes _(decoding + encoding)_ run **as fast as your hardware allows** with no artificial pacing between them. This causes the pipeline to max out your CPU to process the video in the shortest time possible, which may be undesirable.

    You can mitigate this in two ways depending on your use case:

    === "Throttle to Real-Time Speed"

        Pass the `-re` flag via FFGear's `-ffprefixes` parameter to force FFmpeg to read the input at its native framerate. This naturally paces the pipeline to real-time speed and **drastically reduces CPU usage**:

        ```python
        # force input to be read at native framerate
        stream = FFGear(source="foo.mp4", frame_format="bgr24", **{"-ffprefixes": ["-re"]})
        ```

    === "Limit FFmpeg Threads"

        Pass `-threads` to both FFGear and WriteGear to cap the number of CPU threads each FFmpeg process may use. This leaves headroom for other system tasks:

        ```python
        # limit decoder to 2 threads
        stream = FFGear(source="foo.mp4", frame_format="bgr24", **{"-threads": 2})

        # limit encoder to 2 threads
        writer = WriteGear(output="output_foo.mp4", **{"-input_framerate": fps, "-threads": 2})
        ```

    !!! tip "Hardware Acceleration"
        If your machine has a dedicated GPU, you can offload encoding to the GPU entirely — for example by passing `"-vcodec": "h264_nvenc"` to WriteGear _(NVIDIA)_ — shifting the heavy lifting off the CPU.

FFGear integrates seamlessly with VidGear's WriteGear API in Compression Mode for high-quality re-encoding of decoded frames:

!!! tip "You can use FFGear API's `stream.metadata` property object that dumps source Video's metadata information _(as JSON string)_ to retrieve source framerate."

=== "BGR frames"

    WriteGear API by default expects `BGR` format frames in its `write()` class method.

    ```python linenums="1" hl_lines="3 12 17 32 38"
    # import required libraries
    from vidgear.gears import FFGear
    from vidgear.gears import WriteGear
    import cv2, json

    # open source with FFGear and BGR frames
    stream = FFGear(source="myvideo.mp4", frame_format="bgr24", logging=True).start()

    # retrieve framerate from source JSON Metadata and pass it as `-input_framerate` 
    # parameter for controlled framerate
    output_params = {
        "-input_framerate": json.loads(stream.stream.metadata)["source_video_framerate"]
    }

    # Define writer with default parameters and suitable
    # output filename for e.g. `output_foo.mp4`
    writer = WriteGear(output="output_foo.mp4", **output_params)

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # write frame to output
        writer.write(frame)

    # safely close stream
    stream.stop()

    # safely close writer
    writer.close()
    ```

=== "RGB frames"

    In WriteGear API, you can use [`rgb_mode`](https://abhitronix.github.io/vidgear/latest/bonus/reference/writegear/#vidgear.gears.writegear.WriteGear.write) parameter in  `write()` class method to write `RGB` format frames instead of default `BGR` as follows:

    ```python linenums="1" hl_lines="3 12 17 32 38"
    # import required libraries
    from vidgear.gears import FFGear
    from vidgear.gears import WriteGear
    import cv2, json

    # open source with FFGear and RGB frames
    stream = FFGear(source="myvideo.mp4", frame_format="rgb24", logging=True).start()

    # retrieve framerate from source JSON Metadata and pass it as `-input_framerate` 
    # parameter for controlled framerate
    output_params = {
        "-input_framerate": json.loads(stream.stream.metadata)["source_video_framerate"]
    }

    # Define writer with default parameters and suitable
    # output filename for e.g. `output_foo_rgb.mp4`
    writer = WriteGear(output="output_foo_rgb.mp4", **output_params)

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # write frame to output
        writer.write(frame, rgb_mode=True)

    # safely close stream
    stream.stop()

    # safely close writer
    writer.close()
    ```

=== "Grayscale frames"

    WriteGear API also directly consumes `GRAYSCALE` format frames in its `write()` class method. 

    ```python linenums="1" hl_lines="3 12 17 32 38"
    # import required libraries
    from vidgear.gears import FFGear
    from vidgear.gears import WriteGear
    import cv2, json

    # open source with FFGear and GRAYSCALE frames
    stream = FFGear(source="myvideo.mp4", frame_format="gray", logging=True).start()

    # retrieve framerate from source JSON Metadata and pass it as `-input_framerate` 
    # parameter for controlled framerate
    output_params = {
        "-input_framerate": json.loads(stream.stream.metadata)["source_video_framerate"]
    }

    # Define writer with default parameters and suitable
    # output filename for e.g. `output_foo_gray.mp4`
    writer = WriteGear(output="output_foo_gray.mp4", **output_params)

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # write frame to output
        writer.write(frame)

    # safely close stream
    stream.stop()

    # safely close writer
    writer.close()
    ```

=== "YUV420p (Performance Mode) :material-speedometer:"

    !!! success "Performance Mode :zap: — Faster Decoding via YUV420p"

        Ingesting frames as 12-bit **YUV 4:2:0** instead of 24-bit **BGR** halves the bytes moving through the FFmpeg pipe. Enable `-enforce_cv_patch` to auto-convert frames inside FFGear for seamless OpenCV compatibility.

    WriteGear API also directly consume `YUV` _(or basically any other supported pixel format)_ frames in its `write()` class method with its `-input_pixfmt` attribute in compression mode. 

    !!! note "You can also use `yuv422p`(4:2:2 subsampling) or `yuv444p`(4:4:4 subsampling) instead for more higher dynamic ranges."

    ```python linenums="1" hl_lines="3 13-14 19 34 40"
    # import required libraries
    from vidgear.gears import FFGear
    from vidgear.gears import WriteGear
    import cv2, json

    # open source with FFGear stream for YUV420 output
    stream = FFGear(source="myvideo.mp4", frame_format="yuv420p", logging=True).start()

    # retrieve framerate from source JSON Metadata and pass it as 
    # `-input_framerate` parameter for controlled framerate
    # and also add input pixfmt as yuv420p 
    output_params = {
        "-input_framerate": json.loads(stream.stream.metadata)["output_framerate"],
        "-input_pixfmt": "yuv420p"
    }

    # Define writer with default parameters and suitable
    # output filename for e.g. `output_foo_yuv.mp4`
    writer = WriteGear(output="output_foo_yuv.mp4", logging=True, **output_params)

    # loop over
    while True:

        # read OpenCV compatible YUV420p frames
        frame = stream.read()

        # check for frame if NoneType
        if frame is None:
            break

        # {do something with the YUV420p frame here}

        # write frame to output
        writer.write(frame)

    # safely close stream
    stream.stop()

    # safely close writer
    writer.close()
    ```

=== "Grayscale via YUV (fastest) :material-invoice-fast-outline:"

    !!! success "Fastest :zap: RAW-to-Grayscale via `-extract_luma`"

        Every YUV/NV bytestream stores the **Luma (Y) plane** uncompressed at the top of each frame. The exclusive [`-extract_luma`](../params/#b-exclusive-parameters) boolean attribute makes FFGear slice that Y-plane directly and hand back a 2D `(H, W)` grayscale ndarray — **no colorspace conversion in FFmpeg, no `cv2.cvtColor` in Python**. This is strictly faster than `frame_format="gray"`, which still asks FFmpeg to do a `yuv→gray` conversion on every frame.

        Combined with the reduced pipe-bytes of YUV 4:2:0 ingest, this is the fastest :fontawesome-solid-tachometer-alt-fast: grayscale pipeline the API can produce.

    Similar to normal `GRAYSCALE` format frames, you can directly consume these frames in WriteGear API’s `write()` class method:

    ```python linenums="1" hl_lines="3 7 12 20 25 39 45"
    # import required libraries
    from vidgear.gears import FFGear
    from vidgear.gears import WriteGear
    import cv2, json

    # enable direct Luma (Y-plane) extraction
    options = {"-extract_luma": True}

    # stream Grayscale via YUV frames
    stream = FFGear(
        source="myvideo.mp4",
        frame_format="yuv420p",
        logging=True,
        **options
    ).start()

    # retrieve framerate from source JSON Metadata and pass it as `-input_framerate` 
    # parameter for controlled framerate
    output_params = {
        "-input_framerate": json.loads(stream.stream.metadata)["source_video_framerate"]
    }

    # Define writer with default parameters and suitable
    # output filename for e.g. `output_foo_yuv_gray.mp4`
    writer = WriteGear(output="output_foo_yuv_gray.mp4", **output_params)

    # loop over
    while True:
        # read grayscale frames
        frame = stream.read()

        # check for frame if NoneType
        if frame is None:
            break

        # {do something with Luma (Y-plane) extracted grayscale frame here}

        # write frame to output
        writer.write(frame)

    # safely close stream
    stream.stop()

    # safely close writer
    writer.close()
    ```

&nbsp;

## Bonus Examples

!!! example "Checkout more advanced FFGear examples with unusual configuration [here ➶](../../../help/ffgear_ex/)"

&nbsp;

[deffcode]:https://github.com/abhiTronix/deffcode
