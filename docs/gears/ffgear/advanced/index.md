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

    FFmpeg provides a set of platform-specific demuxers to read multimedia streams from different types of video capture devices/sources. Please note that the following example explicitly assumes:

    - You're running a :material-linux: Linux machine with a USB webcam connected at the device node/path `/dev/video0`.
    - You already have the appropriate Linux video drivers and related software installed on your machine.
    - Your machine uses FFmpeg binaries built with the `--enable-libv4l2` flag to support the `video4linux2` (or simply `v4l2`) demuxer. You can list all supported demuxers using the terminal command:

    ```sh
    ffmpeg --list-demuxers
    ```

    These assumptions **MAY/MAY NOT** suit your current setup. Kindly use parameters appropriate for your system platform and hardware configuration only.


In this example, we output **BGR24** video frames from a USB webcam device connected at `/dev/video0` on a Linux machine using the `video4linux2` (or simply `v4l2`) demuxer:

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

    Similar to webcam capturing, FFmpeg provides a set of platform-specific demuxers for desktop capture _(screen recording)_. Please note that the following example explicitly assumes:
    - You're running a :material-linux: Linux machine with the `libxcb` module installed properly.
    - Your machine uses FFmpeg binaries built with the `--enable-libxcb` flag to support the `x11grab` demuxer. You can list all supported demuxers using the terminal command:

    ```sh
    ffmpeg --list-demuxers
    ```
    These assumptions **MAY/MAY NOT** suit your current setup. Kindly use parameters appropriate for your system platform and hardware configuration only.


FFGear API supports screen capture from your entire desktop as well as specific screen regions. Under the hood, it uses platform-specific FFmpeg demuxers for this purpose.

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

    **Please note that following example explicitly assumes:**

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
    "-enforce_cv_patch": True,  # auto-convert YUV420p → OpenCV Compatible in FFGear
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

    The following example explicitly assumes:
    - You're running :fontawesome-brands-linux: Linux operating system with a [**supported NVIDIA GPU**](https://developer.nvidia.com/nvidia-video-codec-sdk).
    - You're using FFmpeg 4.4 or newer, configured with at least `--enable-nonfree --enable-cuda-nvcc --enable-libnpp  --enable-cuvid --enable-nvenc` configuration flags during compilation. For compilation follow [these instructions ➶](https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/#prerequisites)

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

```python linenums="1" hl_lines="7-21 26 44"
# import required libraries
from vidgear.gears import FFGear
import cv2

# CUDA hwaccel: decode in GPU memory, scale & fps in GPU, download as NV12
options = {
    "-vcodec": None,               # skip source decoder, let FFmpeg choose
    "-enforce_cv_patch": True,     # auto-convert NV12 → OpenCV Compatible in FFGear
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
    source=VIDEO,
    frame_format="null",           # discard source pixel format
    logging=True,
    **options
).start()

# loop over
while True:

    # read NV12 frames
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # convert it to OpenCV compatible frame
    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)

    # {do something with the BGR frame here}

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

## Hardware-Accelerated Video Transcoding

FFGear, in conjunction with WriteGear, provides direct access to FFmpeg’s powerful transcoding capabilities while still allowing real-time frame processing with immense flexibility. Together, these APIs can leverage GPU-backed hardware-accelerated decoding and encoding pipelines, dramatically improving video transcoding performance for real-time multimedia workloads.

??? danger "Performance Bottleneck in Hardware-Accelerated Video Transcoding with Real-Time Frame Processing"

    When using FFmpeg with `-hwaccel cuda -hwaccel_output_format cuda`, decoded frames remain in GPU memory. This avoids expensive GPU ↔ CPU memory transfers and enables near-optimal transcoding performance on supported hardware.

    <figure markdown>
    ![HW Acceleration](https://abhitronix.github.io/deffcode/latest/assets/images/hw_accel.png){ width="350" }
    <figcaption>Memory Flow with Hardware Acceleration</figcaption>
    </figure>

    However, when performing real-time frame processing in Python with FFGear and WriteGear, decoded frames must be transferred from GPU memory back to system memory so Python can access and process them.

    This introduces several performance bottlenecks:

    - **Explicit GPU ↔ CPU memory transfers** over the PCIe bus  
    - **Increased latency** from continuous frame movement  
    - **Higher PCIe bandwidth utilization**, especially with uncompressed frames  
    - **Potential PCIe bus saturation**, limiting overall throughput  

    As a result, real-time processing pipelines incur additional overhead that can impact performance.

    <figure markdown>
    ![HW Acceleration Limitation](https://abhitronix.github.io/deffcode/latest/assets/images/hw_accel_limitation.png){ width="350" }
    <figcaption>Memory Flow with Hardware Acceleration and Real-Time Processing</figcaption>
    </figure>

    Even with these limitations, hardware acceleration still provides major advantages over software-only transcoding:

    - **Lower CPU utilization**, since most processing stays on the GPU  
    - **Faster execution** compared to CPU-based pipelines  
    - **Accelerated video operations**, including scaling, filtering, and deinterlacing  
    - **Better overall system resource utilization**, freeing CPU resources for parallel workloads  

    In contrast, software transcoding is entirely CPU-bound and generally less efficient for high-throughput or real-time workloads.

    !!! summary "Although GPU ↔ CPU memory transfers introduce unavoidable overhead in real-time processing pipelines, hardware acceleration still delivers substantial performance and efficiency gains—making it the preferred choice for most modern video workflows."

### CUDA-accelerated Transcoding with OpenCV's VideoWriter API 

???+ alert "Example Assumptions"

    The following example explicitly assumes:

    - You're running :fontawesome-brands-linux: Linux operating system with a [**supported NVIDIA GPU**](https://developer.nvidia.com/nvidia-video-codec-sdk).
    - You're using FFmpeg 4.4 or newer, configured with at least `--enable-nonfree --enable-cuda-nvcc --enable-libnpp  --enable-cuvid --enable-nvenc` configuration flags during compilation. For compilation follow [these instructions ➶](https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/#prerequisites)

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

        ??? danger "Verifying H.264 NVENC encoder support in FFmpeg"

            To use NVENC Video-encoder(`cuda`), remember to check if your FFmpeg compiled with H.264 NVENC encoder support. You can easily do this by executing following one-liner command in your terminal, and observing if output contains something similar as follows:

            ```sh
            $ ffmpeg  -hide_banner -encoders | grep nvenc 

            V....D av1_nvenc            NVIDIA NVENC av1 encoder (codec av1)
            V....D h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
            V....D hevc_nvenc           NVIDIA NVENC hevc encoder (codec hevc)
            ```

            !!! note "You can also use other NVENC encoder in the similar way, if supported."
        

    - You already have appropriate Nvidia video drivers and related softwares installed on your machine.
    - If the stream is not decodable in hardware (for example, it is an unsupported codec or profile) then it will still be decoded in software automatically, but hardware filters won't be applicable.

    These assumptions **MAY/MAY NOT** suit your current setup. Kindly use suitable parameters based your system platform and hardware settings only.


In this example, we will:

1. Use NVIDIA’s **CUDA internal hardware-accelerated decoder (`cuda`)** with the FFGear API to automatically select the best NV-accelerated video codec while keeping decoded frames in GPU memory for hardware-accelerated filtering.
2. Apply GPU-accelerated scaling and cropping directly in GPU memory.
3. Download processed frames into system memory as patched **NV12** frames.
4. Convert **NV12** frames into the **BGR24** pixel format using the [`-enforce_cv_patch`](../params/#b-ffdecoder-parameters) flag together with OpenCV’s `cvtColor()` method for OpenCV compatibility.
5. Encode the resulting **BGR24** frames using OpenCV’s `VideoWriter` API.

!!! tip "You can use FFGear's [`stream.stream.metadata`](../../params/#a-ffgear-parameters) property object that dumps source Video's metadata information _(as JSON string)_ to retrieve source framerate."

!!! info "More information on Nvidia's NVENC Encoder can be found [here ➶](https://developer.nvidia.com/blog/nvidia-ffmpeg-transcoding-guide/)"

```python linenums="1" hl_lines="7-20 26 32 54 59"
# import the necessary packages
from vidgear.gears import FFGear
import cv2, json

# CUDA hwaccel: decode in GPU memory, scale & crop in GPU, download as NV12
options = {
    "-vcodec": None,               # skip source decoder, let FFmpeg choose
    "-enforce_cv_patch": True,     # auto-convert NV12 → OpenCV Compatible in FFGear
    "-ffprefixes": [
        "-vsync", "0",             # prevent duplicate frames
        "-hwaccel", "cuda",        # use CUDA accelerator
        "-hwaccel_output_format", "cuda",  # keep frames in GPU memory
    ],
    "-custom_resolution": "null",  # discard source resolution param
    "-framerate": "null",          # discard source framerate param
    "-vf": (
        "scale_cuda=640:360,"      # scale to 640x360 in GPU memory
        "crop=80:60:200:100,"      # crop a 80×60 section from position (200, 100) in GPU memory
        "hwdownload,"              # download to system memory
        "format=nv12"              # convert to NV12 pixel format
    ),
}

stream = FFGear(
    source="foo.mp4",
    frame_format="null",           # discard source pixel format
    logging=True,
    **options
).start()

# retrieve JSON Metadata and convert it to dict
metadata_dict = json.loads(stream.stream.metadata)

# prepare OpenCV parameters
FOURCC = cv2.VideoWriter_fourcc("M", "J", "P", "G")
FRAMERATE = metadata_dict["output_framerate"]
FRAMESIZE = tuple(metadata_dict["output_frames_resolution"])

# Define writer with parameters and suitable output filename for e.g. `output_foo.avi`
writer = cv2.VideoWriter("output_foo.avi", FOURCC, FRAMERATE, FRAMESIZE)

# loop over
while True:

    # read NV12 frames (auto-converted from NV12)
    frame = stream.read()

    # check if frame is None
    if frame is None:
        break

    # convert it to `BGR` pixel format,
    # since write() method only accepts `BGR` frames
    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)

    # {do something with the BGR frame here}

    # writing BGR frame to writer
    writer.write(frame)

# safely close video stream
stream.stop()

# safely close writer
writer.release()
```

### CUDA-NVENC-accelerated Transcoding with WriteGear API (Compression Mode)

> FFGear API in conjunction with WriteGear API (Compression Mode) creates a high-level **High-performance Lossless FFmpeg Transcoding _(Decoding + Encoding)_ Pipeline** :fire: that is able to exploit almost any FFmpeg parameter for achieving anything imaginable with multimedia video data all while allow us to manipulate the real-time video frames with immense flexibility.

???+ alert "Example Assumptions"

    The following example explicitly assumes:

    - You're running :fontawesome-brands-linux: Linux operating system with a [**supported NVIDIA GPU**](https://developer.nvidia.com/nvidia-video-codec-sdk).
    - You're using FFmpeg 4.4 or newer, configured with at least `--enable-nonfree --enable-cuda-nvcc --enable-libnpp  --enable-cuvid --enable-nvenc` configuration flags during compilation. For compilation follow [these instructions ➶](https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/#prerequisites)

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

        ??? danger "Verifying H.264 NVENC encoder support in FFmpeg"

            To use NVENC Video-encoder(`cuda`), remember to check if your FFmpeg compiled with H.264 NVENC encoder support. You can easily do this by executing following one-liner command in your terminal, and observing if output contains something similar as follows:

            ```sh
            $ ffmpeg  -hide_banner -encoders | grep nvenc 

            V....D av1_nvenc            NVIDIA NVENC av1 encoder (codec av1)
            V....D h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
            V....D hevc_nvenc           NVIDIA NVENC hevc encoder (codec hevc)
            ```

            !!! note "You can also use other NVENC encoder in the similar way, if supported."
        

    - You already have appropriate Nvidia video drivers and related softwares installed on your machine.
    - If the stream is not decodable in hardware (for example, it is an unsupported codec or profile) then it will still be decoded in software automatically, but hardware filters won't be applicable.

    These assumptions **MAY/MAY NOT** suit your current setup. Kindly use suitable parameters based your system platform and hardware settings only.

??? info "Additional Parameters in WriteGear API"
    WriteGear API only requires a valid Output filename _(e.g. `output_foo.mp4`)_ as input, but you can easily control any output specifications _(such as bitrate, codec, framerate, resolution, subtitles, etc.)_ supported by FFmpeg _(in use)_.

!!! tip "You can use FFGear's [`stream.stream.metadata`](../../params/#a-ffgear-parameters) property object that dumps source Video's metadata information _(as JSON string)_ to retrieve source framerate."

=== "Transcoding BGR frames"

    In this example, we will:

    1. Use NVIDIA’s **CUDA internal hardware-accelerated decoder (`cuda`)** with the FFGear API to automatically select the best NV-accelerated video codec while keeping decoded frames in GPU memory for hardware-accelerated filtering.
    2. Apply GPU-accelerated scaling and cropping directly in GPU memory.
    3. Download processed frames into system memory as patched **NV12** frames.
    4. Convert **NV12** frames into the **BGR24** pixel format using the [`-enforce_cv_patch`](../params/#b-ffdecoder-parameters) flag together with OpenCV’s `cvtColor()` method for OpenCV compatibility.
    5. Encode the resulting **BGR24** frames with the WriteGear API using NVIDIA’s hardware-accelerated **H.264 NVENC video encoder (`h264_nvenc`)** to generate a lossless output video.

    ```python linenums="1" hl_lines="8-21 27 35-36 54 59"
    # import the necessary packages
    from vidgear.gears import FFGear
    from vidgear.gears import WriteGear
    import cv2, json

    # CUDA hwaccel: decode in GPU memory, scale & crop in GPU, download as NV12
    options = {
        "-vcodec": None,               # skip source decoder, let FFmpeg choose
        "-enforce_cv_patch": True,     # auto-convert NV12 → OpenCV Compatible in FFGear
        "-ffprefixes": [
            "-vsync", "0",             # prevent duplicate frames
            "-hwaccel", "cuda",        # use CUDA accelerator
            "-hwaccel_output_format", "cuda",  # keep frames in GPU memory
        ],
        "-custom_resolution": "null",  # discard source resolution param
        "-framerate": "null",          # discard source framerate param
        "-vf": (
            "scale_cuda=640:360,"      # scale to 640x360 in GPU memory
            "crop=80:60:200:100,"      # crop a 80×60 section from position (200, 100) in GPU memory
            "hwdownload,"              # download to system memory
            "format=nv12"              # convert to NV12 pixel format
        ),
    }

    stream = FFGear(
        source="foo.mp4",
        frame_format="null",           # discard source pixel format
        logging=True,
        **options
    ).start()

    # retrieve framerate from JSON Metadata and pass it as
    # `-input_framerate` parameter for controlled framerate
    output_params = {
        "-input_framerate": json.loads(stream.stream.metadata)["output_framerate"],
        "-vcodec": "h264_nvenc", # H.264 NVENC Video-encoder
    }

    # Define writer with default parameters and suitable
    # output filename for e.g. `output_foo.mp4`
    writer = WriteGear(output="output_foo.mp4", logging=True, **output_params)

    # loop over
    while True:

        # read NV12 frames (auto-converted from NV12)
        frame = stream.read()

        # check if frame is None
        if frame is None:
            break

        # convert it to `BGR` pixel format
        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)

        # {do something with the BGR frame here}

        # writing BGR frame to writer
        writer.write(frame)

    # safely close video stream
    stream.stop()

    # safely close writer
    writer.close()
    ```
    
=== "Transcoding NV12 frames :material-rocket-launch:"

    In this example, we will:

    1. Use NVIDIA’s **CUDA internal hardware-accelerated decoder (`cuda`)** with the FFGear API to automatically select the best NV-accelerated video codec while keeping decoded frames in GPU memory for hardware-accelerated filtering.
    2. Apply GPU-accelerated scaling and cropping directly in GPU memory.
    3. Download processed frames into system memory as patched **NV12** frames.
    4. Encode the resulting **NV12** frames directly with the WriteGear API using NVIDIA’s hardware-accelerated **H.264 NVENC video encoder (`h264_nvenc`)** to generate a lossless output video.


    ```python linenums="1" hl_lines="8-21 26 34-36 57"
    # import the necessary packages
    from vidgear.gears import FFGear
    from vidgear.gears import WriteGear
    import json

    # CUDA hwaccel: decode in GPU memory, scale & crop in GPU, download as NV12
    options = {
        "-vcodec": None,               # skip source decoder, let FFmpeg choose
        "-ffprefixes": [
            "-vsync", "0",             # prevent duplicate frames
            "-hwaccel", "cuda",        # use CUDA accelerator
            "-hwaccel_output_format", "cuda",  # keep frames in GPU memory
        ],
        "-custom_resolution": "null",  # discard source resolution param
        "-framerate": "null",          # discard source framerate param
        "-vf": (
            "scale_cuda=640:360,"      # scale to 640x360 in GPU memory
            "crop=80:60:200:100,"      # crop a 80×60 section from position (200, 100) in GPU memory
            "hwdownload,"              # download to system memory
            "format=nv12"              # convert to NV12 pixel format
        ),
    }

    stream = FFGear(
        source="foo.mp4",
        frame_format="null",           # discard source pixel format
        logging=True,
        **options
    ).start()

    # retrieve framerate from JSON Metadata and pass it as
    # `-input_framerate` parameter for controlled framerate
    output_params = {
        "-input_framerate": json.loads(stream.stream.metadata)["output_framerate"],
        "-vcodec": "h264_nvenc", # H.264 NVENC Video-encoder
        "-input_pixfmt": "nv12", # input frames pixel format as `NV12`
    }

    # Define writer with default parameters and suitable
    # output filename for e.g. `output_foo.mp4`
    writer = WriteGear(output="output_foo.mp4", logging=True, **output_params)

    # loop over
    while True:

        # read NV12 frames
        frame = stream.read()

        # check if frame is None
        if frame is None:
            break

        # {do something with the NV12 frame here}

        # writing NV12 frame to writer
        writer.write(frame)

    # safely close video stream
    stream.stop()

    # safely close writer
    writer.close()
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

??? warning "Incompatible with `-filter_complex`"

    The `-extract_metadata` cannot be combined with the `-filter_complex` attribute. If both are supplied, a warning is logged and metadata extraction is silently disabled. A pre-existing `-vf` is fine — `showinfo` is automatically appended onto it.

### Keyframe-Only Decoding for AI Inference

> Many Computer Vision workflows — perceptual hashing, scene-change detection, and AI model inference *(YOLO, ResNet, etc.)* — only need to process **Keyframes (I-frames)**. Running inference on every P/B frame wastes compute, since most frames contain only small incremental changes. With the `-extract_metadata` option, you can detect keyframes and run inference only on those frames while skipping all non-keyframes. This can reduce unnecessary decoding and inference workloads by **upto 98%**, significantly improving pipeline efficiency and lowering compute usage.

In this example we run heavy AI model on keyframes only and skipping all non-keyframes:

```python linenums="1" hl_lines="5 17-33"
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

    # returns (frame, metadata) tuple when `-extract_metadata` is enabled
    output = stream.read()

    # check if output is None (end of stream)
    if output is None:
        break
    
    # Unpacking tuple
    frame, meta = output

    # [OPTIMIZATION]: skip non-keyframes entirely
    if meta and not meta["is_keyframe"]: 
        continue # <-- Skips Non-key frames (P, B-frames)

    # Run Heavy AI model inference/Prediction on keyframes (I-frames) only 
    # results = my_heavy_ai_model.predict(frame)
    print(f"Keyframe #{meta['frame_num']} at {meta['pts_time']:.3f}s")

# safely close video stream
stream.stop()
```

!!! success "Depending on the GOP size (Group of Pictures), the optimization pattern shown above can reduce downstream processing time by **10–50×** without missing any scene-boundary information."

!!! example "Checkout this [bonus example ➶](../../../help/ffgear_ex/#using-ffgear-for-real-time-aiml-video-inference) demonstrating FFGear optimizing **YOLOv10-Nano** model inference by processing only Keyframes (I-frames) while skipping all non-keyframes (P/B-frames)."

&nbsp;

### Variable-Frame-Rate (VFR) Synchronization

> Most modern video sources — smartphones, screen recordings, webcams — are **Variable-Frame-Rate**. Assuming a constant frame rate will drift out of sync very quickly.

In this example we use `meta["pts_time"]` to get the **exact presentation timestamp** of every frame and skip frames based on the actual time difference between frames:

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

### Apply custom Watermark Image Overlay Filter

<figure markdown>
  <img src="https://gitlab.com/abhiTronix/Imbakup/-/raw/master/Images/vidgear/ffgear_watermark.gif" alt="Big Buck Bunny with watermark" loading="lazy" width=85%/>
  <figcaption>Big Buck Bunny with custom watermark</figcaption>
</figure>

In this example, we apply a watermark image _(say `watermark.png` with transparent background)_ overlay to the `10` seconds of video file _(say `foo.mp4`)_ using FFmpeg's [`overlay`](https://ffmpeg.org/ffmpeg-filters.html#toc-overlay-1) filter with some additional filtering:

!!! info "You can use FFGear API's [`stream.stream.metadata`](../../params/#a-ffgear-parameters) property object that dumps Source Metadata as JSON to retrieve source framerate and frame-size."

!!! danger "Remember to replace `watermark.png` watermark image absolute file path with yours before executing this example."

```python linenums="1" hl_lines="8-18 27"
# import the necessary packages
from vidgear.gears import FFGear
from vidgear.gears import WriteGear
import json, cv2

# define the Complex Video Filter with additional `watermark.png` image input
options = {
    "-ffprefixes": ["-t", "10"],  # playback time of 10 seconds
    "-clones": [
        "-i",
        "watermark.png",  # !!! [WARNING] define your absolute `watermark.png` path here.
    ],
    "-filter_complex": "[1]scale=256:-1,format=rgba,"  # change 2nd(image) input to 256 pixels wide and yuv444p format
    + "colorchannelmixer=aa=0.7[logo];"  # apply colorchannelmixer to image for controlling alpha [logo]
    + "[0][logo]overlay=W-w-{pixel}:H-h-{pixel}:format=auto,".format(  # apply overlay to 1st(video) with [logo]
        pixel=5  # at 5 pixels from the bottom right corner of the input video
    )
    + "format=bgr24",  # change output format to `bgr24`
}

# open source with FFGear and BGR frames with given params
stream = FFGear(source="myvideo.mp4", frame_format="bgr24", logging=True, **options).start()

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

### Generate Mandelbrot test pattern with Vectorscope & Waveforms

> The [`mandelbrot`](https://ffmpeg.org/ffmpeg-filters.html#toc-mandelbrot) graph generate a [**Mandelbrot set fractal**](https://en.wikipedia.org/wiki/Mandelbrot_set), that progressively zoom towards a specific point.

<figure markdown>
  <img src="https://abhitronix.github.io/deffcode/latest/assets/gifs/mandelbrot_vectorscope_waveforms.gif" alt="Mandelbrot Test Pattern" loading="lazy" />
  <figcaption>Mandelbrot pattern with a Vectorscope & two Waveforms</figcaption>
</figure>

In this example, we generate a `10`-second **Mandelbrot test pattern** *(`1280x720` resolution at `30 FPS`)* using the [`mandelbrot`](https://ffmpeg.org/ffmpeg-filters.html#toc-mandelbrot) filter source with the `lavfi` virtual input device. We then stack a [vectorscope](https://www.studiobinder.com/blog/what-is-a-vectorscope-definition/) *(visualizes color component relationships)* and two [waveform](https://ffmpeg.org/ffmpeg-filters.html#toc-waveform) monitors *(visualize YUV color intensity levels)* alongside the video, and save the final output using the WriteGear API:


```python linenums="1" hl_lines="8-16 22-23 32"
# import required libraries
from vidgear.gears import FFGear
from vidgear.gears import WriteGear
import json

# define parameters
options = {
    "-ffprefixes": ["-t", "10"],  # playback time of 10 seconds
    "-vf": "format=yuv444p," # change input format to yuv444p
    + "split=4[a][b][c][d]," # split input into 4 identical outputs.
    + "[a]waveform[aa],"  # apply waveform on first output
    + "[b][aa]vstack[V],"  # vertical stack 2nd output with waveform [V]
    + "[c]waveform=m=0[cc],"  # apply waveform on 3rd output
    + "[d]vectorscope=color4[dd],"  # apply vectorscope on 4th output
    + "[cc][dd]vstack[V2],"  # vertical stack waveform and vectorscope [V2]
    + "[V][V2]hstack",  # horizontal stack [V] and [V2] vertical stacks
}

# stream with "mandelbrot" source of
# `1280x720` frame size and `30` framerate for BGR24 output
stream = FFGear(
    source="mandelbrot=size=1280x720:rate=30",
    source_demuxer="lavfi",
    frame_format="bgr24",
    logging=True,
    **options
).start()

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

&thinsp;

[deffcode]:https://github.com/abhiTronix/deffcode
