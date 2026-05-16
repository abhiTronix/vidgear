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

# VideoGear API Usage Examples:

!!! example "After going through following Usage Examples, Checkout more of its advanced configurations [here ➶](../../../help/videogear_ex/)"

&thinsp;


## Bare-Minimum Usage with CamGear backend

!!! abstract "VideoGear by **default** provides direct internal access to [CamGear API](../../camgear/). Explicitly select it with `api=Backend.CAMGEAR`."

Following is the bare-minimum code you need to access CamGear API with VideoGear:

```python linenums="1" hl_lines="3 8"
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears.helper import Backend
import cv2


# open any valid video stream(for e.g `myvideo.mp4` file)
stream = VideoGear(api=Backend.CAMGEAR, source="myvideo.mp4").start()

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

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

```

&nbsp; 

## Bare-Minimum Usage with FFGear backend

!!! abstract "Select FFGear backend with `api=Backend.FFGEAR` to access [FFGear API](../../ffgear/) for hardware-accelerated FFmpeg-powered decoding."

Following is the bare-minimum code you need to access FFGear API with VideoGear:

??? danger "FFGear Backend requires the `deffcode` library"

    FFGear API **MUST** have the [`deffcode`][deffcode] library installed, along with a valid FFmpeg executable. Any failure in detection will raise `ImportError`/`RuntimeError` immediately.

    Install via pip:

    ```sh
    pip install deffcode
    ```

    For FFmpeg installation, see [FFmpeg Installation ➶](../../ffgear/advanced/ffmpeg_install/)

!!! note "FFGear API Backend does **not** support `colorspace` or `time_delay` parameters. Use its `options` dict for advanced FFmpeg configuration."

```python linenums="1" hl_lines="3 8"
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears.helper import Backend
import cv2

# select FFGear backend via api parameter
# and open any valid video stream(for e.g `myvideo.mp4` file)
stream = VideoGear(api=Backend.FFGEAR, source="myvideo.mp4").start()

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

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
```

&nbsp;

## Bare-Minimum Usage with PiGear backend

!!! abstract "Select PiGear backend with `api=Backend.PIGEAR` to access [PiGear API](../../pigear/)."

Following is the bare-minimum code you need to access PiGear API with VideoGear:

??? info "Under the hood, PiGear API _(version `0.3.3` onwards)_ prioritizes the new [`picamera2`](https://github.com/raspberrypi/picamera2) API backend."

    However, PiGear API seamlessly switches to the legacy [`picamera`](https://picamera.readthedocs.io/en/release-1.13/index.html) backend, if the `picamera2` library is unavailable or not installed.
    
    !!! tip "It is advised to enable logging(`logging=True`) to see which backend is being used."

    !!! failure "The `picamera` library is built on the legacy camera stack that is NOT _(and never has been)_ supported on 64-bit OS builds."

    !!! note "You could also enforce the legacy picamera API backend in PiGear by using the [`enforce_legacy_picamera`](../params) user-defined optional parameter boolean attribute."

!!! warning "Make sure to [complete Raspberry Pi Camera Hardware-specific settings](https://www.raspberrypi.com/documentation/accessories/camera.html#installing-a-raspberry-pi-camera) prior using this API, otherwise nothing will work."

```python linenums="1" hl_lines="3 7"
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears.helper import Backend
import cv2

# select PiGear backend via api parameter
stream = VideoGear(api=Backend.PIGEAR).start()

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

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
```

&nbsp;

## Using VideoGear with Video Stabilizer backend

!!! abstract "VideoGear API provides a special internal wrapper around VidGear's Exclusive [**Video Stabilizer**](../../stabilizer/) class and provides easy way of activating stabilization for various video-streams _(real-time or not)_ with its [`stabilize`](../params/#stabilize) boolean parameter during initialization."

The usage example is as follows:

!!! tip "For a more detailed information on Video-Stabilizer Class, Read [here ➶](../../stabilizer/)"

!!! warning "The stabilizer might be slower for High-Quality/Resolution videos-frames."

```python linenums="1" hl_lines="7"
# import required libraries
from vidgear.gears import VideoGear
import numpy as np
import cv2

# open any valid video stream with stabilization enabled(`stabilize = True`)
stream_stab = VideoGear(source="test.mp4", stabilize=True).start()

# loop over
while True:

    # read stabilized frames
    frame_stab = stream_stab.read()

    # check for stabilized frame if None-type
    if frame_stab is None:
        break

    # {do something with the frame here}

    # Show output window
    cv2.imshow("Stabilized Output", frame_stab)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close streams
stream_stab.stop()
```

&nbsp;

## Advanced VideoGear usage with CamGear Backend

!!! abstract "VideoGear provides internal access to CamGear, PiGear, and FFGear APIs, and thereby all additional parameters of each backend are also easily accessible within VideoGear API."

The usage example of VideoGear API with Variable Camera Properties is as follows:

!!! info "This example demonstrates how to use the VideoGear API in a similar manner to the CamGear's [example](../../camgear/usage/#using-camgear-with-variable-camera-properties) for controlling variable source properties. Any [CamGear usage example](../../camgear/usage/) can be implemented using the VideoGear API in a similar way."

!!! tip "All the supported Source Tweak Parameters can be found [here ➶](../../camgear/advanced/source_params/#source-tweak-parameters-for-camgear-api)"

```python linenums="1" hl_lines="9-11"
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears.helper import Backend
import cv2


# define suitable tweak parameters for your stream.
options = {
    "CAP_PROP_FRAME_WIDTH": 320, # resolution 320x240
    "CAP_PROP_FRAME_HEIGHT": 240,
    "CAP_PROP_FPS": 60, # framerate 60fps
}

# To open live video stream on webcam at first index(i.e. 0) 
# device and apply source tweak parameters
stream = VideoGear(api=Backend.CAMGEAR, source=0, logging=True, **options).start()

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

## Advanced VideoGear usage with PiGear Backend

!!! abstract "VideoGear provides internal access to both CamGear and PiGear APIs, and thereby all additional parameters of [PiGear API](../params/#parameters-with-pigear-backend) or [CamGear API](../params/#parameters-with-camgear-backend) are also easily accessible within VideoGear API."

The usage example of VideoGear API with Variable Camera Properties is as follows:

!!! info "This example demonstrates how to use the VideoGear API in a similar manner to the PiGear's [example](../../pigear/usage/#using-pigear-with-variable-camera-properties) for using variable camera properties. Any [PiGear usage example](../../pigear/usage/) can be implemented using the VideoGear API in a similar way."

!!! new "Backend PiGear API now fully supports the newer [`picamera2`](https://github.com/raspberrypi/picamera2) python library under the hood for Raspberry Pi :fontawesome-brands-raspberry-pi: camera modules. Follow this [guide ➶](../../../installation/pip_install/#picamera2) for its installation."

!!! warning "Make sure to [complete Raspberry Pi Camera Hardware-specific settings](https://www.raspberrypi.com/documentation/accessories/camera.html#installing-a-raspberry-pi-camera) prior using this backend, otherwise nothing will work."


=== "New Picamera2 backend"

    ```python linenums="1" hl_lines="4 10-14"
    # import required libraries
    from vidgear.gears import VideoGear
    from vidgear.gears.helper import Backend
    from libcamera import Transform
    import cv2

    # formulate various Picamera2 API 
    # configurational parameters
    options = {
        "queue": True,
        "buffer_count": 4,
        "controls": {"Brightness": 0.5, "ExposureValue": 2.0},
        "transform": Transform(hflip=1),
        "auto_align_output_config": True,  # auto-align camera configuration
    }

    # open pi video stream with defined parameters
    stream = VideoGear(api=Backend.PIGEAR, resolution=(640, 480), framerate=60, logging=True, **options).start()

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

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
    ```
    
=== "Legacy Picamera backend"

    ??? info "Under the hood, Backend PiGear API _(version `0.3.3` onwards)_ prioritizes the new [`picamera2`](https://github.com/raspberrypi/picamera2) API backend."

        However, the API seamlessly switches to the legacy [`picamera`](https://picamera.readthedocs.io/en/release-1.13/index.html) backend, if the `picamera2` library is unavailable or not installed.
        
        !!! tip "It is advised to enable logging(`logging=True`) to see which backend is being used."

        !!! failure "The `picamera` library is built on the legacy camera stack that is NOT _(and never has been)_ supported on 64-bit OS builds."

        !!! note "You could also enforce the legacy picamera API backend in PiGear by using the [`enforce_legacy_picamera`](../params) user-defined optional parameter boolean attribute."

    ```python linenums="1" hl_lines="9-14"
    # import required libraries
    from vidgear.gears import VideoGear
    from vidgear.gears.helper import Backend
    import cv2

    # formulate various Picamera API 
    # configurational parameters
    options = {
        "hflip": True,
        "exposure_mode": "auto",
        "iso": 800,
        "exposure_compensation": 15,
        "awb_mode": "horizon",
        "sensor_mode": 0,
    }

    # open pi video stream with defined parameters
    stream = VideoGear(api=Backend.PIGEAR, resolution=(640, 480), framerate=60, logging=True, **options).start()

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

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
    ```

&nbsp; 


## Advanced VideoGear usage with FFGear Backend

??? danger "FFGear Backend requires the `deffcode` library"

    FFGear API **MUST** have the [`deffcode`][deffcode] library installed, along with a valid FFmpeg executable. Any failure in detection will raise `ImportError`/`RuntimeError` immediately.

    Install via pip:

    ```sh
    pip install deffcode
    ```

    For FFmpeg installation, see [FFmpeg Installation ➶](../../ffgear/advanced/ffmpeg_install/)

The usage example below demonstrates hardware-accelerated decoding and stabilization together using the FFGear backend:

!!! info "This example demonstrates how to use the VideoGear API in a similar manner to the FFGear's [NVIDIA CUVID Decoding example](../../ffgear/advanced/#nvidia-cuvid-decoding). Any [FFGear usage example](../../ffgear/usage/) can be implemented using the VideoGear API in a similar way."

!!! tip "For hardware-accelerated decoding options (CUDA/CUVID etc.) and advanced FFmpeg filter pipelines, see [FFGear Hardware-Accelerated Decoding ➶](../../ffgear/advanced/#hardware-accelerated-decoding)"

```python linenums="1" hl_lines="9-11 16"
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears.helper import Backend
import cv2

# pass FFmpeg hardware-acceleration and filter options via `options`
# use H.264 CUVID hardware decoder; enable OpenCV patch for YUV frames
options = {
    "-vcodec": "h264_cuvid",  # NVIDIA CUVID hardware decoder
    "-enforce_cv_patch": True,  # auto-convert YUV420p → BGR in FFGear
    "-vf": "scale=1280:720",  # apply scale filter
}

# open video with FFGear backend using hardware acceleration and stabilization enabled
stream = VideoGear(
    api=Backend.FFGEAR, source="myvideo.mp4", stabilize=True, logging=True, **options
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
    cv2.imshow("Output Frame", frame)

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

## Advanced VideoGear usage with Stabilizer Backend

??? new "New in v0.3.5"
    The `STABILIZER_MODE` option and the `StabilizerMode` enum were added in `v0.3.5`. Omitting `STABILIZER_MODE` keeps the previous behavior — VideoGear defaults to `StabilizerMode.ASW`.

When stabilization is enabled, VideoGear forwards Stabilizer parameters via the [`options`](../params/#options) dictionary:

```python linenums="1" hl_lines="2 8-11"
# import required libraries
from vidgear.gears.stabilizer import StabilizerMode
from vidgear.gears import VideoGear
import cv2

# configure the stabilizer backend + tuning parameters
options = {
    "STABILIZER_MODE": StabilizerMode.ASW,  # default
    "SMOOTHING_RADIUS": 30,
    "BORDER_SIZE": 5,
    "CROP_N_ZOOM": True,
}

# open any valid video stream with stabilization enabled
stream_stab = VideoGear(
    source="test.mp4", stabilize=True, logging=True, **options
).start()

# loop over
while True:

    # read stabilized frames
    frame_stab = stream_stab.read()

    # check for stabilized frame if None-type
    if frame_stab is None:
        break

    # {do something with the frame here}

    cv2.imshow("Stabilized Output", frame_stab)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
stream_stab.stop()
```

!!! warning "`StabilizerMode.KALMAN` is reserved for a future release and currently raises `NotImplementedError`. Non-`StabilizerMode` values passed via `STABILIZER_MODE` silently fall back to `StabilizerMode.ASW`."

&nbsp;

## Using VideoGear with Colorspace Manipulation

VideoGear API also supports **Colorspace Manipulation** but **NOT Direct** like other VideoCapture Gears. 

!!! failure "Important: `color_space` global variable is NOT Supported in VideoGear API"

    * `color_space` global variable is **NOT Supported** in VideoGear API, calling it will result in `AttribueError`. More details can be found [here ➶](../../../bonus/colorspace_manipulation/#source-colorspace-manipulation)
    * Any incorrect or None-type value on [`colorspace`](../params/#colorspace) parameter will be skipped automatically.

!!! danger "FFGear API Backend does **not** support `colorspace` parameter. See [this ➶](../../ffgear/usage/#using-ffgear-with-different-pixel-formats) for source pixel-format conversions."

In following example code, we will convert source colorspace to [**HSV**](https://en.wikipedia.org/wiki/HSL_and_HSV) on initialization:


```python linenums="1" hl_lines="7"
# import required libraries
from vidgear.gears import VideoGear
import cv2

# Open any source of your choice, like Webcam first index(i.e. 0) 
# and change its colorspace to `HSV`
stream = VideoGear(source=0, colorspace="COLOR_BGR2HSV", logging=True).start()

# loop over
while True:

    # read HSV frames
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the HSV frame here}

    # Show output window
    cv2.imshow("Output Frame", frame)

    # check for key if pressed
    key = cv2.waitKey(1) & 0xFF

    # check for 'q' key is pressed
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
```

&nbsp;

## Bonus Examples

!!! example "Checkout more advanced VideoGear examples with unusual configuration [here ➶](../../../help/videogear_ex/)"

&nbsp;

[deffcode]:https://github.com/abhiTronix/deffcode
