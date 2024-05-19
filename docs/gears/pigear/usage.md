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

# PiGear API Usage Examples:

!!! new "PiGear API now fully supports the newer [`picamera2`](https://github.com/raspberrypi/picamera2) python library under the hood for Raspberry Pi :fontawesome-brands-raspberry-pi: camera modules. Follow this [guide ➶](../../../installation/pip_install/#picamera2) for its installation."

!!! warning "Make sure to [complete Raspberry Pi Camera Hardware-specific settings](https://www.raspberrypi.com/documentation/accessories/camera.html#installing-a-raspberry-pi-camera) prior using this API, otherwise nothing will work."

!!! example "After going through following Usage Examples, Checkout more of its advanced configurations [here ➶](../../../help/pigear_ex/)"



&thinsp;


## Bare-Minimum Usage

Following is the bare-minimum code you need to get started with PiGear API:

??? info "Under the hood, PiGear API _(version `0.3.3` onwards)_ prioritizes the new [`picamera2`](https://github.com/raspberrypi/picamera2) API backend."

    However, PiGear API seamlessly switches to the legacy [`picamera`](https://picamera.readthedocs.io/en/release-1.13/index.html) backend, if the `picamera2` library is unavailable or not installed.
    
    !!! tip "It is advised to enable logging(`logging=True`) to see which backend is being used."

    !!! note "You could also enforce the legacy picamera API backend in PiGear by using the [`enforce_legacy_picamera`](../params/#b-user-defined-parameters) user-defined optional parameter boolean attribute."

??? danger "Disabling common `libcamera` API messages in silent mode."
    
    The picamera2 backend can be a bit verbose with logging messages from the underlying `libcamera` library, even when logging is disabled (`logging=False`) in the PiGear API. 

    - [x] To suppress these messages, you'll need to set `LIBCAMERA_LOG_LEVELS=2` environment variable before running your application. This will disable common `libcamera` API messages, keeping your console output cleaner.
    - [x] This can be done on various Operating Systems as follows:

    === ":material-linux: Linux"

        ```sh
        # path to file
        export LIBCAMERA_LOG_LEVELS=2
        ```

    === ":fontawesome-brands-windows: Windows (Powershell)"

        ```powershell
        # path to file
        $Env:LIBCAMERA_LOG_LEVELS=2
        ```

    === ":material-apple: MacOS"
        
        ```sh
        # path to file
        export LIBCAMERA_LOG_LEVELS=2
        ```

```python linenums="1"
# import required libraries
from vidgear.gears import PiGear
import cv2

# open stream with default parameters
stream = PiGear().start()

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

## Using PiGear with Variable Camera Properties

=== "New Picamera2 backend"

    > PiGear provides a user-friendly interface for the underlying picamera2 library, offering access to almost all of its important configurational parameters. It simplifies configuration for developers with even basic knowledge of Raspberry Pi camera modules, allowing them to easily configure and control the camera functionality with just a few lines of code.
    
    This example doc showcases the capabilities of PiGear and demonstrates how it simplifies camera configuration with Picamera2 API backend.

    ??? info "All supported Picamera2 Library Configurational Parameters [IMPORTANT]"

        Following are the list of Picamera2 parameters, i.e. if supported, can be applied to the source stream in PiGear API through its [`options`](../params/#options) dictionary parameter by formatting them as its attributes.

        ???+ warning "Few Important points"
            - These PiCamera2 parameters must be formatted as PiGear API's [`options`](../params/#options) dictionary parameter keys, and their values **MUST** strictly adhere to the specified data types _(see table below)_. If the values do not follow the specified data types, they will be discarded.
            - PiGear API only defines the default `main` stream configuration that is delivered to the PiCamera2 API. You **CANNOT** define other streams _(such as `lores`, `raw`)_ manually.
            -  The `FrameDuration` and `FrameDurationLimits` properties of [`control` configurational parameter](../params/#a-configurational-camera-parameters) are **NOT** supported and will be discarded, since camera FPS is handled by [`framerate`](../params/#framerate) parameter in PiGear API. 
            - The [`resolution`](../params/#resolution) parameter will be **OVERRIDDEN**, if the user explicitly defines the `output_size` property of the [`sensor` configurational parameter](../params/#a-configurational-camera-parameters) in PiGear API.

        | Parameters | Datatype | Description | Supported | Supported on USB Cameras| Remarks |
        |:----------:|:-----:|:-------------:|:-----------:|:--------:|:-------:|
        | `buffer_count` | `int`, `>=1` | number of sets of buffers to allocate for the camera system | :white_check_mark: | :x: | Read Docs [here ➶](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf) |
        | `queue` |  `bool` | whether the system is allowed to queue up a frame ready for a capture request | :white_check_mark: | :x: | Read Docs [here ➶](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)|
        | `controls` | `dict` | specify a set of runtime controls that can be regarded as part of the camera configuration | :white_check_mark: | :x: | Read Docs [here ➶](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf) |
        | `sensor` | `dict` | allow to select a particular mode of operation for the sensor | :white_check_mark: | :white_check_mark: | Read Docs [here ➶](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf) |
        | `format` | `str` | Pixel formats | :white_check_mark: | :white_check_mark: | Read Docs [here ➶](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf) and see [Bonus example ➶](../../../help/pigear_ex/#changing-output-pixel-format-in-pigear-api-with-picamera2-backend) |
        | `transform` | `Transform`[^1] | The 2D plane transform that is applied to all images from all the configured streams.  | :white_check_mark: | :x: | Read Docs [here ➶](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf) |
        | `colour_space` | :octicons-dash-16: | colour space of the output images |  :x: | :no_entry_sign: | Handled by [`colorspace`](../params/#colorspace) parameter of PiGear API |
        | `size` | :octicons-dash-16: | A tuple of two values giving the width and height of the output image. _(Both numbers should be no less than 64)_ | :x: | :no_entry_sign: | Handled by [`resolution`](../params/#resolution)  parameter of PiGear API |
        | `display` | :octicons-dash-16: | name of the stream that will be displayed in the preview window.  | :x: | :no_entry_sign: | Not-Required |
        | `encode` | :octicons-dash-16: | name of the stream that will be used for video recording.   | :x: | :no_entry_sign: | Not-Required |

    ??? failure "Limited support for USB Cameras"

        This example also works with USB Cameras, However:

        - Users should assume that features such as: **Camera controls** (`"controls"`), **Transformations** (`"transform"`), **Queue** (`"queue"`) , and **Buffer Count** (`"buffer_count"`) that are supported on Raspberry Pi cameras, and so forth, are not available on USB Cameras. 
        - Hot-plugging of USB cameras is also **NOT** supported - PiGear API should be completely shut down and restarted when cameras are added or removed.

    ??? tip "Enabling verbose logs for backend PiCamera2 Library"
        The PiGear API allows you to enable more detailed logging from the `picamera2` backend library using the [`enable_verbose_logs`](../params/#b-user-defined-parameters) user-defined optional parameter attribute. This can be used in conjunction with enabling general logging (`logging=True`) in the PiGear API for even more granular control over logging output.

    !!! example "PiGear also support changing parameter at runtime. Checkout this bonus example [here ➶](../../../help/pigear_ex/#dynamically-adjusting-raspberry-pi-camera-parameters-at-runtime-in-pigear-api)"

    ```python linenums="1" hl_lines="3 9-14"
    # import required libraries
    from vidgear.gears import PiGear
    from libcamera import Transform
    import cv2

    # formulate various Picamera2 API 
    # configurational parameters
    options = {
        "queue": True,
        "buffer_count": 4,
        "controls": {"Brightness": 0.5, "ExposureValue": 2.0},
        "transform": Transform(hflip=1),
        "sensor": {"output_size": (480, 320)},  # !!! will override `resolution` !!!
        "auto_align_output_size": True,  # auto-align output size
    }

    # open pi video stream with defined parameters
    stream = PiGear(resolution=(640, 480), framerate=60, logging=True, **options).start()

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

    !!! danger "PiGear API switches to the legacy `picamera`backend if the `picamera2` library is unavailable."
    
        It is advised to enable logging(`logging=True`) to see which backend is being used.

        !!! note "You could also enforce the legacy picamera API backend in PiGear by using the [`enforce_legacy_picamera`](../params/#b-user-defined-parameters) user-defined optional parameter boolean attribute."

    PiGear also supports almost every parameter available within [`picamera`](https://picamera.readthedocs.io/en/release-1.13/api_camera.html) python library. These parameters can be easily applied to the source stream in PiGear API through its [`options`](../params/#options) dictionary parameter by formatting them as its attributes. The complete usage example is as follows:

    !!! tip "All supported parameters are listed in [PiCamera Docs ➶](https://picamera.readthedocs.io/en/release-1.13/api_camera.html)"

    !!! example "PiGear also support changing parameter at runtime. Checkout this bonus example [here ➶](../../../help/pigear_ex/#dynamically-adjusting-raspberry-pi-camera-parameters-at-runtime-in-pigear-api)"

    ```python linenums="1" hl_lines="8-13"
    # import required libraries
    from vidgear.gears import PiGear
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
    stream = PiGear(resolution=(640, 480), framerate=60, logging=True, **options).start()

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

## Using PiGear with Direct Colorspace Manipulation

PiGear API also supports **Direct Colorspace Manipulation**, which is ideal for changing source colorspace on the run. 

!!! info "A more detailed  information on colorspace manipulation can be found [here ➶](../../../bonus/colorspace_manipulation/)"

In following example code, we will start with [**HSV**](https://en.wikipedia.org/wiki/HSL_and_HSV) as source colorspace, and then we will switch to [**GRAY**](https://en.wikipedia.org/wiki/Grayscale)  colorspace when ++"W"++ key is pressed, and then [**LAB**](https://en.wikipedia.org/wiki/CIELAB_color_space) colorspace when ++"E"++ key is pressed, finally default colorspace _(i.e. **BGR**)_ when ++"S"++ key is pressed. Also, quit when ++"Q"++ key is pressed:


!!! warning "Any incorrect or None-Type value will immediately revert the colorspace to default _(i.e. `BGR`)_."


```python linenums="1" hl_lines="9 35 39 43"
# import required libraries
from vidgear.gears import PiGear
import cv2

# open pi video stream with defined parameters and change colorspace to `HSV`
stream = PiGear(
    resolution=(640, 480),
    framerate=60,
    colorspace="COLOR_BGR2HSV",
    logging=True
).start()


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

    # check if 'w' key is pressed
    if key == ord("w"):
        # directly change colorspace at any instant
        stream.color_space = cv2.COLOR_BGR2GRAY  # Now colorspace is GRAY

    # check for 'e' key is pressed
    if key == ord("e"):
        stream.color_space = cv2.COLOR_BGR2LAB  # Now colorspace is CieLAB

    # check for 's' key is pressed
    if key == ord("s"):
        stream.color_space = None  # Now colorspace is default(ie BGR)

    # check for 'q' key is pressed
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
```

&nbsp;

## Using PiGear with WriteGear API

PiGear can be easily used with WriteGear API directly without any compatibility issues. The suitable example is as follows:

=== "New Picamera2 backend"

    ```python linenums="1"
    # import required libraries
    from vidgear.gears import PiGear
    from vidgear.gears import WriteGear
    from libcamera import Transform
    import cv2

    # formulate various Picamera2 API 
    # configurational parameters
    options = {
        "queue": True,
        "buffer_count": 4,
        "controls": {"Brightness": 0.5, "ExposureValue": 2.0},
        "transform": Transform(hflip=1),
        "sensor": {"output_size": (480, 320)},  # will override `resolution`
        "auto_align_output_config": True,  # auto-align camera configuration
    }

    # open pi video stream with defined parameters
    stream = PiGear(resolution=(640, 480), framerate=60, logging=True, **options).start()

    # define suitable (Codec,CRF,preset) FFmpeg parameters for writer
    output_params = {"-vcodec": "libx264", "-crf": 0, "-preset": "fast"}

    # Define writer with defined parameters and suitable output filename for e.g. `Output.mp4`
    writer = WriteGear(output="Output.mp4", logging=True, **output_params)

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
    
=== "Legacy Picamera backend"

    ??? danger "PiGear API switches to the legacy `picamera`backend if the `picamera2` library is unavailable."
    
        It is advised to enable logging(`logging=True`) to see which backend is being used.

        !!! note "You could also enforce the legacy picamera API backend in PiGear by using the [`enforce_legacy_picamera`](../params/#b-user-defined-parameters) user-defined optional parameter boolean attribute."

    ```python linenums="1"
    # import required libraries
    from vidgear.gears import PiGear
    from vidgear.gears import WriteGear
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
    stream = PiGear(resolution=(640, 480), framerate=60, logging=True, **options).start()

    # define suitable (Codec,CRF,preset) FFmpeg parameters for writer
    output_params = {"-vcodec": "libx264", "-crf": 0, "-preset": "fast"}

    # Define writer with defined parameters and suitable output filename for e.g. `Output.mp4`
    writer = WriteGear(output="Output.mp4", logging=True, **output_params)

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

[^1]: A custom `libcamera` API class. Must be imported as `from libcamera import Transform`.
