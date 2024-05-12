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

# PiGear Examples

&thinsp;

## Changing Output Pixel Format in PiGear API with Picamera2 Backend

> With the Picamera2 backend, you can also define a custom `format` _(format of output frame pixels)_ in PiGear API. 

??? info "Handling output frames with a custom pixel format correctly"
    While defining custom `format` as an optional parameter, it is advised to also define the [`colorspace`](../params/#colorspace) parameter in the PiGear API. This is required only under **TWO** conditions:
    
    - If `format` value is not **MPEG** for USB cameras.
    - If `format` value is not **BGR** _(i.e., `RGB888`)_ or **BGRA** _(i.e., `XRGB8888`)_ for Raspberry Pi camera modules.
    
    :warning: Otherwise, output frames might **NOT** be compatible with OpenCV functions, and you need to handle these frames manually!

??? failure "Picamera2 library has an unconventional naming convention for its pixel formats."
    Please note that, Picamera2 takes its pixel format naming from `libcamera`, which in turn takes them from certain underlying Linux components. The results are not always the most intuitive. For example, OpenCV users will typically want each pixel to be a (`B`, `G`, `R`) triple for which the `RGB888` format should be chosen, and not `BGR888`. Similarly, OpenCV users wanting an alpha channel should select `XRGB8888`. 
    
    For more information, refer [Picamera2 docs ➶](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)

=== "YUV420/YVU420"

    !!! abstract "For reducing the size of frames in memory it is advised to use the `YUV420` pixels format." 

    In this example we will be defining custom `YUV420` _(or `YVU420`)_ pixels format of output frame, and converting it back to `BGR` to be able to display with OpenCV.

    !!! tip "You could also instead define [`colorspace="COLOR_YUV420p2RGB"`](../params/#colorspace) parameter in PiGear API for converting it back to `BGR` similarly."

    ```python hl_lines="8 27"
    # import required libraries
    from vidgear.gears import PiGear
    import cv2

    # formulate `format` Picamera2 API 
    # configurational parameters
    options = {
        "format": "YUV420" # or use `YVU420`
    }

    # open pi video stream with defined parameters
    stream = PiGear(resolution=(640, 480), framerate=60, logging=True, **options).start()

    # loop over
    while True:

        # read frames from stream
        yuv420_frame = stream.read()

        # check for frame if Nonetype
        if yuv420_frame is None:
            break

        # {do something with the `YUV420` frame here}

        # convert `YUV420` to `BGR`
        bgr = cv2.cvtColor(yuv420_frame, cv2.COLOR_YUV420p2BGR)

        # {do something with the `BGR` frame here}

        # Show output window
        cv2.imshow("Output Frame", bgr)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()
    ```

=== "YUYV"

    !!! abstract "`YUYV` is a one packed `4:2:2` YUV format that is popularly used by USB cameras." 

    !!! alert "Make sure `YUYV` pixel format is supported by your USB camera."

    In this example we will be defining custom `YUYV` pixels format of output frame, and converting it back to `BGR` to be able to display with OpenCV.

    !!! tip "You could also instead define [`colorspace="COLOR_YUV2BGR_YUYV"`](../params/#colorspace) parameter in PiGear API for converting it back to `BGR` similarly."

    ```python hl_lines="8 27"
    # import required libraries
    from vidgear.gears import PiGear
    import cv2

    # formulate `format` Picamera2 API 
    # configurational parameters
    options = {
        "format": "YUYV"
    }

    # open pi video stream with defined parameters
    stream = PiGear(resolution=(640, 480), framerate=60, logging=True, **options).start()

    # loop over
    while True:

        # read frames from stream
        yuv420_frame = stream.read()

        # check for frame if Nonetype
        if yuv420_frame is None:
            break

        # {do something with the `YUV420` frame here}

        # convert `YUV420` to `BGR`
        bgr = cv2.cvtColor(yuv420_frame, cv2.COLOR_YUV2BGR_YUYV)

        # {do something with the `BGR` frame here}

        # Show output window
        cv2.imshow("Output Frame", bgr)

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

## Dynamically Adjusting Raspberry Pi Camera Parameters at Runtime in PiGear API

=== "New Picamera2 backend"
    
    > With the `picamera2` backend, using `stream` global parameter in the PiGear API, you can change all camera **controls** _(except output **resolution** and **format**)_ at runtime after the camera has started.

    ??? tip "Accessing all available camera controls"
        A complete list of all the available camera controls can be found in the [`picamera2` docs ➶](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf), and also by inspecting the `camera_controls` property of the Picamera2 object available with  `stream` global parameter in PiGear API:

        ```python
        # import required libraries
        from vidgear.gears import PiGear

        # open any pi video stream
        stream = PiGear()

        #display all available camera controls
        print(stream.stream.camera_controls)

        # safely close video stream
        stream.stop()
        ```

        This returns a dictionary with the control names as keys, and each value being a tuple of _(`min`, `max`, `default`)_ values for that control. :warning: _The default value should be interpreted with some caution as in many cases libcamera's default value will be overwritten by the camera tuning as soon as the camera is started._


    In this example, we will set the initial Camera Module's brightness value to `-0.5` _(dark)_, and will change it to `0.5` _(bright)_ when the ++"Z"++ key is pressed at runtime:

    !!! warning "Delay in setting runtime controls"
        There will be a delay of several frames before the controls take effect. This is because there is perhaps quite a large number of requests for camera frames already in flight, and for some controls _(`exposure time` and `analogue gain` specifically)_, the camera may actually take several frames to apply the updates.

    ??? info "Using `with` construct for Guaranteed Camera Control Updates at Runtime"

        While directly modifying using `set_controls` method might seem convenient, it doesn't guarantee that all camera control settings are applied within the same frame at runtime. The `with` construct provides a structured approach to managing camera control updates in real-time. Here's how to use it:

        ```python
        # import required libraries
        from vidgear.gears import PiGear

        # formulate initial configurational parameters
        options = "controls": {"ExposureTime": 5000, "AnalogueGain": 0.5}

        # open pi video stream with these parameters
        stream = PiGear(logging=True, **options).start() 
        
        # Enter context manager and set runtime controls
        # Within this block, the controls are guaranteed to be applied atomically
        with stream.stream.controls as controls:  
            controls.ExposureTime = 10000  # Set new exposure time
            controls.AnalogueGain = 1.0     # Set new analogue gain

        # ...rest of code goes here...

        # safely close video stream
        stream.stop()
        ```

    ```python hl_lines="7 37"
    # import required libraries
    from vidgear.gears import PiGear
    import cv2

    # formulate initial configurational parameters
    # set brightness to -0.5 (dark)
    options = "controls": {"Brightness": -0.5}

    # open pi video stream with these parameters
    stream = PiGear(logging=True, **options).start() 

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
        
        # check for 'z' key if pressed
        if key == ord("z"):
            # change brightness to 0.5 (bright)
            stream.stream.set_controls({"Brightness": 0.5})

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()
    ``` 

=== "Legacy Picamera backend"

    > You can also use the `stream` global parameter in PiGear with the`picamera` backend to feed any [`picamera`](https://picamera.readthedocs.io/en/release-1.10/api_camera.html) parameters at runtime after the camera has started.

    ???+ danger "PiGear API switches to the legacy `picamera`backend if the `picamera2` library is unavailable."
    
        It is advised to enable logging(`logging=True`) to see which backend is being used.

        !!! note "You could also enforce the legacy picamera API backend in PiGear by using the [`enforce_legacy_picamera`](../params/#options) optional parameter boolean attribute."

    In this example we will set initial Camera Module's `brightness` value `80` _(brighter)_, and will change it `30` _(darker)_ when ++"Z"++ key is pressed at runtime:

    ```python hl_lines="7 37"
    # import required libraries
    from vidgear.gears import PiGear
    import cv2

    # formulate initial configurational parameters 
    # set brightness to `80` (bright)
    options = {"brightness": 80} 

    # open pi video stream with these parameters
    stream = PiGear(logging=True, **options).start() 

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

        # check for 'z' key if pressed
        if key == ord("z"):
            # change brightness to `30` (darker)
            stream.stream.brightness = 30

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()
    ``` 

## Accessing Multiple Camera through its Index in PiGear API 

> With the [`camera_num`](../params/#camera_num) parameter in the PiGear API, you can easily select the camera index to be used as the source, allowing you to drive these multiple cameras simultaneously from within a single Python session. 

!!! failure "The `camera_num` value can only be zero or greater, otherwise, PiGear API will throw `ValueError` for any negative value."

=== "New Picamera2 backend"

    With the `picamera2` backend, you can use the `camera_num` parameter in PiGear to select the camera index to be used as the source if you have multiple **Raspberry Pi camera modules _(such as CM4)_** and/or **USB cameras** connected simultaneously to your Raspberry Pi.

    ??? tip "Accessing metadata about connected cameras."
        You can call the `global_camera_info()` method of the Picamera2 object available with `stream` global parameter in PiGear API to find out what cameras are attached. This returns a list containing one dictionary for each camera, ordered according the camera number you would pass to the `camera_num` parameter in PiGear API to open that device. The dictionary contains:

        - `Model` : the model name of the camera, as advertised by the camera driver.
        - `Location` : a number reporting how the camera is mounted, as reported by `libcamera`.
        - `Rotation` : how the camera is rotated for normal operation, as reported by `libcamera`.
        - `Id` : an identifier string for the camera, indicating how the camera is connected. 

        You should always check this list to discover which camera is which as the order can change when the system boots or USB cameras are re-connected as follows:

        ```python
        # import required libraries
        from vidgear.gears import PiGear

        # open any pi video stream
        stream = PiGear()

        #display all available cameras metadata
        print(stream.stream.global_camera_info())

        # safely close video stream
        stream.stop()
        ```

    !!! info "The PiGear API can accurately differentiate between USB and Raspberry Pi camera modules by utilizing the camera's metadata."

    In this example, we will select the Camera Module connected at index `1` on the Raspberry Pi as the primary source for extracting frames in PiGear API:

    !!! alert "This example assumes a Camera Module is connected at index `1`, and some other camera connected at index `0` on your Raspberry Pi."

    ```python hl_lines="19"
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
        "exposure_compensation": 15,
        "sensor": {"output_size": (480, 320)},  # will override `resolution`
        "auto_align_output_config": True,  # auto-align camera configuration
    }

    # open pi video stream at index `1` with defined parameters
    stream = PiGear(camera_num=1, resolution=(640, 480), framerate=60, logging=True, **options).start()

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

    !!! warning "With the Picamera backend, you should not change the `camera_num` parameter unless you are using the [**Raspberry Pi 3/3+/4 Compute Module IO Boards**](https://www.raspberrypi.org/documentation/hardware/computemodule/cmio-camera.md) or third party [**Arducam Camarray Multiple Camera Solutions**](https://www.arducam.com/arducam-camarray-solutions/), which supports attaching multiple camera modules to the same Raspberry Pi board using appropriate I/O connections."
    
    You can use the `camera_num` parameter in PiGear with the `picamera` backend to select the camera index to be used as the source if you have multiple Raspberry Pi camera modules connected.

    ???+ danger "PiGear API switches to the legacy `picamera`backend if the `picamera2` library is unavailable."

        It is advised to enable logging(`logging=True`) to see which backend is being used.

        !!! note "You could also enforce the legacy picamera API backend in PiGear by using the [`enforce_legacy_picamera`](../params/#options) optional parameter boolean attribute."

    In this example, we will select the Camera Module connected at index `1` on the Raspberry Pi as the primary source for extracting frames in PiGear API:

    !!! alert "This example assumes a Camera Module is connected at index `1` on your Raspberry Pi."

    ```python hl_lines="17"
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

    # open pi video stream at index `1` with defined parameters
    stream = PiGear(camera_num=1, resolution=(640, 480), framerate=60, logging=True, **options).start()

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