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

# PiGear API Parameters 

&thinsp;

## **`camera_num`** 

This parameter selects the camera index to be used as the source, allowing you to drive these multiple cameras simultaneously from within a single Python session. Its value can only be zero or greater, otherwise, PiGear API will throw `ValueError` for any negative value.

**Data-Type:** Integer

**Default Value:** Its default value is `0`. 

**Usage:**

```python
# select Camera Module at index `1`
PiGear(camera_num=1)
```

!!! example "The complete usage example demonstrating the usage of the `camera_num` parameter is available [here ➶](../usage/#bare-minimum-usage-with-pigear-backend)."

  
&nbsp;


## **`resolution`** 

This parameter controls the **resolution** - a tuple _(i.e. `(width,height)`)_ of two values giving the width and height of the output frames. 

!!! warning "Make sure both width and height values should be at least `64`."

!!! danger "When using the Picamera2 backend, the `resolution` parameter will be **OVERRIDDEN**, if the user explicitly defines the `output_size` property of the [`sensor` configurational parameter](#a-configurational-camera-parameters) in PiGear API."

**Data-Type:** Tuple

**Default Value:**  Its default value is `(640,480)`. 

**Usage:**

```python
PiGear(resolution=(1280,720)) # sets 1280x720 resolution
```

&nbsp;

## **`framerate`** 


This parameter controls the framerate of the source.

**Data-Type:** integer/float

**Default Value:**  Its default value is `30`. 

**Usage:**

```python
PiGear(framerate=60) # sets 60fps framerate
```

&nbsp;

## **`colorspace`**

This parameter controls the colorspace of the output frames. 

!!! example "With the Picamera2 backend, you can also define a custom `format` _(format of output frame pixels)_ in PiGear API. Checkout this [bonus example ➶](../../../help/pigear_ex/#changing-output-pixel-format-in-pigear-api-with-picamera2-backend)"

**Data-Type:** String

**Default Value:** Its default value is `None` _(i.e. Default `BGR` colorspace)_. 

**Usage:**

!!! tip "All supported `colorspace` values are described [here ➶](../../../bonus/colorspace_manipulation/)"

```python
PiGear(colorspace="COLOR_BGR2HSV")
```

!!! example "Its complete usage example is given [here ➶](../usage/#using-pigear-with-direct-colorspace-manipulation)"

&nbsp;

## **`options`** 

This dictionary parameter in the PiGear API allows you to control various camera settings for both the `picamera2` and legacy `picamera` backends and some internal API tasks. These settings include:

### A. Configurational Camera Parameters
- [x] These parameters are provided by the underlying backend library _(depending upon backend in use)_, and must be applied to the camera system before the camera can be started.
- [x] **These parameter include:** _Brightness, Contrast, Saturation, Exposure, Colour Temperature, Colour Gains, etc._ 
- [x] All supported parameters are listed in this [Usage example ➶](../usage/#using-pigear-with-variable-camera-properties)


### B. User-defined Parameters
- [x] These user-defined parameters control specific internal behaviors of the API and perform certain tasks on the camera objects.
- [x] **All supported User-defined Parameters are listed below:**

    * **`enforce_legacy_picamera`** (bool): This user-defined boolean parameter, if `True`, forces the use of the legacy `picamera` backend in PiGear API, even if the newer `picamera2` backend is available on the system. It's default value is `False`. Its usage is as follows:

        !!! info "PiGear API will verify if the `picamera` Python library is installed before enabling the `enforce_legacy_picamera` parameter."

        ```python
        options = {"enforce_legacy_picamera": True}  # enforces `picamera` backend 
        ```

    * **`enable_verbose_logs`** (bool): **[`picamera2` backend only]** This `picamera2` backend specific parameter, if `True`, will set the logging level to output all debug messages from Picamera2 library. This parameter can be used in conjunction with enabling general logging (`logging=True`) in the PiGear API for even more granular control over logging output. It's default value is `False` _(meaning only warning message will be outputted)_. Its usage is as follows:

        !!! warning "This parameter requires logging to be enabled _(i.e. [`logging=True`](#logging))_ in PiGear API, otherwise it will be discarded."

        ```python
        options = {"enable_verbose_logs": True}  # enables debug logs from `picamera2` backend
        ```

    * **`auto_align_output_size`** (bool): **[`picamera2` backend only]** The Picamera2 backend in PiGear API has certain hardware restrictions and optimal frame size _(or `resolution`)_ for efficient processing. Although user-specified frame sizes are allowed, Picamera2 can make minimal adjustments to the configuration if it detects an invalid or inefficient size. This parameter, if `True`, will request these optimal frame size adjustments from Picamera2. It's default value is `False` _(meaning no changes will be made to user-specified resolution)_. Its usage is explained in detail below:

        !!! danger "This parameter may override any invalid or inefficient size inputted by user through [`resolution`](#resolution) parameter in PiGear API."

        ```python
        # auto-aligns output resolution to optimal
        options = {"auto_align_output_size": True}

        # open pi video stream with user-specified resolution `(808, 606)`
        stream = PiGear(resolution=(808, 606), logging=True, **options).start()

        # read frame from stream
        frame = stream.read()

        # print final resolution of frame
        print('width: ', frame.shape[1]) # height: 800 (changed)
        print('height: ', frame.shape[0]) # height: 606
        # Picamera2 has decided an 800x606 image will be more efficient.
        ```
       **Explanation:** In the example code, Picamera2 adjusts the requested output resolution of `(808, 606)` to the more efficient `(800, 606)` size.

    * **`HWFAILURE_TIMEOUT`** (float): PiGear API provides a ==**Threaded Internal Timer**== that silently keeps track of any frozen threads/hardware failures and exits safely if any occur at a timeout value. This parameter controls the timeout value, which is the maximum waiting time _(in seconds)_ after which API exits itself with a `SystemError` to save resources. Its value can only be set between `1.0` _(min)_ and `10.0` _(max)_, with a default value of `2.0`. Its usage is as follows:

        ```python
        options = {"HWFAILURE_TIMEOUT": 2.5}  # sets timeout to 2.5 seconds
        ```


**Data-Type:** Dictionary

**Default Value:** Its default value is `{}` 

**Usage:**

!!! example "The complete usage example demonstrating the usage of the `options` parameter is available [here ➶](../usage/#using-pigear-with-variable-camera-properties)."

You can format these user-defined and configurational parameters as attributes of this `options` dictionary parameter as follows:

=== "New Picamera2 backend"

    ```python
    # formulate various Picamera2 API parameters
    options = {
        "queue": True,
        "buffer_count": 4,
        "controls": {"Brightness": 0.5, "ExposureValue": 2.0},
        "exposure_compensation": 15,
        "sensor": {"output_size": (480, 320)},  # !!! will override `resolution` !!!
    }

    # open pi video stream with defined parameters
    stream = PiGear(resolution=(640, 480), framerate=60, logging=True, **options).start()
    ```

=== "Legacy Picamera backend"

    ```python
    # formulate various Picamera API parameters
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
    ```

&nbsp;

## **`logging`**

This parameter enables logging _(if `True`)_, essential for debugging. 

**Data-Type:** Boolean

**Default Value:** Its default value is `False`.

**Usage:**

```python
PiGear(logging=True)
```

&nbsp;

## **`time_delay`** 

This parameter set the time delay _(in seconds)_ before the PiGear API start reading the frames. This delay is only required if the source required some warm-up delay before starting up. 

**Data-Type:** Integer

**Default Value:** Its default value is `0`.

**Usage:**

```python
PiGear(time_delay=1)  # set 1 seconds time delay
```

&nbsp; 
