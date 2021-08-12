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

This parameter selects the camera module index which will be used as source, if you're having multiple camera modules connected. Its value can only be greater than zero, otherwise, it will throw `ValueError` for any negative value.

!!! warning "This parameter shouldn't be altered, until unless you using [Raspberry Pi 3/3+ Compute Module IO Board](https://www.raspberrypi.org/documentation/hardware/computemodule/cmio-camera.md).""

**Data-Type:** Integer

**Default Value:** Its default value is `0`. 

**Usage:**

```python
PiGear(camera_num=0)
```
  
&nbsp;


## **`resolution`** 

This parameter sets the resolution (i.e. `(width,height)`) of the source. 

!!! info "For more information read [here ➶](https://picamera.readthedocs.io/en/release-1.13/api_camera.html#picamera.PiCamera.resolution)"


**Data-Type:** Tuple

**Default Value:**  Its default value is `(640,480)`. 

**Usage:**

```python
PiGear(resolution=(1280,720)) # sets 1280x720 resolution
```

&nbsp;

## **`framerate`** 


This parameter sets the framerate of the source.
 

!!! info "For more information read [here ➶](https://picamera.readthedocs.io/en/release-1.13/api_camera.html#picamera.PiCamera.framerate)"


**Data-Type:** integer/float

**Default Value:**  Its default value is `30`. 

**Usage:**

```python
PiGear(framerate=60) # sets 60fps framerate
```

&nbsp;

## **`colorspace`**

This parameter selects the colorspace of the source stream. 

**Data-Type:** String

**Default Value:** Its default value is `None`. 

**Usage:**

!!! tip "All supported `colorspace` values are given [here ➶](../../../bonus/colorspace_manipulation/)"

```python
PiGear(colorspace="COLOR_BGR2HSV")
```

!!! example "Its complete usage example is given [here ➶](../usage/#using-pigear-with-direct-colorspace-manipulation)"

&nbsp;

## **`options`** 

This parameter provides the ability to alter various **Tweak Parameters** `like brightness, saturation, senor_mode, resolution, etc.` available within [**Picamera library**](https://picamera.readthedocs.io/en/release-1.13/api_camera.html).

**Data-Type:** Dictionary

**Default Value:** Its default value is `{}` 

**Usage:**

!!! tip "All supported parameters are listed in [PiCamera Docs](https://picamera.readthedocs.io/en/release-1.13/api_camera.html)"

The desired parameters can be passed to PiGear API by formatting them as this parameter's attributes, as follows:

```python
# formatting parameters as dictionary attributes
options = {
    "hflip": True,
    "exposure_mode": "auto",
    "iso": 800,
    "exposure_compensation": 15,
    "awb_mode": "horizon",
    "sensor_mode": 0,
}
# assigning it
PiGear(logging=True, **options)
```

**User-specific attributes:**

Additionally, `options` parameter also support some User-specific attributes, which are as follows:

* **`HWFAILURE_TIMEOUT`** (float): PiGear contains ==Threaded Internal Timer== - that silently keeps active track of any frozen-threads/hardware-failures and exit safely, if any does occur at a timeout value. This parameter can be used to control that timeout value i.e. the maximum waiting time _(in seconds)_ after which PiGear exits with a `SystemError` to save resources. Its value can only be between `1.0` _(min)_ and `10.0` _(max)_ and its default value is `2.0`. Its usage is as follows: 

    ```python
    options = {"HWFAILURE_TIMEOUT": 2.5}  # sets timeout to 2.5 seconds
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
