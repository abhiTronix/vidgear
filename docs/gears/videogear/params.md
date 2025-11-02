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

# VideoGear API Parameters

!!! cite "VideoGear acts as a Common Video-Capture API that provides internal access for both [CamGear](../../camgear/) and [PiGear](../../pigear/) APIs and their parameters."

&thinsp;

## **`enablePiCamera`** 

This parameter provide direct access to [PiGear](../../pigear/) or [CamGear](../../camgear/) APIs respectively in VideoGear. This means the if `enablePiCamera` flag is `True`, the PiGear API will be accessed, and if `False`, the CamGear API will be accessed. 

**Data-Type:** Boolean

**Default Value:** Its default value is `False`. 

**Usage:**

```python
VideoGear(enablePiCamera=True) # enable access to PiGear API
```

!!! example "Its complete usage example is given [here ➶](../usage/#bare-minimum-usage-with-pigear-backend)."


&nbsp; 

&nbsp; 


## Parameters for Stabilizer Backend


!!! summary "Enable this backend with [`stabilize=True`](#stabilize) in VideoGear."


### **`stabilize`**

This parameter enable access to [Stabilizer Class](../../stabilizer/) for stabilizing frames, i.e. can be set to `True`(_to enable_) or unset to `False`(_to disable_). 

**Data-Type:** Boolean

**Default Value:** Its default value is `False`. 

**Usage:**

```python
VideoGear(stabilize=True) # enable stablization
```

!!! example "Its complete usage example is given [here ➶](../usage/#using-videogear-with-video-stabilizer-backend)."

&nbsp; 

### **`options`**

This parameter can be used in addition, to pass user-defined parameters supported by [Stabilizer Class](../../stabilizer/). These parameters can be formatted as this parameter's attribute.

**Supported dictionary attributes for Stabilizer Class are:**

* **`SMOOTHING_RADIUS`** (_integer_) : This attribute can be used to alter averaging window size. It basically handles the quality of stabilization at the expense of latency and sudden panning. Larger its value, less will be panning, more will be latency and vice-versa. Its default value is `25`. You can easily pass this attribute as follows:

    ```python
    options = {'SMOOTHING_RADIUS': 30}
    ```

* **`BORDER_SIZE`** (_integer_) : This attribute enables the feature to extend border size that compensates for stabilized output video frames motions. Its default value is `0`(no borders). You can easily pass this attribute as follows:

    ```python
    options = {'BORDER_SIZE': 10}
    ```

* **`CROP_N_ZOOM`**(_boolean_): This attribute enables the feature where it crops and zooms frames(to original size) to reduce the black borders from stabilization being too noticeable _(similar to the Stabilized, cropped and Auto-Scaled feature available in Adobe AfterEffects)_. It simply works in conjunction with the `BORDER_SIZE` attribute, i.e. when this attribute is enabled,  `BORDER_SIZE` will be used for cropping border instead of extending them. Its default value is `False`. You can easily pass this attribute as follows:

    ```python
    options = {'BORDER_SIZE': 10, 'CROP_N_ZOOM' : True}
    ```

* **`BORDER_TYPE`** (_string_) : This attribute can be used to change the extended border style. Valid border types are `'black'`, `'reflect'`, `'reflect_101'`, `'replicate'` and `'wrap'`, learn more about it [here](https://docs.opencv.org/3.1.0/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5). Its default value is `'black'`. You can easily pass this attribute as follows:

    !!! warning "Altering `BORDER_TYPE` attribute is **Disabled** while `CROP_N_ZOOM` is enabled."

    ```python
    options = {'BORDER_TYPE': 'black'}
    ```


&nbsp;

&nbsp; 


## Parameters for CamGear backend

!!! summary "Enable this backend with [`enablePiCamera=False`](#enablepicamera) in VideoGear. Default is also `False`."

### **`source`**

!!! warning "VideoGear API will throw `RuntimeError` if `source` provided is invalid."


This parameter defines the source for the input stream.


**Data-Type:** Based on input.

**Default Value:** Its default value is `0`. 


Its valid input can be one of the following: 

- [x] **Index (*integer*):** _Valid index of the connected video device, for e.g `0`, or `1`, or `2` etc. as follows:_

    ```python
    VideoGear(source=0)
    ```

- [x] **Filepath (*string*):** _Valid path of the video file, for e.g `"/home/foo.mp4"` as follows:_

    ```python
    VideoGear(source='/home/foo.mp4')
    ```

- [x] **Streaming Services URL Address (*string*):** _Valid Video URL as input when Stream Mode is enabled(*i.e. `stream_mode=True`*)_ 

    CamGear internally implements `yt_dlp` backend class for pipelining live video-frames and metadata from various streaming services. For example Twitch URL can be used as follows:

    !!! info "Supported Streaming Websites"

        The list of all supported Streaming Websites URLs can be found [here ➶](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md#supported-sites)

    ```python
    CamGear(source='https://www.twitch.tv/shroud', stream_mode=True)
    ```

- [x] **Network Address (*string*):** _Valid (`http(s)`, `rtp`, `rtsp`, `rtmp`, `mms`, etc.) incoming network stream address such as `'rtsp://192.168.31.163:554/'` as input:_

    ```python
    VideoGear(source='rtsp://192.168.31.163:554/')
    ```

- [x] **GStreamer Pipeline:** 
   
    CamGear API also supports GStreamer Pipeline.

    !!! warning "Requirements for GStreamer Pipelining"

        Successful GStreamer Pipelining needs your OpenCV to be built with GStreamer support. Checkout [this FAQ](../../../help/camgear_faqs/#how-to-compile-opencv-with-gstreamer-support) for compiling OpenCV with GStreamer support.

        Thereby, You can easily check GStreamer support by running `print(cv2.getBuildInformation())` python command and see if output contains something similar as follows:

         ```sh
         Video I/O:
         ...
              GStreamer:                   YES (ver 1.8.3)
         ...
         ```

    Be sure convert video output into BGR colorspace before pipelining as follows:

    ```python
    VideoGear(source='udpsrc port=5000 ! application/x-rtp,media=video,payload=96,clock-rate=90000,encoding-name=H264, ! rtph264depay ! decodebin ! videoconvert ! video/x-raw, format=BGR ! appsink')
    ```

&nbsp;

### **`stream_mode`**

This parameter controls the Stream Mode, .i.e if enabled(`stream_mode=True`), the VideoGear API will interpret the given `source` input as YouTube URL address. 

!!! bug "Due to a [**FFmpeg bug**](https://github.com/abhiTronix/vidgear/issues/133#issuecomment-638263225) that causes video to freeze frequently in OpenCV, It is advised to always use [GStreamer backend _(`backend=cv2.CAP_GSTREAMER`)_](#backend) for any livestreams _(such as Twitch)_."

!!! warning "VideoGear automatically enforce GStreamer backend _(backend=`cv2.CAP_GSTREAMER`)_ for YouTube-livestreams!"

!!! failure "VideoGear will exit with `RuntimeError` for YouTube livestreams, if OpenCV is not compiled with GStreamer(`>=v1.0.0`) support. Checkout [this FAQ](../../../help/camgear_faqs/#how-to-compile-opencv-with-gstreamer-support) for compiling OpenCV with GStreamer support."


**Data-Type:** Boolean

**Default Value:** Its default value is `False`. 

**Usage:**

```python
VideoGear(source='https://youtu.be/bvetuLwJIkA', stream_mode=True)
```

!!! example "Its complete usage example is given [here ➶](../usage/#using-camgear-with-youtube-videos)."

&nbsp;


### **`backend`**

This parameter manually selects the backend for OpenCV's VideoCapture class _(only if specified)_. 

**Data-Type:** Integer

**Default Value:** Its default value is `0` 


**Usage:**

!!! tip "All supported backends are listed [here ➶](https://docs.opencv.org/master/d4/d15/group__videoio__flags__base.html#ga023786be1ee68a9105bf2e48c700294d)"

Its value can be for e.g. `backend = cv2.CAP_DSHOW` for selecting Direct Show as backend:

```python
VideoGear(source=0, backend = cv2.CAP_DSHOW)
```

&nbsp;

### **`options`** 

This parameter provides the ability to alter various **Source Tweak Parameters** available within OpenCV's [VideoCapture API properties](https://docs.opencv.org/master/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d). 

**Data-Type:** Dictionary

**Default Value:** Its default value is `{}` 

**Usage:**

!!! tip "All supported parameters are listed [here ➶](../../camgear/advanced/source_params/)"

The desired parameters can be passed to VideoGear API by formatting them as this parameter's attributes, as follows:

```python
# formatting parameters as dictionary attributes
options = {"CAP_PROP_FRAME_WIDTH":320, "CAP_PROP_FRAME_HEIGHT":240, "CAP_PROP_FPS":60}
# assigning it
VideoGear(source=0, **options)
```

&nbsp; 

&nbsp;

## Parameters for PiGear backend 

!!! summary "Enable this backend with [`enablePiCamera=True`](#enablepicamera) in VideoGear."

### **`camera_num`** 

This parameter selects the camera index to be used as the source, allowing you to drive these multiple cameras simultaneously from within a single Python session. Its value can only be zero or greater, otherwise, VideoGear API will throw `ValueError` for any negative value.

**Data-Type:** Integer

**Default Value:** Its default value is `0`. 

**Usage:**

```python
# select Camera Module at index `1`
VideoGear(enablePiCamera=True, camera_num=1)
```

!!! example "The complete usage example demonstrating the usage of the `camera_num` parameter is available [here ➶](../../../help/pigear_ex/#accessing-multiple-camera-through-its-index-in-pigear-api)."

  
&nbsp;


### **`resolution`** 

This parameter controls the **resolution** - a tuple _(i.e. `(width,height)`)_ of two values giving the width and height of the output frames. 

!!! warning "Make sure both width and height values should be at least `64`."

!!! danger "When using the Picamera2 backend, the `resolution` parameter will be **OVERRIDDEN**, if the user explicitly defines the `output_size` property of the [`sensor`](#a-configurational-camera-parameters) configurational parameter."

**Data-Type:** Tuple

**Default Value:**  Its default value is `(640,480)`. 

**Usage:**

```python
VideoGear(enablePiCamera=True, resolution=(1280,720)) # sets 1280x720 resolution
```

&nbsp;

### **`framerate`** 


This parameter controls the framerate of the source.

**Data-Type:** integer/float

**Default Value:**  Its default value is `30`. 

**Usage:**

```python
VideoGear(enablePiCamera=True, framerate=60) # sets 60fps framerate
```

&nbsp;


### **`options`** 

This dictionary parameter in the internal PiGear API backend allows you to control various camera settings for both the `picamera2` and legacy `picamera` backends and some internal API tasks. These settings include:

#### A. Configurational Camera Parameters
- [x] These parameters are provided by the underlying backend library _(depending upon backend in use)_, and must be applied to the camera system before the camera can be started.
- [x] **These parameter include:** _Brightness, Contrast, Saturation, Exposure, Colour Temperature, Colour Gains, etc._ 
- [x] All supported parameters are listed in this [Usage example ➶](../../pigear/usage/#using-pigear-with-variable-camera-properties)


#### B. User-defined Parameters
- [x] These user-defined parameters control specific internal behaviors of the API and perform certain tasks on the camera objects.
- [x] All supported User-defined Parameters are listed [here ➶](../../pigear/params/#b-user-defined-parameters)


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
    stream = VideoGear(enablePiCamera=True, resolution=(640, 480), framerate=60, logging=True, **options).start()
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
    stream = VideoGear(enablePiCamera=True, resolution=(640, 480), framerate=60, logging=True, **options).start()
    ```

&nbsp;

&nbsp;

## Common Parameters


!!! summary "These are common parameters that works with every backend in VideoGear."
 

### **`colorspace`**

This parameter selects the colorspace of the source stream. 

**Data-Type:** String

**Default Value:** Its default value is `None`. 

**Usage:**

!!! tip "All supported `colorspace` values are given [here ➶](../../../bonus/colorspace_manipulation/)"

```python
VideoGear(colorspace="COLOR_BGR2HSV")
```

!!! example "Its complete usage example is given [here ➶](../usage/#using-videogear-with-colorspace-manipulation)"

&nbsp;


### **`logging`**

This parameter enables logging _(if `True`)_, essential for debugging. 

**Data-Type:** Boolean

**Default Value:** Its default value is `False`.

**Usage:**

```python
VideoGear(logging=True)
```

&nbsp;

### **`time_delay`** 

This parameter set the time delay _(in seconds)_ before the VideoGear API start reading the frames. This delay is only required if the source required some warm-up delay before starting up. 

**Data-Type:** Integer

**Default Value:** Its default value is `0`.

**Usage:**

```python
VideoGear(time_delay=1)  # set 1 seconds time delay
```

&nbsp; 
