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

# CamGear API Parameters 


&thinsp;


## **`source`**

!!! warning "CamGear API will throw `RuntimeError` if `source` provided is invalid."


This parameter defines the source for the input stream.


**Data-Type:** Based on input.

**Default Value:** Its default value is `0`. 


Its valid input can be one of the following: 

- [x] **Index (*integer*):** _Valid index of the connected video device, for e.g `0`, or `1`, or `2` etc. as follows:_

    ```python
    CamGear(source=0)
    ```

- [x] **Filepath (*string*):** _Valid path of the video file, for e.g `"/home/foo.mp4"` as follows:_

    ```python
    CamGear(source='/home/foo.mp4')
    ```

- [x] **Streaming Services URL Address (*string*):** _Valid Video URL as input when Stream Mode is enabled(i.e. [`stream_mode=True`](#stream_mode)):_ 

    CamGear internally implements `yt_dlp` backend class for pipelining live video-frames and metadata from various streaming services. For example Twitch URL can be used as follows:

    !!! info "Supported Streaming Websites"

        The complete list of all supported Streaming Websites URLs can be found [here ➶](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md#supported-sites)

    ```python
    CamGear(source='https://www.twitch.tv/shroud', stream_mode=True)
    ```

- [x] **Network Address (*string*):** _Valid (`http(s)`, `rtp`, `rtsp`, `rtmp`, `mms`, etc.) incoming network stream address such as `'rtsp://192.168.31.163:554/'` as input:_

    ```python
    CamGear(source='rtsp://192.168.31.163:554/')
    ```

- [x] **GStreamer Pipeline:** 
   
    CamGear API also supports GStreamer Pipeline.

    !!! warning "Requirements for GStreamer Pipelining"

        GStreamer Pipelining in WriteGear your OpenCV to be built with GStreamer support. Checkout [this FAQ](../../../help/camgear_faqs/#how-to-compile-opencv-with-gstreamer-support) for compiling OpenCV with GStreamer support.

        Thereby, You can easily check GStreamer support by running `print(cv2.getBuildInformation())` python command and see if output contains something similar as follows:

         ```sh
         Video I/O:
         ...
              GStreamer:                   YES (ver 1.8.3)
         ...
         ```

    Be sure convert video output into BGR colorspace before pipelining as follows:

    ```python
    CamGear(source='udpsrc port=5000 ! application/x-rtp,media=video,payload=96,clock-rate=90000,encoding-name=H264, ! rtph264depay ! decodebin ! videoconvert ! video/x-raw, format=BGR ! appsink')
    ```

&nbsp;

## **`stream_mode`**

This parameter controls the Stream Mode, .i.e if enabled(`stream_mode=True`), the CamGear API will interpret the given `source` input as YouTube URL address. 

!!! bug "Due to a [**FFmpeg bug**](https://github.com/abhiTronix/vidgear/issues/133#issuecomment-638263225) that causes video to freeze frequently in OpenCV, It is advised to always use [GStreamer backend](#backend) for any livestream videos. Checkout [this FAQ](../../../help/camgear_faqs/#how-to-compile-opencv-with-gstreamer-support) for compiling OpenCV with GStreamer support."

**Data-Type:** Boolean

**Default Value:** Its default value is `False`. 

**Usage:**

!!! info "Supported Streaming Websites"

    The complete list of all supported Streaming Websites URLs can be found [here ➶](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md#supported-sites)

```python
CamGear(source='https://youtu.be/bvetuLwJIkA', stream_mode=True)
```

!!! example "Its complete usage example is given [here ➶](../usage/#using-camgear-with-youtube-videos)."


&nbsp;

## **`colorspace`**

This parameter selects the colorspace of the input stream. 

**Data-Type:** String

**Default Value:** Its default value is `None`. 

**Usage:**

!!! tip "All supported `colorspace` values are given [here ➶](../../../bonus/colorspace_manipulation/)"

```python
CamGear(source=0, colorspace="COLOR_BGR2HSV")
```

!!! example "Its complete usage example is given [here ➶](../usage/#using-camgear-with-direct-colorspace-manipulation)"

&nbsp;

## **`backend`**

This parameter manually selects the backend for OpenCV's VideoCapture class _(only if specified)_. 

**Data-Type:** Integer

**Default Value:** Its default value is `0` 


**Usage:**

!!! tip "All supported backends are listed [here ➶](https://docs.opencv.org/master/d4/d15/group__videoio__flags__base.html#ga023786be1ee68a9105bf2e48c700294d)"

Its value can be for e.g. `backend = cv2.CAP_DSHOW` for selecting Direct Show as backend:

```python
CamGear(source=0, backend = cv2.CAP_DSHOW)
```

&nbsp;

## **`options`** 

This parameter provides the ability to alter various **Source Tweak Parameters** available within OpenCV's [VideoCapture API properties](https://docs.opencv.org/master/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d). 

**Data-Type:** Dictionary

**Default Value:** Its default value is `{}` 

**Usage:**

!!! tip "All supported parameters are listed [here ➶](../advanced/source_params/)"

The desired parameters can be passed to CamGear API by formatting them as this parameter's attributes, as follows:

```python
# formatting parameters as dictionary attributes
options = {"CAP_PROP_FRAME_WIDTH":320, "CAP_PROP_FRAME_HEIGHT":240, "CAP_PROP_FPS":60}
# assigning it
CamGear(source=0, **options)
```

&nbsp;

## **`logging`**

This parameter enables logging _(if `True`)_, essential for debugging. 

**Data-Type:** Boolean

**Default Value:** Its default value is `False`.

**Usage:**

```python
CamGear(source=0, logging=True)
```

&nbsp;

## **`time_delay`** 

This parameter set the time delay _(in seconds)_ before the CamGear API start reading the frames. This delay is only required if the source required some warm-up delay before starting up. 

**Data-Type:** Integer

**Default Value:** Its default value is `0`.

**Usage:**

```python
CamGear(source=0, time_delay=1) # set 1 seconds time delay
```

&nbsp; 
