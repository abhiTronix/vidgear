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

# WriteGear API Parameters: Non-Compression Mode

&thinsp;

## **`output`**

This parameter sets the valid output Video filename/path for the output video.

!!! warning "WriteGear API will throw `RuntimeError` if `output` provided is empty or invalid."

**Data-Type:** String

**Default Value:** Its default value is `0`. 

**Usage:**

!!! danger "Make sure to provide valid filename with valid file-extension based on the encoder in use _(default is `.mp4`)_."

Its valid input can be one of the following: 

* **Path to directory**: Valid path of the directory to save the output video file. In this case, WriteGear API will automatically assign a unique filename (_with a default extension i.e.`.mp4`_) as follows:

    ```python
    writer = WriteGear(output = '/home/foo/foo1', compression_mode=False) # Define writer 
    ```

* **Filename** _(with/without path)_: Valid filename(_with valid extension_) of the output video file. In case filename is provided without path, then current working directory will be used.

    ```python
    writer = WriteGear(output = 'output.mp4', compression_mode=False) # Define writer 
    ```

* **GStreamer Pipeline:** 
   
    WriteGear API also supports GStreamer Pipeline as input to its `output` parameter in Non-Compression Mode, when [GStreamer Pipeline Mode](#b-exclusive-parameters) is enabled. It can be used as follows:

    !!! warning "Requirement for GStreamer Pipelining"

        GStreamer Pipelining in WriteGear requires your OpenCV to be built with GStreamer support. Checkout [this FAQ](../../../../help/camgear_faqs/#how-to-compile-opencv-with-gstreamer-support) for compiling OpenCV with GStreamer support.


    ??? new "New in v0.2.5" 
        This feature was added in `v0.2.5`.

    ```python
    # enable GStreamer Pipeline Mode for writer
    output_params = {"-gst_pipeline_mode": True}
    # Define writer
    writer = WriteGear(
    output="appsrc ! videoconvert ! avenc_mpeg4 bitrate=100000 ! mp4mux ! filesink location=foo.mp4", compression_mode=False) 
    ```

&nbsp;


## **`compression_mode`**

This parameter selects the WriteGear's Primary [Mode of Operation](../../introduction/#modes-of-operation), i.e. if this parameter is enabled _(.i.e `compression_mode = True`)_ WriteGear will use **FFmpeg** to encode output video, and if disabled _(.i.e `compression_mode = False`)_, the **OpenCV's VideoWriter API** will be used for encoding files/streams. 

**Data-Type:** Boolean

**Default Value:** Its default value is `True`.

**Usage:**

```python
WriteGear(output = 'output.mp4', compression_mode=False)
```

&nbsp;


## **`custom_ffmpeg`**

==**Not supported in Non-Compression Mode!**==

&nbsp;

## **`output_params`**

This parameter allows us to exploit almost all [**OpenCV's VideoWriter API**](https://docs.opencv.org/master/dd/d9e/classcv_1_1VideoWriter.html#ad59c61d8881ba2b2da22cff5487465b5) supported parameters effortlessly and flexibly for video-encoding in Non-Compression Mode, by formatting desired FFmpeg Parameters as this parameter's attributes. All supported parameters and FOURCC codecs for compression mode discussed below:


!!! info "Remember, Non-Compression mode lacks the ability to control output quality and other important features like _lossless video compression, audio encoding, etc._, which are available with WriteGear's [Compression Mode](../../compression/overview/) only."


**Data-Type:** Dictionary

**Default Value:** Its default value is `{}`.

&thinsp; 

### Supported Attributes

Non-Compression Mode only gives access to a limited number of Parameters through its `output_params` parameter's attributes, which are as follows:

#### A. OpenCV Parameters

WriteGear provides access to all available [**OpenCV's VideoWriter API**](https://docs.opencv.org/master/dd/d9e/classcv_1_1VideoWriter.html#ad59c61d8881ba2b2da22cff5487465b5) parameters in Non-Compression Mode.

| Parameters | Description |
|:-----------:|-------------|
|`-fourcc`| _4-character code of codec used to encode frames_ |
|`-fps`| _controls the framerate of output video(Default value: 25)_ |
|`-backend`| (optional) _In case of multiple backends, this parameter allows us to specify VideoWriter API's backends to use. Its valid values are `CAP_FFMPEG` or `CAP_GSTREAMER`(if enabled)_  |
|`-color`| (optional) _If it is not zero(0), the encoder will expect and encode color frames, otherwise it will work with grayscale frames (the flag is currently supported on Windows only)_ |

!!! warning "`-height` and `-width` parameter are no longer supported and are automatically derived from the input frames."

#### B. Exclusive Parameters

In addition to OpenCV Parameters, WriteGear API also provides few exclusive attribute, which are as follows: 

* **`-gst_pipeline_mode`**: a boolean attribute to enable **GStreamer Pipeline Mode** to supports GStreamer Pipeline as input to its `output` parameter in Non-Compression Mode.

    !!! note "Enabling `-gst_pipeline_mode` will enforce `-backend` parameter value to `"CAP_GSTREAMER"`"

    ??? new "New in v0.2.5" 
        `-gst_pipeline_mode` attribute was added in `v0.2.5`.

    !!! example "Its usage example can be found [here ➶](../usage/#using-non-compression-mode-with-gstreamer-pipeline)."


**Usage:**

To assign desired parameters in Non-Compression Mode, you can format it as dictionary attribute and pass through this(`output_params`) parameter as follows:

```python
# format parameter as dictionary attribute
output_params = {"-fps":30} 
# and then, assign it
WriteGear(output = 'output.mp4', **output_params)
```

!!! example "Its usage example can be found [here ➶](../usage/#using-non-compression-mode-with-videocapture-gears)."

&thinsp;

### Supported FOURCC Codecs

FOURCC is a 4-character code of the codec used to encode video in Non-Compression Mode(OpenCV's VideoWriter API) without compression.

!!! tip "List of all supported FOURCC codecs can found [here ➶](http://www.fourcc.org/codecs.php)"

**Usage:**

To select desired FOURCC codec in Non-Compression Mode, you can format it as dictionary attribute and pass through this(`output_params`) parameter. For example, using [`MJPG`](http://www.fourcc.org/mjpg/) as codec, we can:

```python
# format codec as dictionary attribute
output_params = {"-fourcc":"MJPG"} 
# and then, assign it
WriteGear(output = 'output.mp4', **output_params)
```

!!! example "Its usage example can be found [here ➶](../usage/#using-non-compression-mode-with-videocapture-gears)."

&nbsp; 

## **`logging`**

This parameter enables logging _(if `True`)_, essential for debugging. 

**Data-Type:** Boolean

**Default Value:** Its default value is `False`.

**Usage:**

```python
WriteGear(output = 'output.mp4', logging=True)
```

&nbsp;
