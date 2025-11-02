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


# WriteGear API Parameters

&thinsp;

## **`output`**

This parameter sets the valid filename/path/URL for the video output.

!!! warning "WriteGear API will throw `ValueError` if `output` provided is empty or invalid."

**Data-Type:** String

**Usage:**

Its valid input can be one of the following: 

* **Path to directory**: Valid path of the directory to save the output video file. In this case, WriteGear API will automatically assign a unique filename _(with a default extension i.e.`.mp4`)_ as follows:

    ```python
    writer = WriteGear(output = '/home/foo/foo1') #Define writer 
    ```

* **Filename** _(with/without path)_: Valid filename(_with valid extension_) of the output video file. In case filename is provided without path, then current working directory will be used.

    ```python
    writer = WriteGear(output = 'output.mp4') #Define writer 
    ```

    !!! danger "Make sure to provide valid filename with valid file-extension based on the encoder in use."


* **URL**: Valid URL of a network stream with a protocol supported by installed FFmpeg _(verify with command `ffmpeg -protocols`)_ only. This is useful for building a [**Video-Streaming Server**](https://trac.ffmpeg.org/wiki/StreamingGuide) with FFmpeg in WriteGear API. For example, you can stream on a `rtmp` protocol URL as follows:

    ```python
    writer = WriteGear(output = 'rtmp://localhost/live/test') #Define writer 
    ```

&nbsp;


## **`compression_mode`**

This parameter selects the WriteGear's **[Primary Mode of Operation](../introduction/#modes-of-operation)**, i.e. if you want to use Compression Mode, then set this parameter to `True`, otherwise set it to `False` for Non-Compression Mode. 

!!! info "In Compression Mode, WriteGear API will use [**FFmpeg's Video Encoding API**](https://ffmpeg.org/ffmpeg.html#Video-Options) to encode output video, and in Non-Compression Mode, it will use [**OpenCV's VideoWriter API**](https://docs.opencv.org/master/dd/d9e/classcv_1_1VideoWriter.html#ad59c61d8881ba2b2da22cff5487465b5) to encode output video."

**Data-Type:** Boolean

**Default Value:** Its default value is `True`.

**Usage:**

```python
WriteGear(output = 'output.mp4', compression_mode=True)
```

&nbsp;


## **`custom_ffmpeg`**

This parameter assigns the _path_ or _directory_ where the custom FFmpeg executables are located in Compression Mode only.

**Data-Type:** String

**Default Value:** Its default value is `None`.

**Usage:**

!!! info "See Compression Mode's [`custom_ffmpeg` ➶](../compression/params/#custom_ffmpeg) for more information on its usage."

&nbsp;

## **`output_params`**

This parameter allows to customize the output video encoding parameters both in Compression and Non-Compression Mode.

- In non-compression mode, this parameter allows us to exploit almost all [**OpenCV's VideoWriter API**](https://docs.opencv.org/master/dd/d9e/classcv_1_1VideoWriter.html#ad59c61d8881ba2b2da22cff5487465b5) supported parameters effortlessly and flexibly for video-encoding, by formatting desired OpenCV Parameters as this parameter's attributes. 
- In compression mode, this parameter allows us to exploit almost all [**FFmpeg's Video Encoding API**](https://ffmpeg.org/ffmpeg.html#Video-Options) supported parameters effortlessly and flexibly for encoding, by formatting desired FFmpeg Parameters as this parameter's attributes.

**Data-Type:** Dictionary

**Default Value:** Its default value is `{}`.

**Usage:**

!!! info "See [Compression Mode Parameter ➶](compression/index.md) and [Non-Compression Mode Parameter ➶](non_compression/index.md) for more information on its usage and supported parameters."

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
