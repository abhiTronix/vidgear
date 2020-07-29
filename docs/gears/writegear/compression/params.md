<!--
===============================================
vidgear library source-code is deployed under the Apache 2.0 License:

Copyright (c) 2019-2020 Abhishek Thakur(@abhiTronix) <abhi.una12@gmail.com>

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


# WriteGear API Parameters: Compression Mode

## **`output_filename`**

This parameter sets the valid output Video filename/path for the output video.

!!! warning

    WriteGear API will throw `RuntimeError` if `output_filename` provided is empty or invalid.

**Data-Type:** String

**Usage:**

Its valid input can be one of the following: 

* **Path to directory**: Valid path of the directory to save the output video file. In this case, WriteGear API will automatically assign a unique filename (_with a default extension i.e.`.mp4`_) as follows:

    ```python
    writer = WriteGear(output_filename = '/home/foo/foo1') #Define writer 
    ```

* **Filename** _(with/without path)_: Valid filename(_with valid extension_) of the output video file. In case filename is provided without path, then current working directory will be used.

    ```python
    writer = WriteGear(output_filename = 'output.mp4') #Define writer 
    ```

    !!! danger "Make sure to provide valid filename with valid file-extension based on the encoder in use _(default is `.mp4`)_."


* **URL**: Valid URL of a network stream with a protocol supported by installed FFmpeg _(verify with command `ffmpeg -protocols`)_ only. This is useful for building a [**Video-Streaming Server**](https://trac.ffmpeg.org/wiki/StreamingGuide) with FFmpeg in WriteGear API. For example, you can stream on a `rtmp` protocol URL as follows:

    ```python
    writer = WriteGear(output_filename = 'rtmp://localhost/live/test') #Define writer 
    ```

&nbsp;


## **`compression_mode`**

This parameter selects the WriteGear's primary [Mode of Operation](../../introduction/#modes-of-operation), i.e. if this parameter is enabled _(.i.e `compression_mode = True`)_ WriteGear will use **FFmpeg** to encode output video, and if disabled _(.i.e `compression_mode = False`)_, the **OpenCV's VideoWriter API** will be used for encoding. 

**Data-Type:** Boolean

**Default Value:** Its default value is `True`.

**Usage:**

```python
WriteGear(output_filename = 'output.mp4', compression_mode=True)
```

&nbsp;


## **`custom_ffmpeg`**

This parameter assigns the custom _path/directory_ where the custom FFmpeg executables are located in Compression Mode only.

!!! info "Compression Mode Behavior on Windows"
    
    In Compression Mode, if a custom FFmpeg executable's path | directory is not provided through `custom_ffmpeg` parameter on Windows machine, then WriteGear API will ==automatically attempt to download and extract suitable Static FFmpeg binaries at suitable location on your windows machine==. More information can be found [here ➶](../advanced/ffmpeg_install/#a-auto-installation).

**Data-Type:** String

**Default Value:** Its default value is `None`.

**Usage:**

```python
# if ffmpeg executables are located at "/foo/foo1/FFmpeg"
WriteGear(output_filename = 'output.mp4', custom_ffmpeg="/foo/foo1/FFmpeg")
```

&nbsp;

## **`output_params`**

This parameter allows us to exploit almost all FFmpeg supported parameters effortlessly and flexibly for encoding in Compression Mode, by formatting desired FFmpeg Parameters as this parameter's attributes. All supported parameters and encoders for compression mode discussed below:


!!! danger "Kindly read [**FFmpeg Docs**](https://ffmpeg.org/documentation.html) carefully, before passing any values to `output_param` dictionary parameter. Wrong values may result in undesired Errors or no output at all."


**Data-Type:** Dictionary

**Default Value:** Its default value is `{}`.


### Supported Parameters

* **FFmpeg Parameters:** All parameters based on selected [encoder](#supported-encoders) in use, are supported, and can be passed as dictionary attributes in `output_param`. For example, for using `libx264 encoder` to produce a lossless output video, we can pass required FFmpeg parameters as dictionary attributes, as follows:

    !!! warning "**DO NOT** provide additional video-source with `-i` FFmpeg parameter in `output_params`, otherwise it will interfere with frame you input later, and it will break things!"

    !!! tip "Kindly check [H.264 docs ➶](https://trac.ffmpeg.org/wiki/Encode/H.264) and other [FFmpeg Docs ➶](https://ffmpeg.org/documentation.html) for more information on these parameters"

    ```python
     output_params = {"-vcodec":"libx264", "-crf": 0, "-preset": "fast", "-tune": "zerolatency"} 
    ```


* **Special Internal Parameters:** In addition to FFmpeg parameters, WriteGear API also supports some Special Parameters to tweak its internal properties. These parameters are discussed below:

    * **`-ffmpeg_download_path`** _(string)_: sets the custom directory for downloading FFmpeg Static Binaries in Compression Mode, during the [Auto-Installation](../advanced/ffmpeg_install/#a-auto-installation) on Windows Machines Only. If this parameter is not altered, then these binaries will auto-save to the default temporary directory (for e.g. `C:/User/temp`) on your windows machine. It can be used as follows: 

        ```python
        output_params = {"-ffmpeg_download_path": "C:/User/foo/foo1"} # will be saved to "C:/User/foo/foo1"
        ```

    * **`-input_framerate`** _(float/int)_ : sets the constant framerate of the output. It can be used as follows: 

        ```python
        output_params = {"-input_framerate": 60.0} # set the constant framerate to 60fps
        ```

        !!! tip "Its usage example can be found [here ➶](../usage/#using-compression-mode-with-controlled-framerate)"
      
    * **`-output_dimensions`** _(tuple/list)_ : sets the custom dimensions(*size/resolution*) of the output video _(otherwise input video-frame size will be used)_. Its value can either be a **tuple** => `(width,height)` or a **list** => `[width, height]`, Its usage is as follows: 
    
        ```python
        output_params = {"-output_dimensions": (1280,720)} #to produce a 1280x720 resolution/scale output video
        ```
    * **`-clones`** _(list)_: sets the special FFmpeg parameters that are repeated more than once in the command _(For more info., see [this issue](https://github.com/abhiTronix/vidgear/issues/141))_ as **list** only. Its usage is as follows: 

        ```python
        output_params = {"-clones": ['-map', '0:v:0', '-map', '1:a?']}
        ```

    * **`-disable_force_termination`** _(bool)_: sets a special flag to disable the default forced-termination behaviour in WriteGear API when `-i` FFmpeg parameter is used _(For more details, see issue: #149)_. Its usage is as follows:

        ```python
        output_params = {"-disable_force_termination": True} # disable the default forced-termination behaviour
        ```

### Supported Encoders

All the encoders that are compiled with FFmpeg in use, are supported by WriteGear API. You can easily check the compiled encoders by running following command in your terminal:

```sh
ffmpeg -encoders           # use `ffmpeg.exe -encoders` on windows
``` 

&nbsp; 

## **`logging`**

This parameter enables logging _(if `True`)_, essential for debugging. 

**Data-Type:** Boolean

**Default Value:** Its default value is `False`.

**Usage:**

```python
WriteGear(output_filename = 'output.mp4', logging=True)
```

&nbsp;
