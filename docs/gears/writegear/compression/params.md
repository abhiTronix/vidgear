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


# WriteGear API Parameters: Compression Mode

&thinsp;

## **`output`**

This parameter sets the valid filename/path/URL for the video output.

!!! warning

    WriteGear API will throw `ValueError` if `output` provided is empty or invalid.

**Data-Type:** String

**Usage:**

Its valid input can be one of the following: 

* **Path to directory**: Valid path of the directory to save the output video file. In this case, WriteGear API will automatically assign a unique filename (_with a default extension i.e.`.mp4`_) as follows:

    ```python
    writer = WriteGear(output = '/home/foo/foo1') #Define writer 
    ```

* **Filename** _(with/without path)_: Valid filename(_with valid extension_) of the output video file. In case filename is provided without path, then current working directory will be used. Its parent directory must already exist _(WriteGear doesn't create it)_, and an existing file is overwritten _(FFmpeg's `-y`)_.

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

This parameter selects the WriteGear's Primary [Mode of Operation](../introduction/#modes-of-operation), i.e. if this parameter is enabled _(.i.e `compression_mode = True`)_ WriteGear will use **FFmpeg** to encode output video, and if disabled _(.i.e `compression_mode = False`)_, the **OpenCV's VideoWriter API** will be used for encoding files/streams. 

**Data-Type:** Boolean

**Default Value:** Its default value is `True`.

**Usage:**

```python
WriteGear(output = 'output.mp4', compression_mode=True)
```

&nbsp;


## **`custom_ffmpeg`**

This parameter assigns the _path_ or _directory_ where the custom FFmpeg executables are located in Compression Mode only.

!!! info "Compression Mode Behavior on Windows"
    
    In Compression Mode, if a custom FFmpeg executable's path or directory is not provided through `custom_ffmpeg` parameter on Windows machine, then WriteGear API will ==automatically attempt to download and extract suitable Static FFmpeg binaries at suitable location on your windows machine==. More information can be found [here ➶](advanced/ffmpeg_install/#a-auto-installation).

**Data-Type:** String

**Default Value:** Its default value is `""` _(empty string)_.

**Usage:**

=== "Custom Path"
    ```python
    # if ffmpeg executable is located at "./foo/foo1/ffmpeg" path (On Windows: `./foo/foo1/ffmpeg.exe`)
    WriteGear(output = 'output.mp4', compression_mode=True, custom_ffmpeg="./foo/foo1/ffmpeg")
    ```
=== "Custom Directory"
    ```python
    # if ffmpeg executable is located at "./foo/foo1/FFmpeg" directory
    WriteGear(output = 'output.mp4', compression_mode=True, custom_ffmpeg="/foo/foo1/FFmpeg/bin")
    ```

```python
# if ffmpeg executables are located at "/foo/foo1/FFmpeg"
WriteGear(output = 'output.mp4', custom_ffmpeg="/foo/foo1/FFmpeg")
```

&nbsp;

## **`output_params`**

This parameter allows us to exploit almost all FFmpeg supported parameters effortlessly and flexibly for encoding in Compression Mode, by formatting desired FFmpeg Parameters as this parameter's attributes. All supported parameters and encoders for compression mode discussed below:


!!! danger "Kindly read [**FFmpeg Docs**](https://ffmpeg.org/documentation.html) carefully, before passing any values to `output_param` dictionary parameter. Wrong values may result in undesired Errors or no output at all."


**Data-Type:** Dictionary

**Default Value:** Its default value is `{}`.


### Supported Parameters

* **FFmpeg Parameters:** All parameters based on selected [encoder](#supported-encoders) in use, are supported, and can be passed as dictionary attributes in `output_param`. For example, for using `libx264 encoder` to produce a lossless output video, we can pass required FFmpeg parameters as dictionary attributes, as follows:

    ??? info "Extra inputs (`-i`) only add audio, and end with your frames"

        When an extra input is given with `-i` _(in `output_params`, `-clones` or `-ffpreheaders`)_ and you set no `-map`/`-filter_complex`, WriteGear keeps the piped frames as the only video _(`-map <pipe>:v:0`)_, adds audio from every extra input _(`-map <n>:a?`)_, pads shorter audio with silence _(`-af apad`, unless you set an audio filter)_ and adds `-shortest`. So every frame is written, and infinite inputs _(live microphone, `anullsrc`)_ end with your frames on `close()`.

        With copied audio _(`-acodec copy`)_, which can't be padded, WriteGear adds neither `-af apad` nor `-shortest`: a finite audio file then simply ends early, but an infinite input never ends, so add `-shortest` yourself and accept that it cuts the video at the audio's end.

    ??? tip "Valueless FFmpeg flags"

        Set a flag's value to `None` to pass it without a value _(e.g. `{"-shortest": None}` → `-shortest`)_. `-vcodec`/`-c:v` set to `None` is the exception (see below).

    !!! failure "All ffmpeg parameters are case-sensitive. Remember to double check every parameter if any error occurs."

    !!! tip "Kindly check [H.264 docs ➶](https://trac.ffmpeg.org/wiki/Encode/H.264) and other [FFmpeg Docs ➶](https://ffmpeg.org/documentation.html) for more information on these parameters"

    ```python
     output_params = {"-vcodec":"libx264", "-crf": 0, "-preset": "fast", "-tune": "zerolatency"} 
    ```

    ??? info "Discarding `-vcodec` parameter entirely"

        Setting `-vcodec` _(or its alias `-c:v`)_ to `None` in `output_params` will discard the `-vcodec` FFmpeg parameter entirely from WriteGear's encoding pipeline. This allows FFmpeg itself to auto-select the best available video encoder for the given output _(e.g. for GIF outputs)_:

        ```python
        # let FFmpeg auto-select the best video encoder for the output (e.g. GIF)
        output_params = {"-vcodec": None}
        # equivalent using `-c:v` alias
        output_params = {"-c:v": None}
        ```

* **Special Internal Parameters:** In addition to FFmpeg parameters, WriteGear API also supports some Special Parameters to tweak its internal properties. These parameters are discussed below:

    * **`-ffmpeg_download_path`** _(string)_: sets the custom directory for downloading FFmpeg Static Binaries in Compression Mode, during the [Auto-Installation](advanced/ffmpeg_install/#a-auto-installation) on Windows Machines Only. If this parameter is not altered, then these binaries will auto-save to the default temporary directory (for e.g. `C:/User/temp`) on your windows machine. It can be used as follows: 

        ```python
        output_params = {"-ffmpeg_download_path": "C:/User/foo/foo1"} # will be saved to "C:/User/foo/foo1"
        ```

    * **`-input_framerate`** _(float/int)_ : sets the framerate of the incoming frames _(FFmpeg's `-framerate` for the raw pipe input)_, i.e. how many frames make one second of video. ==It defaults to `25` fps when omitted or invalid==, so set it to your source's real framerate. It can be used as follows: 

        ```python
        output_params = {"-input_framerate": 60.0} # 60 frames make one second of video
        ```

        !!! tip "For live sources that don't deliver frames at a steady rate, timestamp frames by arrival time instead: `{"-ffpreheaders": ["-use_wallclock_as_timestamps", "1"], "-fps_mode": "cfr", "-r": 30}` _(use `-vsync cfr` instead of `-fps_mode cfr` on FFmpeg older than 5.1)_."

        !!! tip "Its usage example can be found [here ➶](usage/#using-compression-mode-with-controlled-framerate)"
      
    * **`-output_dimensions`** _(tuple/list)_ : scales the output video to custom dimensions(*size/resolution*) _(otherwise input video-frame size will be used)_. Its value can either be a **tuple** => `(width,height)` or a **list** => `[width, height]`, Its usage is as follows: 
    
        ```python
        output_params = {"-output_dimensions": (1280,720)} # to produce a 1280x720 resolution/scale output video
        ```

        !!! note "With the default `yuv420p` output of `libx264`/`libx265`, odd dimensions are rounded up to the next even number."

    * **`-input_pixfmt`** _(string)_ : sets the pixel format of the incoming frames _(otherwise auto-derived from their channels and datatype, e.g. `bgr24` for 3-channel `uint8` frames)_. It must be a format supported by your FFmpeg _(else it is discarded with a warning)_, and must match the frames' size, else `write()` raises a `ValueError`. Its usage is as follows:

        ```python
        output_params = {"-input_pixfmt": "yuv444p"} # frames are `yuv444p` (3-channel `uint8`)
        ```

    * **`-clones`** _(list)_: required to set special FFmpeg parameters that are repeated more than once in the command _(For more info., see [this issue](https://github.com/abhiTronix/vidgear/issues/141))_ or in cases where you want to preserve order of multiple FFmpeg parameters. This attribute only accepts **list** datatype as value. Its usage is as follows:

        !!! tip "Turn on logging([`logging = True`](#logging)) to see the FFmpeg command that is being executed in WriteGear's pipeline. This helps you debug/address any issues and make adjustments accordingly."

        !!! note "WriteGear by automatically applies `-format` or `-f`, `-pix_fmt` and `-vcodec` or `-v:f` like critical input parameters for every stream. Therefore if you need multiple values for these parameter just define them with `-clones` attribute." 

        ```python
        output_params = {
            "-i": "plug:dsnoopUSB",
            "-f": "alsa",
            "-ac": "1",
            "-ar": "48000",
            "-clones": ["-vcodec", "mpeg1video", "-f", "mpegts"],
        }
        ```

    * **`-ffpreheaders`** _(list)_: required to set special FFmpeg parameters that are present at the starting of command(such as `-re`). This attribute only accepts **list** datatype as value. Its usage is as follows:

        !!! danger "This attribute is quite powerful and can break FFmpeg pipeline easily if not used correctly. User Discretion is advised!" 

        !!! tip "Turn on logging([`logging = True`](#logging)) to see the FFmpeg command that is being executed in WriteGear's pipeline. This helps you debug/address any issues and make adjustments accordingly."

        ```python
        output_params = {
            "-ffpreheaders": ["-re"], # executes as `ffmpeg -re <rest of command>`
        }
        ```

    * **`-disable_ffmpeg_window`** _(bool)_: sets a special flag to enable detached subprocess creation on Windows OS, and can be useful while creating an `.exe` file for a python script that uses WriteGear API. On Windows, in certain cases, even after creating the `.exe` file in windowed mode or no-console mode, the FFmpeg commandline window would pop up while its being used by WriteGear API. Its usage is as follows:

        ??? new "New in v0.3.2" 
            This feature was added in `v0.3.2`.

        !!! warning "`-disable_ffmpeg_window` is only available on Windows OS with logging disabled(`logging=False`) in compression mode."

        ```python
        output_params = {"-disable_ffmpeg_window": True} # disables FFmpeg creation window
        ```
        
    * **`-disable_force_termination`** _(bool)_: **Deprecated, has no effect.** WriteGear no longer force-terminates FFmpeg on `close()` _(which dropped buffered frames)_. Extra inputs get `-shortest` instead _(see above)_.

        ??? danger "Deprecated in v0.3.8" 
            This flag is accepted but ignored since `v0.3.8`, with a warning.

    * **`-capture_stderr`** _(bool)_: **Deprecated, has no effect.** FFmpeg's stderr is now always captured: its last 50 lines _(still mirrored to console when `logging=True`)_ are exposed via WriteGear's `stderr_tail` property and appended to the `ValueError` raised by `write()` when FFmpeg dies early _(e.g. on invalid parameters)_. FFmpeg's exit code is available via the `returncode` attribute after `close()`.

        ??? danger "Deprecated in v0.3.8" 
            Added in `v0.3.7`, always on since `v0.3.8`.

### Supported Encoders

All the encoders that are compiled with FFmpeg in use, are supported by WriteGear API. You can easily check the compiled encoders by running following command in your terminal:

!!! info "Similarily, supported demuxers and filters depends upons compiled FFmpeg in use."

```sh
ffmpeg -encoders           # use `ffmpeg.exe -encoders` on windows
``` 

!!! warning "An encoder not listed by your FFmpeg raises a `ValueError` on first `write()` _(it was silently replaced with `libx264` before `v0.3.8`)_. `copy` is also accepted, to store raw frames _(e.g. in `.nut`/`.avi`)_."

!!! info "When no encoder is set, WriteGear picks the first available of `libx264`, `libx265`, `libxvid`, `mpeg4`. For `libx264`/`libx265` it defaults to `-crf 18 -preset fast` and, for 8-bit frames, `-pix_fmt yuv420p` _(playable in browsers, QuickTime and hardware decoders)_. Any of these can be overridden in `output_params` _(`-c:v` wins over `-vcodec` if both are given)_."

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
