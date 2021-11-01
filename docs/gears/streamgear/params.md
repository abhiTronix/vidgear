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

# StreamGear API Parameters 

&thinsp;

## **`output`**

This parameter sets the valid filename/path for storing the StreamGear assets _(Manifest file (such as MPD in-case of DASH) or a Master Playlist (such as M3U8 in-case of Apple HLS) & Transcoded sequence of segments)_.

!!! warning "StreamGear API will throw `ValueError` if `output` provided is empty or invalid."

!!! error "Make sure to provide _valid filename with valid file-extension_ for selected [`format`](#format) value _(such as `.mpd` in case of MPEG-DASH and `.m3u8` in case of APPLE-HLS)_, otherwise StreamGear will throw `AssertionError`."

!!! note "StreamGear generated sequence of multiple chunks/segments are also stored in the same directory."

!!! tip "You can easily delete all previous assets at `output` location, by using [`-clear_prev_assets`](#a-exclusive-parameters) attribute of [`stream_params`](#stream_params) dictionary parameter."

**Data-Type:** String

**Usage:**

Its valid input can be one of the following: 

* **Path to directory**: Valid path of the directory. In this case, StreamGear API will automatically assign a unique filename for Manifest file. This can be defined as follows:

    === "DASH"

        ```python
        streamer = StreamGear(output = "/home/foo/foo1") # Define streamer with manifest saving directory path 
        ```

    === "HLS"

        ```python
        streamer = StreamGear(output = "/home/foo/foo1", format="hls") # Define streamer with playlist saving directory path 
        ```

* **Filename** _(with/without path)_: Valid filename(_with valid extension_) of the output Manifest file. In case filename is provided without path, then current working directory will be used.

    === "DASH"

        ```python
        streamer = StreamGear(output = "output_foo.mpd") # Define streamer with manifest file name
        ```

    === "HLS"

        ```python
        streamer = StreamGear(output = "output_foo.m3u8", format="hls") # Define streamer with playlist file name
        ```

* **URL**: Valid URL of a network stream with a protocol supported by installed FFmpeg _(verify with command `ffmpeg -protocols`)_ only. This is useful for directly storing assets to a network server. For example, you can use a `http` protocol URL as follows:

    
    === "DASH"

        ```python
        streamer = StreamGear(output = "http://195.167.1.101/live/test.mpd") #Define streamer 
        ```

    === "HLS"

        ```python
        streamer = StreamGear(output = "http://195.167.1.101/live/test.m3u8", format="hls") #Define streamer 
        ```

&nbsp;

## **`format`** 


This parameter select the adaptive HTTP streaming formats. For now, the supported format are: `dash` _(i.e [**MPEG-DASH**](https://www.encoding.com/mpeg-dash/))_ and  `hls` _(i.e [**Apple HLS**](https://developer.apple.com/documentation/http_live_streaming))_.

!!! warning "Any invalid value to `format` parameter will result in ValueError!"

!!! error "Make sure to provide _valid filename with valid file-extension_ in [`output`](#output) for selected `format` value _(such as `.mpd` in case of MPEG-DASH and `.m3u8` in case of APPLE-HLS)_, otherwise StreamGear will throw `AssertionError`."


**Data-Type:** String

**Default Value:** Its default value is `dash`

**Usage:**

=== "DASH"

    ```python
    StreamGear(output = "output_foo.mpd", format="dash")
    ```

=== "HLS"

    ```python
    StreamGear(output = "output_foo.m3u8", format="hls")
    ```


&nbsp; 


## **`custom_ffmpeg`**

This parameter assigns the custom _path/directory_ where the custom/downloaded FFmpeg executables are located.

!!! info "Behavior on Windows"
    
    If a custom FFmpeg executable's path | directory is not provided through `custom_ffmpeg` parameter on Windows machine, then StreamGear API will ==automatically attempt to download and extract suitable Static FFmpeg binaries at suitable location on your windows machine==. More information can be found [here ➶](../ffmpeg_install/#a-auto-installation).

**Data-Type:** String

**Default Value:** Its default value is `None`.

**Usage:**

```python
# If ffmpeg executables are located at "/foo/foo1/ffmpeg"
StreamGear(output = 'output_foo.mpd', custom_ffmpeg="/foo/foo1/ffmpeg")
```

&nbsp;


## **`stream_params`**

This parameter allows us to exploit almost all FFmpeg supported parameters effortlessly and flexibly change its internal settings for transcoding and seamlessly generating high-quality streams. All [supported parameters](#supported-parameters) can formatting as attributes for this dictionary parameter:


!!! danger "Kindly read [**FFmpeg Docs**](https://ffmpeg.org/documentation.html) carefully, before passing any additional values to `stream_params` parameter. Wrong values may result in undesired errors or no output at all."


**Data-Type:** Dictionary

**Default Value:** Its default value is `{}`.


### Supported Parameters

#### A. Exclusive Parameters

StreamGear API provides some exclusive internal parameters to easily generate Streaming Assets and effortlessly tweak its internal properties. These parameters are discussed below:

* **`-streams`** _(list of dicts)_: This important attribute makes it simple and pretty straight-forward to define additional multiple streams as _list of dictionaries_ of different quality levels _(i.e. different bitrates or spatial resolutions)_ for streaming. 

    !!! danger "Important `-streams` attribute facts"
        * ==On top of these additional streams, StreamGear by default, generates a primary stream of same resolution and framerate[^1] as the input Video, at the index `0`.==  
        * You **MUST** need to define `-resolution` value for your stream, otherwise stream will be discarded!
        * You only need either of `-video_bitrate` or `-framerate` for defining a valid stream. Since with `-framerate` value defined, video-bitrate is calculated automatically using `-bpps` and `-resolution` values.
        * If you define both `-video_bitrate` and `-framerate` values at the same time, StreamGear will discard the `-framerate` value automatically.


    **To construct the additional stream dictionaries, you'll will need following sub-attributes:**

    * `-resolution` _(string)_: It is **compulsory** to define the required resolution/dimension/size for the stream, otherwise given stream will be rejected. Its value can be a `"{width}x{height}"` as follows: 
        
        ```python
        "-streams" = [{"-resolution": "1280x720"}] # to produce a 1280x720 resolution/scale 
        ```

    * `-video_bitrate` _(string)_: It is an **optional** _(can be ignored if `-framerate` parameter is defined)_ sub-attribute that generally determines the bandwidth and quality of stream, i.e. the higher the bitrate, the better the quality and the larger will be bandwidth and more will be strain on network. It value is generally in `kbps` _(kilobits per second)_ for OBS (Open Broadcasting Softwares). You can easily define this attribute as follows:

        ```python
        "-streams" : [{"-resolution": "1280x720", "-video_bitrate": "2000k"}] # to produce a 1280x720 resolution and 2000kbps bitrate stream
        ```

    * `-framerate` _(float/int)_: It is another **optional** _(can be ignored if `-video_bitrate` parameter is defined)_ sub-attribute that defines the assumed framerate for the stream. It's value can be float/integer as follows:

        ```python
        "-streams" : [{"-resolution": "1280x720", "-framerate": "60.0"}] # to produce a 1280x720 resolution and 60fps framerate stream
        ```

    **Usage:** You can easily define any number of streams using `-streams` attribute as follows:

    !!! tip "Usage example can be found [here ➶](../ssm/usage/#usage-with-additional-streams)"

    ```python
    stream_params = 
        {"-streams": 
            [{"-resolution": "1920x1080", "-video_bitrate": "4000k"}, # Stream1: 1920x1080 at 4000kbs bitrate
            {"-resolution": "1280x720", "-framerate": "30.0"}, # Stream2: 1280x720 at 30fps
            {"-resolution": "640x360", "-framerate": "60.0"},  # Stream3: 640x360 at 60fps 
            ]}
    ```

&ensp;

* **`-video_source`** _(string)_: This attribute takes valid Video path as input and activates [**Single-Source Mode**](../ssm/overview), for transcoding it into multiple smaller chunks/segments for streaming after successful validation. Its value be one of the following:

    !!! tip "Usage example can be found [here ➶](../ssm/usage/#bare-minimum-usage)"

    * **Video Filename**: Valid path to Video file as follows:
        ```python
        stream_params = {"-video_source": "/home/foo/foo1.mp4"} # set input video source: /home/foo/foo1.mp4
        ```
    * **Video URL**: Valid URL of a network video stream as follows:

        !!! danger "Make sure given Video URL has protocol that is supported by installed FFmpeg. _(verify with `ffmpeg -protocols` terminal command)_"
        
        ```python
        stream_params = {"-video_source": "http://livefeed.com:5050"} # set input video source: http://livefeed.com:5050
        ``` 

&ensp;


* **`-audio`** _(string/list)_: This attribute takes external custom audio path _(as `string`)_ or audio device name followed by suitable demuxer _(as `list`)_ as audio source input for all StreamGear streams. Its value be one of the following:

    !!! failure "Make sure this audio-source is compatible with provided video -source, otherwise you could encounter multiple errors, or even no output at all!"

    * **Audio Filename** _(string)_: Valid path to Audio file as follows:
        ```python
        stream_params = {"-audio": "/home/foo/foo1.aac"} # set input audio source: /home/foo/foo1.aac
        ```
        !!! tip "Usage example can be found [here ➶](../ssm/usage/#usage-with-custom-audio)"

    * **Audio URL** _(string)_: Valid URL of a network audio stream as follows:

        !!! danger "Make sure given Video URL has protocol that is supported by installed FFmpeg. _(verify with `ffmpeg -protocols` terminal command)_"
        
        ```python
        stream_params = {"-audio": "https://exampleaudio.org/example-160.mp3"} # set input audio source: https://exampleaudio.org/example-160.mp3
        ``` 

    * **Device name and Demuxer** _(list)_: Valid audio device name followed by suitable demuxer as follows:
        
        ```python
        stream_params = {"-audio": "https://exampleaudio.org/example-160.mp3"} # set input audio source: https://exampleaudio.org/example-160.mp3
        ``` 
        !!! tip "Usage example can be found [here ➶](../rtfm/usage/#usage-with-device-audio--input)"



&ensp;

* **`-livestream`** _(bool)_: ***(optional)*** specifies whether to enable **Livestream Support**_(chunks will contain information for new frames only)_ for the selected mode, or not. You can easily set it to `True` to enable this feature, and default value is `False`. It can be used as follows: 
    
    !!! tip "Use `window_size` & `extra_window_size` FFmpeg parameters for controlling number of frames to be kept in New Chunks."

    ```python
    stream_params = {"-livestream": True} # enable livestreaming
    ```

&ensp;

* **`-input_framerate`** _(float/int)_ :  ***(optional)*** specifies the assumed input video source framerate, and only works in [Real-time Frames Mode](../usage/#b-real-time-frames-mode). It can be used as follows:

    !!! tip "Usage example can be found [here ➶](../rtfm/usage/#bare-minimum-usage-with-controlled-input-framerate)" 

    ```python
    stream_params = {"-input_framerate": 60.0} # set input video source framerate to 60fps
    ```

&ensp;

* **`-bpp`** _(float/int)_: ***(optional)*** This attribute controls constant _Bits-Per-Pixel_(BPP) value, which is kind of a constant value to ensure good quality of high motion scenes ,and thereby used in calculating desired video-bitrate for streams. Higher the BPP, better will be motion quality. Its default value is `0.1`. Going over `0.1`helps to fill gaps between current bitrate and upload limit/ingest cap. Its value can be anything above `0.001`, can be used as follows:

    !!! tip "Important BPP tips for streaming"
        * `-bpp` a sensitive value, try 0.001, and then make increments in 0.0001 to fine tune
        * If your desired resolution/fps/audio combination is below maximum service bitrate, raise BPP to match it for extra quality.  
        * It is generally better to lower resolution (and/or fps) and raise BPP than raise resolution and loose on BPP.

    ```python
    stream_params = {"-bpp": 0.05} # sets BPP to 0.05
    ```

&ensp;

* **`-gop`** _(float/int)_ : ***(optional)*** specifies the number of frames between two I-frames for accurate GOP length. By increasing the length of the GOP, there will be fewer I-frames per time frame, which minimizes bandwidth consumption. So, for example, with extremely complex subjects such as water sports or action mode, you’ll want to use a shorter GOP length such as 15 or below that results in excellent video quality. For more static video such as talking heads, then much longer GOP sizes are not only sufficient but also more efficient. It can be used as follows:

    !!! tip "The larger the GOP size, the more efficient the compression and the less bandwidth you will need"

    !!! info "By default, StreamGear automatically sets recommended fixed GOP value _(i.e. every two seconds)_ w.r.t input framerate and selected encoder."

    ```python
    stream_params = {"-gop": 70} # set GOP length to 70
    ```

&ensp;


* **`-clones`** _(list)_: ***(optional)*** sets the special FFmpeg parameters that are repeated more than once in the command _(For more info., see [this issue](https://github.com/abhiTronix/vidgear/issues/141))_ as **list** only. Usage is as follows: 

    ```python
    stream_params = {"-clones": ['-map', '0:v:0', '-map', '1:a?']}
    ```

&ensp;

* **`-ffmpeg_download_path`** _(string)_: ***(optional)*** sets the custom directory for downloading FFmpeg Static Binaries in Compression Mode, during the [Auto-Installation](../ffmpeg_install/#a-auto-installation) on Windows Machines Only. If this parameter is not altered, then these binaries will auto-save to the default temporary directory (for e.g. `C:/User/temp`) on your windows machine. It can be used as follows: 

    ```python
    stream_params = {"-ffmpeg_download_path": "C:/User/foo/foo1"} # will be saved to "C:/User/foo/foo1"
    ```

&ensp;

* **`-clear_prev_assets`** _(bool)_: ***(optional)*** specify whether to force-delete any previous copies of StreamGear Assets _(i.e. Manifest files(.mpd) & streaming chunks(.m4s) etc.)_ present at path specified by [`output`](#output) parameter. You can easily set it to `True` to enable this feature, and default value is `False`. It can be used as follows: 

    !!! info "In Single-Source Mode, additional segments _(such as `.webm`, `.mp4` chunks)_ are also cleared automatically."

    ```python
    stream_params = {"-clear_prev_assets": True} # will delete all previous assets
    ```

&ensp;

#### B. FFmpeg Parameters 

Almost all FFmpeg parameter can be passed as dictionary attributes in `stream_params`. For example, for using `libx264 encoder` to produce a lossless output video, we can pass required FFmpeg parameters as dictionary attributes, as follows:

!!! tip "Kindly check [H.264 docs ➶](https://trac.ffmpeg.org/wiki/Encode/H.264) and other [FFmpeg Docs ➶](https://ffmpeg.org/documentation.html) for more information on these parameters"


!!! error "All ffmpeg parameters are case-sensitive. Remember to double check every parameter if any error occurs."


!!! note "In addition to these parameters, almost any FFmpeg parameter _(supported by installed FFmpeg)_ is also supported. But make sure to read [**FFmpeg Docs**](https://ffmpeg.org/documentation.html) carefully first."

```python
stream_params = {"-vcodec":"libx264", "-crf": 0, "-preset": "fast", "-tune": "zerolatency"} 
```

&ensp;

### Supported Encoders and Decoders

All the encoders and decoders that are compiled with FFmpeg in use, are supported by WriteGear API. You can easily check the compiled encoders by running following command in your terminal:

!!! info "Similarily, supported demuxers and filters depends upons compiled FFmpeg in use."

```sh
# for checking encoder
ffmpeg -encoders           # use `ffmpeg.exe -encoders` on windows
# for checking decoders
ffmpeg -decoders           # use `ffmpeg.exe -decoders` on windows
``` 

&nbsp; 

## **`logging`**

This parameter enables logging _(if `True`)_, essential for debugging. 

**Data-Type:** Boolean

**Default Value:** Its default value is `False`.

**Usage:**

```python
StreamGear(logging=True)
```

&nbsp; 

[^1]: 
    :bulb: In Real-time Frames Mode, the Primary Stream's framerate defaults to [`-input_framerate`](../params/#a-exclusive-parameters) attribute value, if defined, else it will be 25fps.