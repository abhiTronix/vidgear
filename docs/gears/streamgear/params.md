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

This parameter sets the valid filename/path for storing the StreamGear assets, including Manifest file _(such as MPD in case of DASH)_ or a Master Playlist _(such as M3U8 in case of Apple HLS)_ and generated sequence of chunks/segments.

!!! warning "StreamGear API will throw `ValueError` if the provided `output` is empty or invalid."

!!! failure "Make sure to provide a valid filename with a valid file extension for the selected `format` value _(such as `.mpd` for MPEG-DASH and `.m3u8` for APPLE-HLS)_, otherwise StreamGear will throw `AssertionError`."

!!! tip "You can easily delete all previous assets at the `output` location by using the [`-clear_prev_assets`](#a-exclusive-parameters) attribute of the [`stream_params`](#stream_params) dictionary parameter."

**Data-Type:** String

**Usage:**

Its valid input can be one of the following: 

* **Path to directory**: Valid path of the directory. In this case, StreamGear API will automatically assign a unique filename for the Manifest file. This can be defined as follows:

    === "DASH"

        ```python
        # Define streamer with output directory path for saving DASH assets 
        streamer = StreamGear(output = "/home/foo/bar") 
        ```

    === "HLS"

        ```python
        # Define streamer with output directory path for saving HLS assets
        streamer = StreamGear(output = "/home/foo/bar", format="hls")  
        ```

* **Filename** _(with/without path)_: Valid filename _(with a valid extension)_ of the output Manifest or Playlist file. If the filename is provided without a path, the current working directory will be used. This can be defined as follows:

    === "DASH"

        ```python
        # Define streamer with output manifest filename
        streamer = StreamGear(output = "output_dash.mpd") 
        ```

    === "HLS"

        ```python
        # Define streamer with output playlist filename
        streamer = StreamGear(output = "output_hls.m3u8", format="hls") 
        ```

* **URL**: Valid URL of a network stream with a protocol supported by the installed FFmpeg _(verify with the `ffmpeg -protocols` command)_. This is useful for directly storing assets to a network server. For example, you can use an `HTTP` protocol URL as follows:
    
    === "DASH"

        ```python
        # Define streamer with output manifest URL
        streamer = StreamGear(output = "http://some_dummy_serverip/live/output_dash.mpd") 
        ```

    === "HLS"

        ```python
        # Define streamer with output playlist URL
        streamer = StreamGear(output = "http://some_dummy_serverip/live/output_hls.m3u8", format="hls")
        ```

&nbsp;

## **`format`** 

This parameter enables the adaptive HTTP streaming format. This parameter currently supported these formats: `dash` _(i.e [**MPEG-DASH**](https://www.encoding.com/mpeg-dash/))_ and `hls` _(i.e [**Apple HLS**](https://developer.apple.com/documentation/http_live_streaming))_.

!!! danger "Make sure to provide a valid filename with a valid file extension in the [`output`](#output) parameter for the selected `format` value _(i.e., `.mpd` for MPEG-DASH and `.m3u8` for APPLE-HLS)_, otherwise StreamGear will throw an `AssertionError`."

!!! warning "Any improper value assigned to `format` parameter will result in a `ValueError`!"

**Data-Type:** String

**Default Value:** Its default value is `dash`

**Usage:**

=== "DASH"

    ```python
    # Define streamer with DASH format
    StreamGear(output = "output_dash.mpd", format="dash")
    ```

=== "HLS"

    ```python
    # Define streamer with HLS format
    StreamGear(output = "output_hls.m3u8", format="hls")
    ```

&nbsp; 


## **`custom_ffmpeg`**

This parameter assigns the custom _path/directory_ where the custom/downloaded FFmpeg executables are located.

!!! info "Behavior on :fontawesome-brands-windows: Windows Systems"

    On Windows, if a custom FFmpeg executable's path/directory is not provided through this `custom_ffmpeg` parameter, the StreamGear API will automatically attempt to download and extract suitable Static FFmpeg binaries at a suitable location on your Windows machine. More information can be found [here ➶](../ffmpeg_install/#a-auto-installation).
    

**Data-Type:** String

**Default Value:** Its default value is `None`.

**Usage:**

```python
# Define streamer with custom ffmpeg binary
StreamGear(output = 'output_foo.mpd', custom_ffmpeg="C://foo//bar//ffmpeg.exe")
```

&nbsp;


## **`stream_params`**

This parameter allows developers to leverage nearly all FFmpeg options, providing effortless and flexible control over its internal settings for transcoding and generating high-quality streams. All [supported parameters](#supported-parameters) can be formatted as attributes within this dictionary parameter.

!!! danger "Please read the [**FFmpeg Documentation**](https://ffmpeg.org/documentation.html) carefully before passing any additional values to the `stream_params` parameter. Incorrect values may cause errors or result in no output."

**Data-Type:** Dictionary

**Default Value:** Its default value is `{}`.


### Supported Parameters

#### A. Exclusive Parameters

StreamGear API provides some exclusive internal parameters to easily generate Streaming Assets and effortlessly tweak its internal properties. These parameters are discussed below:

* **`-streams`** _(list of dicts)_: This important attribute makes it simple and pretty straight-forward to define additional multiple streams as _list of dictionaries_ of different quality levels _(i.e. different bitrate or spatial resolutions)_ for streaming. 

    ???+ danger "Important Information about `-streams` attribute :material-file-document-alert-outline:"

        * In addition to the user-defined Secondary Streams, ==StreamGear automatically generates a Primary Stream _(at index `0`)_ with the same resolution as the input frames and at default framerate[^1], at the index `0`.==  
        * You **MUST** define the `-resolution` value for each stream; otherwise, the stream will be discarded.
        * You only need to define either the `-video_bitrate` or the `-framerate` for a valid stream. 
            * If you specify the `-framerate`, the video bitrate will be calculated automatically.
            * If you define both the `-video_bitrate` and the `-framerate`, the `-framerate` will get discard automatically.


    **To construct the additional stream dictionaries, you will need the following sub-attributes::**

    * `-resolution` _(string)_: It is **compulsory** to define the required resolution/dimension/size for the stream, otherwise, the given stream will be rejected. Its value should be in the format `"{width}x{height}"`, as shown below:
        
        ```python
        # produce a 1280x720 resolution/scale stream
        "-streams" = [{"-resolution": "1280x720"}]  
        ```

    * `-video_bitrate` _(string)_: This is an **optional** sub-attribute _(can be ignored if the `-framerate` parameter is defined)_ that generally determines the bandwidth and quality of the stream. The higher the bitrate, the better the quality and the larger the bandwidth, which can place more strain on the network. Its value is typically in `k` _(kilobits per second)_ or `M` _(Megabits per second)_. Define this attribute as follows:

        ```python
        # produce a 1280x720 resolution and 2000 kbps bitrate stream
        "-streams" : [{"-resolution": "1280x720", "-video_bitrate": "2000k"}] 
        ```

    * `-framerate` _(float/int)_: This is another **optional** sub-attribute _(can be ignored if the `-video_bitrate` parameter is defined)_ that defines the assumed framerate for the stream. Its value can be a float or integer, as shown below:

        ```python
        # produce a 1280x720 resolution and 60fps framerate stream
        "-streams" : [{"-resolution": "1280x720", "-framerate": "60.0"}] 
        ```

    **Usage:** You can easily define any number of streams using `-streams` attribute as follows:

    ```python
    stream_params = 
        {"-streams": 
            [
            {"-resolution": "1920x1080", "-video_bitrate": "4000k"}, # Stream1: 1920x1080 at 4000kbs bitrate
            {"-resolution": "1280x720", "-framerate": 30}, # Stream2: 1280x720 at 30fps
            {"-resolution": "640x360", "-framerate": 60.0},  # Stream3: 640x360 at 60fps 
            ]
        }
    ```

    !!! example "Its usage example can be found [here ➶](../ssm/usage/#usage-with-additional-streams)"

&ensp;

* **`-video_source`** _(string)_: This attribute takes a valid video path as input and activates [**Single-Source Mode**](../ssm/#overview), for transcoding it into multiple smaller chunks/segments for streaming after successful validation. Its value can be one of the following:

    * **Video Filename**: Valid path to a video file as follows:

        ```python
        # set video source as `/home/foo/bar.mp4`
        stream_params = {"-video_source": "/home/foo/bar.mp4"}
        ```

    * **Video URL**: Valid URL of a network video stream as follows:

        !!! danger "Ensure the given video URL uses a protocol supported by the installed FFmpeg _(verify with `ffmpeg -protocols` terminal command)_."
        
        ```python
        # set video source as `http://livefeed.com:5050`
        stream_params = {"-video_source": "http://livefeed.com:5050"} 
        ``` 

    !!! example "Its usage example can be found [here ➶](../ssm/usage/#bare-minimum-usage)"

&ensp;

* **`-audio`** _(string/list)_: This attribute takes an external custom audio path _(as a string)_ or an audio device name followed by a suitable demuxer _(as a list)_ as the audio source input for all StreamGear streams. Its value can be one of the following:

    !!! failure "Ensure the provided `-audio` audio source is compatible with the input video source. Incompatibility can cause multiple errors or result in no output at all."

    * **Audio Filename** _(string)_: Valid path to an audio file as follows:

        ```python
        # set audio source as `/home/foo/foo1.aac`
        stream_params = {"-audio": "/home/foo/foo1.aac"} 
        ```

        !!! example "Its usage examples can be found [here ➶](../ssm/usage/#usage-with-custom-audio) and [here ➶](../ssm/usage/#usage-with-file-audio-input)"

    * **Audio URL** _(string)_: Valid URL of a network audio stream as follows:

        !!! danger "Ensure the given audio URL uses a protocol supported by the installed FFmpeg _(verify with `ffmpeg -protocols` terminal command)_."
        
        ```python
        # set input audio source as `https://exampleaudio.org/example-160.mp3`
        stream_params = {"-audio": "https://exampleaudio.org/example-160.mp3"} 
        ``` 

    * **Device name and Demuxer** _(list)_: Valid audio device name followed by a suitable demuxer as follows:
        
        ```python
        # Assign appropriate input audio-source device (compatible with video source) and its demuxer
        stream_params = {"-audio":  [
            "-f",
            "dshow",
            "-i",
            "audio=Microphone (USB2.0 Camera)",
        ]} 
        ``` 

        !!! example "Its usage example can be found [here ➶](../rtfm/usage/#usage-with-device-audio--input)"

&ensp;

* **`-livestream`** _(bool)_: ***(optional)*** specifies whether to enable **Low-latency Live-Streaming :material-video-wireless-outline:** in [**Real-time Frames Mode**](../rtfm/#overview) only, where chunks will contain information for new frames only and forget previous ones, or not. The default value is `False`. It can be used as follows: 
    
    !!! warning "The `-livestream` optional parameter is **NOT** supported in [Single-Source mode](../ssm/#overview)."

    ```python
    stream_params = {"-livestream": True} # enable live-streaming
    ```

    !!! example "Its usage example can be found [here ➶](../rtfm/usage/#bare-minimum-usage-with-live-streaming)" 

&ensp;

* **`-input_framerate`** _(float/int)_ :  ***(optional)*** This parameter specifies the assumed input video source framerate and only works in [Real-time Frames Mode](../usage/#b-real-time-frames-mode). Its default value is `25.0` fps. Its value can be a float or integer, as shown below:

    ```python
    # set input video source framerate to 60fps
    stream_params = {"-input_framerate": 60.0} 
    ```

    !!! example "Its usage example can be found [here ➶](../rtfm/usage/#bare-minimum-usage-with-controlled-input-framerate)" 

&ensp;

* **`-bpp`** _(float/int)_: ***(optional)*** This attribute controls the constant **BPP** _(Bits-Per-Pixel)_ value, which helps ensure good quality in high motion scenes by determining the desired video bitrate for streams. A higher BPP value improves motion quality. The default value is `0.1`. Increasing the BPP value helps fill the gaps between the current bitrate and the upload limit/ingest cap. Its value can be anything above `0.001` and can be used as follows:

    !!! tip "Important points while tweaking BPP"
        * BPP is a sensitive value; start with `0.001` and make small increments (`0.0001`) to fine-tune.
        * If your desired resolution/fps/audio combination is below the maximum service bitrate, raise BPP to match it for extra quality.
        * It is generally better to lower resolution _(and/or `fps`)_ and raise BPP than to raise resolution and lose BPP.

    ```python
    # sets BPP to 0.05
    stream_params = {"-bpp": 0.05} 
    ```

&ensp;

* **`-gop`** _(float/int)_ : ***(optional)*** This parameter specifies the number of frames between two I-frames for accurate **GOP** _(Group of Pictures)_ length. Increasing the GOP length reduces the number of I-frames per time frame, minimizing bandwidth consumption. For example, with complex subjects such as water sports or action scenes, a shorter GOP length _(e.g., `15` or below)_ results in excellent video quality. For more static video, such as talking heads, much longer GOP sizes are not only sufficient but also more efficient. It can be used as follows:

    !!! tip "The larger the GOP size, the more efficient the compression and the less bandwidth you will need."

    !!! info "By default, StreamGear automatically sets a recommended fixed GOP value _(i.e., every two seconds)_ based on the input framerate and selected encoder."

    ```python
    # set GOP length to 70
    stream_params = {"-gop": 70} 
    ```

&ensp;


* **`-clones`** _(list)_: ***(optional)*** This parameter sets special FFmpeg options that need to be repeated more than once in the command. For more information, see [this issue](https://github.com/abhiTronix/vidgear/issues/141). It accepts values as a **list** only. Usage is as follows:

    ```python
    # sets special FFmpeg options repeated multiple times
    stream_params = {"-clones": ['-map', '0:v:0', '-map', '1:a?']}
    ```

&ensp;

* **`-ffmpeg_download_path`** _(string)_: ***(optional)*** This parameter sets a custom directory for downloading FFmpeg static binaries in Compression Mode during the [**Auto-Installation**](../ffmpeg_install/#a-auto-installation) step on Windows machines only. If this parameter is not altered, the binaries will be saved to the default temporary directory _(e.g., `C:/User/foo/temp`)_ on your Windows machine. It can be used as follows:

    ```python
    # download FFmpeg static binaries to `C:/User/foo/bar`
    stream_params = {"-ffmpeg_download_path": "C:/User/foo/bar"} 
    ```

&ensp;

* **`-clear_prev_assets`** _(bool)_: ***(optional)*** This parameter specifies whether to remove/delete all previous copies of StreamGear assets files for selected [`format`](#format) _(i.e., manifest (`mpd`) in DASH, playlist (`mu38`) in HLS, and respective streaming chunks (`.ts`,`.m4s`), etc.)_ present at the path specified by the [`output`](#output) parameter. The default value is `False`. It can be enabled as follows:

    !!! info "Additional segments _(such as `.webm`, `.mp4` chunks)_ are also removed automatically."

    ```python
    # delete all previous assets
    stream_params = {"-clear_prev_assets": True} 
    ```

&ensp;

* **`-enable_force_termination`** _(bool)_: sets a special flag to enable the forced termination of the FFmpeg process, required only if StreamGear is getting frozen when terminated. Its usage is as follows:

    !!! warning "The `-enable_force_termination` flag can potentially cause unexpected behavior or corrupted output in certain scenarios. It is recommended to use this flag with caution."

    ```python
    # enables forced termination of FFmpeg process
    stream_params = {"-enable_force_termination": True} 
    ```

&ensp;

#### B. FFmpeg Parameters 

Almost all FFmpeg parameters can be passed as dictionary attributes in `stream_params`. For example, to use the `libx264` encoder to produce a lossless output video, you can pass the required FFmpeg parameters as dictionary attributes as follows:

!!! tip "Please check the [H.264 documentation ➶](https://trac.ffmpeg.org/wiki/Encode/H.264) and [FFmpeg Documentation ➶](https://ffmpeg.org/documentation.html) for more information on following parameters."

!!! failure "All FFmpeg parameters are case-sensitive. Double-check each parameter if any errors occur."

!!! note "In addition to these parameters, almost any FFmpeg parameter _(supported by the installed FFmpeg)_ is also supported. Be sure to read the [**FFmpeg Documentation**](https://ffmpeg.org/documentation.html) carefully first."

```python
# libx264 encoder and its supported parameters
stream_params = {"-vcodec":"libx264", "-crf": 0, "-preset": "fast", "-tune": "zerolatency"} 
```

&ensp;

### Supported Encoders and Decoders

All encoders and decoders compiled with the FFmpeg in use are supported by the StreamGear API. You can check the compiled encoders by running the following command in your terminal:

???+ tip "Faster Transcoding with Stream Copy in Single Source Mode"
    
    For faster transcoding of input video, utilize Stream copy (`-vcodec copy`) as the input video encoder in the [**Single-Source Mode**](../ssm/#overview) for creating HLS/DASH chunks of the primary stream efficiently. However, consider the following points:

    - :warning: Stream copy is **NOT** compatible with [**Real-time Frames Mode**](../rtfm/#overview), as this mode necessitates re-encoding of incoming frames. Therefore, the `-vcodec copy` parameter will be ignored.
    - :warning: Stream copying **NOT** compatible with Custom Streams ([`-streams`](#a-exclusive-parameters)), which also require re-encoding for each additional stream. Consequently, the `-vcodec copy` parameter will be ignored.
    - When using the audio stream from the input video, the Audio Stream copy (`-acodec copy`) encoder will be automatically applied.

```sh
# for checking encoder
ffmpeg -encoders           # use `ffmpeg.exe -encoders` on windows
# for checking decoders
ffmpeg -decoders           # use `ffmpeg.exe -decoders` on windows
``` 

!!! info "Similarly, supported audio/video demuxers and filters depend on the FFmpeg binaries in use."

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