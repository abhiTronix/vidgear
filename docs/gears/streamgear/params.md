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

!!! failure "Make sure to provide a valid filename with a valid file extension for the selected `format` value _(such as `.mpd` for MPEG-DASH and `.m3u8` for APPLE-HLS)_, otherwise StreamGear will throw `ValueError`."

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

* **Filename** _(with/without path)_: Valid filename _(with a valid extension)_ of the output Manifest or Playlist file. If the filename is provided without a path, the current working directory will be used. Its parent directory must already exist _(StreamGear doesn't create it)_, and existing assets with the same names are overwritten _(FFmpeg's `-y`)_. This can be defined as follows:

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

!!! danger "Make sure to provide a valid filename with a valid file extension in the [`output`](#output) parameter for the selected `format` value _(i.e., `.mpd` for MPEG-DASH and `.m3u8` for APPLE-HLS)_, otherwise StreamGear will throw a `ValueError`."

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

    On Windows, if a custom FFmpeg executable's path/directory is not provided through this `custom_ffmpeg` parameter, the StreamGear API will automatically attempt to download and extract suitable Static FFmpeg binaries at a suitable location on your Windows machine. More information can be found [here ➶](ffmpeg_install/#a-auto-installation).
    

**Data-Type:** String

**Default Value:** Its default value is `""` _(empty string)_.

!!! warning "StreamGear checks the FFmpeg in use has the `dash` muxer _(with `-seg_duration`)_ or `hls` muxer _(with `-var_stream_map`)_, and throws `RuntimeError` otherwise. FFmpeg **6.0 or newer** is recommended."

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

        * In addition to the user-defined Secondary Streams, ==StreamGear automatically generates a Primary Stream _(at index `0`)_ with the same resolution as the input frames and at default framerate[^1], at the index `0`.== Its resolution and bitrates can be overridden with the `-s:v:0`, `-b:v:0` and `-b:a:0` FFmpeg parameters _(e.g. `{"-s:v:0": "1280x720", "-b:v:0": "2500k"}`)_.
        * You **MUST** define the `-resolution` value for each stream; otherwise, the stream will be discarded.
        * You only need to define either the `-video_bitrate` or the `-framerate` for a valid stream. 
            * If you specify only the `-framerate`, the video bitrate will be calculated automatically _(see [`-bpp`](#a-exclusive-parameters))_.
            * If you define both, the `-video_bitrate` is used as is, and the `-framerate` only sets the stream's framerate.
        * ==Every stream is encoded at its bitrate _(capped with `-maxrate`/`-bufsize`)_==, so the bandwidth declared in the Manifest/Playlist is what clients actually get. StreamGear sets no `-crf` by default; a user-defined `-crf` is still capped by each stream's bitrate.
        * Every stream maps only the first video stream of the source _(cover art and other video tracks are skipped)_ and its first audio stream _(or the [`-audio`](#a-exclusive-parameters) source)_, so subtitles, data tracks and extra audio tracks are left out.
        * The given `-streams` dictionaries are not modified, so they can be reused.


    **To construct the additional stream dictionaries, you will need the following sub-attributes::**

    * `-resolution` _(string)_: It is **compulsory** to define the required resolution/dimension/size for the stream, otherwise, the given stream will be rejected. Its value should be in the format `"{width}x{height}"` _(odd values are rounded up to even, as encoders require; a warning is logged when it doesn't match the source's display aspect ratio)_, as shown below:
        
        ```python
        # produce a 1280x720 resolution/scale stream
        "-streams" = [{"-resolution": "1280x720"}]  
        ```

    * `-video_bitrate` _(string)_: This is an **optional** sub-attribute _(can be ignored if the `-framerate` parameter is defined)_ that generally determines the bandwidth and quality of the stream. The higher the bitrate, the better the quality and the larger the bandwidth, which can place more strain on the network. Its value is typically in `k` _(kilobits per second)_ or `M` _(Megabits per second)_. Define this attribute as follows:

        ```python
        # produce a 1280x720 resolution and 2000 kbps bitrate stream
        "-streams" : [{"-resolution": "1280x720", "-video_bitrate": "2000k"}] 
        ```

    * `-framerate` _(float/int)_: This is another **optional** sub-attribute _(can be ignored if the `-video_bitrate` parameter is defined)_ that sets the framerate of the stream _(FFmpeg's `-r` for this stream)_, and is used to calculate its bitrate when `-video_bitrate` is not given. It can't exceed the source framerate _(higher values are ignored with a warning)_; keyframes then follow time, so segments of all streams stay aligned. Its value can be a float or integer, as shown below:

        ```python
        # produce a 1280x720 resolution and 15fps framerate stream
        "-streams" : [{"-resolution": "1280x720", "-framerate": 15.0}] 
        ```

        ??? new "New in v0.3.8" 
            The framerate is applied to the stream since `v0.3.8` _(it was only used for the bitrate before)_.

    **Usage:** You can easily define any number of streams using `-streams` attribute as follows:

    ```python
    stream_params = 
        {"-streams": 
            [
            {"-resolution": "1920x1080", "-video_bitrate": "4000k"}, # Stream1: 1920x1080 at 4000kbs bitrate
            {"-resolution": "1280x720", "-framerate": 30}, # Stream2: 1280x720 at 30fps (if source is >= 30fps)
            {"-resolution": "640x360", "-framerate": 15.0},  # Stream3: 640x360 at 15fps 
            ]
        }
    ```

    !!! example "Its usage example can be found [here ➶](ssm/usage/#usage-with-additional-streams)"

&ensp;

* **`-video_source`** _(string)_: This attribute takes a valid video path as input and activates [**Single-Source Mode**](ssm/#overview), for transcoding it into multiple smaller chunks/segments for streaming after successful validation. Its value can be one of the following:

    !!! warning "StreamGear throws `ValueError` if the given `-video_source` is neither an existing file nor a supported URL, or has no readable video stream _(it silently switched to Real-time Frames Mode before `v0.3.8`)_."

    ??? info "Source audio, rotation and HDR"

        * Source audio is copied when it is AAC _(which every DASH/HLS client plays)_, and otherwise encoded to AAC _(e.g. PCM, Opus)_. With `-streams`, it is always encoded at each stream's `-audio_bitrate`.
        * Rotated sources _(e.g. portrait phone videos)_ keep their display orientation: the Primary Stream gets the rotated dimensions _(e.g. `720x1280`)_.
        * HDR sources _(PQ/HLG)_ are encoded as 8-bit SDR without tone-mapping and may look washed out, so StreamGear logs a warning. Add a tone-mapping filter yourself, e.g. `{"-vf": "zscale=t=linear,tonemap=hable,zscale=p=bt709:t=bt709:m=bt709,format=yuv420p"}` _(needs FFmpeg built with `--enable-libzimg`)_.

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

    !!! example "Its usage example can be found [here ➶](ssm/usage/#bare-minimum-usage)"

&ensp;

* **`-audio`** _(string/list)_: This attribute takes an external custom audio path _(as a string)_ or an audio device name followed by a suitable demuxer _(as a list)_ as the audio source input for all StreamGear streams. Its value can be one of the following:

    !!! failure "Ensure the provided `-audio` audio source is compatible with the input video source. Incompatibility can cause multiple errors or result in no output at all."

    !!! warning "StreamGear throws `ValueError` if the given `-audio` is neither an existing file nor a supported URL, or has no readable audio stream _(it was silently dropped before `v0.3.8`)_."

    !!! info "The `-audio` source replaces any audio of the `-video_source` _(in both DASH and HLS)_, is encoded with `aac` by default, and ends with the video: shorter audio is padded with silence _(`-af apad`, unless you set an audio filter)_ and `-shortest` is added. So infinite sources _(live microphone, `lavfi`)_ end with your frames on `close()`. With copied audio _(`-acodec copy`)_, neither is added."

    * **Audio Filename** _(string)_: Valid path to an audio file as follows:

        ```python
        # set audio source as `/home/foo/foo1.aac`
        stream_params = {"-audio": "/home/foo/foo1.aac"} 
        ```

        !!! example "Its usage examples can be found [here ➶](ssm/usage/#usage-with-custom-audio-input) and [here ➶](rtfm/usage/#usage-with-file-audio-input)"

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

        !!! example "Its usage example can be found [here ➶](rtfm/usage/#usage-with-device-audio-input)"

&ensp;

* **`-livestream`** _(bool)_: ***(optional)*** specifies whether to enable **Live-Streaming :material-video-wireless-outline:**, where chunks will contain information for new frames only and forget previous ones, or not. The default value is `False`. It can be used as follows: 
    
    ??? new "New in v0.3.8" 
        `-livestream` also works in [Single-Source mode](ssm/#overview) since `v0.3.8`, e.g. for restreaming a live RTSP/RTMP source. Local video files are then read at their native speed _(FFmpeg's `-re`)_.

    !!! info "Live defaults: 4-second segments in both formats _(DASH `-seg_duration 4`, `-window_size 5`, `-extra_window_size 5`; HLS `-hls_time 4`, `-hls_list_size 6`, `-hls_flags delete_segments+independent_segments+temp_file`)_. Durations can be fractional _(e.g. `{"-seg_duration": 2.5}`)_. For lower latency, lower them and see FFmpeg's DASH `-ldash`, `-streaming`, `-target_latency` and HLS `-lhls` options, which can be passed in `stream_params`."

    ```python
    stream_params = {"-livestream": True} # enable live-streaming
    ```

    !!! example "Its usage example can be found [here ➶](rtfm/usage/#bare-minimum-usage-with-live-streaming)" 

&ensp;

* **`-input_framerate`** _(float/int)_ :  ***(optional)*** This parameter specifies the framerate of the incoming frames _(i.e. how many frames make one second of video)_ and only works in [Real-time Frames Mode](rtfm/). Its default value is `25.0` fps _(also used when the value is invalid)_. Its value can be a float, integer or numeric string, as shown below:

    ```python
    # set input video source framerate to 60fps
    stream_params = {"-input_framerate": 60.0} 
    ```

    !!! example "Its usage example can be found [here ➶](rtfm/usage/#bare-minimum-usage-with-controlled-input-framerate)" 

&ensp;

* **`-bpp`** _(float/int)_: ***(optional)*** This attribute controls the constant **BPP** _(Bits-Per-Pixel)_ value, which helps ensure good quality in high motion scenes by determining the desired video bitrate for streams. A higher BPP value improves motion quality. The default value is `0.1`. Increasing the BPP value helps fill the gaps between the current bitrate and the upload limit/ingest cap. Its value can be anything above `0.001` and can be used as follows:

    !!! tip "Important points while tweaking BPP"
        * The resulting bitrate is `width × height × framerate × BPP`, e.g. `0.1` gives ~2304 kbps for 1280x720 at 25fps.
        * If your desired resolution/fps/audio combination is below the maximum service bitrate, raise BPP to match it for extra quality.
        * It is generally better to lower resolution _(and/or `fps`)_ and raise BPP than to raise resolution and lose BPP.

    ```python
    # sets BPP to 0.05
    stream_params = {"-bpp": 0.05} 
    ```

&ensp;

* **`-gop`** _(float/int)_ : ***(optional)*** This parameter specifies the number of frames between two I-frames for accurate **GOP** _(Group of Pictures)_ length. Increasing the GOP length reduces the number of I-frames per time frame, minimizing bandwidth consumption. For example, with complex subjects such as water sports or action scenes, a shorter GOP length _(e.g., `15` or below)_ results in excellent video quality. For more static video, such as talking heads, much longer GOP sizes are not only sufficient but also more efficient. It can be used as follows:

    !!! tip "The larger the GOP size, the more efficient the compression and the less bandwidth you will need."

    !!! info "By default, StreamGear automatically sets a recommended fixed GOP value _(i.e., every two seconds)_ based on the input framerate and selected encoder. Values below `1` are invalid. For `libx265`, the GOP is merged into your `-x265-params` _(e.g. `crf=28:keyint=50:min-keyint=50`)_."

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

* **`-ffmpeg_download_path`** _(string)_: ***(optional)*** This parameter sets a custom directory for downloading FFmpeg static binaries in Compression Mode during the [**Auto-Installation**](ffmpeg_install/#a-auto-installation) step on Windows machines only. If this parameter is not altered, the binaries will be saved to the default temporary directory _(e.g., `C:/User/foo/temp`)_ on your Windows machine. It can be used as follows:

    ```python
    # download FFmpeg static binaries to `C:/User/foo/bar`
    stream_params = {"-ffmpeg_download_path": "C:/User/foo/bar"} 
    ```

&ensp;

* **`-clear_prev_assets`** _(bool)_: ***(optional)*** This parameter specifies whether to remove/delete previous StreamGear assets in the directory of the [`output`](#output) parameter before transcoding. The default value is `False`. It can be enabled as follows:

    !!! warning "It deletes the output manifest/playlist itself, plus every file named like StreamGear's assets: `chunk-stream*`, `init-stream*` _(DASH segments)_, `stream_*.m3u8` _(HLS variant playlists)_ and `init*.mp4` _(HLS fMP4 init segments)_. Other files _(including other `.mpd`/`.m3u8`)_ are kept. Don't store your own files with these names there."

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

* **`-progress_callback`** _(callable)_: ***(optional)*** This parameter sets a function called as `callback(done_s, total_s)` while [Single-Source Mode](ssm/#overview) transcodes, alongside the built-in progress bar, where `done_s` is seconds of source transcoded so far and `total_s` is the source duration _(`0` when unknown)_. If it raises, the error is logged and the callback is disabled for that run _(transcoding continues)_. The default value is `None`. Its usage is as follows:

    ??? new "New in v0.3.7" 
        This feature was added in `v0.3.7`.

    ```python
    # print transcoding progress
    stream_params = {"-progress_callback": lambda done, total: print(f"{done:.1f}/{total:.1f}s")} 
    ```

* **`-capture_stderr`** _(bool)_: **Deprecated, has no effect.** FFmpeg's stderr is now always captured in [Real-time Frames Mode](rtfm/#overview): its last 50 lines _(still mirrored to console when `logging=True`)_ are exposed via StreamGear's `stderr_tail` property and appended to the `ValueError` raised by `stream()` when FFmpeg dies early. FFmpeg's exit code is available via the `returncode` attribute after `close()` _(or after `transcode_source()` in Single-Source Mode)_.

    ??? danger "Deprecated in v0.3.8" 
        Added in `v0.3.7`, always on since `v0.3.8`.

&ensp;

#### B. FFmpeg Parameters 

Almost all FFmpeg parameters can be passed as dictionary attributes in `stream_params`. For example, to use the `libx264` encoder with a quality target _(capped by each stream's bitrate)_, you can pass the required FFmpeg parameters as dictionary attributes as follows:

!!! info "Defaults: `libx264` with `-preset veryfast`, `-vf format=yuv420p`, GOP of 2 seconds _(`-g`, `-keyint_min`, `-sc_threshold 0`, `-bf 1`)_ and `-tune zerolatency` _(Real-time Frames Mode and live-streaming only)_. VOD segments are 5 seconds in DASH _(`-seg_duration`)_ and 2 seconds in HLS _(FFmpeg's `-hls_time` default)_. No `-crf` is set by default."

!!! tip "Set a parameter's value to `None` to pass it as a valueless flag _(e.g. `{"-shortest": None}`)_."

!!! tip "Please check the [H.264 documentation ➶](https://trac.ffmpeg.org/wiki/Encode/H.264) and [FFmpeg Documentation ➶](https://ffmpeg.org/documentation.html) for more information on following parameters."

!!! failure "All FFmpeg parameters are case-sensitive. Double-check each parameter if any errors occur."

!!! note "In addition to these parameters, almost any FFmpeg parameter _(supported by the installed FFmpeg)_ is also supported. Be sure to read the [**FFmpeg Documentation**](https://ffmpeg.org/documentation.html) carefully first."

```python
# libx264 encoder and its supported parameters
stream_params = {"-vcodec":"libx264", "-crf": 20, "-preset": "fast", "-tune": "zerolatency"} 
```

&ensp;

### Supported Encoders and Decoders

All encoders and decoders compiled with the FFmpeg in use are supported by the StreamGear API. You can check the compiled encoders by running the following command in your terminal:

???+ tip "Faster Transcoding with Stream Copy in Single Source Mode"
    
    For faster transcoding of input video, utilize Stream copy (`-vcodec copy`) as the input video encoder in the [**Single-Source Mode**](ssm/#overview) for creating HLS/DASH chunks of the primary stream efficiently. However, consider the following points:

    - :warning: Stream copy is **NOT** compatible with [**Real-time Frames Mode**](rtfm/#overview), as this mode necessitates re-encoding of incoming frames. Therefore, the `-vcodec copy` parameter will be ignored.
    - :warning: Stream copying **NOT** compatible with Custom Streams ([`-streams`](#a-exclusive-parameters)), which also require re-encoding for each additional stream. Consequently, the `-vcodec copy` parameter will be ignored.
    - When using the AAC audio stream from the input video, the Audio Stream copy (`-acodec copy`) encoder will be automatically applied _(other audio codecs are encoded to `aac`)_.

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
    :bulb: In Real-time Frames Mode, the Primary Stream's framerate defaults to [`-input_framerate`](params/#a-exclusive-parameters) attribute value, if defined, else it will be 25fps.