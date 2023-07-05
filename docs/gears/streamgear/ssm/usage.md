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

# StreamGear API Usage Examples: Single-Source Mode

!!! warning "Important Information"
    
    * StreamGear **MUST** requires FFmpeg executables for its core operations. Follow these dedicated [Platform specific Installation Instructions ➶](../../ffmpeg_install/) for its installation.

    * StreamGear API will throw **RuntimeError**, if it fails to detect valid FFmpeg executables on your system.

    * By default, ==StreamGear generates a primary stream of same resolution and framerate[^1] as the input video  _(at the index `0`)_.==

    * Always use `terminate()` function at the very end of the main code.


!!! experiment "After going through following Usage Examples, Checkout more of its advanced configurations [here ➶](../../../help/streamgear_ex/)"


&thinsp;

## Bare-Minimum Usage

Following is the bare-minimum code you need to get started with StreamGear API in Single-Source Mode:

!!! note "If input video-source _(i.e. `-video_source`)_ contains any audio stream/channel, then it automatically gets mapped to all generated streams without any extra efforts."

=== "DASH"

    ```python
    # import required libraries
    from vidgear.gears import StreamGear

    # activate Single-Source Mode with valid video input
    stream_params = {"-video_source": "foo.mp4"}
    # describe a suitable manifest-file location/name and assign params
    streamer = StreamGear(output="dash_out.mpd", **stream_params)
    # trancode source
    streamer.transcode_source()
    # terminate
    streamer.terminate()
    ```

=== "HLS"

    ```python
    # import required libraries
    from vidgear.gears import StreamGear

    # activate Single-Source Mode with valid video input
    stream_params = {"-video_source": "foo.mp4"}
    # describe a suitable master playlist location/name and assign params
    streamer = StreamGear(output="hls_out.m3u8", format = "hls", **stream_params)
    # trancode source
    streamer.transcode_source()
    # terminate
    streamer.terminate()
    ```


!!! success "After running this bare-minimum example, StreamGear will produce a Manifest file _(`dash.mpd`)_ with streamable chunks that contains information about a Primary Stream of same resolution and framerate as the input."

&thinsp;

## Bare-Minimum Usage with Live-Streaming

You can easily activate ==Low-latency Livestreaming in Single-Source Mode==, where chunks will contain information for few new frames only and forgets all previous ones), using exclusive [`-livestream`](../../params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter as follows:

!!! note "If input video-source _(i.e. `-video_source`)_ contains any audio stream/channel, then it automatically gets mapped to all generated streams without any extra efforts."

=== "DASH"

    !!! tip "Chunk size in DASH"
        Use `-window_size` & `-extra_window_size` FFmpeg parameters for controlling number of frames to be kept in Chunks in DASH stream. Less these value, less will be latency.

    !!! alert "After every few chunks _(equal to the sum of `-window_size` & `-extra_window_size` values)_, all chunks will be overwritten in Live-Streaming. Thereby, since newer chunks in manifest will contain NO information of any older ones, and therefore resultant DASH stream will play only the most recent frames."

    ```python hl_lines="5"
    # import required libraries
    from vidgear.gears import StreamGear

    # activate Single-Source Mode with valid video input and enable livestreaming
    stream_params = {"-video_source": 0, "-livestream": True}
    # describe a suitable manifest-file location/name and assign params
    streamer = StreamGear(output="dash_out.mpd", **stream_params)
    # trancode source
    streamer.transcode_source()
    # terminate
    streamer.terminate()
    ```

=== "HLS"

    !!! tip "Chunk size in HLS"
    
        Use `-hls_init_time` & `-hls_time` FFmpeg parameters for controlling number of frames to be kept in Chunks in HLS stream. Less these value, less will be latency.

    !!! alert "After every few chunks _(equal to the sum of `-hls_init_time` & `-hls_time` values)_, all chunks will be overwritten in Live-Streaming. Thereby, since newer chunks in playlist will contain NO information of any older ones, and therefore resultant HLS stream will play only the most recent frames."

    ```python hl_lines="5"
    # import required libraries
    from vidgear.gears import StreamGear

    # activate Single-Source Mode with valid video input and enable livestreaming
    stream_params = {"-video_source": 0, "-livestream": True}
    # describe a suitable master playlist location/name and assign params
    streamer = StreamGear(output="hls_out.m3u8", format = "hls", **stream_params)
    # trancode source
    streamer.transcode_source()
    # terminate
    streamer.terminate()
    ```

&thinsp;

## Usage with Additional Streams

In addition to Primary Stream, you can easily generate any number of additional Secondary Streams of variable bitrates or spatial resolutions, using exclusive [`-streams`](../../params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter. You just need to add each resolution and bitrate/framerate as list of dictionaries to this attribute, and rest is done automatically.

!!! info "A more detailed information on `-streams` attribute can be found [here ➶](../../params/#a-exclusive-parameters)" 

The complete example is as follows:

!!! note "If input video-source contains any audio stream/channel, then it automatically gets assigned to all generated streams without any extra efforts."

??? danger "Important `-streams` attribute Information"
    
    * On top of these additional streams, StreamGear by default, generates a primary stream of same resolution and framerate as the input, at the index `0`.
    * :warning: Make sure your System/Machine/Server/Network is able to handle these additional streams, discretion is advised! 
    * You **MUST** need to define `-resolution` value for your stream, otherwise stream will be discarded!
    * You only need either of `-video_bitrate` or `-framerate` for defining a valid stream. Since with `-framerate` value defined, video-bitrate is calculated automatically.
    * If you define both `-video_bitrate` and `-framerate` values at the same time, StreamGear will discard the `-framerate` value automatically.

!!! fail "Always use `-stream` attribute to define additional streams safely, any duplicate or incorrect definition can break things!"


=== "DASH"

    ```python hl_lines="6-12"
    # import required libraries
    from vidgear.gears import StreamGear

    # activate Single-Source Mode and also define various streams
    stream_params = {
        "-video_source": "foo.mp4",
        "-streams": [
            {"-resolution": "1920x1080", "-video_bitrate": "4000k"},  # Stream1: 1920x1080 at 4000kbs bitrate
            {"-resolution": "1280x720", "-framerate": 30.0},  # Stream2: 1280x720 at 30fps framerate
            {"-resolution": "640x360", "-framerate": 60.0},  # Stream3: 640x360 at 60fps framerate
            {"-resolution": "320x240", "-video_bitrate": "500k"},  # Stream3: 320x240 at 500kbs bitrate
        ],
    }
    # describe a suitable manifest-file location/name and assign params
    streamer = StreamGear(output="dash_out.mpd", **stream_params)
    # trancode source
    streamer.transcode_source()
    # terminate
    streamer.terminate()
    ```

=== "HLS"

    ```python hl_lines="6-12"
    # import required libraries
    from vidgear.gears import StreamGear

    # activate Single-Source Mode and also define various streams
    stream_params = {
        "-video_source": "foo.mp4",
        "-streams": [
            {"-resolution": "1920x1080", "-video_bitrate": "4000k"},  # Stream1: 1920x1080 at 4000kbs bitrate
            {"-resolution": "1280x720", "-framerate": 30.0},  # Stream2: 1280x720 at 30fps framerate
            {"-resolution": "640x360", "-framerate": 60.0},  # Stream3: 640x360 at 60fps framerate
            {"-resolution": "320x240", "-video_bitrate": "500k"},  # Stream3: 320x240 at 500kbs bitrate
        ],
    }
    # describe a suitable master playlist location/name and assign params
    streamer = StreamGear(output="hls_out.m3u8", format = "hls", **stream_params)
    # trancode source
    streamer.transcode_source()
    # terminate
    streamer.terminate()
    ```

&thinsp;

## Usage with Custom Audio

By default, if input video-source _(i.e. `-video_source`)_ contains any audio, then it gets automatically mapped to all generated streams. But, if you want to add any custom audio, you can easily do it by using exclusive [`-audio`](../../params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter. You just need to input the path of your audio file to this attribute as `string`, and the API will automatically validate as well as map it to all generated streams. 

The complete example is as follows:

!!! failure "Make sure this `-audio` audio-source it compatible with provided video-source, otherwise you could encounter multiple errors or no output at all."

!!! tip "You can also assign a valid Audio URL as input, rather than filepath. More details can be found [here ➶](../../params/#a-exclusive-parameters)"


=== "DASH"

    ```python hl_lines="12"
    # import required libraries
    from vidgear.gears import StreamGear

    # activate Single-Source Mode and various streams, along with custom audio
    stream_params = {
        "-video_source": "foo.mp4",
        "-streams": [
            {"-resolution": "1920x1080", "-video_bitrate": "4000k"},  # Stream1: 1920x1080 at 4000kbs bitrate
            {"-resolution": "1280x720", "-framerate": 30.0},  # Stream2: 1280x720 at 30fps
            {"-resolution": "640x360", "-framerate": 60.0},  # Stream3: 640x360 at 60fps
        ],
        "-audio": "/home/foo/foo1.aac" # assigns input audio-source: "/home/foo/foo1.aac"
    }
    # describe a suitable manifest-file location/name and assign params
    streamer = StreamGear(output="dash_out.mpd", **stream_params)
    # trancode source
    streamer.transcode_source()
    # terminate
    streamer.terminate()
    ```

=== "HLS"

    ```python hl_lines="12"
    # import required libraries
    from vidgear.gears import StreamGear

    # activate Single-Source Mode and various streams, along with custom audio
    stream_params = {
        "-video_source": "foo.mp4",
        "-streams": [
            {"-resolution": "1920x1080", "-video_bitrate": "4000k"},  # Stream1: 1920x1080 at 4000kbs bitrate
            {"-resolution": "1280x720", "-framerate": 30.0},  # Stream2: 1280x720 at 30fps
            {"-resolution": "640x360", "-framerate": 60.0},  # Stream3: 640x360 at 60fps
        ],
        "-audio": "/home/foo/foo1.aac" # assigns input audio-source: "/home/foo/foo1.aac"
    }
    # describe a suitable master playlist location/name and assign params
    streamer = StreamGear(output="hls_out.m3u8", format = "hls", **stream_params)
    # trancode source
    streamer.transcode_source()
    # terminate
    streamer.terminate()
    ```


&thinsp;


## Usage with Variable FFmpeg Parameters

For seamlessly generating these streaming assets, StreamGear provides a highly extensible and flexible wrapper around [**FFmpeg**](https://ffmpeg.org/) and access to almost all of its parameter. Thereby, you can access almost any parameter available with FFmpeg itself as dictionary attributes in [`stream_params` dictionary parameter](../../params/#stream_params), and use it to manipulate transcoding as you like. 

For this example, let us use our own [H.265/HEVC](https://trac.ffmpeg.org/wiki/Encode/H.265) video and [AAC](https://trac.ffmpeg.org/wiki/Encode/AAC) audio encoder, and set custom audio bitrate, and various other optimizations:


!!! tip "This example is just conveying the idea on how to use FFmpeg's encoders/parameters with StreamGear API. You can use any FFmpeg parameter in the similar manner."

!!! danger "Kindly read [**FFmpeg Docs**](https://ffmpeg.org/documentation.html) carefully, before passing any FFmpeg values to `stream_params` parameter. Wrong values may result in undesired errors or no output at all."

!!! fail "Always use `-streams` attribute to define additional streams safely, any duplicate or incorrect stream definition can break things!"

=== "DASH"

    ```python hl_lines="6-10 15-17"
    # import required libraries
    from vidgear.gears import StreamGear

    # activate Single-Source Mode and various other parameters
    stream_params = {
        "-video_source": "foo.mp4", # define Video-Source
        "-vcodec": "libx265", # assigns H.265/HEVC video encoder
        "-x265-params": "lossless=1", # enables Lossless encoding
        "-crf": 25, # Constant Rate Factor: 25
        "-bpp": "0.15", # Bits-Per-Pixel(BPP), an Internal StreamGear parameter to ensure good quality of high motion scenes
        "-streams": [
            {"-resolution": "1280x720", "-video_bitrate": "4000k"}, # Stream1: 1280x720 at 4000kbs bitrate
            {"-resolution": "640x360", "-framerate": 60.0},  # Stream2: 640x360 at 60fps
        ],
        "-audio": "/home/foo/foo1.aac",  # define input audio-source: "/home/foo/foo1.aac",
        "-acodec": "libfdk_aac", # assign lossless AAC audio encoder
        "-vbr": 4, # Variable Bit Rate: `4`
    }

    # describe a suitable manifest-file location/name and assign params
    streamer = StreamGear(output="dash_out.mpd", logging=True, **stream_params)
    # trancode source
    streamer.transcode_source()
    # terminate
    streamer.terminate()
    ```

=== "HLS"

    ```python hl_lines="6-10 15-17"
    # import required libraries
    from vidgear.gears import StreamGear

    # activate Single-Source Mode and various other parameters
    stream_params = {
        "-video_source": "foo.mp4", # define Video-Source
        "-vcodec": "libx265", # assigns H.265/HEVC video encoder
        "-x265-params": "lossless=1", # enables Lossless encoding
        "-crf": 25, # Constant Rate Factor: 25
        "-bpp": "0.15", # Bits-Per-Pixel(BPP), an Internal StreamGear parameter to ensure good quality of high motion scenes
        "-streams": [
            {"-resolution": "1280x720", "-video_bitrate": "4000k"}, # Stream1: 1280x720 at 4000kbs bitrate
            {"-resolution": "640x360", "-framerate": 60.0},  # Stream2: 640x360 at 60fps
        ],
        "-audio": "/home/foo/foo1.aac",  # define input audio-source: "/home/foo/foo1.aac",
        "-acodec": "libfdk_aac", # assign lossless AAC audio encoder
        "-vbr": 4, # Variable Bit Rate: `4`
    }

    # describe a suitable master playlist file location/name and assign params
    streamer = StreamGear(output="hls_out.m3u8", format = "hls", logging=True, **stream_params)
    # trancode source
    streamer.transcode_source()
    # terminate
    streamer.terminate()
    ```

&nbsp;

[^1]: 
    :bulb: In Real-time Frames Mode, the Primary Stream's framerate defaults to [`-input_framerate`](../../params/#a-exclusive-parameters) attribute value, if defined, else it will be 25fps.