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

# StreamGear API Usage Examples: Single-Source Mode :material-file-video-outline:

!!! warning "Important Information :fontawesome-solid-person-military-pointing:"
    
    - [x] StreamGear **MUST** requires FFmpeg executables for its core operations. Follow these dedicated [Platform specific Installation Instructions ➶](../../ffmpeg_install/) for its installation. API will throw **RuntimeError**, if it fails to detect valid FFmpeg executables on your system.
    - [x] In this mode, ==API auto generates a primary stream of same resolution and framerate[^1] as the input video  _(at the index `0`)_.==
    - [x] In this mode, if input video-source _(i.e. `-video_source`)_ contains any audio stream/channel, then it automatically gets mapped to all generated streams.
    - [x] Always use `close()` function at the very end of the main code.

???+ danger "DEPRECATION NOTICES for `v0.3.3` and above"
    
    - [ ] The `terminate()` method in StreamGear is now deprecated and will be removed in a future release. Developers should use the new [`close()`](../../../../bonus/reference/streamgear/#vidgear.gears.streamgear.StreamGear.close) method instead, as it offers a more descriptive name, similar to the WriteGear API, for safely terminating StreamGear processes.
    - [ ] The [`-livestream`](../../params/#a-exclusive-parameters) optional parameter is NOT supported in this Single-Source Mode.


!!! example "After going through following Usage Examples, Checkout more of its advanced configurations [here ➶](../../../help/streamgear_ex/)"


&thinsp;

## Bare-Minimum Usage

Following is the bare-minimum code you need to get started with StreamGear API in Single-Source Mode:

!!! note "If input video-source _(i.e. `-video_source`)_ contains any audio stream/channel, then it automatically gets mapped to all generated streams."

=== "DASH"

    ```python linenums="1"
    # import required libraries
    from vidgear.gears import StreamGear

    # activate Single-Source Mode with valid video input
    stream_params = {"-video_source": "foo.mp4"}
    # describe a suitable manifest-file location/name and assign params
    streamer = StreamGear(output="dash_out.mpd", **stream_params)
    # transcode source
    streamer.transcode_source()
    # close
    streamer.close()
    ```

    !!! success "After running this bare-minimum example, StreamGear will produce a Manifest file (`dash_out.mpd`) with streamable chunks, containing information about a Primary Stream with the same resolution and framerate as the input."

=== "HLS"

    ```python linenums="1"
    # import required libraries
    from vidgear.gears import StreamGear

    # activate Single-Source Mode with valid video input
    stream_params = {"-video_source": "foo.mp4"}
    # describe a suitable master playlist location/name and assign params
    streamer = StreamGear(output="hls_out.m3u8", format = "hls", **stream_params)
    # transcode source
    streamer.transcode_source()
    # close
    streamer.close()
    ```

    !!! success "After running this bare-minimum example, StreamGear will produce a Master Playlist file (`hls_out.mpd`) with streamable chunks, containing information about a Primary Stream with the same resolution and framerate as the input."


&thinsp;

## Usage with Additional Streams

> In addition to the Primary Stream, you can easily generate any number of additional Secondary Streams with variable bitrates or spatial resolutions, using the exclusive [`-streams`](../../params/#a-exclusive-parameters) attribute of the `stream_params` dictionary parameter. 

To generate Secondary Streams, add each desired resolution and bitrate/framerate as a list of dictionaries to the `-streams` attribute. StreamGear will handle the rest automatically. The complete example is as follows:

!!! info "A more detailed information on `-streams` attribute can be found [here ➶](../../params/#a-exclusive-parameters)" 

!!! note "If input video-source _(i.e. `-video_source`)_ contains any audio stream/channel, then it automatically gets mapped to all generated streams without any extra efforts."

???+ danger "Important Information about `-streams` attribute :material-file-document-alert-outline:"

    * In addition to the user-defined Secondary Streams, StreamGear automatically generates a Primary Stream _(at index `0`)_ with the same resolution and framerate as the input video-source _(i.e. `-video_source`)_.
    * :warning: Ensure that your system, machine, server, or network can handle the additional resource requirements of the Secondary Streams. Exercise discretion when configuring multiple streams.
    * You **MUST** define the `-resolution` value for each stream; otherwise, the stream will be discarded.
    * You only need to define either the `-video_bitrate` or the `-framerate` for a valid stream. 
        * If you specify the `-framerate`, the video bitrate will be calculated automatically.
        * If you define both the `-video_bitrate` and the `-framerate`, the `-framerate` will get discard automatically.

!!! failure "Always use the `-streams` attribute to define additional streams safely. Duplicate or incorrect definitions can break the transcoding pipeline and corrupt the output chunks."

=== "DASH"

    ```python linenums="1" hl_lines="6-12"
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
    # transcode source
    streamer.transcode_source()
    # close
    streamer.close()
    ```

=== "HLS"

    ```python linenums="1" hl_lines="6-12"
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
    # transcode source
    streamer.transcode_source()
    # close
    streamer.close()
    ```

&thinsp;

## Usage with Custom Audio-Input

> In single source mode, by default, if the input video source (i.e., `-video_source`) contains audio, it gets automatically mapped to all generated streams. However, if you want to add a custom audio source, you can use the exclusive [`-audio`](../../params/#a-exclusive-parameters) attribute of the `stream_params` dictionary parameter.

To add a custom audio source, provide the path to your audio file as a string to the `-audio` attribute. The API will automatically validate and map the audio to all generated streams. The complete example is as follows:

!!! failure "Ensure the provided `-audio` audio source is compatible with the input video source (`-video_source`). Incompatibility can cause multiple errors or result in no output at all."

!!! tip "You can also assign a valid audio URL as input instead of a file path. More details can be found [here ➶](../../params/#a-exclusive-parameters)"


=== "DASH"

    ```python linenums="1" hl_lines="11-12"
    # import required libraries
    from vidgear.gears import StreamGear

    # activate Single-Source Mode and various streams, along with custom audio
    stream_params = {
        "-video_source": "foo.mp4",
        "-streams": [
            {"-resolution": "1280x720", "-video_bitrate": "4000k"},  # Stream1: 1280x720 at 4000kbs bitrate
            {"-resolution": "640x360", "-framerate": 60.0},  # Stream2: 640x360 at 60fps
        ],
        "-audio": "/home/foo/foo1.aac", # define custom audio-source
        "-acodec": "copy", # define copy audio encoder
    }
    # describe a suitable manifest-file location/name and assign params
    streamer = StreamGear(output="dash_out.mpd", **stream_params)
    # transcode source
    streamer.transcode_source()
    # close
    streamer.close()
    ```

=== "HLS"

    ```python linenums="1" hl_lines="11-12"
    # import required libraries
    from vidgear.gears import StreamGear

    # activate Single-Source Mode and various streams, along with custom audio
    stream_params = {
        "-video_source": "foo.mp4",
        "-streams": [
            {"-resolution": "1280x720", "-video_bitrate": "4000k"},  # Stream1: 1280x720 at 4000kbs bitrate
            {"-resolution": "640x360", "-framerate": 60.0},  # Stream2: 640x360 at 60fps
        ],
        "-audio": "/home/foo/foo1.aac",  # define custom audio-source
        "-acodec": "copy", # define copy audio encoder
    }
    # describe a suitable master playlist location/name and assign params
    streamer = StreamGear(output="hls_out.m3u8", format = "hls", **stream_params)
    # transcode source
    streamer.transcode_source()
    # close
    streamer.close()
    ```


&thinsp;


## Usage with Variable FFmpeg Parameters

> For fine-grained control over the transcoding process, StreamGear provides a highly extensible and flexible wrapper around [**FFmpeg**](https://ffmpeg.org/) library and access to almost all of its configurational parameter. 

In this example, we'll use the [H.265/HEVC](https://trac.ffmpeg.org/wiki/Encode/H.265) video encoder and [AAC](https://trac.ffmpeg.org/wiki/Encode/AAC) audio encoder, apply various optimal FFmpeg configurational parameters.

!!! warning "This example assumes that the given input video source (`-video_source`) contains at least one audio stream."

!!! info "This example is just conveying the idea on how to use FFmpeg's internal encoders/parameters with StreamGear API. You can use any FFmpeg parameter in the similar manner."

!!! danger "Refer to the FFmpeg Documentation (https://ffmpeg.org/documentation.html) before passing FFmpeg values to `stream_params`. Incorrect values may result in errors or no output."

=== "DASH"

    ```python linenums="1" hl_lines="6-9 14"
    # import required libraries
    from vidgear.gears import StreamGear

    # activate Single-Source Mode and various other parameters
    stream_params = {
        "-video_source": "foo.mp4", # define Video-Source
        "-vcodec": "libx265", # specify H.265/HEVC video encoder
        "-x265-params": "lossless=1", # enables Lossless encoding
        "-bpp": 0.15, # Bits-Per-Pixel(BPP), an Internal StreamGear parameter to ensure good quality of high motion scenes
        "-streams": [
            {"-resolution": "640x360", "-video_bitrate": "4000k"}, # Stream1: 1280x720 at 4000kbs bitrate
            {"-resolution": "320x240", "-framerate": 60.0},  # Stream2: 640x360 at 60fps
        ],
        "-acodec": "aac", # specify AAC audio encoder
    }

    # describe a suitable manifest-file location/name and assign params
    streamer = StreamGear(output="dash_out.mpd", logging=True, **stream_params)
    # transcode source
    streamer.transcode_source()
    # close
    streamer.close()
    ```

=== "HLS"

    ```python linenums="1" hl_lines="6-9 14"
    # import required libraries
    from vidgear.gears import StreamGear

    stream_params = {
        "-video_source": "foo.mp4", # define Video-Source
        "-vcodec": "libx265", # specify H.265/HEVC video encoder
        "-x265-params": "lossless=1", # enables Lossless encoding
        "-bpp": 0.15, # Bits-Per-Pixel(BPP), an Internal StreamGear parameter to ensure good quality of high motion scenes
        "-streams": [
            {"-resolution": "640x360", "-video_bitrate": "4000k"}, # Stream1: 1280x720 at 4000kbs bitrate
            {"-resolution": "320x240", "-framerate": 60.0},  # Stream2: 640x360 at 60fps
        ],
        "-acodec": "aac", # specify AAC audio encoder
    }

    # describe a suitable master playlist file location/name and assign params
    streamer = StreamGear(output="hls_out.m3u8", format = "hls", logging=True, **stream_params)
    # transcode source
    streamer.transcode_source()
    # close
    streamer.close()
    ```

&nbsp;

[^1]: 
    :bulb: In Real-time Frames Mode, the Primary Stream's framerate defaults to [`-input_framerate`](../../params/#a-exclusive-parameters) attribute value, if defined, else it will be 25fps.