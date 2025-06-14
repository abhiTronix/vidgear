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

# StreamGear API Usage Examples: Real-time Frames Mode :material-camera-burst:


!!! alert "Real-time Frames Mode itself is NOT Live-Streaming :material-video-wireless-outline:"

    To enable live-streaming in Real-time Frames Mode, use the exclusive [`-livestream`](../params/#a-exclusive-parameters) attribute of the `stream_params` dictionary parameter in the StreamGear API. Checkout following [usage example ➶](#bare-minimum-usage-with-live-streaming) for more information.

!!! warning "Important Information :fontawesome-solid-person-military-pointing:"
    
    - [x] StreamGear API **MUST** requires FFmpeg executables for its core operations. Follow these dedicated [Platform specific Installation Instructions ➶](../../ffmpeg_install/) for its installation. API will throw **RuntimeError**, if it fails to detect valid FFmpeg executables on your system.
    - [x] In this mode, ==API by default generates a primary stream _(at the index `0`)_ of same resolution as the input frames and at default framerate[^1].==
    - [x] In this mode, API **DOES NOT** automatically maps video-source audio to generated streams. You need to manually assign separate audio-source through [`-audio`](../../params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter.
    - [x] In this mode, Stream copy (`-vcodec copy`) encoder is unsupported as it requires re-encoding of incoming frames.
    - [x] Always use `close()` function at the very end of the main code.

??? danger "DEPRECATION NOTICES for `v0.3.3` and above"
    
    - [ ] The `terminate()` method in StreamGear is now deprecated and will be removed in a future release. Developers should use the new [`close()`](../../../../bonus/reference/streamgear/#vidgear.gears.streamgear.StreamGear.close) method instead, as it offers a more descriptive name, similar to the WriteGear API, for safely terminating StreamGear processes.
    - [ ] The `rgb_mode` parameter in [`stream()`](../../../bonus/reference/streamgear/#vidgear.gears.streamgear.StreamGear.stream) method, which earlier used to support RGB frames in Real-time Frames Mode is now deprecated, and will be removed in a future version. Only BGR format frames will be supported going forward. Please update your code to handle BGR format frames.

!!! example "After going through following Usage Examples, Checkout more of its advanced configurations [here ➶](../../../help/streamgear_ex/)"


&thinsp;


## Bare-Minimum Usage

Following is the bare-minimum code you need to get started with StreamGear API in Real-time Frames Mode:

!!! note "We are using [CamGear](../../../camgear/) in this Bare-Minimum example, but any [VideoCapture Gear](../../../#a-videocapture-gears) will work in the similar manner."

!!! danger "In this mode, StreamGear **DOES NOT** automatically maps video-source audio to generated streams. You need to manually assign separate audio-source through [`-audio`](../../params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter."

=== "DASH"

    ```python linenums="1"
    # import required libraries
    from vidgear.gears import CamGear
    from vidgear.gears import StreamGear
    import cv2

    # open any valid video stream(for e.g `foo1.mp4` file)
    stream = CamGear(source='foo1.mp4').start() 

    # describe a suitable manifest-file location/name
    streamer = StreamGear(output="dash_out.mpd")

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break


        # {do something with the frame here}


        # send frame to streamer
        streamer.stream(frame)

        # Show output window
        cv2.imshow("Output Frame", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()

    # safely close streamer
    streamer.close()
    ```

=== "HLS"

    ```python linenums="1"
    # import required libraries
    from vidgear.gears import CamGear
    from vidgear.gears import StreamGear
    import cv2

    # open any valid video stream(for e.g `foo1.mp4` file)
    stream = CamGear(source='foo1.mp4').start() 

    # describe a suitable manifest-file location/name
    streamer = StreamGear(output="hls_out.m3u8", format = "hls")

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break


        # {do something with the frame here}


        # send frame to streamer
        streamer.stream(frame)

        # Show output window
        cv2.imshow("Output Frame", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()

    # safely close streamer
    streamer.close()
    ```

!!! success "After running this bare-minimum example, StreamGear will produce a Manifest file _(`dash.mpd`)_ with streamable chunks that contains information about a Primary Stream of same resolution and framerate[^1] as input _(without any audio)_."

&thinsp;

## Bare-Minimum Usage with controlled Input-framerate

> In Real-time Frames Mode, StreamGear API provides the exclusive [`-input_framerate`](../../params/#a-exclusive-parameters) attribute for the `stream_params` dictionary parameter, which allows you to set the assumed constant framerate for incoming frames.

In this example, we will retrieve the framerate from a webcam video stream and set it as the value for the `-input_framerate` attribute in StreamGear.

!!! danger "Remember, the input framerate defaults to 25.0 fps if the `-input_framerate` attribute value is not defined in Real-time Frames mode."

=== "DASH"

    ```python linenums="1" hl_lines="10"
    # import required libraries
    from vidgear.gears import CamGear
    from vidgear.gears import StreamGear
    import cv2

    # Open live video stream on webcam at first index(i.e. 0) device
    stream = CamGear(source=0).start()

    # retrieve framerate from CamGear Stream and pass it as `-input_framerate` value
    stream_params = {"-input_framerate":stream.framerate}

    # describe a suitable manifest-file location/name and assign params
    streamer = StreamGear(output="dash_out.mpd", **stream_params)

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # send frame to streamer
        streamer.stream(frame)

        # Show output window
        cv2.imshow("Output Frame", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()

    # safely close streamer
    streamer.close()
    ```

=== "HLS"

    ```python linenums="1" hl_lines="10"
    # import required libraries
    from vidgear.gears import CamGear
    from vidgear.gears import StreamGear
    import cv2

    # Open live video stream on webcam at first index(i.e. 0) device
    stream = CamGear(source=0).start()

    # retrieve framerate from CamGear Stream and pass it as `-input_framerate` value
    stream_params = {"-input_framerate":stream.framerate}

    # describe a suitable manifest-file location/name and assign params
    streamer = StreamGear(output="hls_out.m3u8", format = "hls", **stream_params)

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # send frame to streamer
        streamer.stream(frame)

        # Show output window
        cv2.imshow("Output Frame", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()

    # safely close streamer
    streamer.close()
    ```

&thinsp;

## Bare-Minimum Usage with Live-Streaming

You can easily activate **Low-latency Live-Streaming :material-video-wireless-outline:** in Real-time Frames Mode, where chunks will contain information for new frames only and forget previous ones, using the exclusive [`-livestream`](../../params/#a-exclusive-parameters) attribute of the `stream_params` dictionary parameter.
The complete example is as follows:

!!! danger "In this mode, StreamGear **DOES NOT** automatically maps video-source audio to generated streams. You need to manually assign separate audio-source through [`-audio`](../../params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter."

=== "DASH"

    !!! tip "Controlling chunk size in DASH"
        To control the number of frames kept in Chunks for the DASH stream _(controlling latency)_, you can use the `-window_size` and `-extra_window_size` FFmpeg parameters. Lower values for these parameters will result in lower latency.

    !!! alert "After every few chunks _(equal to the sum of `-window_size` and `-extra_window_size` values)_, all chunks will be overwritten while Live-Streaming. This means that newer chunks in the manifest will contain NO information from older chunks, and the resulting DASH stream will only play the most recent frames, reducing latency."

    ```python linenums="1" hl_lines="11"
    # import required libraries
    from vidgear.gears import CamGear
    from vidgear.gears import StreamGear
    import cv2

    # open any valid video stream(from web-camera attached at index `0`)
    stream = CamGear(source=0).start()

    # enable livestreaming and retrieve framerate from CamGear Stream and
    # pass it as `-input_framerate` parameter for controlled framerate
    stream_params = {"-input_framerate": stream.framerate, "-livestream": True}

    # describe a suitable manifest-file location/name
    streamer = StreamGear(output="dash_out.mpd", **stream_params)

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # send frame to streamer
        streamer.stream(frame)

        # Show output window
        cv2.imshow("Output Frame", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()

    # safely close streamer
    streamer.close()
    ```

=== "HLS"

    !!! tip "Controlling chunk size in HLS"
        To control the number of frames kept in Chunks for the HLS stream _(controlling latency)_, you can use the `-hls_init_time` & `-hls_time` FFmpeg parameters. Lower values for these parameters will result in lower latency.

    !!! alert "After every few chunks _(equal to the sum of `-hls_init_time` & `-hls_time` values)_, all chunks will be overwritten while Live-Streaming. This means that newer chunks in the master playlist will contain NO information from older chunks, and the resulting HLS stream will only play the most recent frames, reducing latency."

    ```python linenums="1" hl_lines="11"
    # import required libraries
    from vidgear.gears import CamGear
    from vidgear.gears import StreamGear
    import cv2

    # open any valid video stream(from web-camera attached at index `0`)
    stream = CamGear(source=0).start()

    # enable livestreaming and retrieve framerate from CamGear Stream and
    # pass it as `-input_framerate` parameter for controlled framerate
    stream_params = {"-input_framerate": stream.framerate, "-livestream": True}

    # describe a suitable manifest-file location/name
    streamer = StreamGear(output="hls_out.m3u8", format = "hls", **stream_params)

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # send frame to streamer
        streamer.stream(frame)

        # Show output window
        cv2.imshow("Output Frame", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()

    # safely close streamer
    streamer.close()
    ```


&thinsp;

## Bare-Minimum Usage with OpenCV

> You can easily use the StreamGear API directly with any other Video Processing library _(for e.g. [OpenCV](https://github.com/opencv/opencv))_ in Real-time Frames Mode.

The following is a complete StreamGear API usage example with OpenCV:

!!! note "This is a bare-minimum example with OpenCV, but any other Real-time Frames Mode feature or example will work in a similar manner."

=== "DASH"

    ```python linenums="1"
    # import required libraries
    from vidgear.gears import StreamGear
    import cv2

    # Open suitable video stream, such as webcam on first index(i.e. 0)
    stream = cv2.VideoCapture(0) 

    # describe a suitable manifest-file location/name
    streamer = StreamGear(output="dash_out.mpd")

    # loop over
    while True:

        # read frames from stream
        (grabbed, frame) = stream.read()

        # check for frame if not grabbed
        if not grabbed:
          break

        # {do something with the frame here}
        # lets convert frame to gray for this example
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # send frame to streamer
        streamer.stream(gray)

        # Show output window
        cv2.imshow("Output Gray Frame", gray)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.release()

    # safely close streamer
    streamer.close()
    ```

=== "HLS"

    ```python linenums="1"
    # import required libraries
    from vidgear.gears import StreamGear
    import cv2

    # Open suitable video stream, such as webcam on first index(i.e. 0)
    stream = cv2.VideoCapture(0) 

    # describe a suitable manifest-file location/name
    streamer = StreamGear(output="hls_out.m3u8", format = "hls")

    # loop over
    while True:

        # read frames from stream
        (grabbed, frame) = stream.read()

        # check for frame if not grabbed
        if not grabbed:
          break

        # {do something with the frame here}
        # lets convert frame to gray for this example
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # send frame to streamer
        streamer.stream(gray)

        # Show output window
        cv2.imshow("Output Gray Frame", gray)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.release()

    # safely close streamer
    streamer.close()
    ```

&thinsp;

## Usage with Additional Streams

> Similar to Single-Source Mode, in addition to the Primary Stream, you can easily generate any number of additional Secondary Streams with variable bitrate or spatial resolution, using the exclusive [`-streams`](../../params/#a-exclusive-parameters) attribute of the `stream_params` dictionary parameter.

To generate Secondary Streams, add each desired resolution and bitrate/framerate as a list of dictionaries to the `-streams` attribute. StreamGear will handle the rest automatically. The complete example is as follows:

!!! info "A more detailed information on `-streams` attribute can be found [here ➶](../../params/#a-exclusive-parameters)" 

!!! alert "In this mode, StreamGear DOES NOT automatically maps video-source audio to generated streams. You need to manually assign separate audio-source through [`-audio`](../../params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter."

???+ danger "Important Information about `-streams` attribute :material-file-document-alert-outline:"

    * In addition to the user-defined Secondary Streams, StreamGear automatically generates a Primary Stream _(at index `0`)_ with the same resolution as the input frames and at default framerate[^1].
    * :warning: Ensure that your system, machine, server, or network can handle the additional resource requirements of the Secondary Streams. Exercise discretion when configuring multiple streams.
    * You **MUST** define the `-resolution` value for each stream; otherwise, the stream will be discarded.
    * You only need to define either the `-video_bitrate` or the `-framerate` for a valid stream. 
        * If you specify the `-framerate`, the video bitrate will be calculated automatically.
        * If you define both the `-video_bitrate` and the `-framerate`, the `-framerate` will get discard automatically.

!!! failure "Always use the `-streams` attribute to define additional streams safely. Duplicate or incorrect definitions can break the transcoding pipeline and corrupt the output chunks."

=== "DASH"

    ```python linenums="1" hl_lines="11-15"
    # import required libraries
    from vidgear.gears import CamGear
    from vidgear.gears import StreamGear
    import cv2

    # Open suitable video stream, such as webcam on first index(i.e. 0)
    stream = CamGear(source=0).start() 

    # define various streams
    stream_params = {
        "-streams": [
            {"-resolution": "1280x720", "-framerate": 30.0},  # Stream1: 1280x720 at 30fps framerate
            {"-resolution": "640x360", "-framerate": 60.0},  # Stream2: 640x360 at 60fps framerate
            {"-resolution": "320x240", "-video_bitrate": "500k"},  # Stream3: 320x240 at 500kbs bitrate
        ],
    }

    # describe a suitable manifest-file location/name and assign params
    streamer = StreamGear(output="dash_out.mpd")

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # send frame to streamer
        streamer.stream(frame)

        # Show output window
        cv2.imshow("Output Frame", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()

    # safely close streamer
    streamer.close()
    ```

=== "HLS"

    ```python linenums="1" hl_lines="11-15"
    # import required libraries
    from vidgear.gears import CamGear
    from vidgear.gears import StreamGear
    import cv2

    # Open suitable video stream, such as webcam on first index(i.e. 0)
    stream = CamGear(source=0).start() 

    # define various streams
    stream_params = {
        "-streams": [
            {"-resolution": "1280x720", "-framerate": 30.0},  # Stream1: 1280x720 at 30fps framerate
            {"-resolution": "640x360", "-framerate": 60.0},  # Stream2: 640x360 at 60fps framerate
            {"-resolution": "320x240", "-video_bitrate": "500k"},  # Stream3: 320x240 at 500kbs bitrate
        ],
    }

    # describe a suitable manifest-file location/name and assign params
    streamer = StreamGear(output="hls_out.m3u8", format = "hls")

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # send frame to streamer
        streamer.stream(frame)

        # Show output window
        cv2.imshow("Output Frame", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()

    # safely close streamer
    streamer.close()
    ```

&thinsp;

## Usage with File Audio-Input

> In Real-time Frames Mode, if you want to add audio to your streams, you need to use the exclusive [`-audio`](../../params/#a-exclusive-parameters) attribute of the `stream_params` dictionary parameter.

To add a audio source, provide the path to your audio file as a string to the `-audio` attribute. The API will automatically validate and map the audio to all generated streams. The complete example is as follows:

!!! failure "Ensure the provided `-audio` audio source is compatible with the input video source. Incompatibility can cause multiple errors or result in no output at all."

!!! warning "You **MUST** use [`-input_framerate`](../../params/#a-exclusive-parameters) attribute to set exact value of input framerate when using external audio in Real-time Frames mode, otherwise audio delay will occur in output streams."

!!! tip "You can also assign a valid audio URL as input instead of a file path. More details can be found [here ➶](../../params/#a-exclusive-parameters)"

=== "DASH"

    ```python linenums="1" hl_lines="16-17"
    # import required libraries
    from vidgear.gears import CamGear
    from vidgear.gears import StreamGear
    import cv2

    # open any valid video stream(for e.g `foo1.mp4` file)
    stream = CamGear(source='foo1.mp4').start() 

    # add various streams, along with custom audio
    stream_params = {
        "-streams": [
            {"-resolution": "1920x1080", "-video_bitrate": "4000k"},  # Stream1: 1920x1080 at 4000kbs bitrate
            {"-resolution": "1280x720", "-framerate": 30.0},  # Stream2: 1280x720 at 30fps
            {"-resolution": "640x360", "-framerate": 60.0},  # Stream3: 640x360 at 60fps
        ],
        "-input_framerate": stream.framerate, # controlled framerate for audio-video sync !!! don't forget this line !!!
        "-audio": "/home/foo/foo1.aac" # assign external audio-source
    }

    # describe a suitable manifest-file location/name and assign params
    streamer = StreamGear(output="dash_out.mpd", **stream_params)

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break


        # {do something with the frame here}


        # send frame to streamer
        streamer.stream(frame)

        # Show output window
        cv2.imshow("Output Frame", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()

    # safely close streamer
    streamer.close()
    ```

=== "HLS"

    ```python linenums="1" hl_lines="16-17"
    # import required libraries
    from vidgear.gears import CamGear
    from vidgear.gears import StreamGear
    import cv2

    # open any valid video stream(for e.g `foo1.mp4` file)
    stream = CamGear(source='foo1.mp4').start() 

    # add various streams, along with custom audio
    stream_params = {
        "-streams": [
            {"-resolution": "1920x1080", "-video_bitrate": "4000k"},  # Stream1: 1920x1080 at 4000kbs bitrate
            {"-resolution": "1280x720", "-framerate": 30.0},  # Stream2: 1280x720 at 30fps
            {"-resolution": "640x360", "-framerate": 60.0},  # Stream3: 640x360 at 60fps
        ],
        "-input_framerate": stream.framerate, # controlled framerate for audio-video sync !!! don't forget this line !!!
        "-audio": "/home/foo/foo1.aac" # assign external audio-source
    }

    # describe a suitable manifest-file location/name and assign params
    streamer = StreamGear(output="hls_out.m3u8", format = "hls", **stream_params)

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break


        # {do something with the frame here}


        # send frame to streamer
        streamer.stream(frame)

        # Show output window
        cv2.imshow("Output Frame", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()

    # safely close streamer
    streamer.close()
    ```

&thinsp;

## Usage with Device Audio-Input

> In Real-time Frames Mode, you can also use the exclusive [`-audio`](../../params/#a-exclusive-parameters) attribute of the `stream_params` dictionary parameter for streaming live audio from an external device.

To stream live audio, format your audio device name followed by a suitable demuxer as a list, and assign it to the `-audio` attribute. The API will automatically validate and map the audio to all generated streams. The complete example is as follows:

!!! alert "Example Assumptions :octicons-checklist-24:"

    - [x] You're running a Windows machine with all necessary audio drivers and software installed.
    - [x] There's an audio device named "Microphone (USB2.0 Camera)" connected to your Windows machine. Check instructions below to use device sources with the `-audio` attribute on different OS platforms.


??? info "Using devices sources with `-audio` attribute on different OS platforms"

    To use device sources with the `-audio` attribute on different OS platforms, follow these instructions:
    
    === ":fontawesome-brands-windows: Windows"

        Windows OS users can use the [dshow](https://trac.ffmpeg.org/wiki/DirectShow) (DirectShow) to list audio input device which is the preferred option for Windows users. You can refer following steps to identify and specify your sound card:

        - [x] **[OPTIONAL] Enable sound card(if disabled):** First enable your Stereo Mix by opening the "Sound" window and select the "Recording" tab, then right click on the window and select "Show Disabled Devices" to toggle the Stereo Mix device visibility. **Follow this [post ➶](https://forums.tomshardware.com/threads/no-sound-through-stereo-mix-realtek-hd-audio.1716182/) for more details.**

        - [x] **Identify Sound Card:** Then, You can locate your soundcard using `dshow` as follows:

            ```sh
            c:\> ffmpeg -list_devices true -f dshow -i dummy
            ffmpeg version N-45279-g6b86dd5... --enable-runtime-cpudetect
              libavutil      51. 74.100 / 51. 74.100
              libavcodec     54. 65.100 / 54. 65.100
              libavformat    54. 31.100 / 54. 31.100
              libavdevice    54.  3.100 / 54.  3.100
              libavfilter     3. 19.102 /  3. 19.102
              libswscale      2.  1.101 /  2.  1.101
              libswresample   0. 16.100 /  0. 16.100
            [dshow @ 03ACF580] DirectShow video devices
            [dshow @ 03ACF580]  "Integrated Camera"
            [dshow @ 03ACF580]  "USB2.0 Camera"
            [dshow @ 03ACF580] DirectShow audio devices
            [dshow @ 03ACF580]  "Microphone (Realtek High Definition Audio)"
            [dshow @ 03ACF580]  "Microphone (USB2.0 Camera)"
            dummy: Immediate exit requested
            ```


        - [x] **Specify Sound Card:** Then, you can specify your located soundcard in StreamGear as follows:

            ```python linenums="1"
            # assign appropriate input audio-source device and demuxer device and demuxer
            stream_params = {"-audio": ["-f","dshow", "-i", "audio=Microphone (USB2.0 Camera)"]}
            ```

        !!! failure "If audio still doesn't work then [checkout this troubleshooting guide ➶](https://www.maketecheasier.com/fix-microphone-not-working-windows10/) or reach us out on [Gitter ➶](https://gitter.im/vidgear/community) Community channel"


    === ":material-linux: Linux"

        Linux OS users can use the [alsa](https://ffmpeg.org/ffmpeg-all.html#alsa) to list input device to capture live audio input such as from a webcam. You can refer following steps to identify and specify your sound card:

        - [x] **Identify Sound Card:** To get the list of all installed cards on your machine, you can type `arecord -l` or `arecord -L` _(longer output)_.

            ```sh
            arecord -l

            **** List of CAPTURE Hardware Devices ****
            card 0: ICH5 [Intel ICH5], device 0: Intel ICH [Intel ICH5]
              Subdevices: 1/1
              Subdevice #0: subdevice #0
            card 0: ICH5 [Intel ICH5], device 1: Intel ICH - MIC ADC [Intel ICH5 - MIC ADC]
              Subdevices: 1/1
              Subdevice #0: subdevice #0
            card 0: ICH5 [Intel ICH5], device 2: Intel ICH - MIC2 ADC [Intel ICH5 - MIC2 ADC]
              Subdevices: 1/1
              Subdevice #0: subdevice #0
            card 0: ICH5 [Intel ICH5], device 3: Intel ICH - ADC2 [Intel ICH5 - ADC2]
              Subdevices: 1/1
              Subdevice #0: subdevice #0
            card 1: U0x46d0x809 [USB Device 0x46d:0x809], device 0: USB Audio [USB Audio]
              Subdevices: 1/1
              Subdevice #0: subdevice #0
            ```


        - [x] **Specify Sound Card:** Then, you can specify your located soundcard in WriteGear as follows:

            !!! info "The easiest thing to do is to reference sound card directly, namely "card 0" (Intel ICH5) and "card 1" (Microphone on the USB web cam), as `hw:0` or `hw:1`"

            ```python linenums="1"
            # assign appropriate input audio-source device and demuxer device and demuxer 
            stream_params = {"-audio": ["-f","alsa", "-i", "hw:1"]}
            ```

        !!! failure "If audio still doesn't work then reach us out on [Gitter ➶](https://gitter.im/vidgear/community) Community channel"


    === ":material-apple: MacOS"

        MAC OS users can use the [avfoundation](https://ffmpeg.org/ffmpeg-devices.html#avfoundation) to list input devices for grabbing audio from integrated iSight cameras as well as cameras connected via USB or FireWire. You can refer following steps to identify and specify your sound card on MacOS/OSX machines:


        - [x] **Identify Sound Card:** Then, You can locate your soundcard using `avfoundation` as follows:

            ```sh
            ffmpeg -f avfoundation -list_devices true -i ""
            ffmpeg version N-45279-g6b86dd5... --enable-runtime-cpudetect
              libavutil      51. 74.100 / 51. 74.100
              libavcodec     54. 65.100 / 54. 65.100
              libavformat    54. 31.100 / 54. 31.100
              libavdevice    54.  3.100 / 54.  3.100
              libavfilter     3. 19.102 /  3. 19.102
              libswscale      2.  1.101 /  2.  1.101
              libswresample   0. 16.100 /  0. 16.100
            [AVFoundation input device @ 0x7f8e2540ef20] AVFoundation video devices:
            [AVFoundation input device @ 0x7f8e2540ef20] [0] FaceTime HD camera (built-in)
            [AVFoundation input device @ 0x7f8e2540ef20] [1] Capture screen 0
            [AVFoundation input device @ 0x7f8e2540ef20] AVFoundation audio devices:
            [AVFoundation input device @ 0x7f8e2540ef20] [0] Blackmagic Audio
            [AVFoundation input device @ 0x7f8e2540ef20] [1] Built-in Microphone
            ```


        - [x] **Specify Sound Card:** Then, you can specify your located soundcard in StreamGear as follows:

            ```python linenums="1"
            # assign appropriate input audio-source device and demuxer
            stream_params = {"-audio": ["-f","avfoundation", "-audio_device_index", "0"]}
            ```

        !!! failure "If audio still doesn't work then reach us out on [Gitter ➶](https://gitter.im/vidgear/community) Community channel"

!!! tip "It is advised to use this example with live-streaming enabled(`True`) by using StreamGear API's exclusive [`-livestream`](../../params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter."

!!! failure "Ensure the provided `-audio` audio source is compatible with the video source device. Incompatibility can cause multiple errors or result in no output at all."

!!! warning "You **MUST** use [`-input_framerate`](../../params/#a-exclusive-parameters) attribute to set exact value of input framerate when using external audio in Real-time Frames mode, otherwise audio delay will occur in output streams."

=== "DASH"

    ```python linenums="1" hl_lines="18-25"
    # import required libraries
    from vidgear.gears import CamGear
    from vidgear.gears import StreamGear
    import cv2

    # open any valid DEVICE video stream
    stream = CamGear(source=0).start()

    # add various streams, along with custom audio
    stream_params = {
        "-streams": [
            {
                "-resolution": "640x360",
                "-video_bitrate": "4000k",
            },  # Stream1: 640x360 at 4000kbs bitrate
            {"-resolution": "320x240", "-framerate": 30.0},  # Stream2: 320x240 at 30fps
        ],
        "-input_framerate": stream.framerate,  # controlled framerate for audio-video sync !!! don't forget this line !!!
        "-livestream": True,
        "-audio": [
            "-f",
            "dshow",
            "-i",
            "audio=Microphone (USB2.0 Camera)",
        ],  # assign appropriate input audio-source device(compatible with video source) and its demuxer
    }

    # describe a suitable manifest-file location/name and assign params
    streamer = StreamGear(output="dash_out.mpd", **stream_params)

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # send frame to streamer
        streamer.stream(frame)

        # Show output window
        cv2.imshow("Output Frame", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()

    # safely close streamer
    streamer.close()
    ```

=== "HLS"

    ```python linenums="1" hl_lines="18-25"
    # import required libraries
    from vidgear.gears import CamGear
    from vidgear.gears import StreamGear
    import cv2

    # open any valid DEVICE video stream
    stream = CamGear(source=0).start()

    # add various streams, along with custom audio
    stream_params = {
        "-streams": [
            {
                "-resolution": "640x360",
                "-video_bitrate": "4000k",
            },  # Stream1: 640x360 at 4000kbs bitrate
            {"-resolution": "320x240", "-framerate": 30.0},  # Stream2: 320x240 at 30fps
        ],
        "-input_framerate": stream.framerate,  # controlled framerate for audio-video sync !!! don't forget this line !!!
        "-livestream": True,
        "-audio": [
            "-f",
            "dshow",
            "-i",
            "audio=Microphone (USB2.0 Camera)",
        ],  # assign appropriate input audio-source device(compatible with video source) and its demuxer
    }

    # describe a suitable manifest-file location/name and assign params
    streamer = StreamGear(output="hls_out.m3u8", format="hls", **stream_params)

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # send frame to streamer
        streamer.stream(frame)

        # Show output window
        cv2.imshow("Output Frame", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()

    # safely close streamer
    streamer.close()
    ```

&thinsp;

## Usage with Hardware Video-Encoder

> In Real-time Frames Mode, you can easily change the video encoder according to your requirements by passing the `-vcodec` FFmpeg parameter as an attribute in the `stream_params` dictionary parameter. Additionally, you can specify additional properties, features, and optimizations for your system's GPU.

In this example, we will be using `h264_vaapi` as our Hardware Encoder and specifying the device hardware's location and compatible video filters  by formatting them as attributes in the `stream_params` dictionary parameter.

!!! danger "This example is just conveying the idea of how to use FFmpeg's hardware encoders with the StreamGear API in Real-time Frames Mode, which MAY OR MAY NOT suit your system. Please use suitable parameters based on your supported system and FFmpeg configurations only."

???+ info "Checking VAAPI Support for Hardware Encoding"

    To use **VAAPI** (Video Acceleration API) as a hardware encoder in this example, follow these steps to ensure your FFmpeg supports VAAPI:

    ```sh
    ffmpeg  -hide_banner -encoders | grep vaapi 

     V..... h264_vaapi           H.264/AVC (VAAPI) (codec h264)
     V..... hevc_vaapi           H.265/HEVC (VAAPI) (codec hevc)
     V..... mjpeg_vaapi          MJPEG (VAAPI) (codec mjpeg)
     V..... mpeg2_vaapi          MPEG-2 (VAAPI) (codec mpeg2video)
     V..... vp8_vaapi            VP8 (VAAPI) (codec vp8)
    ```

!!! failure "Please read the [**FFmpeg Documentation**](https://ffmpeg.org/documentation.html) carefully before passing any additional values to the `stream_params` parameter. Incorrect values may cause errors or result in no output."


=== "DASH"

    ```python linenums="1" hl_lines="16-18"
    # import required libraries
    from vidgear.gears import VideoGear
    from vidgear.gears import StreamGear
    import cv2

    # Open suitable video stream, such as webcam on first index(i.e. 0)
    stream = VideoGear(source=0).start() 

    # add various streams with custom Video Encoder and optimizations
    stream_params = {
        "-streams": [
            {"-resolution": "1920x1080", "-video_bitrate": "4000k"},  # Stream1: 1920x1080 at 4000kbs bitrate
            {"-resolution": "1280x720", "-framerate": 30.0},  # Stream2: 1280x720 at 30fps
            {"-resolution": "640x360", "-framerate": 60.0},  # Stream3: 640x360 at 60fps
        ],
        "-vcodec": "h264_vaapi", # define custom Video encoder
        "-vaapi_device": "/dev/dri/renderD128", # define device location
        "-vf": "format=nv12,hwupload",  # define video filters
    }

    # describe a suitable manifest-file location/name and assign params
    streamer = StreamGear(output="dash_out.mpd", **stream_params)

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break


        # {do something with the frame here}


        # send frame to streamer
        streamer.stream(frame)

        # Show output window
        cv2.imshow("Output Frame", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()

    # safely close streamer
    streamer.close()
    ```

=== "HLS"

    ```python linenums="1" hl_lines="16-18"
    # import required libraries
    from vidgear.gears import VideoGear
    from vidgear.gears import StreamGear
    import cv2

    # Open suitable video stream, such as webcam on first index(i.e. 0)
    stream = VideoGear(source=0).start() 

    # add various streams with custom Video Encoder and optimizations
    stream_params = {
        "-streams": [
            {"-resolution": "1920x1080", "-video_bitrate": "4000k"},  # Stream1: 1920x1080 at 4000kbs bitrate
            {"-resolution": "1280x720", "-framerate": 30.0},  # Stream2: 1280x720 at 30fps
            {"-resolution": "640x360", "-framerate": 60.0},  # Stream3: 640x360 at 60fps
        ],
        "-vcodec": "h264_vaapi", # define custom Video encoder
        "-vaapi_device": "/dev/dri/renderD128", # define device location
        "-vf": "format=nv12,hwupload",  # define video pixformat
    }

    # describe a suitable manifest-file location/name and assign params
    streamer = StreamGear(output="hls_out.m3u8", format = "hls", **stream_params)

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break


        # {do something with the frame here}


        # send frame to streamer
        streamer.stream(frame)

        # Show output window
        cv2.imshow("Output Frame", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()

    # safely close streamer
    streamer.close()
    ```

&nbsp;

[^1]: 
    :bulb: In Real-time Frames Mode, the Primary Stream's framerate defaults to the value of the [`-input_framerate`](../../params/#a-exclusive-parameters) attribute, if defined. Otherwise, it will be set to 25 fps.