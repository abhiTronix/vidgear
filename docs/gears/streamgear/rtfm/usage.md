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

# StreamGear API Usage Examples: Real-time Frames Mode


!!! alert "Real-time Frames Mode is NOT Live-Streaming."

    Rather you can easily enable live-streaming in Real-time Frames Mode by using StreamGear API's exclusive [`-livestream`](../../params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter. Checkout following [usage example](#bare-minimum-usage-with-live-streaming).

!!! warning "Important Information"
    
    * StreamGear **MUST** requires FFmpeg executables for its core operations. Follow these dedicated [Platform specific Installation Instructions ➶](../../ffmpeg_install/) for its installation.

    * StreamGear API will throw **RuntimeError**, if it fails to detect valid FFmpeg executables on your system.

    * By default, ==StreamGear generates a primary stream of same resolution and framerate[^1] as the input video  _(at the index `0`)_.==

    * Always use `terminate()` function at the very end of the main code.

!!! experiment "After going through following Usage Examples, Checkout more of its advanced configurations [here ➶](../../../help/streamgear_ex/)"


&thinsp;


## Bare-Minimum Usage

Following is the bare-minimum code you need to get started with StreamGear API in Real-time Frames Mode:

!!! note "We are using [CamGear](../../../camgear/overview/) in this Bare-Minimum example, but any [VideoCapture Gear](../../../#a-videocapture-gears) will work in the similar manner."


=== "DASH"

    ```python
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
    streamer.terminate()
    ```

=== "HLS"

    ```python
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
    streamer.terminate()
    ```

!!! success "After running this bare-minimum example, StreamGear will produce a Manifest file _(`dash.mpd`)_ with streamable chunks that contains information about a Primary Stream of same resolution and framerate[^1] as input _(without any audio)_."


&thinsp;

## Bare-Minimum Usage with Live-Streaming

You can easily activate ==Low-latency Livestreaming in Real-time Frames Mode==, where chunks will contain information for few new frames only and forgets all previous ones), using exclusive [`-livestream`](../../params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter as follows:

!!! note "In this mode, StreamGear **DOES NOT** automatically maps video-source audio to generated streams. You need to manually assign separate audio-source through [`-audio`](../../params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter."

=== "DASH"

    !!! tip "Chunk size in DASH"
        Use `-window_size` & `-extra_window_size` FFmpeg parameters for controlling number of frames to be kept in Chunks in DASH stream. Less these value, less will be latency.

    !!! alert "After every few chunks _(equal to the sum of `-window_size` & `-extra_window_size` values)_, all chunks will be overwritten in Live-Streaming. Thereby, since newer chunks in manifest will contain NO information of any older ones, and therefore resultant DASH stream will play only the most recent frames."

    ```python hl_lines="11"
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
    streamer.terminate()
    ```

=== "HLS"

    !!! tip "Chunk size in HLS"
    
        Use `-hls_init_time` & `-hls_time` FFmpeg parameters for controlling number of frames to be kept in Chunks in HLS stream. Less these value, less will be latency.

    !!! alert "After every few chunks _(equal to the sum of `-hls_init_time` & `-hls_time` values)_, all chunks will be overwritten in Live-Streaming. Thereby, since newer chunks in playlist will contain NO information of any older ones, and therefore resultant HLS stream will play only the most recent frames."

    ```python hl_lines="11"
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
    streamer.terminate()
    ```


&thinsp;

## Bare-Minimum Usage with RGB Mode

In Real-time Frames Mode, StreamGear API provide [`rgb_mode`](../../../../../bonus/reference/streamgear/#vidgear.gears.streamgear.StreamGear.stream) boolean parameter with its `stream()` function, which if enabled _(i.e. `rgb_mode=True`)_, specifies that incoming frames are of RGB format _(instead of default BGR format)_, thereby also known as ==RGB Mode==.

The complete usage example is as follows:

=== "DASH"

    ```python hl_lines="28"
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


        # {simulating RGB frame for this example}
        frame_rgb = frame[:,:,::-1]


        # send frame to streamer
        streamer.stream(frame_rgb, rgb_mode = True) #activate RGB Mode

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
    streamer.terminate()
    ```

=== "HLS"

    ```python hl_lines="28"
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


        # {simulating RGB frame for this example}
        frame_rgb = frame[:,:,::-1]


        # send frame to streamer
        streamer.stream(frame_rgb, rgb_mode = True) #activate RGB Mode

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
    streamer.terminate()
    ```


&thinsp;

## Bare-Minimum Usage with controlled Input-framerate

In Real-time Frames Mode, StreamGear API provides exclusive [`-input_framerate`](../../params/#a-exclusive-parameters)  attribute for its `stream_params` dictionary parameter, that allow us to set the assumed constant framerate for incoming frames. 

In this example, we will retrieve framerate from webcam video-stream, and set it as value for `-input_framerate` attribute in StreamGear:

!!! danger "Remember, Input framerate default to `25.0` fps if [`-input_framerate`](../../params/#a-exclusive-parameters) attribute value not defined in Real-time Frames mode."


=== "DASH"

    ```python hl_lines="10"
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
    streamer.terminate()
    ```

=== "HLS"

    ```python hl_lines="10"
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
    streamer.terminate()
    ```

&thinsp;

## Bare-Minimum Usage with OpenCV

You can easily use StreamGear API directly with any other Video Processing library(_For e.g. [OpenCV](https://github.com/opencv/opencv) itself_) in Real-time Frames Mode. 

The complete usage example is as follows:

!!! tip "This just a bare-minimum example with OpenCV, but any other Real-time Frames Mode feature/example will work in the similar manner."

=== "DASH"

    ```python
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
    streamer.terminate()
    ```

=== "HLS"

    ```python
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
    streamer.terminate()
    ```


&thinsp;

## Usage with Additional Streams

Similar to Single-Source Mode, you can easily generate any number of additional Secondary Streams of variable bitrates or spatial resolutions, using exclusive [`-streams`](../../params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter. You just need to add each resolution and bitrate/framerate as list of dictionaries to this attribute, and rest is done automatically.

!!! info "A more detailed information on `-streams` attribute can be found [here ➶](../../params/#a-exclusive-parameters)" 

The complete example is as follows:

??? danger "Important `-streams` attribute Information"
    * On top of these additional streams, StreamGear by default, generates a primary stream of same resolution and framerate[^1] as the input, at the index `0`.
    * :warning: Make sure your System/Machine/Server/Network is able to handle these additional streams, discretion is advised! 
    * You **MUST** need to define `-resolution` value for your stream, otherwise stream will be discarded!
    * You only need either of `-video_bitrate` or `-framerate` for defining a valid stream. Since with `-framerate` value defined, video-bitrate is calculated automatically.
    * If you define both `-video_bitrate` and `-framerate` values at the same time, StreamGear will discard the `-framerate` value automatically.

!!! fail "Always use `-stream` attribute to define additional streams safely, any duplicate or incorrect definition can break things!"


=== "DASH"

    ```python hl_lines="11-15"
    # import required libraries
    from vidgear.gears import CamGear
    from vidgear.gears import StreamGear
    import cv2

    # Open suitable video stream, such as webcam on first index(i.e. 0)
    stream = CamGear(source=0).start() 

    # define various streams
    stream_params = {
        "-streams": [
            {"-resolution": "1280x720", "-framerate": 30.0},  # Stream2: 1280x720 at 30fps framerate
            {"-resolution": "640x360", "-framerate": 60.0},  # Stream3: 640x360 at 60fps framerate
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
    streamer.terminate()
    ```

=== "HLS"

    ```python hl_lines="11-15"
    # import required libraries
    from vidgear.gears import CamGear
    from vidgear.gears import StreamGear
    import cv2

    # Open suitable video stream, such as webcam on first index(i.e. 0)
    stream = CamGear(source=0).start() 

    # define various streams
    stream_params = {
        "-streams": [
            {"-resolution": "1280x720", "-framerate": 30.0},  # Stream2: 1280x720 at 30fps framerate
            {"-resolution": "640x360", "-framerate": 60.0},  # Stream3: 640x360 at 60fps framerate
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
    streamer.terminate()
    ```

&thinsp;

## Usage with File Audio-Input

In Real-time Frames Mode, if you want to add audio to your streams, you've to use exclusive [`-audio`](../../params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter. You just need to input the path of your audio file to this attribute as `string` value, and the API will automatically validate as well as maps it to all generated streams. 

The complete example is as follows:

!!! failure "Make sure this `-audio` audio-source it compatible with provided video-source, otherwise you could encounter multiple errors or no output at all."

!!! warning "You **MUST** use [`-input_framerate`](../../params/#a-exclusive-parameters) attribute to set exact value of input framerate when using external audio in Real-time Frames mode, otherwise audio delay will occur in output streams."

!!! tip "You can also assign a valid Audio URL as input, rather than filepath. More details can be found [here ➶](../../params/#a-exclusive-parameters)"


=== "DASH"

    ```python hl_lines="16-17"
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
        "-audio": "/home/foo/foo1.aac" # assigns input audio-source: "/home/foo/foo1.aac"
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
    streamer.terminate()
    ```

=== "HLS"

    ```python hl_lines="16-17"
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
        "-audio": "/home/foo/foo1.aac" # assigns input audio-source: "/home/foo/foo1.aac"
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
    streamer.terminate()
    ```

&thinsp;

## Usage with Device Audio-Input

In Real-time Frames Mode, you've can also use exclusive [`-audio`](../../params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter for streaming live audio from external device. You just need to format your audio device name followed by suitable demuxer as `list` and assign to this attribute, and the API will automatically validate as well as map it to all generated streams. 

The complete example is as follows:


!!! alert "Example Assumptions"

    * You're running are Windows machine with all neccessary audio drivers and software installed.
    * There's a audio device with named `"Microphone (USB2.0 Camera)"` connected to your windows machine.


??? tip "Using devices with `-audio` attribute on different OS platforms"
    
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

            ```python
            # assign appropriate input audio-source device and demuxer device and demuxer
            stream_params = {"-audio": ["-f","dshow", "-i", "audio=Microphone (USB2.0 Camera)"]}
            ```

        !!! fail "If audio still doesn't work then [checkout this troubleshooting guide ➶](https://www.maketecheasier.com/fix-microphone-not-working-windows10/) or reach us out on [Gitter ➶](https://gitter.im/vidgear/community) Community channel"


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

            ```python
            # assign appropriate input audio-source device and demuxer device and demuxer 
            stream_params = {"-audio": ["-f","alsa", "-i", "hw:1"]}
            ```

        !!! fail "If audio still doesn't work then reach us out on [Gitter ➶](https://gitter.im/vidgear/community) Community channel"


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

            ```python
            # assign appropriate input audio-source device and demuxer
            stream_params = {"-audio": ["-f","avfoundation", "-audio_device_index", "0"]}
            ```

        !!! fail "If audio still doesn't work then reach us out on [Gitter ➶](https://gitter.im/vidgear/community) Community channel"


!!! danger "Make sure this `-audio` audio-source it compatible with provided video-source, otherwise you could encounter multiple errors or no output at all."

!!! warning "You **MUST** use [`-input_framerate`](../../params/#a-exclusive-parameters) attribute to set exact value of input framerate when using external audio in Real-time Frames mode, otherwise audio delay will occur in output streams."

!!! note "It is advised to use this example with live-streaming enabled(True) by using StreamGear API's exclusive [`-livestream`](../../params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter."


=== "DASH"

    ```python hl_lines="18-24"
    # import required libraries
    from vidgear.gears import CamGear
    from vidgear.gears import StreamGear
    import cv2

    # open any valid video stream(for e.g `foo1.mp4` file)
    stream = CamGear(source="foo1.mp4").start()

    # add various streams, along with custom audio
    stream_params = {
        "-streams": [
            {
                "-resolution": "1280x720",
                "-video_bitrate": "4000k",
            },  # Stream1: 1280x720 at 4000kbs bitrate
            {"-resolution": "640x360", "-framerate": 30.0},  # Stream2: 640x360 at 30fps
        ],
        "-input_framerate": stream.framerate,  # controlled framerate for audio-video sync !!! don't forget this line !!!
        "-audio": [
            "-f",
            "dshow",
            "-i",
            "audio=Microphone (USB2.0 Camera)",
        ],  # assign appropriate input audio-source device and demuxer
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
    streamer.terminate()
    ```

=== "HLS"

    ```python hl_lines="18-24"
    # import required libraries
    from vidgear.gears import CamGear
    from vidgear.gears import StreamGear
    import cv2

    # open any valid video stream(for e.g `foo1.mp4` file)
    stream = CamGear(source="foo1.mp4").start()

    # add various streams, along with custom audio
    stream_params = {
        "-streams": [
            {
                "-resolution": "1280x720",
                "-video_bitrate": "4000k",
            },  # Stream1: 1280x720 at 4000kbs bitrate
            {"-resolution": "640x360", "-framerate": 30.0},  # Stream2: 640x360 at 30fps
        ],
        "-input_framerate": stream.framerate,  # controlled framerate for audio-video sync !!! don't forget this line !!!
        "-audio": [
            "-f",
            "dshow",
            "-i",
            "audio=Microphone (USB2.0 Camera)",
        ],  # assign appropriate input audio-source device and demuxer
    }

    # describe a suitable manifest-file location/name and assign params
    streamer = StreamGear(output="dash_out.m3u8", format="hls", **stream_params)

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
    streamer.terminate()
    ```

&thinsp;

## Usage with Hardware Video-Encoder


In Real-time Frames Mode, you can also easily change encoder as per your requirement just by passing `-vcodec` FFmpeg parameter as an attribute in `stream_params` dictionary parameter. In addition to this, you can also specify the additional properties/features/optimizations for your system's GPU similarly. 

In this example, we will be using `h264_vaapi` as our hardware encoder and also optionally be specifying our device hardware's location (i.e. `'-vaapi_device':'/dev/dri/renderD128'`) and other features such as `'-vf':'format=nv12,hwupload'` like properties by formatting them as `option` dictionary parameter's attributes, as follows:

!!! warning "Check VAAPI support"

    **This example is just conveying the idea on how to use FFmpeg's hardware encoders with WriteGear API in Compression mode, which MAY/MAY-NOT suit your system. Kindly use suitable parameters based your supported system and FFmpeg configurations only.**

    To use `h264_vaapi` encoder, remember to check if its available and your FFmpeg compiled with VAAPI support. You can easily do this by executing following one-liner command in your terminal, and observing if output contains something similar as follows:

    ```sh
    ffmpeg  -hide_banner -encoders | grep vaapi 

     V..... h264_vaapi           H.264/AVC (VAAPI) (codec h264)
     V..... hevc_vaapi           H.265/HEVC (VAAPI) (codec hevc)
     V..... mjpeg_vaapi          MJPEG (VAAPI) (codec mjpeg)
     V..... mpeg2_vaapi          MPEG-2 (VAAPI) (codec mpeg2video)
     V..... vp8_vaapi            VP8 (VAAPI) (codec vp8)
    ```


=== "DASH"

    ```python hl_lines="16-18"
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
    streamer.terminate()
    ```

=== "HLS"

    ```python hl_lines="16-18"
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
    streamer.terminate()
    ```

&nbsp;

[^1]: 
    :bulb: In Real-time Frames Mode, the Primary Stream's framerate defaults to [`-input_framerate`](../../params/#a-exclusive-parameters) attribute value, if defined, else it will be 25fps.