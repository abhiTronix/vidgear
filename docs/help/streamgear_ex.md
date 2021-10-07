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

# StreamGear Examples

&thinsp;

## StreamGear Live-Streaming Usage with PiGear

In this example, we will be Live-Streaming video-frames from Raspberry Pi _(with Camera Module connected)_  using PiGear API and StreamGear API's Real-time Frames Mode:

??? new "New in v0.2.2" 
    This example was added in `v0.2.2`.

!!! tip "Use `-window_size` & `-extra_window_size` FFmpeg parameters for controlling number of frames to be kept in Chunks. Less these value, less will be latency."

!!! alert "After every few chunks _(equal to the sum of `-window_size` & `-extra_window_size` values)_, all chunks will be overwritten in Live-Streaming. Thereby, since newer chunks in manifest/playlist will contain NO information of any older ones, and therefore resultant DASH/HLS stream will play only the most recent frames."

!!! note "In this mode, StreamGear **DOES NOT** automatically maps video-source audio to generated streams. You need to manually assign separate audio-source through [`-audio`](../../gears/streamgear/params/#a-exclusive-parameters) attribute of `stream_params` dictionary parameter."

=== "DASH"

    ```python
    # import required libraries
    from vidgear.gears import PiGear
    from vidgear.gears import StreamGear
    import cv2

    # add various Picamera tweak parameters to dictionary
    options = {
        "hflip": True,
        "exposure_mode": "auto",
        "iso": 800,
        "exposure_compensation": 15,
        "awb_mode": "horizon",
        "sensor_mode": 0,
    }

    # open pi video stream with defined parameters
    stream = PiGear(resolution=(640, 480), framerate=60, logging=True, **options).start()

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

    ```python
    # import required libraries
    from vidgear.gears import PiGear
    from vidgear.gears import StreamGear
    import cv2

    # add various Picamera tweak parameters to dictionary
    options = {
        "hflip": True,
        "exposure_mode": "auto",
        "iso": 800,
        "exposure_compensation": 15,
        "awb_mode": "horizon",
        "sensor_mode": 0,
    }

    # open pi video stream with defined parameters
    stream = PiGear(resolution=(640, 480), framerate=60, logging=True, **options).start()

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