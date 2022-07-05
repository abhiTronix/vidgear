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

# CamGear API Usage Examples:

!!! experiment "After going through following Usage Examples, Checkout more of its advanced configurations [here ➶](../../../help/camgear_ex/)"

&thinsp;

## Bare-Minimum Usage

Following is the bare-minimum code you need to get started with CamGear API:

```python
# import required libraries
from vidgear.gears import CamGear
import cv2


# open any valid video stream(for e.g `myvideo.avi` file)
stream = CamGear(source="myvideo.avi").start()

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # Show output window
    cv2.imshow("Output", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
```

&nbsp; 

## Using Camgear with Streaming Websites

CamGear internally implements [`yt_dlp`]() backend class for seamlessly pipelining live video-frames and metadata from various streaming services like [Twitch](https://www.twitch.tv/), [Vimeo](https://vimeo.com/), [Dailymotion](https://www.dailymotion.com), and [many more ➶](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md#supported-sites). All you have to do is to provide the desired Video's URL to its `source` parameter, and enable its [`stream_mode`](../params/#stream_mode) parameter. 

The complete usage example for Dailymotion and Twitch URLs are as follows:


??? bug "Bug in OpenCV's FFmpeg"

    To workaround a [**FFmpeg bug**](https://github.com/abhiTronix/vidgear/issues/133#issuecomment-638263225) that causes video to freeze frequently in OpenCV, It is advised to always use [GStreamer backend](../params/#backend) for Livestream videos.

    **Checkout [this FAQ ➶](../../../help/camgear_faqs/#how-to-compile-opencv-with-gstreamer-support) for compiling OpenCV with GStreamer support.**

    !!! fail "Not all resolutions are supported with GStreamer Backend. See issue #244"

???+ info "Exclusive CamGear Attributes for `yt_dlp` backend"
    
    CamGear also provides exclusive attributes: 
    
    - `STREAM_RESOLUTION` _(for specifying stream resolution)_
    - `STREAM_PARAMS` _(for specifying  `yt_dlp` parameters)_ 
    
    with its [`options`](../params/#options) dictionary parameter. **More information can be found [here ➶](../advanced/source_params/#exclusive-camgear-parameters)**


??? note "Supported Streaming Websites"

    The list of all supported Streaming Websites URLs can be found [here ➶](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md#supported-sites)


??? tip "Accessing Stream's Metadata :material-database-eye:"

    CamGear now provides `ytv_metadata` global parameter for accessing given Video's metadata as JSON Object. It can used as follows:

    ??? new "New in v0.2.4" 
        `ytv_metadata` global parameter was added in `v0.2.4`.

    ```python
    # import required libraries
    from vidgear.gears import CamGear

    # Add YouTube Video URL as input source (for e.g https://www.dailymotion.com/video/x2yrnum)
    # and enable Stream Mode (`stream_mode = True`)
    stream = CamGear(
        source="https://www.dailymotion.com/video/x2yrnum", stream_mode=True, logging=True, **options
    ).start()

    # get Video's metadata as JSON object
    video_metadata =  stream.ytv_metadata

    # print all available keys
    print(video_metadata.keys())

    # get data like `title`
    print(video_metadata["title"])
    ```

=== "Dailymotion :fontawesome-brands-dailymotion:"
    ```python  hl_lines="12-13"
    # import required libraries
    from vidgear.gears import CamGear
    import cv2

    # set desired quality as 720p
    options = {"STREAM_RESOLUTION": "720p"}

    # Add any desire Video URL as input source
    # for e.g https://vimeo.com/151666798
    # and enable Stream Mode (`stream_mode = True`)
    stream = CamGear(
        source="https://www.dailymotion.com/video/x2yrnum",
        stream_mode=True,
        logging=True,
        **options
    ).start()

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # Show output window
        cv2.imshow("Output", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()
    ```
=== "Twitch :fontawesome-brands-twitch:"

    !!! warning "If Twitch user is offline, CamGear will throw ValueError."

    ```python  hl_lines="12-13"
    # import required libraries
    from vidgear.gears import CamGear
    import cv2

    # set desired quality as 720p
    options = {"STREAM_RESOLUTION": "720p"}

    # Add any desire Video URL as input source
    # for e.g hhttps://www.twitch.tv/shroud
    # and enable Stream Mode (`stream_mode = True`)
    stream = CamGear(
        source="https://www.twitch.tv/shroud",
        stream_mode=True,
        logging=True,
        **options
    ).start()

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # Show output window
        cv2.imshow("Output", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()
    ```


&nbsp; 

## Using Camgear with Youtube Videos

CamGear API also provides out-of-the-box support for pipelining live video-frames and metadata from **:fontawesome-brands-youtube: YouTube (Livestream + Normal) Videos**. 

!!! fail "YouTube Playlists :material-youtube-subscription: are not supported yet."

The complete usage example is as follows:

??? bug "Bug in OpenCV's FFmpeg"

    To workaround a [**FFmpeg bug**](https://github.com/abhiTronix/vidgear/issues/133#issuecomment-638263225) that causes video to freeze frequently in OpenCV, It is advised to always use [GStreamer backend](../params/#backend) for Livestream videos.

    **Checkout [this FAQ ➶](../../../help/camgear_faqs/#how-to-compile-opencv-with-gstreamer-support) for compiling OpenCV with GStreamer support.**

    !!! fail "Not all resolutions are supported with GStreamer Backend. See issue #244"
    

??? info "Exclusive CamGear Attributes for `yt_dlp` backend"
    
    CamGear also provides exclusive attributes: 
    
    - `STREAM_RESOLUTION` _(for specifying stream resolution)_
    - `STREAM_PARAMS` _(for specifying  `yt_dlp` parameters)_ 
    
    with its [`options`](../params/#options) dictionary parameter. **More information can be found [here ➶](../advanced/source_params/#exclusive-camgear-parameters)**


??? tip "Accessing Stream's Metadata :material-database-eye:"

    CamGear now provides `ytv_metadata` global parameter for accessing given Video's metadata as JSON Object. It can used as follows:

    ??? new "New in v0.2.4" 
        `ytv_metadata` global parameter was added in `v0.2.4`.

    ```python
    # import required libraries
    from vidgear.gears import CamGear

    # Add YouTube Video URL as input source (for e.g https://youtu.be/uCy5OuSQnyA)
    # and enable Stream Mode (`stream_mode = True`)
    stream = CamGear(
        source="https://youtu.be/uCy5OuSQnyA", stream_mode=True, logging=True, **options
    ).start()

    # get Video's metadata as JSON object
    video_metadata =  stream.ytv_metadata

    # print all available keys
    print(video_metadata.keys())

    # get data like `title`
    print(video_metadata["title"])
    ```

```python hl_lines="8-9"
# import required libraries
from vidgear.gears import CamGear
import cv2

# Add YouTube Video URL as input source (for e.g https://youtu.be/uCy5OuSQnyA)
# and enable Stream Mode (`stream_mode = True`)
stream = CamGear(
    source="https://youtu.be/uCy5OuSQnyA", 
    stream_mode=True,
    logging=True
).start()

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # Show output window
    cv2.imshow("Output", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
```

&nbsp; 


## Using CamGear with Variable Camera Properties

CamGear API also flexibly support various **Source Tweak Parameters** available within [OpenCV's VideoCapture API](https://docs.opencv.org/master/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d). These tweak parameters can be used to transform input source Camera-Device properties _(such as its brightness, saturation, framerate, resolution, gain etc.)_ seamlessly, and can be easily applied in CamGear API through its `options` dictionary parameter by formatting them as its attributes. 

The complete usage example is as follows:

!!! tip "All the supported Source Tweak Parameters can be found [here ➶](../advanced/source_params/#source-tweak-parameters-for-camgear-api)"

```python hl_lines="8-10"
# import required libraries
from vidgear.gears import CamGear
import cv2


# define suitable tweak parameters for your stream.
options = {
    "CAP_PROP_FRAME_WIDTH": 320, # resolution 320x240
    "CAP_PROP_FRAME_HEIGHT": 240,
    "CAP_PROP_FPS": 60, # framerate 60fps
}

# To open live video stream on webcam at first index(i.e. 0) 
# device and apply source tweak parameters
stream = CamGear(source=0, logging=True, **options).start()

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # Show output window
    cv2.imshow("Output", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
```

&nbsp; 

## Using Camgear with Direct Colorspace Manipulation

CamGear API also supports **Direct Colorspace Manipulation**, which is ideal for changing source colorspace on the run. 

!!! info "A more detailed  information on colorspace manipulation can be found [here ➶](../../../bonus/colorspace_manipulation/)"

In following example code, we will start with [**HSV**](https://en.wikipedia.org/wiki/HSL_and_HSV) as source colorspace, and then we will switch to [**GRAY**](https://en.wikipedia.org/wiki/Grayscale)  colorspace when `w` key is pressed, and then [**LAB**](https://en.wikipedia.org/wiki/CIELAB_color_space) colorspace when `e` key is pressed, finally default colorspace _(i.e. **BGR**)_ when `s` key is pressed. Also, quit when `q` key is pressed:


!!! fail "Any incorrect or None-type value, will immediately revert the colorspace to default i.e. `BGR`."


```python hl_lines="7 30 34 38"
# import required libraries
from vidgear.gears import CamGear
import cv2

# Open any source of your choice, like Webcam first index(i.e. 0)
# and change its colorspace to `HSV`
stream = CamGear(source=0, colorspace="COLOR_BGR2HSV", logging=True).start()

# loop over
while True:

    # read HSV frames
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the HSV frame here}

    # Show output window
    cv2.imshow("Output", frame)

    # check for key if pressed
    key = cv2.waitKey(1) & 0xFF

    # check if 'w' key is pressed
    if key == ord("w"):
        # directly change colorspace at any instant
        stream.color_space = cv2.COLOR_BGR2GRAY  # Now colorspace is GRAY

    # check for 'e' key is pressed
    if key == ord("e"):
        stream.color_space = cv2.COLOR_BGR2LAB  # Now colorspace is CieLAB

    # check for 's' key is pressed
    if key == ord("s"):
        stream.color_space = None  # Now colorspace is default(ie BGR)

    # check for 'q' key is pressed
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
```

&nbsp; 

