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

# Source Tweak Parameters for CamGear API 

<figure>
  <img src="../../../../assets/images/stream_tweak.png" loading="lazy" alt="Source Tweak Parameters" />
</figure>

## Overview

With CamGear's [`options`](../../params/#options) dictionary parameter, the user has the ability to alter various tweak parameters available within [OpenCV's VideoCapture Class](https://docs.opencv.org/master/d8/dfe/classcv_1_1VideoCapture.html#a57c0e81e83e60f36c83027dc2a188e80) by formatting them as its attributes. 

These tweak parameters can be used to transform input Camera-Source properties _(such as its brightness, saturation, resolution, iso, gain etc.)_ seamlessly. All parameters supported by CamGear API are disscussed in this document.

&emsp; 


### Exclusive CamGear Attributes

??? abstract "CamGear's Exclusive Attributes"

	In addition to Source Tweak Parameters, CamGear also provides some exclusive attributes for its [`options`](../../params/#options) dictionary parameters. 

	These attributes are as follows:

	- [X] `STREAM_RESOLUTION` _(string)_: This attribute can be used in CamGear's Stream Mode (`stream_mode=True`) for specifying supported stream resolution. Its possible values can be: `144p`, `240p`, `360p`, `480p`, `720p`, `1080p`, `1440p`, `2160p`, `4320p`, `worst`, `best`, and its default value is `best`. Its usage is as follows:

		!!! warning "In case specificed `STREAM_RESOLUTION` value is unavailable within Source Stream, it defaults to `best`!"

		```python
		options = {"STREAM_RESOLUTION": "720p"} # 720p stream will be used. 
		```

		!!! example "Its complete usage example is given [here ➶](../../../../help/camgear_faqs/#how-to-change-quality-and-parameters-of-youtube-streams-with-camgear)"

	- [X] `STREAM_PARAMS` _(dict)_: This dictionary attribute can be used in CamGear's Stream Mode (`stream_mode=True`) for specifying parameters for its internal `yt_dlp` backend class. Its usage is as follows:

		!!! info "All `STREAM_PARAMS` Supported Parameters"
			- All yt_dlp parameter can be found [here ➶](https://github.com/yt-dlp/yt-dlp/blob/bd1c7923274962e3027acf63111ccb0d766b9725/yt_dlp/__init__.py#L594-L749)

		```python
		options = {"STREAM_PARAMS": {"nocheckcertificate": True}} # disables verifying SSL certificates in yt_dlp
		```

	- [X] `THREADED_QUEUE_MODE` _(boolean)_: This attribute can be used to override Threaded-Queue-Mode mode to manually disable it:

	   	!!! danger "Disabling Threaded-Queue-Mode can be dangerous! Read more [here ➶](../../../../bonus/TQM/#manually-disabling-threaded-queue-mode)"

		```python
		options = {"THREADED_QUEUE_MODE": False} # disable Threaded Queue Mode. 
		```

	- [X] `THREAD_TIMEOUT` _(int/float)_: This attribute can be used to override the timeout value(positive number), that blocks the video-thread for at most ==timeout seconds== if no video-frame was available within that time, and otherwise raises the [Empty exception](https://docs.python.org/3/library/queue.html#queue.Empty) to prevent any never-ending deadlocks. Its default value is `None`, meaning no timeout at all.  Its usage is as follows:

		??? new "New in v0.2.1" 
			`THREAD_TIMEOUT` attribute added in `v0.2.1`.

		```python
		options = {"THREAD_TIMEOUT": 300} # set Video-Thread Timeout for 5mins. 
		```


&nbsp; 


### Supported Source Tweak Parameters

**All Source Tweak Parameters supported by CamGear API are as follows:**

!!! bug "Remember, Not all parameters are supported by all cameras devices, which is one of the most troublesome thing with OpenCV library. Each camera type, from android cameras, to USB cameras , to professional ones, offers a different interface to modify its parameters. Therefore, there are many branches in OpenCV code to support as many of them, but of course, not all possible devices are covered, and thereby works. Furthermore, OpenCV does not return any helpful error messages regarding this problem, so it’s pretty much based on _trial and error_."

!!! tip "You can easily check parameter values supported by your webcam, by hooking it to a Linux machine, and using the command `#!sh v4l2-ctl -d 0 --list-formats-ext` _(where 0 is an index of the given camera)_ to list the supported video parameters and their values. If that doesn't works, refer to its datasheet _(if available)_."

!!! info "These parameters can be passed to CamGear's [`options`](../../params/#options) dictionary parameter by formatting them as its string attributes. Its complete usage example is [here ➶](../../usage/#using-camgear-with-variable-camera-properties)"

&thinsp;

|Values|Description|
|:--------------------------------------:	|:--------------------------------------------------------------------------------------------------------------------	|
| CAP_PROP_POS_MSEC 	| Current position of the video file in milliseconds. 	|
| CAP_PROP_POS_FRAMES 	| 0-based index of the frame to be decoded/captured next. 	|
| CAP_PROP_POS_AVI_RATIO 	| Relative position of the video file: 0=start of the film, 1=end of the film. 	|
| CAP_PROP_FRAME_WIDTH 	| Width of the frames in the video stream. 	|
| CAP_PROP_FRAME_HEIGHT 	| Height of the frames in the video stream. 	|
| CAP_PROP_FPS 	| Frame rate. 	|
| CAP_PROP_FOURCC 	| 4-character code of codec. see [VideoWriter::fourcc](https://docs.opencv.org/master/dd/d9e/classcv_1_1VideoWriter.html#afec93f94dc6c0b3e28f4dd153bc5a7f0). 	|
| CAP_PROP_FRAME_COUNT 	| Number of frames in the video file. 	|
| CAP_PROP_FORMAT 	| Format of the Mat objects returned by [VideoCapture::retrieve()](https://docs.opencv.org/master/d8/dfe/classcv_1_1VideoCapture.html#a9ac7f4b1cdfe624663478568486e6712). 	|
| CAP_PROP_MODE 	| Backend-specific value indicating the current capture mode. 	|
| CAP_PROP_BRIGHTNESS 	| Brightness of the image (only for those cameras that support). 	|
| CAP_PROP_CONTRAST 	| Contrast of the image (only for cameras). 	|
| CAP_PROP_SATURATION 	| Saturation of the image (only for cameras). 	|
| CAP_PROP_HUE 	| Hue of the image (only for cameras). 	|
| CAP_PROP_GAIN 	| Gain of the image (only for those cameras that support). 	|
| CAP_PROP_EXPOSURE 	| Exposure (only for those cameras that support). 	|
| CAP_PROP_CONVERT_RGB 	| Boolean flags indicating whether images should be converted to RGB. 	|
| CAP_PROP_WHITE_BALANCE_BLUE_U 	| Currently unsupported. 	|
| CAP_PROP_RECTIFICATION 	| Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently). 	|
| CAP_PROP_MONOCHROME 	|  	|
| CAP_PROP_SHARPNESS 	|  	|
| CAP_PROP_AUTO_EXPOSURE 	| DC1394: exposure control done by camera, user can adjust reference level using this feature. 	|
| CAP_PROP_GAMMA 	|  	|
| CAP_PROP_TEMPERATURE 	|  	|
| CAP_PROP_TRIGGER 	|  	|
| CAP_PROP_TRIGGER_DELAY 	|  	|
| CAP_PROP_WHITE_BALANCE_RED_V 	|  	|
| CAP_PROP_ZOOM 	|  	|
| CAP_PROP_FOCUS 	|  	|
| CAP_PROP_GUID 	|  	|
| CAP_PROP_ISO_SPEED 	|  	|
| CAP_PROP_BACKLIGHT 	|  	|
| CAP_PROP_PAN 	|  	|
| CAP_PROP_TILT 	|  	|
| CAP_PROP_ROLL 	|  	|
| CAP_PROP_IRIS 	|  	|
| CAP_PROP_SETTINGS 	| Pop up video/camera filter dialog (note: only supported by DSHOW backend currently. The property value is ignored) 	|
| CAP_PROP_BUFFERSIZE 	|  	|
| CAP_PROP_AUTOFOCUS 	|  	|
| CAP_PROP_SAR_NUM 	| Sample aspect ratio: num/den (num) 	|
| CAP_PROP_SAR_DEN 	| Sample aspect ratio: num/den (den) 	|
| CAP_PROP_BACKEND 	| Current backend (enum VideoCapture APIs). Read-only property. 	|
| CAP_PROP_CHANNEL 	| Video input or Channel Number (only for those cameras that support) 	|
| CAP_PROP_AUTO_WB 	| enable/ disable auto white-balance 	|
| CAP_PROP_WB_TEMPERATURE 	| white-balance color temperature 	|

&nbsp; 
