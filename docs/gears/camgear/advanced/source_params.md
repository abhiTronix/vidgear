<!--
===============================================
vidgear library source-code is deployed under the Apache 2.0 License:

Copyright (c) 2019-2020 Abhishek Thakur(@abhiTronix) <abhi.una12@gmail.com>

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

## Overview

The CamGear API's [`option`](../../params/#options) dictionary parameter, provides the ability to alter various **Source Tweak Parameters** available within [OpenCV's VideoCapture Class](https://docs.opencv.org/master/d8/dfe/classcv_1_1VideoCapture.html#a57c0e81e83e60f36c83027dc2a188e80). These tweak parameters can be used to manipulate input source Camera-Device properties _(such as its brightness, saturation, size, iso, gain etc.)_ seemlessly. Thereby, All Source Tweak Parameters supported by CamGear API are disscussed in this document.


!!! bug "Remember, Not all parameters are supported by all cameras devices, which is one of the most troublesome thing with OpenCV library. Each camera type, from android cameras, to USB cameras , to professional ones, offers a different interface to modify its parameters. Therefore, there are many branches in OpenCV code to support as many of them, but of course, not all possible devices are covered, and thereby works. Furthermore, OpenCV does not return any helpful error messages regarding this problem, so it’s pretty much based on _trial and error_."


!!! tip "You can easily check parameter values supported by your webcam, by hooking it to a Linux machine, and using the command `#!sh v4l2-ctl -d 0 --list-formats-ext` _(where 0 is an index of the given camera)_ to list the supported video parameters and their values. If that doesn't works, refer to its datasheet _(if available)_."


&nbsp; 


## Supported Source Tweak Parameters

**All Source Tweak Parameters supported by CamGear API are as follows:**

!!! info "These parameters can be passed to CamGear's [`option`](../../params/#options) dictionary parameter by formatting them as its string attributes."

!!! tip "Its complete usage example is [here ➶](../../usage/#using-camgear-with-variable-camera-properties)"

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
| CAP_PROP_BACKEND 	| Current backend (enum VideoCaptureAPIs). Read-only property. 	|
| CAP_PROP_CHANNEL 	| Video input or Channel Number (only for those cameras that support) 	|
| CAP_PROP_AUTO_WB 	| enable/ disable auto white-balance 	|
| CAP_PROP_WB_TEMPERATURE 	| white-balance color temperature 	|

&nbsp; 
