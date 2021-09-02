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

# Colorspace Manipulation for VideoCapture Gears

<figure>
  <img src="../../assets/images/colorspace.png" alt="Colorspace" loading="lazy" class="center" />
</figure>


## Source ColorSpace manipulation

> All VidGear's Videocapture Gears _(namely CamGear, ScreenGear, VideoGear)_ and some Streaming Gears _(namely WebGear, WebGear_RTC)_ and Network Gears _(Client's end)_ - provides exclusive internal support for ==Source [Color Space](https://en.wikipedia.org/wiki/Color_space) manipulation==. 

**There are two ways to alter source colorspace:**

### Using `colorspace` parameter
  
Primarily, the safest way is by **`colorspace`** _(string)_ parameter of the respective VideoCapture API, that can be used to easily alter the colorspace of the input source, during initialization. But on the downside, `colorspace` parameter value **CANNOT** be changed/altered at runtime. 

All possible values for this parameter are discussed [below ➶](#supported-colorspace-parameter-values)

### Using `color_space` global variable

Alternatively, a more direct approach is by using **`color_space`** _(integer)_ global variable the respective VideoCapture API, can be used for directly changing the source colorspace at runtime. It can be used in conjunction with `colorspace` parameter easily. 

&nbsp; 

!!! info "Supported Colorspace Conversions"

	Any conversion from default Source colorspace _(i.e. **BGR** in case of OpenCV)_, to any other colorspace and vice-versa _(use `None` to revert)_, is supported.


!!! warning "Important Information"

  	* Using `color_space` global variable is **NOT Supported** in VideoGear API, calling it will result in `AttribueError`.

	* Any incorrect or None-type value, will immediately revert the colorspace to default _(i.e. `BGR`)_.

	* Using `color_space` global variable with [Threaded Queue Mode](../../bonus/TQM/) may have minor lag, User discretion is advised.

!!! tip
	It is advised to enable logging(`logging = True`) on the first run for easily identifying any runtime errors.


&nbsp; 


## Supported `colorspace` parameter values

All supported string values for `colorspace` parameter are as follows:

!!! info "You can check all OpenCV Colorspace Conversion Codes [here ➶](https://docs.opencv.org/master/d8/d01/group__imgproc__color__conversions.html#ga4e0972be5de079fed4e3a10e24ef5ef0)."

| Supported Conversion Values | Description |
|:----:|:----| 
| COLOR_BGR2BGRA     | BGR to BGRA |
| COLOR_BGR2RGBA     | BGR to RGBA |
| COLOR_BGR2RGB      | BGR to RGB backward conversions to RGB/BGR |
| COLOR_BGR2GRAY     | BGR to GRAY |
| COLOR_BGR2BGR565   | BGR to BGR565 |
| COLOR_BGR2BGR555   | BGR to BGR555 |
| COLOR_BGR2XYZ      | BGR to CIE XYZ |
| COLOR_BGR2YCrCb    | BGR to luma-chroma (aka YCC) |
| COLOR_BGR2HSV      | BGR to HSV (hue saturation value) |
| COLOR_BGR2Lab      | BGR to CIE Lab |
| COLOR_BGR2Luv      | BGR to CIE Luv |
| COLOR_BGR2HLS      | BGR to HLS (hue lightness saturation) |
| COLOR_BGR2HSV_FULL | BGR to HSV_FULL |
| COLOR_BGR2HLS_FULL | BGR to HLS_FULL |
| COLOR_BGR2YUV      | BGR to YUV |
| COLOR_BGR2YUV_I420 | BGR to YUV 4:2:0 family |
| COLOR_BGR2YUV_IYUV | BGR to IYUV |
| COLOR_BGR2YUV_YV12 | BGR to YUV_YV12 |
| None | Back to default colorspace (i.e. BGR) |


&nbsp; 


## Usage examples

### Using CamGear with Direct Colorspace Manipulation
[The complete usage example can be found here ➶](../../gears/camgear/usage/#using-camgear-with-direct-colorspace-manipulation)

&nbsp; 

### Using PiGear with Direct Colorspace Manipulation
[The complete usage example can be found here ➶](../../gears/pigear/usage/#using-pigear-with-direct-colorspace-manipulation)

&nbsp; 

### Using VideoGear with Colorspace Manipulation
[The complete usage example can be found here ➶](../../gears/videogear/usage/#using-videogear-with-colorspace-manipulation)

&nbsp; 

### Using ScreenGear with Direct Colorspace Manipulation
[The complete usage example can be found here ➶](../../gears/screengear/usage/#using-screengear-with-direct-colorspace-manipulation)

&nbsp; 