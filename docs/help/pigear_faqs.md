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

# PiGear FAQs

&nbsp;

## What is PiGear API and what does it do?

**Answer:** PiGear is similar to CamGear but exclusively made to support various Raspberry Pi Camera Modules (such as [OmniVision OV5647 Camera Module](https://github.com/techyian/MMALSharp/wiki/OmniVision-OV5647-Camera-Module) and [Sony IMX219 Camera Module](https://github.com/techyian/MMALSharp/wiki/Sony-IMX219-Camera-Module)). _For more info. see [PiGear doc ➶](../../gears/pigear/overview/)_

&nbsp;

## I'm only familiar with OpenCV, how to get started with PiGear API?

**Answer:** First, see [Switching from OpenCV](../../switch_from_cv/#switching-videocapture-apis), then go through [PiGear doc](../../gears/pigear/overview/). Still in doubt, then ask us on [Gitter ➶](https://gitter.im/vidgear/community) Community channel.

&nbsp;

## Why my camera module is not detected by PiGear?

**Answer:** Make sure to [enable Raspberry Pi hardware-specific settings ➶](https://picamera.readthedocs.io/en/release-1.13/quickstart.html) before using PiGear. Also, recheck/change your Camera Module's ribbon-cable and Camera Module itself, if it damaged or got broken somehow.

&nbsp;

## How to select camera index on Pi Compute IO board with two Cameras attached?

**Answer:** See [PiGear's `camera_num` parameter ➶](../../gears/pigear/params/#camera_num)

&nbsp;

## Why PiGear is throwing `SystemError`?

**Answer:** This means your Raspberry Pi CSI ribbon-cable is not connected properly to your Camera Module, or damaged, or even both. 

&nbsp;

## How to assign `picamera` settings for Camera Module with PiGear?

**Answer:** See [this usage example ➶](../../gears/pigear/usage/#using-pigear-with-variable-camera-properties)

&nbsp;

## "Video output is too dark with PiGear", Why?

**Answer:** Seems like the settings are wrong. Kindly see [picamera docs](https://picamera.readthedocs.io/en/release-1.13/api_camera.html) for available parameters, and look for parameters are `sensor_mode`, `shutter_speed` and `exposure_mode`, try changing those values. Also, maybe your `framerate` value is too high. Try lowering it.

&nbsp;


## How to change `picamera` settings for Camera Module at runtime?

**Answer:** You can use `stream` global parameter in PiGear to feed any `picamera` setting at runtime. See [this bonus example ➶](../pigear_ex/#setting-variable-picamera-parameters-for-camera-module-at-runtime)

&nbsp;