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

&thinsp;

## What is PiGear API and what does it do?

**Answer:** PiGear is a specialized API similar to the [CamGear API](../../gears/camgear/overview/) but optimized for **Raspberry Pi Boards**, offering comprehensive **support for camera modules** _(e.g., OmniVision OV5647, Sony IMX219)_, along with **limited compatibility for USB cameras**. _For more info. see [PiGear doc ➶](../../gears/pigear/overview/)_

&nbsp;

## I'm only familiar with OpenCV, how to get started with PiGear API?

**Answer:** First, refer to the [Switching from OpenCV](../../switch_from_cv/#switching-videocapture-apis) guide, then go through [PiGear documentation](../../gears/pigear/overview/). If you still have doubts, ask us on [Gitter ➶](https://gitter.im/vidgear/community) Community channel.

&nbsp;

## Why my camera module is not detected by PiGear?

**Answer:** Make sure to [complete Raspberry Pi Camera Hardware-specific settings](https://www.raspberrypi.com/documentation/accessories/camera.html#installing-a-raspberry-pi-camera) prior using PiGear API. Also, recheck/change your Camera Module's ribbon-cable and Camera Module itself, if it damaged or got broken somehow.

&nbsp;

## How to select camera index on Pi Compute IO board with two Cameras attached?

**Answer:** Refer [this bonus example ➶](../../help/pigear_ex/#accessing-multiple-camera-through-its-index-in-pigear-api)

&nbsp;

## Why PiGear is throwing `SystemError`?

**Answer:** This means your Raspberry Pi CSI ribbon-cable is not connected properly to your Camera Module, or damaged, or even both. 

&nbsp;

## How to assign various configurational settings for Camera Module with PiGear?

**Answer:** See [this usage example ➶](../../gears/pigear/usage/#using-pigear-with-variable-camera-properties)

&nbsp;

## "Video output is too dark with PiGear", Why?

**Answer:** The camera configuration settings might be incorrect. Check [this usage example ➶](../../gears/pigear/usage/#using-pigear-with-variable-camera-properties) and try tinkering parameters like `sensor_mode`, `shutter_speed`, and `exposure_mode`. Additionally, if your `framerate` parameter value is too high, try lowering it.

&nbsp;


## How to dynamically adjust Raspberry Pi Camera Parameters at runtime with PiGear?

**Answer:** See [this bonus example ➶](../../help/pigear_ex/#dynamically-adjusting-raspberry-pi-camera-parameters-at-runtime-in-pigear-api)

&nbsp;


## Is it possible to change output frames Pixel Format in PiGear API?

**Answer:** Yes it is possible with Picamera2 Backend. See [this bonus example ➶](../../help/pigear_ex/#changing-output-pixel-format-in-pigear-api-with-picamera2-backend)

&nbsp;

