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

# PiGear API 

<figure>
  <img src="../../../assets/images/picam2.webp" alt="Pi Zero with Camera Module" loading="lazy" class="center" />
  <figcaption>Raspberry Pi Camera Module</figcaption>
</figure>

## Overview

> PiGear is a specialized API similar to the [CamGear API](../../camgear/overview/) but optimized for **Raspberry Pi :fontawesome-brands-raspberry-pi: Boards**, offering comprehensive **support for camera modules** _(e.g., OmniVision OV5647, Sony IMX219)_, along with **limited compatibility for USB cameras**.


PiGear implements a seamless and robust wrapper around the [picamera2](https://github.com/raspberrypi/picamera2) python library, simplifying integration with minimal code changes and ensuring a smooth transition for developers already familiar with the Picamera2 API. PiGear leverages the `libcamera` API under the hood with multi-threading, providing high-performance :fire:, enhanced control and functionality for Raspberry Pi camera modules. 

PiGear handles common configuration parameters and non-standard settings for various camera types, simplifying the integration process. PiGear currently supports PiCamera2 API parameters such as `sensor`, `controls`, `transform`, and `stride`, with internal type and sanity checks for robust performance.

While primarily focused on Raspberry Pi camera modules, PiGear also provides basic functionality for USB webcams only with Picamera2 API, along with the ability to accurately differentiate between USB and Raspberry Pi cameras using metadata. 

???+ info "Backward compatibility with `picamera` library"
	  PiGear seamlessly switches to the legacy [`picamera`](https://picamera.readthedocs.io/en/release-1.13/index.html) library if the `picamera2` library is unavailable, ensuring seamless backward compatibility. For this, PiGear also provides a flexible multi-threaded framework around complete `picamera` API, allowing developers to effortlessly exploit a wide range of parameters, such as `brightness`, `saturation`, `sensor_mode`, `iso`, `exposure`, and more. 

    !!! note "You could also enforce the legacy picamera API backend in PiGear by using the [`enforce_legacy_picamera`](../params/#b-user-defined-parameters) user-defined optional parameter boolean attribute."

Furthermore, PiGear supports the use of multiple camera modules, including those found on Raspberry Pi Compute Module IO boards and USB cameras _(only with Picamera2 API)_.

???+ new "Threaded Internal Timer :material-camera-timer:"
	PiGear ensures proper resource release during the termination of the API, preventing potential issues or resource leaks. PiGear API internally implements a ==Threaded Internal Timer== that silently keeps active track of any frozen-threads or hardware-failures and exits safely if any do occur. This means that if you're running the PiGear API in your script and someone accidentally pulls the Camera-Module cable out, instead of going into a possible kernel panic, the API will exit safely to save resources.

!!! failure "Make sure to [complete Raspberry Pi Camera Hardware-specific settings](https://www.raspberrypi.com/documentation/accessories/camera.html#installing-a-raspberry-pi-camera) prior using this API, otherwise nothing will work."

!!! tip "Helpful Tips"
    * Follow [PiCamera2 documentation](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf) and [Picamera documentation](https://picamera.readthedocs.io/en/release-1.13/) which should help you quickly get started.

    * If you're already familiar with [OpenCV](https://github.com/opencv/opencv) library, then see [Switching from OpenCV ➶](../../../switch_from_cv/#switching-videocapture-apis).
  
    * It is advised to enable logging(`logging = True`) on the first run for easily identifying any runtime errors.


&thinsp; 

## Usage Examples

<div>
<a href="../usage/">See here 🚀</a>
</div>

!!! example "After going through PiGear Usage Examples, Checkout more of its advanced configurations [here ➶](../../../help/pigear_ex/)"

## Parameters

<div>
<a href="../params/">See here 🚀</a>
</div>

## References

<div>
<a href="../../../bonus/reference/pigear/">See here 🚀</a>
</div>


## FAQs

<div>
<a href="../../../help/pigear_faqs/">See here 🚀</a>
</div>  


&thinsp;