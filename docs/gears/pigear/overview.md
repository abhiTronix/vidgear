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

> PiGear is similar to [CamGear API](../../camgear/overview/) but exclusively made to support various Raspberry Pi Camera Modules _(such as OmniVision OV5647 Camera Module and Sony IMX219 Camera Module)_.

PiGear provides a flexible multi-threaded framework around complete [picamera](https://picamera.readthedocs.io/en/release-1.13/index.html) python library, and provide us the ability to exploit almost all of its parameters like `brightness, saturation, sensor_mode, iso, exposure, etc.` effortlessly. Furthermore, PiGear also supports multiple camera modules, such as in the case of Raspberry-Pi Compute Module IO boards.

Best of all, PiGear contains ==Threaded Internal Timer== - that silently keeps active track of any frozen-threads/hardware-failures and exit safely, if any does occur. That means that if you're running PiGear API in your script and someone accidentally pulls the Camera-Module cable out, instead of going into possible kernel panic, API will exit safely to save resources.

!!! error "Make sure to [enable Raspberry Pi hardware-specific settings](https://picamera.readthedocs.io/en/release-1.13/quickstart.html) prior using this API, otherwise nothing will work."

!!! tip "Helpful Tips"

	* If you're already familar with [OpenCV](https://github.com/opencv/opencv) library, then see [Switching from OpenCV ➶](../../../switch_from_cv/#switching-videocapture-apis)

	* It is advised to enable logging(`logging = True`) on the first run for easily identifying any runtime errors.


&thinsp; 

## Importing

You can import PiGear API in your program as follows:

```python
from vidgear.gears import PiGear
```

&thinsp;

## Usage Examples

<div>
<a href="../usage/">See here 🚀</a>
</div>

!!! experiment "After going through PiGear Usage Examples, Checkout more of its advanced configurations [here ➶](../../../help/pigear_ex/)"



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