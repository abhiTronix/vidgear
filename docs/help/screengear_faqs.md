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

# ScreenGear FAQs

&thinsp;

## What is ScreenGear API and what does it do?

**Answer:** ScreenGear is designed exclusively for targeting rapid Screencasting Capabilities, which means it can grab frames from your monitor in real-time, either by defining an area on the computer screen or full-screen, at the expense of inconsiderable latency. ScreenGear also seamlessly support frame capturing from multiple monitors as well as supports multiple backends. _For more info. see [ScreenGear doc ➶](../../gears/screengear/)_

&nbsp;

## I'm only familiar with OpenCV, how to get started with ScreenGear API?

**Answer:** First, refer to the [Switching from OpenCV](../../switch_from_cv/#switching-the-videocapture-apis) guide, then go through [ScreenGear documentation](../../gears/screengear/). If you still have doubts, ask us on [Gitter ➶](https://gitter.im/vidgear/community) Community channel.

&nbsp;

## ScreenGear is Slow?

**Answer:** This maybe due to selected [`backend`](../../gears/screengear/params/#backend) for ScreenGear API is not compatible with your machine. See [this usage example to change backend ➶](../../gears/screengear/usage/#using-screengear-with-variable-backend). Try different backends, and select which works the best for your machine.

&nbsp;

## How to define area on screen to record with ScreenGear?

**Answer:** See [this usage example ➶](../../gears/screengear/usage/#using-screengear-with-variable-screen-dimensions)

&nbsp;

## How to record video from all connected screens?

**Answer:** With `mss` backend, see ScreenGear's [`monitor`](../../gears/screengear/params/#monitor) parameter that sets the index of the monitor to grab a frame from. If its value is `-1`, it will record from all monitors. _More information can be found [here  ➶](https://python-mss.readthedocs.io/examples.html#a-screen-shot-to-grab-them-all)_

&nbsp;

## I'm getting "AttributeError: 'DXCamera' object has no attribute 'is_capturing'" Error?

**Answer:** This is a [well-known error](https://github.com/ra1nty/DXcam/issues/38) in backend `dxcam` library which occurs when you've multiple GPUs on your Windows machine. To workaround this, you need select Internal GPU in settings as follows:

=== "On :fontawesome-brands-windows: Windows 11"

    In **Settings**, go to `System > Display > Graphics` and add your `Python.exe` as _"Desktop App"_, then select _"Power saving"_ as follows:
    <figure>
        <img src="../../assets/images/screengear_error11.png" alt="AttributeError: 'DXCamera'" loading="lazy" class="center-small" width="50%" />
    </figure>

    And finally press **Save** button.  

=== "On :fontawesome-brands-windows: Windows 10"

    In **Settings**, go to `Graphics Settings` and add your `Python.exe` as _"Desktop App"_, then select _"Power saving"_ as follows:
    
    <figure>
        <img src="https://user-images.githubusercontent.com/93147937/199585781-a1ec316d-c6dd-48e8-bf35-1dec1b725071.png" alt="AttributeError: 'DXCamera'" loading="lazy" class="center" />
    </figure>

    And finally press **Save** button.  

&nbsp;