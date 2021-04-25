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

# ScreenGear FAQs

&nbsp;

## What is ScreenGear API and what does it do?

**Answer:** ScreenGear is designed exclusively for ultra-fast Screencasting, that means it can grab frames from your monitor in real-time, either by define an area on the computer screen, or full-screen, at the expense of inconsiderable latency. ScreenGear also seamlessly support frame capturing from multiple monitors as well as supports multiple backends. _For more info. see [ScreenGear doc ➶](../../gears/screengear/overview/)_

&nbsp;

## I'm only familiar with OpenCV, how to get started with ScreenGear API?

**Answer:** First, see [Switching from OpenCV](../../switch_from_cv/#switching-videocapture-apis), then go through [ScreenGear doc](../../gears/screengear/overview/). Still in doubt, then ask us on [Gitter ➶](https://gitter.im/vidgear/community) Community channel.

&nbsp;

## ScreenGear is Slow?

**Answer:** This maybe due to selected [`backend`](../../gears/screengear/params/#backend) for ScreenGear API is not compatibile with your machine. See [this usage example to change backend ➶](../../gears/screengear/usage/#using-screengear-with-variable-backend). Try different backends, and select which works the best for your machine.

&nbsp;

## How to define area on screen to record with ScreenGear?

**Answer:** See [this usage example ➶](../../gears/screengear/usage/#using-screengear-with-variable-screen-dimensions)

&nbsp;

## How to record video from all connected screens?

**Answer:** See [ScreenGear's `monitor` parameter](../../gears/screengear/params/#monitor) that sets the index of the monitor to grab a frame from. If its value is `-1`, it will record from all monitors. _More information can be found [here  ➶](https://python-mss.readthedocs.io/examples.html#a-screen-shot-to-grab-them-all)_

&nbsp;