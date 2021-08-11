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

# VideoGear FAQs

&nbsp;

## What is VideoGear API and what does it do?

**Answer:** VideoGear provides a special internal wrapper around VidGear's exclusive [**Video Stabilizer**](../../gears/stabilizer/overview/) class. It also act as a Common API, that provided an internal access to both [CamGear](../../gears/camgear/overview/) and [PiGear](../../gears/pigear/overview/) APIs and their parameters, with a special [`enablePiCamera`](../../gears/videogear/params/#enablepicamera) boolean flag. _For more info. see [VideoGear doc ➶](../../gears/videogear/overview/)_

&nbsp;

## What's the need of VideoGear API?

**Answer:** VideoGear is basically ideal when you need to switch to different video sources without changing your code much. Also, it enables easy stabilization for various video-streams _(real-time or not)_  with minimum efforts and using way fewer lines of code. It also serve as backend for other powerful APIs, such [WebGear](../../gears/webgear/overview/) and [NetGear_Async](../../gears/netgear_async/overview/).

&nbsp;

## Which APIs are accessible with VideoGear API?

**Answer:** VideoGear provided an internal access to both [CamGear](../../gears/camgear/overview/) and [PiGear](../../gears/pigear/overview/) APIs and their parameters, also it contains wrapper around [**Video Stabilizer**](../../gears/stabilizer/overview/) class.

&nbsp;

## Can we access WriteGear API or NetGear API too with VideoGear?

**Answer:** No, only selected VideoCapture APIs _(anwsered above)_ are accessible.

&nbsp;

## Does using VideoGear instead of CamGear API directly, affects performance?

**Answer:** No, there's no difference, as VideoGear just a high-level wrapper around CamGear API and without any modifications in-between.

&nbsp;