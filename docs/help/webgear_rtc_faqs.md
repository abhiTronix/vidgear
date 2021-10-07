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

# WebGear_RTC FAQs

&nbsp;

## What is WebGear_RTC API and what does it do?

**Answer:** WebGear_RTC utilizes [WebRTC](https://webrtc.org/) technology under the hood, which makes it suitable for building powerful video-streaming solutions for all modern browsers as well as native clients available on all major platforms. _For more info. see [WebGear_RTC doc ➶](../../gears/webgear_rtc/overview/)_

&nbsp;

## How to get started with WebGear_RTC API?

**Answer:** See [WebGear_RTC doc ➶](../../gears/webgear_rtc/overview/). Still in doubt, then ask us on [Gitter ➶](https://gitter.im/vidgear/community) Community channel.

&nbsp;

## How WebGear_RTC is different to WebGear API, which should I choose?

**Answer:** WebGear_RTC is similar to [WeGear API](../../gears/webgear/overview/) in many aspects but utilizes [WebRTC](https://webrtc.org/) technology under the hood instead of Motion JPEG. You can choose any API according to your application, but the quality would be better on WebGear API, on-the-other-hand latency would be better on WebGear_RTC API. Also, WebRTC protocol accepts a wide range of devices, whereas WebGear is limited only to modern browsers. 

&nbsp;

## "WebGear_RTC is throwing `ModuleNotFoundError` on importing", Why?

**Answer:** This error means, VidGear is installed **WITHOUT asyncio package support** on your machine. For this support, see [Requirements ➶](../../gears/webgear_rtc/usage/#requirements).

&nbsp;

## Can WebGear_RTC always need Active Internet Connection?

**Answer:** No, it just need internet only once during its [Auto-Generation Process ➶](../../gears/webgear_rtc/overview/#auto-generation-process) to download default data-files and it takes few seconds. You can also download files manually from [**Github Server**](https://github.com/abhiTronix/vidgear-vitals), otherwise you can also add [your own custom files](../../gears/webgear_rtc/advanced/#rules-for-altering-webgear_rtc-files-and-folders). _For more information see [Data-Files Auto-Generation WorkFlow ➶](../../gears/webgear_rtc/overview/#data-files-auto-generation-workflow-for-webgear_rtc)_

&nbsp;

## Is it possible to stream on a different device on the network with WebGear_RTC?

!!! note "If you set `"0.0.0.0"` as host value instead of `"localhost"` on Host Machine, then you must still use http://localhost:8000/ to access stream on your host machine browser."

For accessing WebGear_RTC on different Client Devices on the network, use `"0.0.0.0"` as host value instead of `"localhost"` on Host Machine. Then type the IP-address of source machine followed by the defined `port` value in your desired Client Device's browser (for e.g. http://192.27.0.101:8000) to access the stream.

&nbsp;

## Can I manually place default files for WebGear_RTC?

**Answer:** Yes, you can either download default files from [Github Server](https://github.com/abhiTronix/webgear_data), and manually place at [default location](../../gears/webgear_rtc/overview/#default-location), OR, you can yourself create the require three critical files _(i.e `index.html`, `404.html` & `500.html`)_  inside `templates` folder at the [default location](../../gears/webgear_rtc/overview/#default-location), thereby you don't need any internet connection at all. _For more information see [Data-Files Auto-Generation WorkFlow ➶](../../gears/webgear_rtc/overview/#data-files-auto-generation-workflow)_

&nbsp;

## How to stream Webgear_RTC Server output to multiple clients?

**Answer:** See [this usage example ➶](../../gears/webgear_rtc/advanced/#using-webgear_rtc-as-real-time-broadcaster).

&nbsp;

## How to send OpenCV frames directly to Webgear_RTC Server?

**Answer:** See [this usage example ➶](../../gears/webgear_rtc/advanced/#using-webgear_rtc-with-a-custom-sourceopencv).

&nbsp;

## How can I add my custom WebPage to WebGear_RTC?

**Answer:** See [this usage example ➶](../../gears/webgear_rtc/advanced/#using-webgear_rtc-with-custom-webpage-routes).

&nbsp;

## How can to add CORS headers to WebGear_RTC?

**Answer:** See [this usage example ➶](../../gears/webgear_rtc/advanced/#using-webgear_rtc-with-middlewares).

&nbsp;


## Can I change the default location?

**Answer:** Yes, you can use WebGear_RTC's [`custom_data_location`](../../gears/webgear_rtc/params/#webgear_rtc-specific-attributes) attribute of `option` parameter in WebGear_RTC API, to change [default location](../../gears/webgear_rtc/overview/#default-location) to somewhere else.

&nbsp;

## Can I delete/rename the WebGear_RTC default data?

**Answer:** Yes, but you've to follow [these rules ➶](../../gears/webgear_rtc/advanced/#rules-for-altering-webgear_rtc-files-and-folders)

&nbsp;
