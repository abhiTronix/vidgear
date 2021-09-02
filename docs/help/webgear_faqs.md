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

# WebGear FAQs

&nbsp;

## What is WebGear API and what does it do?

**Answer:** WebGear is a powerful [ASGI](https://asgi.readthedocs.io/en/latest/) Video-Broadcaster API ideal for transmitting [Motion-JPEG](https://en.wikipedia.org/wiki/Motion_JPEG)-frames from a single source to multiple recipients via the browser. _For more info. see [WebGear doc ➶](../../gears/webgear/overview/)_

&nbsp;

## How to get started with WebGear API?

**Answer:** See [WebGear doc ➶](../../gears/webgear/overview/). Still in doubt, then ask us on [Gitter ➶](https://gitter.im/vidgear/community) Community channel.

&nbsp;

## "WebGear is throwing `ModuleNotFoundError` on importing", Why?

**Answer:** This error means, VidGear is installed **WITHOUT asyncio package support** on your machine. For this support, see [Requirements ➶](../../gears/webgear/usage/#requirements).

&nbsp;

## Can WebGear always need Active Internet Connection?

**Answer:** No, it just need internet only once during its [Auto-Generation Process ➶](../../gears/webgear/overview/#auto-generation-process) to download default data-files and it takes few seconds. You can also download files manually from [**Github Server**](https://github.com/abhiTronix/vidgear-vitals), otherwise you can also add [your own custom files](../../gears/webgear/advanced/#rules-for-altering-webgear-files-and-folders). _For more information see [Data-Files Auto-Generation WorkFlow ➶](../../gears/webgear/overview/#data-files-auto-generation-workflow-for-webgear)_

&nbsp;

## Is it possible to stream on a different device on the network with WebGear?

!!! alert "If you set `"0.0.0.0"` as host value instead of `"localhost"` on Host Machine, then you must still use http://localhost:8000/ to access stream on that same host machine browser."

For accessing WebGear on different Client Devices on the network, use `"0.0.0.0"` as host value instead of `"localhost"` on Host Machine. Then type the IP-address of source machine followed by the defined `port` value in your desired Client Device's browser (for e.g. http://192.27.0.101:8000) to access the stream.

&nbsp;

## Can I manually place default files for WebGear?

**Answer:** Yes, you can either download default files from [Github Server](https://github.com/abhiTronix/webgear_data), and manually place at [default location](../../gears/webgear/overview/#default-location), OR, you can yourself create the require three critical files _(i.e `index.html`, `404.html` & `500.html`)_  inside `templates` folder at the [default location](../../gears/webgear/overview/#default-location), thereby you don't need any internet connection at all. _For more information see [Data-Files Auto-Generation WorkFlow ➶](../../gears/webgear/overview/#data-files-auto-generation-workflow)_

&nbsp;

## How to send OpenCV frames directly to Webgear Server?

**Answer:** See [this usage example ➶](../../gears/webgear/advanced/#using-webgear-with-a-custom-sourceopencv).

&nbsp;

## How can I add my custom WebPage to WebGear?

**Answer:** See [this usage example ➶](../../gears/webgear/advanced/#using-webgear-with-custom-webpage-routes).

&nbsp;

## How can to add CORS headers to WebGear?

**Answer:** See [this usage example ➶](../../gears/webgear/advanced/#using-webgear-with-middlewares).

&nbsp;

## Can I change the default location?

**Answer:** Yes, you can use WebGear's [`custom_data_location`](../../gears/webgear/params/#webgear-specific-attributes) attribute of `option` parameter in WebGear API, to change [default location](../../gears/webgear/overview/#default-location) to somewhere else.

&nbsp;

## Can I delete/rename the WebGear default data?

**Answer:** Yes, but you've to follow [these rules ➶](../../gears/webgear/advanced/#rules-for-altering-webgear-files-and-folders)

&nbsp;

## What Web browser are supported by WebGear API?

**Answer:** All modern browser with Javascript support are supported by WebGear. If not, then discuss with us on [Gitter ➶](https://gitter.im/vidgear/community) Community channel.

&nbsp;
