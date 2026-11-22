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

# NetGear_Async FAQs

&thinsp;

## What is NetGear_Async API and what does it do?

**Answer:** NetGear_Async is an asyncio videoframe messaging framework, built on [`zmq.asyncio`](https://pyzmq.readthedocs.io/en/latest/api/zmq.asyncio.html), for lag-free video streaming over the network with minimal resource constraints. When it creates its own event loop, it uses the high-performance [**`uvloop`**](https://github.com/MagicStack/uvloop) event loop _(if installed, not on Windows)_. _For more info. see [NetGear_Async doc ➶](../gears/netgear_async/)_

&thinsp;

## How to get started with NetGear_Async API?

**Answer:** First, refer to the [Switching from OpenCV](../switch_from_cv/#switching-the-videocapture-apis) guide, then go through [NetGear_Async documentation ➶](../gears/netgear_async/). If you still have doubts, ask us on [Gitter ➶](https://gitter.im/vidgear/community) Community channel.

&thinsp;

## "NetGear_Async is throwing `ImportError` on initialization", Why?

**Answer:** This error means, VidGear is installed **WITHOUT asyncio package support** on your machine _(i.e. `pyzmq` or `msgspec` is missing)_. For this support, see [Requirements ➶](../gears/netgear_async/usage/#requirement).

&thinsp;

## What is the key difference between NetGear_Async and NetGear APIs?

**Answer:** 

* **NetGear:** implements a high-level wrapper around [PyZmQ](https://github.com/zeromq/pyzmq) python library that contains python bindings for [ZeroMQ](http://zeromq.org/) - a high-performance asynchronous distributed messaging library that provides a message queue, but unlike message-oriented middleware, its system can run without a dedicated message broker. 

* **NetGear_Async:** is an asyncio videoframe messaging framework, built on [`zmq.asyncio`](https://pyzmq.readthedocs.io/en/latest/api/zmq.asyncio.html), for lag-free video streaming over the network with minimal resource constraints. It uses the high-performance [**`uvloop`**](https://github.com/MagicStack/uvloop) event loop when it creates its own loop _(if installed, not on Windows)_.

**Key Difference:** NetGear_Async integrates natively with asyncio applications, but has less features as compared to NetGear API _(for e.g. no Multi-Server/Multi-Client, SSH Tunneling, or Frame Compression modes, no Secure Mode `allowed_ips` whitelist, and no Lazy Pirate reconnection)_. No benchmark comparing their speed or memory usage is available.

&thinsp;

## Can I use Multi-Server, Bi-Directional like modes in NetGear_Async?

**Answer:** NetGear_Async supports only [Bidirectional Mode ➶](../gears/netgear_async/advanced/bidirectional_mode/) and [Secure Mode ➶](../gears/netgear_async/params/#options) _(CURVE encryption, without `allowed_ips` whitelist)_. Other NetGear's [Exclusive modes](../gears/netgear/#exclusive-modes) _(such as Multi-Server, Multi-Client, and SSH Tunneling modes)_ are NOT supported in NetGear_Async yet, so use [NetGear API ➶](../gears/netgear/) for them.

&thinsp;

## How to use NetGear_Async with custom Server Source from OpenCV?

**Answer:** See [this usage example ➶](../gears/netgear_async/usage/#using-netgear_async-with-a-custom-sourceopencv). 

&thinsp;

## Why NetGear_Async is running slow?

**Answer:** Checkout tips suggested in [this answer ➶](netgear_faqs/#why-netgear-is-running-slow)

&thinsp;
