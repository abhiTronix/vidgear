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

&nbsp;

## What is NetGear_Async API and what does it do?

**Answer:** NetGear_Async is an asyncio videoframe messaging framework, built on [`zmq.asyncio`](https://pyzmq.readthedocs.io/en/latest/api/zmq.asyncio.html), and powered by high-performance asyncio event loop called [**`uvloop`**](https://github.com/MagicStack/uvloop) to achieve unmatchable high-speed and lag-free video streaming over the network with minimal resource constraints. Basically, this API is able to transfer thousands of frames in just a few seconds without causing any significant load on your system. _For more info. see [NetGear_Async doc ➶](../../gears/netgear_async/overview/)_

&nbsp;

## How to get started with NetGear_Async API?

**Answer:** See [NetGear_Async doc ➶](../../gears/netgear_async/overview/). Still in doubt, then ask us on [Gitter ➶](https://gitter.im/vidgear/community) Community channel.

&nbsp;

## "NetGear_Async is throwing `ModuleNotFoundError` on importing", Why?

**Answer:** This error means, VidGear is installed **WITHOUT asyncio package support** on your machine. For this support, see [Requirements ➶](../../gears/netgear_async/usage/#requirement).

&nbsp;

## What is the key difference between NetGear_Async and NetGear APIs?

**Answer:** 

* **NetGear:** implements a high-level wrapper around [PyZmQ](https://github.com/zeromq/pyzmq) python library that contains python bindings for [ZeroMQ](http://zeromq.org/) - a high-performance asynchronous distributed messaging library that provides a message queue, but unlike message-oriented middleware, its system can run without a dedicated message broker. 

* **NetGear_Async:** is an asyncio videoframe messaging framework, built on [`zmq.asyncio`](https://pyzmq.readthedocs.io/en/latest/api/zmq.asyncio.html), and powered by high-performance asyncio event loop called [**`uvloop`**](https://github.com/MagicStack/uvloop) to high-speed and lag-free video streaming over the network with minimal resource constraints.

**Key Difference:** NetGear_Async is highly memory efficient, but has less features as compared to NetGear API which is marginally faster too. 

&nbsp;

## Can I use Multi-Server, Bi-Directional like modes in NetGear_Async?

**Answer:** No, NetGear_Async does NOT provide support for any NetGear's [Exclusive modes](../../gears/netgear/overview/#exclusive-modes) yet.

&nbsp;

## How to use NetGear_Async with custom Server Source from OpenCV?

**Answer:** See [this usage example ➶](../../gears/netgear_async/usage/#using-netgear_async-with-a-custom-sourceopencv). 

&nbsp;

## Why NetGear_Async is running slow?

**Answer:** Checkout tips suggested in [this answer ➶](../netgear_faqs/#why-netgear-is-slow)

&nbsp;
