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

# NetGear_Async API 

<figure>
  <img src="../../../assets/images/zmq_asyncio.webp" alt="NetGear_Async generalized" loading="lazy" width="70%" />
</figure>

## Overview

> NetGear_Async is an asyncio videoframe messaging framework, built on [`zmq.asyncio`](https://pyzmq.readthedocs.io/en/latest/api/zmq.asyncio.html), and powered by high-performance asyncio event loop called [**`uvloop`**](https://github.com/MagicStack/uvloop) to achieve unmatchable high-speed and lag-free video streaming over the network with minimal resource constraints. Basically, this API is able to transfer thousands of frames in just a few seconds without causing any significant load on your system. 

NetGear_Async can generate double performance as compared to [NetGear API](../../netgear/overview/) at about 1/3rd of memory consumption, and also provide complete server-client handling with various options to use variable protocols/patterns similar to NetGear, but it doesn't support any [NetGear's Exclusive Modes](../../netgear/overview/#exclusive-modes) yet. 

Furthermore, NetGear_Async allows us to  define our own custom Server Source to manipulate frames easily before sending them across the network(see this [usage example](../usage/#using-netgear_async-with-a-custom-sourceopencv)). In addition to all this, NetGear_Async also **provides a special internal wrapper around [VideoGear API](../../videogear/overview/)**, which itself provides internal access to both [CamGear](../../camgear/overview/) and [PiGear](../../pigear/overview/) APIs thereby granting it exclusive power for streaming frames incoming from any connected device/source to the network.

NetGear_Async as of now supports four ZeroMQ messaging patterns:

- [x] [`zmq.PAIR`](https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/pair.html) _(ZMQ Pair Pattern)_
- [x] [`zmq.REQ/zmq.REP`](https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/client_server.html) _(ZMQ Request/Reply Pattern)_
- [x] [`zmq.PUB/zmq.SUB`](https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/pubsub.html) _(ZMQ Publish/Subscribe Pattern)_ 
- [x] [`zmq.PUSH/zmq.PULL`](https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/pushpull.html#push-pull) _(ZMQ Push/Pull Pattern)_

Whereas supported protocol are: `tcp` and `ipc`.

&thinsp; 


!!! tip "Helpful Tips"

	* It is advised to enable logging(`logging = True`) on the first run for easily identifying any runtime errors.

	* It is advised to comprehend [NetGear API](../../netgear/overview/) before using this API.


&thinsp; 

## Importing

You can import NetGear_Async API in your program as follows:

```python
from vidgear.gears import NetGear_Async
```

&thinsp;

## Usage Examples

<div class="zoom">
<a href="../usage/">See here 🚀</a>
</div>


## Parameters

<div class="zoom">
<a href="../params/">See here 🚀</a>
</div>

## Reference

<div class="zoom">
<a href="../../../bonus/reference/netgear_async/">See here 🚀</a>
</div>


## FAQs

<div class="zoom">
<a href="../../../help/netgear_async_faqs/">See here 🚀</a>
</div> 


&thinsp;