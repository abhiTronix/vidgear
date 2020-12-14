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

# Threaded Queue Mode

## Overview

<figure>
  <img src="../../assets/images/tqm.webp" loading="lazy" alt="Threading timing diagram"/>
  <figcaption>Threaded-Queue-Mode: generalized timing diagram</figcaption>
</figure>

Threaded Queue Mode is designed exclusively for VidGear's Videocapture and Network Gears _(namely CamGear, ScreenGear, VideoGear, and NetGear(Client's end))_, for achieving high-performance, synchronized, and error-free frame handling with its Multi-Threaded APIs. 


!!! danger "Threaded-Queue-Mode is enabled by default for any input video stream - including Network Streams, except for any Camera Device."

&nbsp; 

## What does Threaded-Queue-Mode exactly do?


Threaded-Queue-Mode helps VidGear do the Video Processing tasks in a well-organized, and most competent way possible. 

### Multi-Threading

Depending on the video quality, source, and physical hardware of our machine, much of the I/O bound memory is consumed by blocking [OpenCV VideoCapture API's `read()`](https://docs.opencv.org/master/d8/dfe/classcv_1_1VideoCapture.html#a473055e77dd7faa4d26d686226b292c1) method, for reading/decoding next frame from source. In simple words, the main thread of our script is completely blocked, until the frame is read from source and thereby returned.

VidGear's APIs in layman's words employs [Multi-Threading](https://docs.python.org/3/library/threading.html) to separate various tasks(_such as frame-decoding_) to multiple independent threads. These multi-threads helps VidGear execute different Video Processing I/O-bound operations at the same time, by overlapping the waiting times. In this way, VidGear keeps on processing frames faster in background([_daemon_](https://en.wikipedia.org/wiki/Daemon_(computing))) without having to wait for blocking I/O operations, and doesn't get effected by how sluggish our main python thread is.

### Monitored Fix-Sized Deques

Multi-threading is easy, but it may have some undesired effects like _frame-skipping, deadlocks, and race conditions, etc._ that frequently result in random, intermittent bugs that can be quite difficult to find. Therefore to prevent these problem, in VidGear, we introduced this Thread-Queue-Mode, that utilizes monitored, thread-safe, memory efficient, and fixed-sized [`deques`](https://docs.python.org/3.8/library/collections.html#collections.deque) _(with approximately the same O(1) performance in either direction)_, that always maintains a fixed-length of frame buffer in the memory, and blocks the thread if the queue is full, or otherwise pops out the frames synchronously and efficiently without any obstructions. In this way, monitored Deques stops multiple threads from accessing the same source simultaneously, and thus preventing Global Interpreter Lock (aka GIL) too.


&nbsp; 

## Features

- [x] _Enables Blocking, Sequential and Threaded LIFO Frame Handling._

- [x] _Sequentially adds and releases frames to/from `deque` and handles the overflow of this queue._

- [x] _Utilizes thread-safe, memory efficient `deques` that appends and pops frames with same O(1) performance from either side._

- [x] _Requires less RAM at due to buffered frames in the `deque`._


&nbsp;


## Manually disabling Threaded Queue Mode

To manually disable Threaded Queue Mode, VidGear provides following attribute for `options` dictionary parameter in respective API:  

!!! warning "Important Warning"

	* This **`THREADED_QUEUE_MODE`** attribute does **NOT** work with Live feed, such as Camera Devices/Modules.

	* This **`THREADED_QUEUE_MODE`** attribute is **NOT** supported by ScreenGear & NetGear APIs, as Threaded Queue Mode is essential for their core operations.

	* Disabling Threaded Queue Mode may result in **UNDESIRED BEHAVIORS AND BUGS** such as Non-Blocking frame handling, Frame-skipping, etc. *More insight can be found [here ➶](https://github.com/abhiTronix/vidgear/issues/20#issue-452339596).*


**`THREADED_QUEUE_MODE`** _(boolean)_: This attribute can be used to override Threaded-Queue-Mode mode, to manually disable it:

```python
options = {'THREADED_QUEUE_MODE': False} #to disable Threaded Queue Mode. 
```

and you can pass it to `options` dictionary parameter of the respective API.

&nbsp; 