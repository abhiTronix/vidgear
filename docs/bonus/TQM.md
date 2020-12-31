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
  <img src="../../assets/images/tqm.webp" loading="lazy" alt="Threading timing diagram" class="center"/>
  <figcaption>Threaded-Queue-Mode: generalized timing diagram</figcaption>
</figure>

> Threaded Queue Mode is designed exclusively for VidGear's Videocapture Gears _(namely CamGear, ScreenGear, VideoGear)_ and few Network Gears _(such as NetGear(Client's end))_ for achieving high-performance, synchronized, and error-free video-frames handling with their **Internal Multi-Threaded Frame Extractor Daemons**. 

!!! info "Threaded-Queue-Mode is enabled by default, but a user [can disable it](#manually-disabling-threaded-queue-mode), if extremely necessary."

&nbsp; 

## What does Threaded-Queue-Mode exactly do?


Threaded-Queue-Mode helps VidGear do the Threaded Video-Processing tasks in a well-organized, and most competent way possible: 

### A. Enables Multi-Threading

In case you don't already know, OpenCV's' [`read()`](https://docs.opencv.org/master/d8/dfe/classcv_1_1VideoCapture.html#a473055e77dd7faa4d26d686226b292c1) is a [blocking method](https://luminousmen.com/post/asynchronous-programming-blocking-and-non-blocking) for reading/decoding the next video-frame and consumes much of the I/O bound memory depending upon our Source-properties & System-hardware. This means, it blocks the function from returning until the next frame. As a result, this behavior halts our python script's main thread completely for that moment.

Threaded-Queue-Mode employs [**Multi-Threading**](https://docs.python.org/3/library/threading.html) to separate frame-decoding like tasks to multiple independent threads in layman's word. Multiple-Threads helps it execute different Video Processing I/O-bound operations all at the same time by overlapping the waiting times. In this way, Threaded-Queue-Mode keeps on processing frames faster in the [background(daemon)](https://en.wikipedia.org/wiki/Daemon_(computing)) without waiting for blocked I/O operations and doesn't get affected by how sluggish our main python thread is.

### B. Monitors Fix-Sized Deques

> Although Multi-threading is fast & easy, it may lead to undesired effects like _frame-skipping, deadlocks, and race conditions, etc._

Threaded-Queue-Mode utilizes **Monitored, Thread-Safe, Memory-Efficient, and Fixed-Sized [`Deques`](https://docs.python.org/3.8/library/collections.html#collections.deque)** _(with approximately the same O(1) performance in either direction)_, that always maintains a fixed-length of frames buffer in the memory. It blocks the thread if the queue is full or otherwise pops out the frames synchronously and efficiently without any obstructions. Its fixed-length Deques stops multiple threads from accessing the same source simultaneously and thus preventing Global Interpreter Lock _(a.k.a GIL)_.


&nbsp; 

## What are the advantages of Threaded-Queue-Mode?

- [x] _Enables Blocking, Sequential and Threaded LIFO Frame Handling._

- [x] _Sequentially adds and releases frames to/from `deque` and handles the overflow of this queue._

- [x] _Utilizes thread-safe, memory efficient `deques` that appends and pops frames with same O(1) performance from either side._

- [x] _Requires less RAM at due to buffered frames in the `deque`._


&nbsp;


## Manually disabling Threaded-Queue-Mode

To manually disable Threaded-Queue-Mode, VidGear provides `THREADED_QUEUE_MODE` boolean attribute for `options` dictionary parameter in respective [VideoCapture APIs](../../gears/#a-videocapture-gears):  

!!! warning "Important Warning"

	* This **`THREADED_QUEUE_MODE`** attribute does **NOT** work with Live feed, such as Camera Devices/Modules.

	* This **`THREADED_QUEUE_MODE`** attribute is **NOT** supported by ScreenGear & NetGear APIs, as Threaded Queue Mode is essential for their core operations.

	* Disabling Threaded-Queue-Mode will **NOT** disable Multi-Threading.

	* Disabling Threaded-Queue-Mode may lead to **Random Intermittent Bugs** that can be quite difficult to discover. *More insight can be found [here ➶](https://github.com/abhiTronix/vidgear/issues/20#issue-452339596)*


**`THREADED_QUEUE_MODE`** _(boolean)_: This attribute can be used to override Threaded-Queue-Mode mode to manually disable it:

```python
options = {'THREADED_QUEUE_MODE': False} # to disable Threaded Queue Mode. 
```

and you can pass it to `options` dictionary parameter of the respective API.

&nbsp; 