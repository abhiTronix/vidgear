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

# Threaded Queue Mode

## Overview

<figure>
  <img src="../../assets/images/tqm.webp" loading="lazy" alt="Threading timing diagram" class="center"/>
  <figcaption>Threaded-Queue-Mode: generalized timing diagram</figcaption>
</figure>

> Threaded Queue Mode is designed exclusively for VidGear's Videocapture Gears _(namely CamGear, VideoGear)_ and few Network Gears _(such as NetGear(Client's end))_ for achieving high-performance, asynchronous, error-free video-frames handling. 

!!! tip "Threaded-Queue-Mode is enabled by default, but [can be disabled](#manually-disabling-threaded-queue-mode), only if extremely necessary."

!!! info "Threaded-Queue-Mode is **NOT** required and thereby automatically disabled for Live feed such as Camera Devices/Modules, since ."

&nbsp; 

## What does Threaded-Queue-Mode exactly do?

Threaded-Queue-Mode helps VidGear do the Threaded Video-Processing tasks in highly optimized, well-organized, and most competent way possible: 

### A. Enables Multi-Threading

> In case you don't already know, OpenCV's' [`read()`](https://docs.opencv.org/master/d8/dfe/classcv_1_1VideoCapture.html#a473055e77dd7faa4d26d686226b292c1) is a [**Blocking I/O**](https://luminousmen.com/post/asynchronous-programming-blocking-and-non-blocking) function for reading and decoding the next video-frame, and consumes much of the I/O bound memory depending upon our video source properties & system hardware. This essentially means, the corresponding thread that reads data from it, is continuously blocked from retrieving the next frame. As a result, our python program appears slow and sluggish even without any type of computationally expensive image processing operations. This problem is far more severe on low memory SBCs like Raspberry Pis.

In Threaded-Queue-Mode, VidGear creates several [**Python Threads**](https://docs.python.org/3/library/threading.html) within one process to offload the frame-decoding task to a different thread. Thereby,  VidGear is able to execute different Video I/O-bounded operations at the same time by overlapping there waiting times. Moreover,  threads are managed by operating system itself and is capable of distributing them between available CPU cores efficiently. In this way, Threaded-Queue-Mode keeps on processing frames faster in the [background](https://en.wikipedia.org/wiki/Daemon_(computing)) without affecting by sluggishness in our main python program thread.

### B. Utilizes Fixed-Size Queues

> Although Multi-threading is fast, easy, and efficient, it can lead to some serious undesired effects like frame-skipping, [**Global Interpreter Lock**](https://realpython.com/python-gil/), race conditions, etc. This is because there is no isolation whatsoever in python threads, and in case there is any crash it will cause the whole process to crash. That's not all, the memory of the process is shared by different threads and that may result in random process crashes due to unwanted race conditions.

These problems are avoided in Threaded-Queue-Mode by utilizing **Thread-Safe, Memory-Efficient, and Fixed-Size [`Queues`](https://docs.python.org/3/library/queue.html#module-queue)** _(with approximately same O(1) performance in both directions)_, that isolates the frame-decoding thread from other parallel threads and provide synchronized access to incoming frames without any obstruction. 

### C. Accelerates Frame Processing

With queues, VidGear always maintains a fixed-length frames buffer in the memory and blocks the thread temporarily if the queue is full to avoid possible frame drops or otherwise pops out the frames synchronously without any obstructions. This significantly accelerates frame processing rate (and therefore our overall video processing pipeline) comes from dramatically reducing latency — since we don’t have to wait for the `read()` method to finish reading and decoding a frame; instead, there is always a pre-decoded frame ready for us to process.


&nbsp; 

## What are the advantages of Threaded-Queue-Mode?

- [x] _Enables Blocking, Sequential and Threaded LIFO Frame Handling._

- [x] _Sequentially adds and releases frames from `queues` and handles the overflow._

- [x] _Utilizes thread-safe, memory efficient `queues` that appends and pops frames with same O(1) performance from either side._

- [x] _Faster frame access due to buffered frames in the `queue`._

- [x] _Provides isolation for source thread and prevents GIL._


&nbsp;


## Manually disabling Threaded-Queue-Mode

To manually disable Threaded-Queue-Mode, VidGear provides `THREADED_QUEUE_MODE` boolean attribute for `options` dictionary parameter in respective [VideoCapture APIs](../../gears/#a-videocapture-gears):  

!!! warning "Important Warnings"

	* Disabling Threaded-Queue-Mode does **NOT disables Multi-Threading.**

	* `THREADED_QUEUE_MODE` attribute does **NOT** work with Live feed, such as Camera Devices/Modules.

	* `THREADED_QUEUE_MODE` attribute is **NOT** supported by ScreenGear & NetGear APIs, as Threaded Queue Mode is essential for their core operations.


!!! danger "Disabling Threaded-Queue-Mode may lead to Random Intermittent Bugs that can be quite difficult to discover. More insight can be found [here ➶](https://github.com/abhiTronix/vidgear/issues/20#issue-452339596)"


**`THREADED_QUEUE_MODE`** _(boolean)_: This attribute can be used to override Threaded-Queue-Mode mode to manually disable it:

```python
options = {'THREADED_QUEUE_MODE': False} # to disable Threaded Queue Mode. 
```

and you can pass it to `options` dictionary parameter of the respective API.

&nbsp; 