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

# Stabilizer Class


## Overview

<p align="center">
  <img src="../../../assets/gifs/stabilized.gif" alt="VideoGear Stabilizer in action!" width="120%" />
  <br>
  <sub><i>Original Video Courtesy <a href="http://liushuaicheng.org/SIGGRAPH2013/database.html" title="opensourced video samples database">@SIGGRAPH2013</a></i></sub>
</p>

This is an auxiliary class that enables Video Stabilization for vidgear with minimalistic latency, and at the expense of little to no additional computational requirements. 

The basic idea behind it is to tracks and save the salient feature array for the given number of frames and then uses these anchor point to cancel out all perturbations relative to it for the incoming frames in the queue. This class relies heavily on [**Threaded Queue mode**](../../../bonus/TQM/) for error-free & ultra-fast frame handling.

&nbsp; 


## Features

- [x] _Real-time stabilization with low latency and no extra resources._

- [x] _Works exceptionally well with low-frequency jitter._

- [x] _Integrated with `VideoGear` API, therefore, can be applied to any incoming stream._

- [x] _Also seamlessly works standalone._


&nbsp;


!!! danger "The stabilizer may not perform well against High-frequency jitter in video. Use at your own risk!"

!!! warning "The stabilizer might be slower for High-Quality videos-frames."

!!! tip "It is advised to enable logging on the first run for easily identifying any runtime errors."




&nbsp; 

## Importing

You can import Stabilizer Class in your program as follows:

```python
from vidgear.gears.stabilizer import Stabilizer
```

&nbsp; 