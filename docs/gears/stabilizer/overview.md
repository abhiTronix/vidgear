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

<div id="player_stab" align="middle"></div>
<p align="middle">VidGear's Stabilizer in Action<br><i>(Original Video Courtesy <a href="http://liushuaicheng.org/SIGGRAPH2013/database.html" title="opensourced video samples database">@SIGGRAPH2013</a>)</i></p>

!!! info "This video is transcoded with [**StreamGear API**](../../streamgear/overview/) and hosted on [GitHub Repository](https://github.com/abhiTronix/vidgear-docs-additionals) and served with [raw.githack.com](https://raw.githack.com)"



## Overview

> Stabilizer is an auxiliary class that enables Video Stabilization for vidgear with minimalistic latency, and at the expense of little to no additional computational requirements. 

The basic idea behind it is to tracks and save the salient feature array for the given number of frames and then uses these anchor point to cancel out all perturbations relative to it for the incoming frames in the queue. This class relies heavily on [**Threaded Queue mode**](../../../bonus/TQM/) for error-free & ultra-fast frame handling.

&thinsp; 

## Features

- [x] _Real-time stabilization with low latency and no extra resources._

- [x] _Works exceptionally well with low-frequency jitter._

- [x] _Integrated with [VideoGear](../usage/#using-videogear-with-stabilizer-backend), therefore, can be applied to any incoming stream._

- [x] _Also seamlessly works standalone._


&thinsp;


!!! danger "Important" 

	- The stabilizer may not perform well against High-frequency jitter in video. Use at your own risk!

	- :warning: The stabilizer might be slower for High-Quality videos-frames.

	- It is advised to enable logging on the first run for easily identifying any runtime errors.

&thinsp; 

## Importing

You can import Stabilizer Class in your program as follows:

```python
from vidgear.gears.stabilizer import Stabilizer
```

&thinsp;

## Usage Examples

<div>
<a href="../usage/">See here 🚀</a>
</div>


## Parameters

<div>
<a href="../params/">See here 🚀</a>
</div>

## References

<div>
<a href="../../../bonus/reference/stabilizer/">See here 🚀</a>
</div>


## FAQs

<div>
<a href="../../../help/stabilizer_faqs/">See here 🚀</a>
</div>  

&thinsp; 