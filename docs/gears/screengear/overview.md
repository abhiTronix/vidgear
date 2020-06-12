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

# ScreenGear API 


## Overview

<p align="center">
  <img src="/assets/gifs/screengear.gif" alt="ScreenGear in action!"/>
  <br>
  <sub><i>ScreenGear in action</i></sub>
</p>


ScreenGear is designed exclusively for ultra-fast Screencasting, that means it can grab frames from your monitor in real-time, either by define an area on the computer screen, or full-screen, at the expense of inconsiderable latency. ScreenGear also seamlessly support frame capturing from multiple monitors.

ScreenGear API implements a multi-threaded wrapper around [**python-mss**](https://github.com/BoboTiG/python-mss) python library, and also flexibly supports its internal parameter. 

Furthermore, ScreenGear API relies on [**Threaded Queue mode**](../../../bonus/TQM/) for threaded, error-free and synchronized frame handling.

&nbsp; 


!!! tip "It is advised to enable logging(`logging = True`) on the first run for easily identifying any runtime errors."


&nbsp; 

## Importing

You can import ScreenGear API in your program as follows:

```python
from vidgear.gears import ScreenGear
```

&nbsp; 