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

# ScreenGear API 

<figure>
  <img src="../../../assets/gifs/screengear.gif" loading="lazy" alt="ScreenGear in action!" class="center-small"/>
  <figcaption>ScreenGear API in action</figcaption>
</figure>

## Overview

> *ScreenGear is designed exclusively for targeting rapid Screencasting Capabilities, which means it can grab frames from your monitor in real-time, either by defining an area on the computer screen or full-screen, at the expense of inconsiderable latency. ScreenGear also seamlessly support frame capturing from multiple monitors as well as supports multiple backends.*

ScreenGear API implements a multi-threaded wrapper around [**dxcam**](https://github.com/ra1nty/DXcam), [**pyscreenshot**](https://github.com/ponty/pyscreenshot) & [**python-mss**](https://github.com/BoboTiG/python-mss) python library, and also flexibly supports its internal parameter. 

&thinsp; 


!!! tip "Helpful Tips"

	* If you're already familar with [OpenCV](https://github.com/opencv/opencv) library, then see [Switching from OpenCV Library ➶](../../../switch_from_cv/#switching-videocapture-apis)

	* It is advised to enable logging(`logging = True`) on the first run for easily identifying any runtime errors.


&thinsp; 

## Importing

You can import ScreenGear API in your program as follows:

```python
from vidgear.gears import ScreenGear
```

&thinsp;

## Usage Examples

<div>
<a href="../usage/">See here 🚀</a>
</div>

!!! experiment "After going through ScreenGear Usage Examples, Checkout more of its advanced configurations [here ➶](../../../help/screengear_ex/)"



## Parameters

<div>
<a href="../params/">See here 🚀</a>
</div>

## References

<div>
<a href="../../../bonus/reference/screengear/">See here 🚀</a>
</div>


## FAQs

<div>
<a href="../../../help/screengear_faqs/">See here 🚀</a>
</div>  

&thinsp; 