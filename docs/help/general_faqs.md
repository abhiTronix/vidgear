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

# General FAQs


&nbsp;


## "I'm new to Python Programming or its usage in Computer Vision", How to use vidgear in my projects?

**Answer:** It's recommended to firstly go through following dedicated tutorials/websites thoroughly, and learn how OpenCV-Python actually works(_with examples_):

- [**PyImageSearch.com** ➶](https://www.pyimagesearch.com/) is the best resource for learning OpenCV and its Python implementation. Adrian Rosebrock provides many practical OpenCV techniques with tutorials, code examples, blogs and books at PyImageSearch.com.  I also learned a lot about computer vision methods, and various useful techniques, when I first started with Computer Vision. Highly recommended!

- [**learnopencv.com** ➶](https://www.learnopencv.com)  Maintained by OpenCV CEO Satya Mallick. This blog is for programmers, hackers, engineers, scientists, students and self-starters who are interested in Computer Vision and Machine Learning.

- There's also the official [**OpenCV Tutorials** ➶](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html), provided by the OpenCV folks themselves.

Finally, when you're done, go through [Gears ➶](../../gears/#gears-what-are-these) to see how VidGear works. If you run into any trouble, or have any questions, then join our [Gitter ➶](https://gitter.im/vidgear/community) Community channel, or else [report an Issue ➶](../../contribution/issue/). 

&nbsp;

## How to install VidGear python library on my machine?

**Answer:** See [Installation Notes ➶](../../installation/).

&nbsp;

## "VidGear is using Multi-threading, but Python is notorious for its poor performance in multithreading?"

**Answer:** 

Most of the VidGear task are I/O bounded, meaning that the thread spends most of its time handling I/O processes such as performing network requests, pooling frames out of Camera devices etc. It is perfectly fine to use multi-threading for these tasks, as the thread is most of the time being blocked and put into blocked queue by the OS automatically. For further reading, see [Threaded-Queue-Mode ➶](../../bonus/TQM/)

&nbsp;

## Why VidGear APIs not working for me?

**Answer:** VidGear docs contains a lot of detailed information, please take your time to read it. Please see [Getting Help ➶](../../help/get_help/) for troubleshooting your problems.

&nbsp;

## How do I report an issue?

**Answer:** See [Reporting an Issue ➶](../../contribution/issue/)

&nbsp;

## Can I ask my question directly without raising an issue?

**Answer:** Yes, please join our [Gitter ➶](https://gitter.im/vidgear/community) Community channel.

&nbsp;

## How to contribute to VidGear development?

**Answer:** See our [Contribution Guidelines ➶](../../contribution/PR/)

&nbsp;

## What OSes are supported by VidGear?

**Answer:** See [Supported Systems ➶](../../installation/#supported-systems)

&nbsp;

## What Python versions are supported by VidGear?

**Answer:** See [Supported Python legacies ➶](../../installation/#supported-python-legacies)

&nbsp;

## Can I include VidGear in my project commercially or not?

**Answer:** Yes, you can, *but strictly under the Terms and Conditions given in [VidGear License ➶](https://github.com/abhiTronix/vidgear/blob/master/LICENSE)*

&nbsp;

## "I Love using VidGear", How can I support it?

**Answer:** Thank you! See [Helping VidGear ➶](../../help/#helping-vidgear)  

&nbsp;