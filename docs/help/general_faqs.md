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

# General FAQs

&thinsp;

## "I'm new to Python Programming or its usage in OpenCV Library", How to use vidgear in my projects?

**Answer:** Before using vidgear, It's recommended to first go through the following dedicated blog sites and learn how OpenCV-Python syntax works _(with examples)_:

- [**PyImageSearch.com** ➶](https://www.pyimagesearch.com/) is the best resource for learning OpenCV and its Python implementation. Adrian Rosebrock provides many practical OpenCV techniques with tutorials, code examples, blogs, and books at PyImageSearch.com. Highly recommended!

- [**learnopencv.com** ➶](https://www.learnopencv.com)  Maintained by OpenCV CEO Satya Mallick. This blog is for programmers, hackers, engineers, scientists, students, and self-starters interested in Computer Vision and Machine Learning.

- There's also the official [**OpenCV Tutorials** ➶](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html) curated by the OpenCV developers.

Once done, visit [Switching from OpenCV ➶](../../switch_from_cv/) to easily replace OpenCV APIs with suitable [Gears ➶](../../gears/#gears-what-are-these) in your project. All the best! :smiley:

!!! tip "If you run into any trouble or have any questions, then refer our [**Help**](../get_help) section."

&thinsp;

## "VidGear is using Multi-threading, but Python is notorious for its poor performance in multithreading?"

**Answer:** Refer vidgear's [Threaded-Queue-Mode ➶](../../bonus/TQM/)

&thinsp;

## ModuleNotFoundError: No module named 'vidgear.gears'. 'vidgear' is not a package?

**Answer:** This error means you either have a file named `vidgear.py` in your python path or you've named your python script `vidgear.py`. Replace `vidgear` name with anything else to fix this error.

&thinsp;

## How to containerize VidGear with Docker?

**Answer:** Checkout this guide that addresses the unique challenges of containerizing VidGear applications, particularly around building OpenCV with proper GStreamer and FFmpeg support. Learn the patterns and pitfalls before you start: [**Containerizing VidGear with Docker ➶**](../bonus/docker_example.md)

&thinsp;

## How to log to a file in VidGear?

**Answer:** VidGear provides exclusive **`VIDGEAR_LOGFILE`** environment variable to enable logging to a file while logging is enabled _(i.e. `logging=True`)_ on respective Gear. You just have to set ==directory pathname _(automatically creates `vidgear.log` file)_== or a ==log file pathname== itself as value for this  environment variable. This can be done on various Operating Systems as follows:

!!! info "Remember enabling this logging to a file will completely disable any output on the terminal." 

=== ":material-linux: Linux"

	```sh
	# path to file
	export VIDGEAR_LOGFILE="$HOME/foo.log"

	# or just directory path 
	# !!! Make sure `foo` path already exists !!!
	export VIDGEAR_LOGFILE="$HOME/foo"

	# to remove
	unset VIDGEAR_LOGFILE
	```

=== ":fontawesome-brands-windows: Windows (Powershell)"

	```powershell
	# path to file
	$Env:VIDGEAR_LOGFILE = "D:\foo.log"

	# or just directory path 
	# !!! Make sure `foo` path already exists !!!
	$Env:VIDGEAR_LOGFILE = "D:\foo"

	# to remove
	$Env:VIDGEAR_LOGFILE = ""
	```

=== ":material-apple: MacOS"
	
	```sh
	# path to file
	export VIDGEAR_LOGFILE="$HOME/foo.log"
	
	# or just directory path 
	# !!! Make sure `foo` path already exists !!!
	export VIDGEAR_LOGFILE="$HOME/foo"

	# to remove
	unset VIDGEAR_LOGFILE
	```

&thinsp;

## Can I perform Deep Learning task with VidGear?

**Answer:** VidGear is a powerful Video Processing library _(similar to OpenCV, FFmpeg, etc.)_ that can read, write, process, send & receive a sequence of video-frames in an optimized manner. But for Deep Learning or Machine Learning tasks, you have to use a third-party library. That being said, all VidGear's APIs can be used with any third-party Library(such as PyTorch, Tensorflow, etc.) that can leverage the overall performance if you're processing video/audio streams/frames in your application with Deep Learning tasks. Also, it eases the workflow since you have to write way fewer lines of code to read/store/process output videos.

&thinsp;

## Can I ask my question directly without raising an issue?

**Answer:** Yes, please join our [Gitter ➶](https://gitter.im/vidgear/community) Community channel.

&thinsp;

## How to contribute to VidGear development?

**Answer:** See our [Contribution Guidelines ➶](../../contribution/PR/)

&thinsp;

## What OSes are supported by VidGear?

**Answer:** See [Supported Systems ➶](../../installation/#supported-systems)

&thinsp;

## What Python versions are supported by VidGear?

**Answer:** See [Supported Python legacies ➶](../../installation/#supported-python-legacies)

&thinsp;

## Can I include VidGear in my project commercially or not?

**Answer:** Yes, you can, *but strictly under the Terms and Conditions given in [VidGear License ➶](https://github.com/abhiTronix/vidgear/blob/master/LICENSE)*

&thinsp;

## "I Love using VidGear for my projects", How can I support it?

**Answer:** See [Helping VidGear ➶](../../help/#helping-vidgear)  

&thinsp;