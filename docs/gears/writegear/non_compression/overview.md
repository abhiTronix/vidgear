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

# WriteGear API: Non-Compression Mode


## Overview


When [`compression_mode`](/gears/writegear/non_compression/params/#compression_mode) parameter is disabled _(.i.e `compression_mode = False`)_, WriteGear API uses basic OpenCV's inbuilt [**VideoWriter API**](https://docs.opencv.org/master/dd/d9e/classcv_1_1VideoWriter.html#ad59c61d8881ba2b2da22cff5487465b5) tools for encoding multimedia files but without compression, Thereby, also known as Non-Compression Mode.

This mode provides flexible access to [**OpenCV's VideoWriter API**](https://docs.opencv.org/master/dd/d9e/classcv_1_1VideoWriter.html#ad59c61d8881ba2b2da22cff5487465b5),and also supports various parameters available within this API, but lacks the ability to control output quality, compression, and other important features like _lossless video compression, audio encoding, etc._, which are available in [Compression Mode](/gears/writegear/compression/overview/) only. Thereby, the resultant output videofile size of this Mode, will be many times larger as compared to Compression Mode.

&nbsp; 


!!! danger "Important Information"
		
	* In case WriteGear API fails to detect valid FFmpeg executables on your system, it will automatically switches to this(Non-Compression) Mode.

	* It is advised to enable logging(`logging = True`) on the first run for easily identifying any runtime errors.