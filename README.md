<table align="center"><tr><td align="center" width="100%">
	
<img alt="vidgear Logo" src="https://raw.githubusercontent.com/abhiTronix/Imbakup/master/Images/vidgear.png" width="50%">
	
&nbsp; 

[![PyPi version](https://img.shields.io/pypi/v/vidgear.svg?style=popout-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAABC0lEQVRYhdWVPQoCMRCFX6HY2ghaiZUXsLW0EDyBrbWtN/EUHsHTWFnYyCL4gxibVZZlZzKTnWz0QZpk5r0vIdkF/kBPAMOKeddE+CQPKoc5Yt5cTjBMdQSwDQToWgBJAn3jmhqgltapAV6E6b5U17MGGAUaUj07TficMfIBZDV6vxowBm1BP9WbSQE4o5h9IjPJmy73TEPDDxVmoZdQrQ5jRhly9Q8tgMUXkIIWn0oG4GYQfAXQzz1PGoCiQndM7b4RgJay/h7zBLT3hASgoKjamQJMreKf0gfuAGyYtXEIAKcL/Dss15iq6ohXghozLYiAMxPuACwtIT4yeQUxAaLrZwAoqGRKGk7qDSYTfYQ8LuYnAAAAAElFTkSuQmCC)](https://pypi.org/project/vidgear/)
[![Last Commit](https://img.shields.io/github/last-commit/abhiTronix/vidgear.svg?style=popout-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAABC0lEQVRYhdWVPQoCMRCFX6HY2ghaiZUXsLW0EDyBrbWtN/EUHsHTWFnYyCL4gxibVZZlZzKTnWz0QZpk5r0vIdkF/kBPAMOKeddE+CQPKoc5Yt5cTjBMdQSwDQToWgBJAn3jmhqgltapAV6E6b5U17MGGAUaUj07TficMfIBZDV6vxowBm1BP9WbSQE4o5h9IjPJmy73TEPDDxVmoZdQrQ5jRhly9Q8tgMUXkIIWn0oG4GYQfAXQzz1PGoCiQndM7b4RgJay/h7zBLT3hASgoKjamQJMreKf0gfuAGyYtXEIAKcL/Dss15iq6ohXghozLYiAMxPuACwtIT4yeQUxAaLrZwAoqGRKGk7qDSYTfYQ8LuYnAAAAAElFTkSuQmCC)](https://github.com/abhiTronix/vidgear/commits/master)
[![License](https://img.shields.io/github/license/abhiTronix/vidgear.svg?style=popout-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAABC0lEQVRYhdWVPQoCMRCFX6HY2ghaiZUXsLW0EDyBrbWtN/EUHsHTWFnYyCL4gxibVZZlZzKTnWz0QZpk5r0vIdkF/kBPAMOKeddE+CQPKoc5Yt5cTjBMdQSwDQToWgBJAn3jmhqgltapAV6E6b5U17MGGAUaUj07TficMfIBZDV6vxowBm1BP9WbSQE4o5h9IjPJmy73TEPDDxVmoZdQrQ5jRhly9Q8tgMUXkIIWn0oG4GYQfAXQzz1PGoCiQndM7b4RgJay/h7zBLT3hASgoKjamQJMreKf0gfuAGyYtXEIAKcL/Dss15iq6ohXghozLYiAMxPuACwtIT4yeQUxAaLrZwAoqGRKGk7qDSYTfYQ8LuYnAAAAAElFTkSuQmCC)](https://github.com/abhiTronix/vidgear/blob/master/LICENSE)

<a align="center"> **VidGear is a Python library for Turbo Multi-Threaded Video Processing**</a>
</td></tr></table>

* VidGear is a **high-level All-in-One Video Processing framework build on top of OpenCV, PiCamera and FFmpeg**. It's simple to use and easy to extend. 

* It contains following **Powerful Gears (Multi-Threaded Video Processing classes)** to handle/control different device-specific video streams and writer:
	
    |Gear|Function|
    |:------:|---------|
    |[**CamGear**](https://github.com/abhiTronix/vidgear/wiki/CamGear-Class)|*Targets any IP-Camera/USB-Camera/Network-Stream/YouTube-Video*|
    |[**PiGear**](https://github.com/abhiTronix/vidgear/wiki/PiGear-Class)|*Targets any Raspberry Pi Camera Modules*|
    |[**VideoGear**](https://github.com/abhiTronix/vidgear/wiki/VideoGear-Class)|*Common Gear to access any of the video capture gear*|
    |[**WriteGear**](https://github.com/abhiTronix/vidgear/wiki/WriteGear-Class)|*enables Lossless Video Writer with flexible Video Compression capabilities*|

* It comes with various in-built features like **Flexible Control over Video Source properties** `like resolution, framerate, colorspace, etc.` manipulation and supports easy direct Network stream pipelining (*support for GStreamer, YouTube and other network streams like `http(s), rtp, rstp, mms, etc.`*).

* This library is also very well compatible with the Raspberry-Pi Camera modules (RPiCam) and provides us the ability to exploit its various features like `brightness, saturation, sensor_mode,` etc. easily.

* Furthermore, VidGear utilizes **FFmpeg**'s powerful encoders to encode and reduce the output to a smaller size, without sacrificing the video quality. It provide us full-control over FFmpeg output parameters.

&nbsp; 

## Key Features 

#### What makes **VidGear** stand out from all other Video Processing python tools?

- [x]  *Multi-Threaded high-speed Frame capturing **(High FPS)***
- [x]  *Flexible & Direct control over Video Source properties*
- [x]  *Lossless Video Encoding and Writing* <img src="https://raw.githubusercontent.com/abhiTronix/Imbakup/master/Images/new.gif"/>
- [x]  *Flexible Output Video Encoder, Compression & Quality Control* <img src="https://raw.githubusercontent.com/abhiTronix/Imbakup/master/Images/new.gif"/>
- [x]  *Direct YouTube Video pipelining using its URL address* <img src="https://raw.githubusercontent.com/abhiTronix/Imbakup/master/Images/new.gif"/>
- [x]  *Easy Video Source ColorSpace Conversion* <img src="https://raw.githubusercontent.com/abhiTronix/Imbakup/master/Images/new.gif"/>
- [x]  *Automated prerequisites installation* <img src="https://raw.githubusercontent.com/abhiTronix/Imbakup/master/Images/new.gif" />
- [x]  *Built-in robust Error and Frame Synchronization handling*
- [x]  *Multi-Devices Compatibility (including RpiCamera)*
- [x]  *Support for Live Network Video Streams (including Gstreamer Raw Pipeline)* 

&nbsp; 

## Documentation and Usage

<h3><img src="http://www.animatedimages.org/data/media/81/animated-hand-image-0021.gif" width="25" height="20"/> You can checkout VidGear <a href = https://github.com/abhiTronix/vidgear/wiki>WIKI-SECTION</a> for in-depth documentation with examples for each VidGear Class<img src="https://raw.githubusercontent.com/abhiTronix/Imbakup/master/Images/new.gif" /></h3>


#### Basic example: 

The bare minimum basic example of `WriteGear`(*Compression Mode*) with `VideoGear` class for a common *Live WebCamera stream* is as follows :

```python
from vidgear.gears import VideoGear
from vidgear.gears import WriteGear
import cv2

stream = VideoGear(source=0).start() #Open live webcam video stream on first index(i.e. 0) device

writer = WriteGear(output_filename = 'Output.mp4') #Define writer with output filename 'Output.mp4'

# infinite loop
while True:
	
	frame = stream.read()
	# read frames

	# check if frame is None
	if frame is None:
		#if True break the infinite loop
		break
	

	# do something with frame here

	# write frame to writer
        writer.write(frame) 
       
        # Show output window
	cv2.imshow("Output Frame", frame)

	key = cv2.waitKey(1) & 0xFF
	# check for 'q' key-press
	if key == ord("q"):
		#if 'q' key-pressed break out
		break

cv2.destroyAllWindows()
# close output window

stream.stop()
# safely close video stream
writer.close()
# safely close writer
```
&nbsp; 

## Prerequisites

<h3><img src="http://www.animatedimages.org/data/media/81/animated-hand-image-0021.gif" width="25" height="20"/> Note: Vidgear automatically handles all required prerequisites(except FFmpeg) installation according to your system requirements<img src="https://raw.githubusercontent.com/abhiTronix/Imbakup/master/Images/new.gif"/></h3>

### Critical: 

* **OpenCV(with contrib):** VidGear must require *`OpenCV with contrib`* (3.0+) python library to be installed on your machine which is critical for its core algorithm functioning. You can build it from from [scratch](https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/) ([Raspberry Pi](https://www.pyimagesearch.com/2018/09/26/install-opencv-4-on-your-raspberry-pi/)) or Vidgear automatically installs *`OpenCV with contrib`* python library for you based on your system from [PyPi](https://pypi.org/project/opencv-contrib-python/).

* **FFmpeg:** VidGear requires FFmpeg installation for Compression capabilities. ***Follow this [WIKI Page](https://github.com/abhiTronix/vidgear/wiki/FFmpeg-Installation) for latest FFmpeg installation.*** :warning:

### Additional:

* **PiCamera:** If you are using Raspberry Pi Camera Modules such as *OmniVision OV5647 Camera Module* and *Sony IMX219 Camera Module*. It requires additional [Picamera](https://picamera.readthedocs.io/en/release-1.13/install.html) library installation on your Raspberry Pi machine prior to its installation (*Latest versions recommended*).

<img src="http://www.animatedimages.org/data/media/81/animated-hand-image-0021.gif" width="25" height="20"/> ***Also, make sure to [enable Raspberry Pi hardware specific settings](https://picamera.readthedocs.io/en/release-1.13/quickstart.html) prior using this library.***

* **pafy**: For direct YouTube Video Pipelining into OpenCV, Vidgear requires [Pafy](https://pypi.org/project/pafy/) python library.


&nbsp; 

## Installation
- **From PyPI(Stable Only):** `VidGear` can be easily installed as follows(*available on [Python Package Index (PyPI)](https://pypi.org/project/vidgear/)*):

 
  ```sh
  $ pip install vidgear
  ```

&nbsp; 


## Contribution and Development

You are welcome to contribute with [suggestions, feature requests and pull requests](https://github.com/abhiTronix/vidgear/pulls).

**Clone this repository(development):** You can clone this repository for latest releases (*maybe experimental*) for development purposes and thereby can install as follows:


  ```sh
  $ git clone https://github.com/abhiTronix/vidgear.git
  $ cd vidgear
  $ pip install .
  ```
   
&nbsp; 

## Supported Python versions

* Python 2.7 is the only supported version in 2.x series. ***Python 2.7 support will be dropped in the end of 2019.***

* Python 3.x releases follow `OpenCV` releases.

## Author

- **Abhishek Thakur** [@abhiTronix](https://github.com/abhiTronix)

## License

Copyright © 2019 AbhiTronix

This project is under the MIT License. See the LICENSE file for the full license text.
