<table align="center"><tr><td align="center" width="100%">
	
<img alt="vidgear Logo" src="https://raw.githubusercontent.com/abhiTronix/Imbakup/master/Images/vidgear.png" width="50%">
	
&nbsp; 

[![PyPi version](https://img.shields.io/pypi/v/vidgear.svg?style=popout-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAABC0lEQVRYhdWVPQoCMRCFX6HY2ghaiZUXsLW0EDyBrbWtN/EUHsHTWFnYyCL4gxibVZZlZzKTnWz0QZpk5r0vIdkF/kBPAMOKeddE+CQPKoc5Yt5cTjBMdQSwDQToWgBJAn3jmhqgltapAV6E6b5U17MGGAUaUj07TficMfIBZDV6vxowBm1BP9WbSQE4o5h9IjPJmy73TEPDDxVmoZdQrQ5jRhly9Q8tgMUXkIIWn0oG4GYQfAXQzz1PGoCiQndM7b4RgJay/h7zBLT3hASgoKjamQJMreKf0gfuAGyYtXEIAKcL/Dss15iq6ohXghozLYiAMxPuACwtIT4yeQUxAaLrZwAoqGRKGk7qDSYTfYQ8LuYnAAAAAElFTkSuQmCC)](https://pypi.org/project/vidgear/)
[![Last Commit](https://img.shields.io/github/last-commit/abhiTronix/vidgear.svg?style=popout-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAABC0lEQVRYhdWVPQoCMRCFX6HY2ghaiZUXsLW0EDyBrbWtN/EUHsHTWFnYyCL4gxibVZZlZzKTnWz0QZpk5r0vIdkF/kBPAMOKeddE+CQPKoc5Yt5cTjBMdQSwDQToWgBJAn3jmhqgltapAV6E6b5U17MGGAUaUj07TficMfIBZDV6vxowBm1BP9WbSQE4o5h9IjPJmy73TEPDDxVmoZdQrQ5jRhly9Q8tgMUXkIIWn0oG4GYQfAXQzz1PGoCiQndM7b4RgJay/h7zBLT3hASgoKjamQJMreKf0gfuAGyYtXEIAKcL/Dss15iq6ohXghozLYiAMxPuACwtIT4yeQUxAaLrZwAoqGRKGk7qDSYTfYQ8LuYnAAAAAElFTkSuQmCC)](https://github.com/abhiTronix/vidgear/commits/master)
[![License](https://img.shields.io/github/license/abhiTronix/vidgear.svg?style=popout-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAABC0lEQVRYhdWVPQoCMRCFX6HY2ghaiZUXsLW0EDyBrbWtN/EUHsHTWFnYyCL4gxibVZZlZzKTnWz0QZpk5r0vIdkF/kBPAMOKeddE+CQPKoc5Yt5cTjBMdQSwDQToWgBJAn3jmhqgltapAV6E6b5U17MGGAUaUj07TficMfIBZDV6vxowBm1BP9WbSQE4o5h9IjPJmy73TEPDDxVmoZdQrQ5jRhly9Q8tgMUXkIIWn0oG4GYQfAXQzz1PGoCiQndM7b4RgJay/h7zBLT3hASgoKjamQJMreKf0gfuAGyYtXEIAKcL/Dss15iq6ohXghozLYiAMxPuACwtIT4yeQUxAaLrZwAoqGRKGk7qDSYTfYQ8LuYnAAAAAElFTkSuQmCC)](https://github.com/abhiTronix/vidgear/blob/master/LICENSE)
</td></tr></table>

<a text-align="justify"> **VidGear** is a Python library for **Turbo Video Processing** - built on top of OpenCV, Picamera and FFmpeg - **contains powerful [Multi-thread Video Processing Gears](https://github.com/abhiTronix/vidgear#gears) (*classes*) to enables High-Speed video frames capture(*FPS*) and Lossless Video Encoding and Compression functionality across various devices and platforms.** It comes with various inherent features like flexible and direct Source (like `resolution, framerate, colorspace`, etc.) Transforms in OpenCV and supports express Live Network Stream (including `Live GStreamer and YouTube video streams`) pipelining on-the-fly. This library is also very well compatible with the Raspberry Pi Camera module's provides us the ability to exploit its various features like `brightness, saturation, sensor_mode`, etc. easily. VidGear additionally contains various embedded Video Write tools(*built around [FFmpeg](https://ffmpeg.org/) and [OpenCV's VideoWriter Class](https://docs.opencv.org/3.4.3/dd/d9e/classcv_1_1VideoWriter.html)*) that enables *Lossless Video Writing and provide flexible Output Video Compression control*. </a>

&nbsp; 

## Key Features 

<h3 align="center"><img src="https://media.giphy.com/media/1qfKW7QIbLaYYXpTJX/giphy.gif" width="40%" height="40%"/></h3>

Key features that differentiates VidGear from the other existing OpenCV based video processing solutions are:
- [x]  *Multi-Threaded high-speed OpenCV video-frame capturing **(resulting in significantly High FPS)***
- [x]  *Flexible & **Direct control** over the video source*
- [x]  ***In-Built Lossless Video Encoding with FFmpeg & OpenCV*** <img src="https://raw.githubusercontent.com/abhiTronix/Imbakup/master/Images/new.gif"/>
- [x]  ***Flexible output Video Compression control capabilities*** <img src="https://raw.githubusercontent.com/abhiTronix/Imbakup/master/Images/new.gif"/>
- [x]  ***Direct YouTube Video pipelining** into OpenCV by using its URL address* <img src="https://raw.githubusercontent.com/abhiTronix/Imbakup/master/Images/new.gif"/>
- [x]  ***On-the-fly Video Source ColorSpace Conversion*** <img src="https://raw.githubusercontent.com/abhiTronix/Imbakup/master/Images/new.gif"/>
- [x]  ***Automated prerequisites installation*** <img src="https://raw.githubusercontent.com/abhiTronix/Imbakup/master/Images/new.gif" />
- [x]  *Built-in Robust Error and frame synchronization Handling*
- [x]  *Multi-Platform compatibility*
- [x]  *Full Support for Live Network Video Streams(including Gstreamer Raw Video Capture Pipeline)* 

&nbsp; 

## Gears 

Vidgear contains following  ***powerful gears (multi-threaded Video Processing classes)*** to handle/control different device-specific Video Streams:
	
|Gear|Function|
|:------:|---------|
|[**CamGear**](https://github.com/abhiTronix/vidgear/wiki/CamGear-Class)|*Targets any IP-Camera/USB-Camera/Network-Stream/YouTube-Video*|
|[**PiGear**](https://github.com/abhiTronix/vidgear/wiki/PiGear-Class)|*Targets any Raspberry Pi Camera Modules*|
|[**VideoGear**](https://github.com/abhiTronix/vidgear/wiki/VideoGear-Class)|*Common Gear to access any of the above gear*|
|[**WriteGear**](https://github.com/abhiTronix/vidgear/wiki/WriteGear-Class)|*Contains advanced tools that enable Lossless Video Writing with flexible Video Compression capabilities*|

&nbsp; 

## Prerequisites

<h3><img src="http://www.animatedimages.org/data/media/81/animated-hand-image-0021.gif" width="25" height="20"/> Note: Vidgear automatically handles all required prerequisites(except FFmpeg) installation according to your system requirements<img src="https://raw.githubusercontent.com/abhiTronix/Imbakup/master/Images/new.gif"/></h3>

* **Critical:** VidGear must require *`OpenCV with contrib`* (3.0+) python library to be installed on your machine which is critical for its core algorithm functioning. You can build it from from [scratch](https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/) ([Raspberry Pi](https://www.pyimagesearch.com/2018/09/26/install-opencv-4-on-your-raspberry-pi/)) or Vidgear automatically installs *`OpenCV with contrib`* python library for you based on your system from [PyPi](https://pypi.org/project/opencv-contrib-python/).

* **Additional:** 

   	* **PiCamera:** If you are using Raspberry Pi Camera Modules such as *OmniVision OV5647 Camera Module* and *Sony IMX219 Camera Module*. It requires additional [Picamera](https://picamera.readthedocs.io/en/release-1.13/install.html) library installation on your Raspberry Pi machine prior to its installation (*Latest versions recommended*).

     <img src="http://www.animatedimages.org/data/media/81/animated-hand-image-0021.gif" width="25" height="20"/> ***Also, make sure to [enable Raspberry Pi hardware specific settings](https://picamera.readthedocs.io/en/release-1.13/quickstart.html) prior using this library.***

   	* **pafy**: For direct YouTube Video Pipelining into OpenCV, Vidgear requires [Pafy](https://pypi.org/project/pafy/) python library.

	* **FFmpeg:** VidGear requires FFmpeg installation for Compression capabilities. ***Follow this [WIKI Page](https://github.com/abhiTronix/vidgear/wiki/FFmpeg-Installation) for latest FFmpeg installation.*** :warning:


&nbsp; 

## Installation
- **From PyPI(Stable Only):** `VidGear` can be easily installed as follows(*available on [Python Package Index (PyPI)](https://pypi.org/project/vidgear/)*):


  ```sh
  sudo pip install vidgear
  ```
- **Clone this repository(development/experimental):** You can also directly clone this repository for latest patches(*maybe experimental*) and development purposes and thereby can install as follows:


  ```sh
  git clone https://github.com/abhiTronix/vidgear.git
  cd vidgear
  sudo pip install .
  ```
- **Conda Install:**  Anaconda prefers to use its own `conda package manager`, but it’s also possible to install packages using `pip` as follows:


   ```sh
   pip install vidgear
   ```
   
&nbsp; 

## Documentation and Usage

<h3><img src="http://www.animatedimages.org/data/media/81/animated-hand-image-0021.gif" width="25" height="20"/> You can checkout VidGear's <a href = https://github.com/abhiTronix/vidgear/wiki>WIKI-SECTION</a> for detailed documentation with examples for each Multi-Threaded Class(Gear).<img src="https://raw.githubusercontent.com/abhiTronix/Imbakup/master/Images/new.gif" /></h3>

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

## Supported Python versions

* Python 2.7 is the only supported version in 2.x series. ***Python 2.7 support will be dropped in the end of 2019.***

* Python 3.x releases follow `OpenCV` releases.

&nbsp; 

## Contribution and Development
You are welcome to contribute with [suggestions, feature requests and pull requests](https://github.com/abhiTronix/vidgear/pulls).

## Author

- **Abhishek Thakur** [@abhiTronix](https://github.com/abhiTronix)

## License

Copyright © 2019 AbhiTronix

This project is under the MIT License. See the LICENSE file for the full license text.
