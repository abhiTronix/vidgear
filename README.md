# VidGear

VidGear is a small python wrapper around OpenCV [Video I/O module](https://docs.opencv.org/master/d0/da7/videoio_overview.html) that contains powerful multi-thread modules(gears) to enable high-speed video frames read functionality across various devices and platforms. It is a reworked implementation of [imutils](https://github.com/jrosebr1/imutils) library's video modules with all major bugs fixed and comes with addition features like source parameters manipulation in OpenCV on the go. This library also well compatible with Raspberry Pi Camera module's [Picamera library](http://picamera.readthedocs.io/) and provides us the ability exploit its various features like `brightness, saturation, sensor_mode` etc. easily. This library is compatible with both *Python 2.7 and Python 3.*


## Installation
VidGear requires [OpenCV](https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/) and [Picamera](https://picamera.readthedocs.io/en/release-1.13/install.html) library installation prior to its installation(*Latest versions recommended*). Must install them.

Then VidGear could be easily be downloaded from release page(*not available on pip yet*) and then can be installed as follows:
```
sudo pip install vidgear-0.1.0-py2.py3-none-any.whl
```
## Usage

### Simple example:

```
# import libraries
f# import libraries
from vidgear.gears import VideoGear
import cv2

stream = VideoGear(enablePiCamera = False, source=0).start() # define various attributes and start the stream

# infinite loop
while True:
	
	frame = stream.read()
	# read frames

	# check if frame is None
	if frame is None:
		#if True break the infinite loop
		break
	
	# do something with frame here
	
	cv2.imshow("Output Frame", frame)
	# Show output window

	key = cv2.waitKey(1) & 0xFF
	# check for 'q' key-press
	if key == ord("q"):
		if 'q' key-pressed break out
		break

cv2.destroyAllWindows()
# close output window

stream.stop()
# safely close video stream.
```
***Checkout [wiki](https://github.com/abhiTronix/vidgear/wiki/VidGear-Gears(Classes)) for extensive usage documentation and examples.*** 



## Salient Features:
- Multi-Threaded high-speed OpenCV video-frame capturing(resulting in High FPS)
- Built-in Robust Error and frame synchronization Handling
- Multi-Platform compatibility
- Flexible control over the output stream
- Small Size 

## Contributing and licenses:
This Project is licensed under the MIT license. You are welcome to contribute with suggestions, feature requests and [pull requests](https://github.com/abhiTronix/vidgear/pulls).
