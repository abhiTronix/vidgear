# VidGear

VidGear is a lightweight python wrapper around OpenCV [Video I/O module](https://docs.opencv.org/master/d0/da7/videoio_overview.html) that contains powerful multi-thread modules(gears) to enable high-speed video frames read functionality across various devices and platforms. It is a reworked implementation of [imutils](https://github.com/jrosebr1/imutils) library's video modules with all major bugs fixed and comes with addition features like direct network streaming(*GStreamer Pipeline supported*) and flexible source parameters/attributes manipulation of `OpenCV` and `picamera` libraries on the go. This library is very well compatible with Raspberry Pi Camera module's [Picamera library](http://picamera.readthedocs.io/) and provides us the ability exploit its various features like `brightness, saturation, sensor_mode` etc. easily. This library supports *Python 2.7 and all above versions.*


## Installation
**VidGear requires [OpenCV](https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/) and [Picamera](https://picamera.readthedocs.io/en/release-1.13/install.html) (*Only if you want to use Raspberry Pi Camera Module*) libraries installation prior to its installation (*Latest versions recommended*).**

- **Download from PIP:** `vidgear` wheel file could be easily be downloaded and can be installed as follows(**Now available on [Python Package Index (PyPI)](https://pypi.org/project/vidgear/)**):
  ```bash
  sudo pip install vidgear
  ```
- **Clone This Repo(Latest but experimental):** Or you can clone this repository and install using pip as follows:
  ```bash
  git clone https://github.com/abhiTronix/vidgear.git
  cd vidgear
  sudo pip install .
  ```
- **Conda Install:**  Anaconda prefers to use its own `conda package manager`, but it’s also possible to install packages using `pip`. You can install as follows:
   ```bash
   pip install vidgear
   ```


## Usage

### Basic example:

```python
# import required libraries
from vidgear.gears import VideoGear
import cv2

stream = VideoGear(enablePiCamera = False, source=0).start() 
# define various attributes and start the stream

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
### Important: Checkout [wiki](https://github.com/abhiTronix/vidgear/wiki/VidGear-Gears(Classes)) for extensive usage documentation and examples.

---

## Salient Features:
- Multi-Threaded high-speed OpenCV video-frame capturing(resulting in High FPS)
- Lightweight
- Built-in Robust Error and frame synchronization Handling
- Multi-Platform compatibility
- Flexible control over the output stream
- Full Support for Network Video Streams(*Including Gstreamer Raw Video Capture Pipeline*) 

## Contributing and licenses:
This Project is licensed under the MIT license. You are welcome to contribute with suggestions, feature requests and [pull requests](https://github.com/abhiTronix/vidgear/pulls).
