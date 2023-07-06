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

# ScreenGear API Usage Examples:

!!! experiment "After going through ScreenGear Usage Examples, Checkout more of its advanced configurations [here ➶](../../../help/screengear_ex/)"

!!! success "Recommended: Install DXcam library on Windows :fontawesome-brands-windows: Machines"

    On Windows Machines, if installed, ScreenGear API uses `dxcam` backend machines for higher FPS performance. Thereby, it is **highly recommended** to install it via pip as follows:

    ```sh
    pip install dxcam
    ```

&thinsp;

## Bare-Minimum Usage

Following is the bare-minimum code you need to get started with ScreenGear API:

```python
# import required libraries
from vidgear.gears import ScreenGear
import cv2

# open video stream with default parameters
stream = ScreenGear().start()

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # Show output window
    cv2.imshow("Output Frame", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
```

&nbsp; 

## Using ScreenGear with Variable Screen Dimensions

ScreenGear API provides us the flexibility to directly set the dimensions of capturing-area of the screen. These dimensions can be easily applied to ScreenGear API through its [`options`](../params/#options) dictionary parameter by formatting them as its attributes. 

The complete usage example is as follows:


```python hl_lines="6"
# import required libraries
from vidgear.gears import ScreenGear
import cv2

# define dimensions of screen w.r.t to given monitor to be captured
options = {"top": 40, "left": 0, "width": 100, "height": 100}

# open video stream with defined parameters
stream = ScreenGear(logging=True, **options).start()

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # Show output window
    cv2.imshow("Output Frame", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
``` 

&nbsp; 

## Using ScreenGear with Multiple Screens

ScreenGear API provides us the flexibility to select any connected display for fetching frames, with its [`monitor`](../params/#monitor) parameter:

!!! warning "Implication of using `monitor` parameter"
    Any value on `monitor` parameter other than `None` in ScreenGear API: 
    
    * Will enforce `dxcam` library backend on Windows platform _(if installed)_, and `mss` library backend otherwise.
    * Will discard any value on its [`backend`](../params/#backend) parameter.


=== "With `dxcam` on Windows :fontawesome-brands-windows:"

    ??? tip "Using GPU acceleration on Windows :fontawesome-brands-windows:"
        With  `dxcam` library backend, you can also assign which GPU devices ids to use along with monitor device ids as tuple `(monitor_idx, gpu_idx)`, as follows:

        ```python
        # open video stream with defined parameters with 
        # monitor at index `1` and GPU at index `0`.
        stream = ScreenGear(monitor=(1,0), logging=True).start()
        ```
        
        !!! info "Getting a complete list of monitor devices and GPUs"

            To get a complete list of monitor devices and outputs(GPUs), you can use `dxcam` library itself:
            ```sh
            >>> import dxcam
            >>> dxcam.device_info()
            'Device[0]:<Device Name:NVIDIA GeForce RTX 3090 Dedicated VRAM:24348Mb VendorId:4318>\n'
            >>> dxcam.output_info()
            'Device[0] Output[0]: Res:(1920, 1080) Rot:0 Primary:True\nDevice[0] Output[1]: Res:(1920, 1080) Rot:0 Primary:False\n'
            ```

    ```python hl_lines="6"
    # import required libraries
    from vidgear.gears import ScreenGear
    import cv2

    # open video stream with defined parameters with monitor at index `1` selected
    stream = ScreenGear(monitor=1, logging=True).start()

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # Show output window
        cv2.imshow("Output Frame", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()
    ``` 

=== "With `mss` backend "


    !!! tip "With `mss` library backend, You can also assign `monitor` value to `-1` to fetch frames from all connected multiple monitor screens with `mss` backend."

    !!! danger "With `mss` library backend, API will output [`BGRA`](https://en.wikipedia.org/wiki/RGBA_color_model) colorspace frames instead of default `BGR`."

    ```python hl_lines="6"
    # import required libraries
    from vidgear.gears import ScreenGear
    import cv2

    # open video stream with defined parameters with monitor at index `1` selected
    stream = ScreenGear(monitor=1, logging=True).start()

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # Show output window
        cv2.imshow("Output Frame", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()
    ``` 

&nbsp; 

## Using ScreenGear with Variable Backend

With ScreenGear API, you can select from many different backends that generates best performance as well as the most compatible with our machine by employing its [`backend`](../params/#backend) parameter that supports many different backends:

??? tip "Supported `backend` values" 

    **Its possible values are:** `dxcam` _(Windows only)_, `pil`, `mss`, `scrot`, `maim`, `imagemagick`, `pyqt5`, `pyqt`, `pyside2`, `pyside`, `wx`, `pygdk3`, `mac_screencapture`, `mac_quartz`, `gnome_dbus`, `gnome-screenshot`, `kwin_dbus`. 

    !!! warning "Remember to install backend library and all of its dependencies you're planning to use with ScreenGear API. More information on all these backends _(except `dxcam`)_ can be found [here ➶](https://github.com/ponty/pyscreenshot)"

!!! note "Backend defaults to `dxcam` library on Windows _(if installed)_, and `pyscreenshot` otherwise."

!!! error "Any value on `monitor` parameter will disable the `backend` parameter. You cannot use them simultaneously."

```python hl_lines="7"
# import required libraries
from vidgear.gears import ScreenGear
import cv2

# open video stream with defined parameters and `mss` backend 
# for extracting frames.
stream = ScreenGear(backend="mss", logging=True).start()

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # Show output window
    cv2.imshow("Output Frame", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
``` 

&nbsp; 

## Using ScreenGear with Direct Colorspace Manipulation

ScreenGear API also supports **Direct Colorspace Manipulation**, which is ideal for changing source colorspace on the run. 

!!! info "A more detailed  information on colorspace manipulation can be found [here ➶](../../../bonus/colorspace_manipulation/)"

In following example code, we will start with [**HSV**](https://en.wikipedia.org/wiki/HSL_and_HSV) as source colorspace, and then we will switch to [**GRAY**](https://en.wikipedia.org/wiki/Grayscale)  colorspace when `w` key is pressed, and then [**LAB**](https://en.wikipedia.org/wiki/CIELAB_color_space) colorspace when `e` key is pressed, finally default colorspace _(i.e. **BGR**)_ when `s` key is pressed. Also, quit when `q` key is pressed:


!!! warning "Any incorrect or None-type value, will immediately revert the colorspace to default i.e. `BGR`."


```python hl_lines="6 29 33 37"
# import required libraries
from vidgear.gears import ScreenGear
import cv2

# Change colorspace to `HSV`
stream = ScreenGear(colorspace="COLOR_BGR2HSV", logging=True).start()

# loop over
while True:

    # read HSV frames
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the HSV frame here}

    # Show output window
    cv2.imshow("Output Frame", frame)

    # check for key if pressed
    key = cv2.waitKey(1) & 0xFF

    # check if 'w' key is pressed
    if key == ord("w"):
        # directly change colorspace at any instant
        stream.color_space = cv2.COLOR_BGR2GRAY  # Now colorspace is GRAY

    # check for 'e' key is pressed
    if key == ord("e"):
        stream.color_space = cv2.COLOR_BGR2LAB  # Now colorspace is CieLAB

    # check for 's' key is pressed
    if key == ord("s"):
        stream.color_space = None  # Now colorspace is default(ie BGR)

    # check for 'q' key is pressed
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
```

&nbsp;


## Using ScreenGear with WriteGear API

ScreenGear can be used in conjunction with WriteGear API directly without any compatibility issues. The suitable example is as follows:

```python
# import required libraries
from vidgear.gears import ScreenGear
from vidgear.gears import WriteGear
import cv2


# define dimensions of screen w.r.t to given monitor to be captured
options = {"top": 40, "left": 0, "width": 100, "height": 100}

# define suitable (Codec,CRF,preset) FFmpeg parameters for writer
output_params = {"-vcodec": "libx264", "-crf": 0, "-preset": "fast"}

# open video stream with defined parameters
stream = ScreenGear(monitor=1, logging=True, **options).start()

# Define writer with defined parameters and suitable output filename for e.g. `Output.mp4`
writer = WriteGear(output="Output.mp4", logging=True, **output_params)

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}
    # lets convert frame to gray for this example
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # write gray frame to writer
    writer.write(gray)

    # Show output window
    cv2.imshow("Output Gray Frame", gray)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()

# safely close writer
writer.close()
``` 

&nbsp; 