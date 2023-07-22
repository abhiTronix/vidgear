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

# ScreenGear API Parameters 

&thinsp;

## **`monitor`**

This parameter enforces [`dxcam`](https://github.com/ra1nty/DXcam) _(if installed)_ and [`mss`](https://github.com/BoboTiG/python-mss) _(otherwise)_ usage, and it is suitable for selecting index of a specific screen/monitor device _(from where you want retrieve frames)_ in multi-monitor setup. For example, its value can be assign to `2`, to fetch frames from a secondary monitor screen.

!!! warning "Implication of using `monitor` parameter"
    Any value on `monitor` parameter other than `None` in ScreenGear API: 
    
    * Will enforce `dxcam` library backend on Windows platform _(if installed)_, and `mss` library backend otherwise.
    * Will discard any value on its [`backend`](../params/#backend) parameter.



**Data-Type:** Integer, Tuple _(only if `dxcam` backend on Windows)_

**Default Value:** Its default value is `None` _(i.e. disabled by default)_.

**Usage:**

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

    ```python
    # open video stream with defined parameters 
    # with monitor at index `1` selected
    ScreenGear(monitor=1)
    ``` 

=== "With `mss` backend "

    !!! tip "With `mss` library backend, You can also assign `monitor` value to `-1` to fetch frames from all connected multiple monitor screens with `mss` backend."

    !!! danger "With `mss` library backend, API will output [`BGRA`](https://en.wikipedia.org/wiki/RGBA_color_model) colorspace frames instead of default `BGR`."

    ```python
    # open video stream with defined parameters 
    # with monitor at index `1` selected
    ScreenGear(monitor=1)
    ``` 

&nbsp;

## **`backend`**

This parameter enables [`pyscreenshot`](https://github.com/BoboTiG/python-mss) usage and select suitable backend for extracting frames in ScreenGear. The user have the authority of selecting suitable backend which generates best performance as well as the most compatible with their machines. The possible values are: `dxcam` _(Windows only)_, `pil`, `mss`, `scrot`, `maim`, `imagemagick`, `pyqt5`, `pyqt`, `pyside2`, `pyside`, `wx`, `pygdk3`, `mac_screencapture`, `mac_quartz`, `gnome_dbus`, `gnome-screenshot`, `kwin_dbus`. 

!!! note "Performance Benchmarking of all backend can be found [here ➶](https://github.com/ra1nty/DXcam#benchmarks) and [here ➶](https://github.com/ponty/pyscreenshot#performance)"

!!! warning "Remember to install backend library and all of its dependencies you're planning to use with ScreenGear API."

!!! error "Any value on [`monitor`](#monitor) parameter will disable the `backend` parameter. You cannot use both parameters at same time."

!!! info "Backend defaults to `dxcam` library on Windows _(if installed)_, and `pyscreenshot` otherwise."

**Data-Type:** String

**Default Value:** Its default value is `""` _(i.e. default backend)_.

**Usage:**

```python
ScreenGear(backend="pil") # to enforce `pil` as backend for extracting frames.
```

&nbsp;

## **`colorspace`**

This parameter selects the colorspace of the source stream. 

**Data-Type:** String

**Default Value:** Its default value is `None`. 

**Usage:**

!!! tip "All supported `colorspace` values are given [here ➶](../../../bonus/colorspace_manipulation/)."

```python
ScreenGear(colorspace="COLOR_BGR2HSV")
```

!!! example "Its complete usage example is given [here ➶](../usage/#using-screengear-with-direct-colorspace-manipulation)"

&nbsp;


## **`options`** 

This parameter provides the flexibility to manually set the dimensions of capture screen area. 

!!! info "Supported Dimensional Parameters"

	Supported Dimensional Parameters are as follows: 

	* **`left`:** the x-coordinate of the upper-left corner of the region
	* **`top`:** the y-coordinate of the upper-left corner of the region
	* **`width`:** the width of the region
	* **`height`:** the height of the region

!!! note "Additional Exclusive Attribute such as [`THREAD_TIMEOUT`](../../camgear/advanced/source_params/#exclusive-camgear-parameters) is also supported for this parameter."


**Data-Type:** Dictionary

**Default Value:** Its default value is `{}` 

**Usage:**

The desired dimensional parameters can be passed to ScreenGear API by formatting them as attributes, as follows:

!!! tip "More information about screen dimensioning can be found [here ➶](https://python-mss.readthedocs.io/api.html#mss.tools.mss.base.MSSMixin.monitors)"

```python
# formatting dimensional parameters as dictionary attributes
options = {'top': 40, 'left': 0, 'width': 100, 'height': 100}
# assigning it w.r.t monitor=1
ScreenGear(monitor=1, **options)
```

&nbsp;

## **`logging`**

This parameter enables logging _(if `True`)_, essential for debugging. 

**Data-Type:** Boolean

**Default Value:** Its default value is `False`.

**Usage:**

```python
ScreenGear(logging=True)
```

&nbsp;