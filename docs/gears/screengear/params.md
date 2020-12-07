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

# ScreenGear API Parameters 

## **`monitor`**

This parameter activates [`mss`](https://github.com/BoboTiG/python-mss) backend and sets the index of the monitor screen. This parameter is the most suitable for selecting index of multiple monitor screen from where you want get frames from. For example, its value can be assign to `-1`, to fetch frames from all connected multiple monitor screens. More information can be found [here ➶](https://python-mss.readthedocs.io/examples.html#a-screen-shot-to-grab-them-all)

!!! warning "Any value on `monitor` parameter,  will disable the [`backend`](#backend) parameter."

**Data-Type:** Integer

**Default Value:** Its default value is `None` _(i.e. disabled by default)_.

**Usage:**

```python
ScreenGear(monitor=-1) # to fetch frames from all connected multiple screens
```

&nbsp;

## **`backend`**

This parameter activates [`pyscreenshot`](https://github.com/BoboTiG/python-mss) in ScreenGear API that enables us to select any backend _(for extracting frames)_ of our choice. This parameter give us the authority of selecting the best backend which generates best performance as well as the most compatible with our machine. It's possible values/backends are: `default` ,`pil` ,`mss` ,`scrot` ,`maim` ,`imagemagick` ,`pyqt5` ,`pyqt` ,`pyside2` ,`pyside` ,`wx` ,`pygdk3` ,`mac_screencapture` ,`mac_quartz` ,`gnome_dbus` ,`gnome-screenshot` ,`kwin_dbus`. More information on these backends can be found [here ➶](https://github.com/ponty/pyscreenshot)

!!! note "Performance Benchmarking of each backend can be found [here](https://github.com/ponty/pyscreenshot#performance)"

!!! warning "Remember to install backend library and all of its dependencies you're planning to use with ScreenGear API."

!!! error "Any value on [`monitor`](#monitor) parameter,  will disable the `backend` parameter. You cannot use both parameters at same time."

**Data-Type:** string

**Default Value:** Its default value is `""` _(i.e. default backend)_.

**Usage:**

```python
ScreenGear(backend="mss") # to enforce `mss` as backend for extracting frames.
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

Its complete usage example is given [here ➶](../usage/#using-screengear-with-direct-colorspace-manipulation)

&nbsp;


## **`options`** 

This parameter provides the flexibility to manually set the dimensions of capture screen area. 

!!! info "Supported Dimensional Parameters"
    
    Supported Dimensional Parameters are as follows: 
  
      * **left:** the x-coordinate of the upper-left corner of the region
      * **top:** the y-coordinate of the upper-left corner of the region
      * **width:** the width of the region
      * **height:** the height of the region


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