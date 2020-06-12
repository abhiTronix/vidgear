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

This parameter sets the index of the monitor screen, where to grab live frames from. More information can be found [here ➶](https://python-mss.readthedocs.io/api.html#mss.tools.mss.base.MSSMixin.monitors). Its value can be assign to `0`, to fetch frames from all connected monitor screens. 

!!! warning "`monitor` value cannot be negative, Otherwise, ScreenGear API will throw `ValueError`."

**Data-Type:** Integer

**Default Value:** Its default value is `1` _(means the current monitor will be used)_.

**Usage:**

```python
ScreenGear(monitor=2) # to fetch frames from second screen
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

This parameter provides the flexibility to manually set the dimensions of capture screen w.r.t selected [`monitor`](#monitor) value. 

!!! info "Dimensional Parameters"
    
    Supported Dimensional Parameters for selected `monitor` value are as follows: 
  
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