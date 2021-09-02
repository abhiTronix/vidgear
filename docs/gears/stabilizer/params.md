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

# Stabilizer Class Parameters 

&thinsp;

## **`smoothing_radius`**

This parameter can be used to alter averaging window size. It basically handles the quality of stabilization at the expense of latency and sudden panning. Larger its value, less will be panning, more will be latency and vice-versa.

**Data-Type:** Integer

**Default Value:** Its default value is `25`. 

**Usage:** 

You can easily pass this parameter as follows:

```python
Stabilizer(smoothing_radius=30)
```

&nbsp;


## **`border_size`**

This parameter enables and set the value for extended border size that compensates for reduction of black borders during stabilization. 

**Data-Type:** Integer

**Default Value:** Its default value is `0`(no borders).

**Usage:**

 You can easily pass this parameter as follows:

```python
Stabilizer(border_size=10)
```

&nbsp;


## **`crop_n_zoom`**

This parameter enables cropping and zooming of frames _(to original size)_ to reduce the black borders from being too noticeable _(similar to the Stabilized, cropped and Auto-Scaled feature available in Adobe AfterEffects)_ during stabilization. It simply works in conjunction with the `border_size` parameter, i.e. when this parameter is enabled,  `border_size` will be used for cropping border instead of extending them. 

**Data-Type:** Boolean

**Default Value:** Its default value is `False`.

**Usage:**

You can easily pass this parameter as follows:

```python
Stabilizer(border_size=10, crop_n_zoom=True)
```

&nbsp;


## **`border_type`**

This parameter can be used to change the extended border type. Valid border types are `'black'`, `'reflect'`, `'reflect_101'`, `'replicate'` and `'wrap'`, learn more about it [here](https://docs.opencv.org/3.1.0/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5). 


!!! danger "Altering `border_type` parameter is **DISABLED** when `crop_n_zoom` is enabled!"


**Data-Type:** String

**Default Value:** Its default value is `'black'`.

**Usage:**

You can easily pass this parameter as follows:

```python
Stabilizer(border_type='reflect')
```

&nbsp;


## **`logging`**

This parameter enables logging _(if `True`)_, essential for debugging. 

**Data-Type:** Boolean

**Default Value:** Its default value is `False`.

**Usage:**

```python
Stabilizer(logging=True)
```

&nbsp;
