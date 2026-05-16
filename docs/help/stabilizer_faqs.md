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

# Stabilizer Class FAQs

&thinsp;

## What is Stabilizer Class and what does it do?

**Answer:** Stabilizer Class is an auxiliary class that enables Video Stabilization for vidgear with minimalistic latency, and at the expense of little to no additional computational requirements. _For more info. see [Stabilizer Class doc ➶](../../gears/stabilizer/)_

&thinsp;

## How much latency you would typically expect with Stabilizer Class?

**Answer:** The stabilizer will be Slower for High-Quality videos-frames. Try reducing frames size _(Use [`reducer()`](../../bonus/reference/helper/#vidgear.gears.helper.reducer--reducer) method)_ before feeding them for reducing latency. Also, see [`smoothing_radius`](../../gears/stabilizer/params/#smoothing_radius) parameter of Stabilizer class that handles the quality of stabilization at the expense of latency and sudden panning. The larger its value, the less will be panning, more will be latency, and vice-versa.

&thinsp;

## How to remove black borders in output video after stabilizing it?

**Answer:** See [`crop_n_zoom`](../../gears/stabilizer/params/#crop_n_zoom) parameter of Stabilizer class, that enables the feature, where it crops and zooms frames(to original size) to reduce the black borders from stabilization being too noticeable _(similar to the feature available in Adobe AfterEffects)_. It works in conjunction with the [`border_size`](../../gears/stabilizer/params/#border_size) parameter, i.e. when this parameter is enabled border_size will be used for cropping border instead of making them. Its default value is `False`.

&thinsp;

## Can I use Stabilizer directly with OpenCV?

**Answer:** Yes, see [this usage example ➶](../../gears/stabilizer/usage/#bare-minimum-usage-with-opencv).

&thinsp;

## Why stabilization is not working properly for my video?

**Answer:** The Stabilizer may not perform well against High-frequency jitter in video. But,you can check if increasing [`smoothing_radius`](../../gears/stabilizer/params/#smoothing_radius) parameter value helps but it will add latency too.

&thinsp;

## Which stabilization algorithms are available, and how do I switch between them?

**Answer:** Since `v0.3.5`, the Stabilizer module ships as a plugin-style package with a `Stabilizer(...)` factory backed by a `StabilizerMode` enum. The default is `StabilizerMode.ASW` _(Average Sliding-Window)_, which is what older versions used. `StabilizerMode.KALMAN` is reserved as a placeholder and currently raises `NotImplementedError` — it will be implemented in a future release. _For more info. see the [`mode`](../../gears/stabilizer/params/#mode) parameter._

```python
from vidgear.gears.stabilizer import Stabilizer, StabilizerMode

stab = Stabilizer(mode=StabilizerMode.ASW, smoothing_radius=30)
```

&thinsp;

## Does the Stabilizer's memory grow unbounded on long streams?

**Answer:** No. Since `v0.3.5`, the ASW backend stores its transform window in bounded fixed-size deques, capping memory at `O(smoothing_radius)` regardless of how many frames have been processed. Earlier versions kept an unbounded list that grew with `total_frames_seen`.

&thinsp;