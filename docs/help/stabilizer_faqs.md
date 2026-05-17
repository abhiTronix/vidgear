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

## What is the Stabilizer Class and what does it do?

**Answer:** The Stabilizer Class is an auxiliary class that enables video stabilization for Vidgear with minimal latency and little to no additional computational overhead. Since `v0.3.5`, the Stabilizer module follows a plugin-style design with a `Stabilizer(...)` factory function backed by a [`StabilizerMode`](../../bonus/reference/stabilizer/#vidgear.gears.stabilizer.StabilizerMode) enum. The default (and currently only) backend is the **Average Sliding-Window (ASW)** stabilizer. For more details, see the [Stabilizer Class documentation ➶](../../gears/stabilizer/).

&thinsp;

## Which stabilization algorithms are available, and how do I switch between them?

**Answer:** Since `v0.3.5`, the Stabilizer module is implemented as a plugin-style package with a [`Stabilizer(...)`](../../bonus/reference/stabilizer/#vidgear.gears.stabilizer.Stabilizer) factory backed by a [`StabilizerMode`](../../bonus/reference/stabilizer/#vidgear.gears.stabilizer.StabilizerMode) enum.

* The default mode is `StabilizerMode.ASW` (Average Sliding-Window), which matches the behavior of earlier versions.
* `StabilizerMode.KALMAN` is currently a placeholder and raises `NotImplementedError`. It is planned for a future release.

Refer to the [`mode`](../../gears/stabilizer/params/#mode) for more details.

```python
from vidgear.gears.stabilizer import Stabilizer, StabilizerMode

stab = Stabilizer(mode=StabilizerMode.ASW, smoothing_radius=30)
```

&thinsp;

## What happens if I pass an invalid or unsupported `mode` value?

**Answer:** The [`Stabilizer(...)`](../../bonus/reference/stabilizer/#vidgear.gears.stabilizer.Stabilizer) factory strictly validates the [`mode`](../../gears/stabilizer/params/#mode) parameter.

* Passing a value that is not a [`StabilizerMode`](../../bonus/reference/stabilizer/#vidgear.gears.stabilizer.StabilizerMode) enum member raises a `TypeError`.
* Selecting `StabilizerMode.KALMAN` raises a `NotImplementedError`, as it is reserved for a future implementation.

Always use a valid enum value.

&thinsp;

## How does the Average Sliding-Window (ASW) stabilizer work?

**Answer:** The ASW backend tracks salient feature points using OpenCV’s *goodFeaturesToTrack* across consecutive frames and computes affine transformations using Lucas-Kanade optical flow.

* Per-frame transformations are accumulated into a trajectory (path).
* This path is smoothed using a **normalized box filter** of width [`smoothing_radius`](../../gears/stabilizer/params/#smoothing_radius).
* The difference between the smoothed path and the original path is applied back to each frame to reduce jitter while preserving intentional motion.

&thinsp;

## How much latency should I expect with the ASW Stabilizer?

**Answer:** The ASW Stabilizer introduces:

* A warm-up latency of [`smoothing_radius`](../../gears/stabilizer/params/#smoothing_radius) frames, during which [`stabilize`](../../bonus/reference/stabilizer/#vidgear.gears.stabilizer.ASWStabilizer.stabilize) method returns `None`.
* Per-frame processing time proportional to `O(smoothing_radius)`.

**Performance considerations:** Higher-resolution frames increase processing time. You can reduce latency by:

- [x] Resizing frames using the [`reducer`](../../bonus/reference/helper/#vidgear.gears.helper.reducer) method before stabilization.
- [x] The [`smoothing_radius`](../../gears/stabilizer/params/#smoothing_radius) parameter balances quality and latency:
   * **Larger values:** smoother output, higher latency, reduced sudden panning
   * **Smaller values:** lower latency, less smoothing

&thinsp;

## Does the ASW Stabilizer's memory grow unbounded on long streams?

**Answer:** No. Since `v0.3.5`, the ASW backend uses bounded fixed-size [`deques`](https://docs.python.org/3/library/collections.html#collections.deque) for frame buffering and transform history. Memory usage is capped at `O(smoothing_radius)` regardless of stream length.

Earlier versions used unbounded lists that grew with the total number of processed frames.

&thinsp;

## Why does the `stabilize` method return `None` for the first several frames?

**Answer:** The ASW Stabilizer requires a full sliding window of size [`smoothing_radius`](../../gears/stabilizer/params/#smoothing_radius) before it can compute a smoothed trajectory.

* Until the window is filled, [`stabilize`](../../bonus/reference/stabilizer/#vidgear.gears.stabilizer.ASWStabilizer.stabilize) returns `None`.
* This is expected behavior. Skip `None` outputs and continue processing frames.

&thinsp;

## How can I remove black borders after stabilization?

**Answer:** Use the [`crop_n_zoom`](../../gears/stabilizer/params/#crop_n_zoom) parameter to crop and zoom frames back to their original size, reducing visible black borders _(similar to effects in Adobe After Effects)_.

* Works together with [`border_size`](../../gears/stabilizer/params/#border_size).
* When enabled, `border_size` is used for cropping instead of padding.
* Default value: `False`.

&thinsp;

## Can I use Stabilizer directly with OpenCV?

**Answer:** Yes. Refer to the [bare minimum OpenCV usage example ➶](../../gears/stabilizer/usage/#bare-minimum-usage-with-opencv).

&thinsp;

## Why is stabilization not working properly for my video?

**Answer:** The ASW Stabilizer is designed for **low-frequency motion** and may not perform well with high-frequency jitter.

Possible causes and solutions:
* Increase [`smoothing_radius`](../../gears/stabilizer/params/#smoothing_radius) for stronger smoothing (at the cost of latency)
* Very dark frames may prevent feature detection. Enable `logging=True` to check for `"Video-Frame is too dark"` warnings

&thinsp;

## Can I reuse a Stabilizer instance after calling `clean()`?

**Answer:** Yes. Calling [`clean()`](../../bonus/reference/stabilizer/#vidgear.gears.stabilizer.ASWStabilizer.clean):

- [x] Clears the internal frame queue
- [x] Resets transform history
- [x] Resets the warm-up state

The same instance can then be reused for a new video stream without reinitialization.

&thinsp;

## Which OpenCV versions are supported by the Stabilizer?

**Answer:** The Stabilizer module supports:

* **OpenCV 3.x**
* **OpenCV 4.x and later**

Internally, the implementation automatically selects:

* `cv2.estimateRigidTransform` for OpenCV 3
* `cv2.estimateAffinePartial2D` for OpenCV 4+

No manual configuration is required.

&thinsp;

## Does the existing `Stabilizer()` API from before v0.3.5 still work?

**Answer:** Yes. The `Stabilizer(...)` API is fully backward-compatible.

- [x] If [`mode`](../../gears/stabilizer/params/#mode) is not specified, it defaults to `StabilizerMode.ASW`
- [x] Behavior remains consistent with the earlier monolithic implementation

!!! success "No migration is required for existing users."
