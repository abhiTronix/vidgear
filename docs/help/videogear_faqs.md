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

# VideoGear FAQs

&thinsp;

## What is VideoGear API and what does it do?

**Answer:** VideoGear provides a special internal wrapper around VidGear's exclusive [**Video Stabilizer**](../../gears/stabilizer/) class. It also acts as a Common Video-Capture API that provides unified internal access to [CamGear](../../gears/camgear/), [PiGear](../../gears/pigear/), and [FFGear](../../gears/ffgear/) APIs and their parameters, selectable via the [`api`](../../gears/videogear/params/#api) parameter using the [`Backend`](../../bonus/reference/helper/#vidgear.gears.helper.Backend) enum. _For more info. see [VideoGear doc ➶](../../gears/videogear/)_

&thinsp;

## What's the need of VideoGear API?

**Answer:** VideoGear is basically ideal when you need to switch between different video-capture backends without changing your code much. Also, it enables easy stabilization for various video-streams _(real-time or not)_ with minimum effort and using way fewer lines of code. It also serves as backend for other powerful APIs, such as [WebGear](../../gears/webgear/) and [NetGear_Async](../../gears/netgear_async/).

&thinsp;

## Which APIs are accessible with VideoGear API?

**Answer:** VideoGear provides internal access to [CamGear](../../gears/camgear/), [PiGear](../../gears/pigear/), and [FFGear](../../gears/ffgear/) APIs and their parameters via the `Backend` enum (`Backend.CAMGEAR`, `Backend.PIGEAR`, `Backend.FFGEAR`). It also contains a wrapper around the [**Video Stabilizer**](../../gears/stabilizer/) class.

&thinsp;

## How do I select a specific backend in VideoGear?

**Answer:** Use the [`api`](../../gears/pigear/param/#api) parameter with the [`Backend`](../../bonus/reference/helper/#vidgear.gears.helper.Backend) enum:

```python
from vidgear.gears import VideoGear
from vidgear.gears.helper import Backend

stream = VideoGear(api=Backend.CAMGEAR, source=0).start()   # CamGear (default)
stream = VideoGear(api=Backend.PIGEAR).start()              # PiGear
stream = VideoGear(api=Backend.FFGEAR, source="myvideo.mp4").start()  # FFGear
```

The old `enablePiCamera` boolean flag is **deprecated** — use `api=Backend.PIGEAR` instead.

&thinsp;

## Can we access WriteGear API or NetGear API too with VideoGear?

**Answer:** No, only selected VideoCapture APIs _(answered above)_ are accessible.

&thinsp;

## Does using VideoGear instead of CamGear API directly, affects performance?

**Answer:** No, there's no difference, as VideoGear is just a high-level wrapper around CamGear API and without any modifications in-between.

&thinsp;

## When should I use FFGear backend instead of CamGear?

**Answer:** Use `Backend.FFGEAR` when you need hardware-accelerated decoding (CUDA/CUVID, VAAPI, VideoToolbox), complex FFmpeg filtergraph pipelines (`-vf`, `-filter_complex`), per-frame metadata extraction, or specific pixel formats (e.g. `yuv420p`, `gray`). CamGear via OpenCV is simpler and sufficient for most standard use cases.

&thinsp;