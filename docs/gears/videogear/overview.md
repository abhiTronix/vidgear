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

# VideoGear API 


## Overview

<p align="center">
  <img src="../../../assets/images/videogear_workflow.png" alt="VideoGear Flow Diagram"  width="80%" title="Designed by Abhishek Thakur(@abhiTronix), under CC-BY-NC-SA 4.0 License"/>
  <br>
  <sub><i>VideoGear API's generalized workflow</i></sub>
</p>


VideoGear provides a special internal wrapper around VidGear's exclusive [**Video Stabilizer**](/gears/stabilizer/overview/) class. 

VideoGear also act as a Common API, that provides an internal access to both [CamGear](/gears/camgear/overview/) and [PiGear](/gears/pigear/overview/) APIs and their parameters, with a special [`enablePiCamera`](/gears/videogear/params/#enablepicamera) boolean flag.

VideoGear is basically ideal when you need to switch to different video sources without changing your code much. Also, it enables easy stabilization for various video-streams _(real-time or not)_  with minimum effort and using way fewer lines of code.

&nbsp; 


!!! tip "Helpful Tips"

	* It is advised to enable logging(`logging = True`) on the first run for easily identifying any runtime errors.

	* You can use `framerate` class variable to retrieve framerate of the input source.


&nbsp; 

## Importing

You can import VideoGear API in your program as follows:

```python
from vidgear.gears import VideoGear
```

&nbsp; 