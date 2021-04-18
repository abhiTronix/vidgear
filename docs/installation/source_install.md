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

# Install from source


> _Best option for trying latest patches(maybe experimental), forking for Pull Requests, or automatically installing all dependencies(with a few exceptions)._


## Prerequisites

When installing VidGear from source, FFmpeg and Aiortc is the only dependency you need to install manually:

!!! question "What about rest of the dependencies?"

    Any other python dependencies will be automatically installed based on your OS specifications.

### FFmpeg

Must require for the video compression and encoding compatibilities within [StreamGear](#streamgear) and [**Compression Mode**](../../gears/writegear/compression/overview/) in [WriteGear](#writegear) API. 

!!! tip "FFmpeg Installation"

    Follow this dedicated [**FFmpeg Installation doc**](../../gears/writegear/compression/advanced/ffmpeg_install/) for its installation.


### Aiortc

Must Required only if you're using the [WebGear_RTC API](../../gears/webgear_rtc/overview/). You can easily install it via pip:

??? error "Microsoft Visual C++ 14.0 is required."
    
    Installing `aiortc` on windows requires Microsoft Build Tools for Visual C++ libraries installed. You can easily fix this error by installing any **ONE** of these choices:

    !!! info "While the error is calling for VC++ 14.0 - but newer versions of Visual C++ libraries works as well."

      - Microsoft [Build Tools for Visual Studio](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16).
      - Alternative link to Microsoft [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019).
      - Offline installer: [vs_buildtools.exe](https://aka.ms/vs/16/release/vs_buildtools.exe)

    Afterwards, Select: Workloads → Desktop development with C++, then for Individual Components, select only:

      - [x] Windows 10 SDK
      - [x] C++ x64/x86 build tools

    Finally, proceed installing `aiortc` via pip.

```sh
  pip install aiortc
``` 


&nbsp;

## Installation

If you want to just install and try out the checkout the latest beta [`testing`](https://github.com/abhiTronix/vidgear/tree/testing) branch , you can do so with the following command. This can be useful if you want to provide feedback for a new feature or want to confirm if a bug you have encountered is fixed in the `testing` branch. 

!!! warning "DO NOT clone or install `development` branch, as it is not tested with CI environments and is possibly very unstable or unusable."

??? tip "Windows Installation"
  
    * Install [git for windows](https://gitforwindows.org/).

    * Use following commands to clone and install VidGear:

      ```sh
        # clone the repository and get inside
        git clone https://github.com/abhiTronix/vidgear.git && cd vidgear

        # checkout the latest testing branch
        git checkout testing

        # install normally
        python -m pip install .

        # OR install with asyncio support
        python - m pip install .[asyncio]
      ```

```sh
  # clone the repository and get inside
  git clone https://github.com/abhiTronix/vidgear.git && cd vidgear

  # checkout the latest testing branch
  git checkout testing

  # install normally
  pip install .

  # OR install with asyncio support
  pip install .[asyncio]
```

**Or just install directly without cloning:**

```sh
pip install git+git://github.com/abhiTronix/vidgear@testing#egg=vidgear

# or with asyncio support
pip install git+git://github.com/abhiTronix/vidgear@testing#egg=vidgear[asyncio]
```

&nbsp;
