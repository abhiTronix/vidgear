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

# Install using pip


> _Best option for easily getting stable VidGear installed._


## Prerequisites

When installing VidGear with [pip](https://pip.pypa.io/en/stable/installing/), you need to check manually if following dependencies are installed:


### Core Prerequisites

* #### OpenCV 

    Must require OpenCV(3.0+) python binaries installed for all core functions. You easily install it directly via [pip](https://pypi.org/project/opencv-python/):

    ??? tip "OpenCV installation from source"

        You can also follow online tutorials for building & installing OpenCV on [Windows](https://www.learnopencv.com/install-opencv3-on-windows/), [Linux](https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/), [MacOS](https://www.pyimagesearch.com/2018/08/17/install-opencv-4-on-macos/) and [Raspberry Pi](https://www.pyimagesearch.com/2018/09/26/install-opencv-4-on-your-raspberry-pi/) machines manually from its source. 

        :warning: Make sure not to install both *pip* and *source* version together. Otherwise installation will fail to work!

    ??? info "Other OpenCV binaries"

        OpenCV mainainers also provide additional binaries via pip that contains both main modules and contrib/extra modules [`opencv-contrib-python`](https://pypi.org/project/opencv-contrib-python/), and for server (headless) environments like [`opencv-python-headless`](https://pypi.org/project/opencv-python-headless/) and [`opencv-contrib-python-headless`](https://pypi.org/project/opencv-contrib-python-headless/). You can also install ==any one of them== in similar manner. More information can be found [here](https://github.com/opencv/opencv-python#installation-and-usage).


    ```sh
    pip install opencv-python       
    ```

### API Specific Prerequisites

* #### FFmpeg 

    Require for the video compression and encoding compatibilities within [**StreamGear**](#streamgear) API and [**WriteGear API's Compression Mode**](../../gears/writegear/compression/overview/). 

    !!! tip "FFmpeg Installation"

        Follow this dedicated [**FFmpeg Installation doc**](../../gears/writegear/compression/advanced/ffmpeg_install/) for its installation.

* #### Picamera

    Required only if you're using Raspberry Pi Camera Modules with its [**PiGear**](../../gears/pigear/overview/) API. You can easily install it via pip:


    !!! warning "Make sure to [**enable Raspberry Pi hardware-specific settings**](https://picamera.readthedocs.io/en/release-1.13/quickstart.html) prior to using this library, otherwise it won't work."

    ```sh
    pip install picamera
    ``` 

* #### Aiortc

    Required only if you're using the [**WebGear_RTC**](../../gears/webgear_rtc/overview/) API. You can easily install it via pip:

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

* #### Uvloop

    Required only if you're using the [**NetGear_Async**](../../gears/netgear_async/overview/) API on UNIX machines for maximum performance. You can easily install it via pip:

    !!! error "uvloop is **[NOT yet supported on Windows Machines](https://github.com/MagicStack/uvloop/issues/14).**"
    !!! warning "Python-3.6 legacies support [**dropped in version `>=1.15.0`**](https://github.com/MagicStack/uvloop/releases/tag/v0.15.0). Kindly install previous `0.14.0` version instead."

    ```sh
    pip install uvloop
    ```

&nbsp;

## Installation

**Installation is as simple as:**

??? warning "Windows Installation"

    If you are using Windows, some of the commands given below, may not work out-of-the-box.

    A quick solution may be to preface every Python command with `python -m` like this:

    ```sh
    python -m pip install vidgear

    # or with asyncio support
    python -m pip install vidgear[asyncio]
    ```

    If you don't have the privileges to the directory you're installing package. Then use `--user` flag, that makes pip install packages in your home directory instead:

    ``` sh
    python -m pip install --user vidgear

    # or with asyncio support
    python -m pip install --user vidgear[asyncio]
    ```

```sh
# Install stable release
pip install vidgear

# Or Install stable release with Asyncio support
pip install vidgear[asyncio]
```

**And if you prefer to install VidGear directly from the repository:**

```sh
pip install git+git://github.com/abhiTronix/vidgear@master#egg=vidgear

# or with asyncio support
pip install git+git://github.com/abhiTronix/vidgear@master#egg=vidgear[asyncio]
```

**Or you can also download its wheel (`.whl`) package from our repository's [releases](https://github.com/abhiTronix/vidgear/releases) section, and thereby can be installed as follows:**

```sh
pip install vidgear-0.2.2-py3-none-any.whl

# or with asyncio support
pip install vidgear-0.2.2-py3-none-any.whl[asyncio]
```

&nbsp;
