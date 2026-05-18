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

# Install using Poetry

> _Best option for managing VidGear as a dependency in a [Poetry](https://python-poetry.org/)-managed project._

VidGear's [`pyproject.toml`](https://github.com/abhiTronix/vidgear/blob/master/pyproject.toml) is PEP 517/621 compliant, so it can be consumed directly by [Poetry](https://python-poetry.org/docs/#installation).

??? info "Don't have Poetry installed?"

    Follow the [official Poetry installation guide](https://python-poetry.org/docs/#installation) before proceeding. You can verify your install with:

    ```sh
    poetry --version
    ```

&thinsp;

## Prerequisites

When installing VidGear with Poetry, you need to manually install the following prerequisites:

### Critical Prerequisites :warning:

* #### :simple-opencv: OpenCV

    Must require OpenCV(3.0+) python binaries installed for all core functions. You can easily install it directly via `poetry`:

    ??? tip "OpenCV installation from source"

        You can also follow online tutorials for building & installing OpenCV on [Windows](https://www.learnopencv.com/install-opencv3-on-windows/), [Linux](https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/), [MacOS](https://www.pyimagesearch.com/2018/08/17/install-opencv-4-on-macos/) and [Raspberry Pi](https://www.pyimagesearch.com/2018/09/26/install-opencv-4-on-your-raspberry-pi/) machines manually from its source. 

        :warning: Make sure not to install both *poetry* and *source* version together. Otherwise installation will fail to work!

    ??? info "Other OpenCV binaries"

        OpenCV maintainers also provide additional binaries via *poetry* that contains both main modules and contrib/extra modules [`opencv-contrib-python`](https://pypi.org/project/opencv-contrib-python/), and for server (headless) environments like [`opencv-python-headless`](https://pypi.org/project/opencv-python-headless/) and [`opencv-contrib-python-headless`](https://pypi.org/project/opencv-contrib-python-headless/). You can also install ==any one of them== in similar manner. More information can be found [here](https://github.com/opencv/opencv-python#installation-and-usage).

    ```sh
    poetry add opencv-python       
    ```

&thinsp;

### API Specific Prerequisites 

* #### :simple-ffmpeg: FFmpeg 

    Require only for the video compression and encoding compatibility within [**StreamGear API**](../../gears/streamgear/introduction/) and [**WriteGear API's Compression Mode**](../../gears/writegear/compression/), and for hardware-accelerated decoding within [**FFGear API**](../../gears/ffgear/). 

    !!! tip "FFmpeg Installation"

        * **For WriteGear API's Compression Mode**: Follow this dedicated [**FFmpeg Installation doc**](../../gears/writegear/compression/advanced/ffmpeg_install/) for its installation.
        * **For StreamGear API**: Follow this dedicated [**FFmpeg Installation doc**](../../gears/streamgear/ffmpeg_install/) for its installation.
        * **For FFGear API**: Follow this dedicated [**FFmpeg Installation doc**](../../gears/ffgear/advanced/ffmpeg_install/) for its installation.

&thinsp;

* #### :simple-raspberrypi: Picamera2

    Required only if you're using Raspberry Pi :fontawesome-brands-raspberry-pi: Camera Modules _(or USB webcams)_ with the [**PiGear**](../../gears/pigear/) API. Here's how to install [Picamera2](https://github.com/raspberrypi/picamera2) python library:

    ??? tip "Using Legacy `picamera` library with PiGear (`v0.3.3` and above)"

        PiGear API _(version `0.3.3` onwards)_ prioritizes the newer Picamera2 library under the hood for Raspberry Pi :fontawesome-brands-raspberry-pi: camera modules. However, if your operating system doesn't support Picamera2, you can still use the legacy [`picamera`](https://picamera.readthedocs.io/en/release-1.13/) library. Here's how to easily install it using poetry:

        ```sh
        poetry add picamera
        ```

        !!! note "You could also enforce the legacy picamera API backend in PiGear by using the [`enforce_legacy_picamera`](../../gears/pigear/params/#b-user-defined-parameters) user-defined optional parameter boolean attribute."

    
    ??? warning "Picamera2 is only supported on Raspberry Pi OS Bullseye (or later) images, both 32 and 64-bit."
        
        Picamera2 is **NOT** supported on:

        - [ ] Images based on Buster or earlier releases.
        - [ ] Raspberry Pi OS Legacy images.
        - [ ] Bullseye (or later) images where the legacy camera stack has been re-enabled.

    === "Installation using `apt` (Recommended)"

        ??? success "As of September 2022, Picamera2 is pre-installed on images downloaded from Raspberry Pi. So you don't have to install it manually."

            - [x] On **Raspberry Pi OS images**, Picamera2 is now installed with all the GUI (Qt and OpenGL) dependencies.
            - [x] On **Raspberry Pi OS Lite**, it is installed without the GUI dependencies, although preview images can still be displayed using DRM/KMS. If these users wish to use the additional X-Windows GUI features, they will need to run:

                ```sh
                sudo apt install -y python3-pyqt5 python3-opengl
                ```

        If Picamera2 is not already installed, then your image is presumably older and you should start with system upgrade:
        ```sh
        sudo apt update && upgrade
        ```

        !!! failure "If you have installed Picamera2 previously using pip, then you should also uninstall this (`#!sh python3 -m pip uninstall picamera2`)."

        Thereafter, you can install Picamera2 with all the GUI (Qt and OpenGL) dependencies using:

        ```sh
        sudo apt install -y python3-picamera2
        ```

        Or, If you **DON'T** want the GUI dependencies, use:

        ```sh
        sudo apt install -y python3-picamera2 --no-install-recommends
        ```

    === "Installation using `poetry`"

        !!! danger "This is **NOT** the recommended way to install Picamera2."
        
        However, if you wish to install Picamera2 with all the GUI (Qt and OpenGL) dependencies with poetry, use:

        ```sh
        sudo apt install -y python3-libcamera python3-kms++
        sudo apt install -y python3-pyqt5 python3-prctl 
        sudo apt install -y libatlas-base-dev ffmpeg python3-pip
        python3 -m poetry add numpy
        python3 -m poetry add picamera2[gui]
        ```

        Or, If you **DON'T** want the GUI dependencies, use:

        ```sh
        sudo apt install -y python3-libcamera python3-kms++
        sudo apt install -y python3-prctl libatlas-base-dev
        sudo apt install -y ffmpeg libopenjp2-7 python3-pip
        python3 -m poetry add numpy
        python3 -m poetry add picamera2
        ```


&nbsp;

## Installation

**Add VidGear to an existing Poetry project:**

??? example "Adding vidgear with only selective dependencies"

    Starting with version `v0.2.2`, you can run any VidGear API by installing only the specific dependencies required by the API in use (except for some Core dependencies).

    This is useful when you want to manually review, select and install minimal API-specific dependencies on bare-minimum vidgear from scratch:

    - Add bare-minimum vidgear as follows:

        ```sh
        # Add stable release with bare-minimum dependencies
        poetry add vidgear
        ```

    - Then, you must install **Critical dependencies** (if not already):

        ```sh
        # Install opencv (only if not installed previously)
        poetry add opencv-python 
        ```

    - Finally, manually install your **API-specific dependencies** as required by your API (in use):

        ```sh
        # Just copy-&-paste from table below
        poetry add <API-specific dependencies>
        ```

        | APIs | Dependencies |
        |:---:|:---|
        | CamGear | `yt_dlp` |
        | PiGear | `picamera`, `picamera2` |
        | VideoGear | *Based on CamGear or PiGear or FFGear backend in use*  |
        | ScreenGear | `dxcam`, `mss`, `pyscreenshot`, `Pillow` |
        | WriteGear | **FFmpeg:** See [this doc ➶](../../gears/writegear/compression/advanced/ffmpeg_install/#ffmpeg-installation-instructions)  |
        | StreamGear | **FFmpeg:** See [this doc ➶](../../gears/streamgear/ffmpeg_install/#ffmpeg-installation-instructions) |
        | FFGear | **FFmpeg:** See [this doc ➶](../../gears/ffgear/advanced/ffmpeg_install/#ffmpeg-installation-instructions) |
        | NetGear | `pyzmq`, `simplejpeg` |
        | WebGear | `starlette`, `jinja2`, `uvicorn`, `simplejpeg` |
        | WebGear_RTC | `aiortc`, `starlette`, `jinja2`, `uvicorn` |
        | NetGear_Async | `pyzmq`, `msgpack`, `msgpack_numpy`, `uvloop` |
        | Stabilizer Class | - |


```sh
# Add latest stable release with all Core dependencies
poetry add vidgear[core]

# Or add latest stable release with all Core & Asyncio dependencies
poetry add vidgear[asyncio]
```

**Or, install directly from source in a Poetry-managed environment:**

```sh
# clone the repository and get inside
git clone https://github.com/abhiTronix/vidgear.git && cd vidgear

# Install it into Poetry's virtualenv with all Core dependencies
poetry install --extras core

# Or with all Core & Asyncio dependencies
poetry install --extras asyncio
```

??? tip "Running commands inside Poetry's virtualenv"

    Use `poetry run` to execute VidGear-powered scripts without activating the shell:

    ```sh
    poetry run python your_script.py
    ```

    Or spawn a shell inside the virtualenv:

    ```sh
    poetry shell
    ```

&thinsp;
