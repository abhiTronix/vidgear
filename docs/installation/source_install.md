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

# Install from source


> _Best option for trying latest patches(maybe experimental), forking for Pull Requests, or automatically installing all prerequisites(with a few exceptions)._


## Prerequisites

When installing VidGear from source, following are some API specific prerequisites you may need to install manually:


!!! question "What about rest of the prerequisites?"

    Any other python prerequisites _(Critical/API specific)_ will be automatically installed based on your OS/System specifications.
    

??? alert ":fontawesome-brands-python: Upgrade your `pip`"

    ==It strongly advised to upgrade to latest `pip` before installing vidgear to avoid any undesired installation error(s).==

    There are two mechanisms to upgrade `pip`:

    === "`pip`"

        You can use existing `pip` to upgrade itself:

        ??? info "Install `pip` if not present"

            * Download the script, from https://bootstrap.pypa.io/get-pip.py.
            * Open a terminal/command prompt, `cd` to the folder containing the `get-pip.py` file and run:

            === "Linux/MacOS"

                ```sh
                python get-pip.py
                
                ```

            === "Windows"

                ```sh
                py get-pip.py
                
                ```
            More details about this script can be found in [pypa/get-pip’s README](https://github.com/pypa/get-pip).


        === "Linux/MacOS"

            ```sh
            python -m pip install pip --upgrade
            
            ```

        === "Windows"

            ```sh
            py -m pip install pip --upgrade
            
            ```

    === "`ensurepip`"

        Python also comes with an [`ensurepip`](https://docs.python.org/3/library/ensurepip.html#module-ensurepip) module[^1], which can easily upgrade/install `pip` in any Python environment.

        === "Linux/MacOS"

            ```sh
            python -m ensurepip --upgrade
            
            ```

        === "Windows"

            ```sh
            py -m ensurepip --upgrade
            
            ```
    

### API Specific Prerequisites

* #### FFmpeg 

    Require only for the video compression and encoding compatibility within [**StreamGear API**](../../gears/streamgear/introduction/) and [**WriteGear API's Compression Mode**](../../gears/writegear/compression/overview/). 

    !!! tip "FFmpeg Installation"

        * **For WriteGear API's Compression Mode**: Follow this dedicated [**FFmpeg Installation doc**](../../gears/writegear/compression/advanced/ffmpeg_install/) for its installation.
        * **For StreamGear API**: Follow this dedicated [**FFmpeg Installation doc**](../../gears/streamgear/ffmpeg_install/) for its installation.


&thinsp;

* #### Picamera2

    Required only if you're using Raspberry Pi :fontawesome-brands-raspberry-pi: Camera Modules _(or USB webcams)_ with the [**PiGear**](../../gears/pigear/overview/) API. Here's how to install [Picamera2](https://github.com/raspberrypi/picamera2) python library:

    ??? tip "Using Legacy `picamera` library with PiGear (`v0.3.3` and above)"

        PiGear API _(version `0.3.3` onwards)_ prioritizes the newer Picamera2 library under the hood for Raspberry Pi :fontawesome-brands-raspberry-pi: camera modules. However, if your operating system doesn't support Picamera2, you can still use the  legacy [`picamera`](https://picamera.readthedocs.io/en/release-1.13/) library. Here's how to easily install it using pip:

        ```sh
        pip install picamera
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

        !!! failure "If you have installed Picamera2 previously using pip, then you should also uninstall this (`pip3 uninstall picamera2`)."

        Thereafter, you can install Picamera2 with all the GUI (Qt and OpenGL) dependencies using:

        ```sh
        sudo apt install -y python3-picamera2
        ```

        Or, If you **DON'T** want the GUI dependencies, use:

        ```sh
        sudo apt install -y python3-picamera2 --no-install-recommends
        ```

    === "Installation using `pip`"

        !!! danger "This is **NOT** the recommended way to install Picamera2."
        
        However, if you wish to install Picamera2 with all the GUI (Qt and OpenGL) dependencies with pip, use:

        ```sh
        sudo apt install -y python3-libcamera python3-kms++
        sudo apt install -y python3-pyqt5 python3-prctl 
        sudo apt install -y libatlas-base-dev ffmpeg python3-pip
        pip3 install numpy --upgrade
        pip3 install picamera2[gui]
        ```

        Or, If you **DON'T** want the GUI dependencies, use:

        ```sh
        sudo apt install -y python3-libcamera python3-kms++
        sudo apt install -y python3-prctl libatlas-base-dev
        sudo apt install -y ffmpeg libopenjp2-7 python3-pip
        pip3 install numpy --upgrade
        pip3 install picamera2
        ```

&nbsp;


## Installation


**If you want to checkout the latest beta [`testing`](https://github.com/abhiTronix/vidgear/tree/testing) branch , you can do so with the following commands:**


!!! info "This can be useful if you want to provide feedback for a new feature or bug fix in the `testing` branch."

!!! danger "DO NOT clone or install any other branch other than `testing` unless advised, as it is not tested with CI environments and possibly very unstable or unusable."

??? example "Installing vidgear with only selective dependencies"

    Starting with version `v0.2.2`, you can now run any VidGear API by installing only just specific dependencies required by the API in use(except for some Core dependencies). 

    This is useful when you want to manually review, select and install minimal API-specific dependencies on bare-minimum vidgear from scratch on your system:
    
    - To clone and install bare-minimum vidgear without any dependencies do as follows:

        ```sh
        # clone the repository and get inside
        git clone https://github.com/abhiTronix/vidgear.git && cd vidgear

        # checkout the latest testing branch
        git checkout testing

        # Install stable release with bare-minimum dependencies
        pip install .
        ```

    - Then, you must install **Critical dependencies**(if not already):

        ```sh
        # Install opencv(only if not installed previously)
        pip install opencv-python 
        ```

    - Finally, manually install your **API-specific dependencies** as required by your API(in use):

        ```sh
        # Just copy-&-paste from table below
        pip install <API-specific dependencies>
        ```

        | APIs | Dependencies |
        |:---:|:---|
        | CamGear | `yt_dlp` |
        | PiGear | `picamera`, `picamera2` _(see [this](#picamera2) for its installation)_ |
        | VideoGear | *Based on CamGear or PiGear backend in use*  |
        | ScreenGear | `dxcam`, `mss`, `pyscreenshot`, `Pillow` |
        | WriteGear | **FFmpeg:** See [this doc ➶](https://abhitronix.github.io/vidgear/dev/gears/writegear/compression/advanced/ffmpeg_install/#ffmpeg-installation-instructions)  |
        | StreamGear | **FFmpeg:** See [this doc ➶](https://abhitronix.github.io/vidgear/dev/gears/streamgear/ffmpeg_install/#ffmpeg-installation-instructions) |
        | NetGear | `pyzmq`, `simplejpeg` |
        | WebGear | `starlette`, `jinja2`, `uvicorn`, `simplejpeg` |
        | WebGear_RTC | `aiortc`, `starlette`, `jinja2`, `uvicorn` |
        | NetGear_Async | `pyzmq`, `msgpack`, `msgpack_numpy`, `uvloop` |
        | Stabilizer Class | - |
                    

??? warning ":fontawesome-brands-windows: Windows Installation"

    If you are using Windows, some of the commands given below, may not work out-of-the-box.

    A quick solution may be to preface every Python command with `python -m` like this:

    ```sh
    # Install latest stable release with all Core dependencies
    python -m pip install -U .[core]

    # Or Install latest stable release with all Core & Asyncio dependencies
    python -m pip install -U .[asyncio]
    ```

    And, If you don't have the privileges to the directory you're installing package. Then use `--user` flag, that makes pip install packages in your home directory instead:

    ```sh
    # Install latest stable release with all Core dependencies
    python -m pip install --upgrade --user .[core]

    # Or Install latest stable release with all Core & Asyncio dependencies
    python -m pip install --upgrade --user .[asyncio]
    ```

    Or, If you're using `py` as alias for installed python, then:

    ```sh
    # Install latest stable release with all Core dependencies
    py -m pip install --upgrade --user .[core]

    # Or Install latest stable release with all Core & Asyncio dependencies
    py -m pip install --upgrade --user .[asyncio]
    ```
    

```sh
# clone the repository and get inside
git clone https://github.com/abhiTronix/vidgear.git && cd vidgear

# checkout the latest testing branch
git checkout testing

# Install latest stable release with all Core dependencies
pip install -U .[core]

# Or Install latest stable release with all Core & Asyncio dependencies
pip install -U .[asyncio]
```

&nbsp;


[^1]: :warning: The `ensurepip` module is missing/disabled on Ubuntu. Use `pip` method only.
