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

When installing VidGear with [pip](https://pip.pypa.io/en/stable/installing/), you need to manually install following prerequisites:


??? alert ":fontawesome-brands-python: Upgrade your `pip`"

    ==It strongly advised to upgrade to latest `pip` before installing vidgear to avoid any undesired installation error(s).==

    There are two mechanisms to upgrade `pip`:

    === "`pip`"

        You can use existing `pip` to upgrade itself:

        ??? info "Install `pip` if not present"

            * Download the script, from https://bootstrap.pypa.io/get-pip.py.
            * Open a terminal/command prompt, `cd` to the folder containing the `get-pip.py` file and run:

            === ":material-linux: Linux / :material-apple: MacOS"

                ```sh
                python get-pip.py
                
                ```

            === ":fontawesome-brands-windows: Windows"

                ```sh
                py get-pip.py
                
                ```
            More details about this script can be found in [pypa/get-pip’s README](https://github.com/pypa/get-pip).


        === ":material-linux: Linux / :material-apple: MacOS"

            ```sh
            python -m pip install pip --upgrade
            
            ```

        === ":fontawesome-brands-windows: Windows"

            ```sh
            py -m pip install pip --upgrade
            
            ```

    === "`ensurepip`"

        Python also comes with an [`ensurepip`](https://docs.python.org/3/library/ensurepip.html#module-ensurepip) module[^1], which can easily upgrade/install `pip` in any Python environment.

        === ":material-linux: Linux / :material-apple: MacOS"

            ```sh
            python -m ensurepip --upgrade
            
            ```

        === ":fontawesome-brands-windows: Windows"

            ```sh
            py -m ensurepip --upgrade
            
            ```

### Critical Prerequisites :warning:

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

    Require only for the video compression and encoding compatibility within [**StreamGear API**](../../gears/streamgear/introduction/) and [**WriteGear API's Compression Mode**](../../gears/writegear/compression/overview/). 

    !!! tip "FFmpeg Installation"

        * **For WriteGear API's Compression Mode**: Follow this dedicated [**FFmpeg Installation doc**](../../gears/writegear/compression/advanced/ffmpeg_install/) for its installation.
        * **For StreamGear API**: Follow this dedicated [**FFmpeg Installation doc**](../../gears/streamgear/ffmpeg_install/) for its installation.


* #### Picamera

    Required only if you're using Raspberry Pi :fontawesome-brands-raspberry-pi: Camera Modules with its [**PiGear**](../../gears/pigear/overview/) API. You can easily install it via pip:


    !!! warning "Make sure to [**enable Raspberry Pi hardware-specific settings**](https://picamera.readthedocs.io/en/release-1.13/quickstart.html) prior to using this library, otherwise it won't work."

    ```sh
    pip install picamera
    ```  

* #### Uvloop

    Required only if you're using the [**NetGear_Async**](../../gears/netgear_async/overview/) API on UNIX machines for maximum performance. You can easily install it via pip:

    !!! fail "uvloop is **[NOT yet supported on Windows :fontawesome-brands-windows: Machines](https://github.com/MagicStack/uvloop/issues/14).**"

    ```sh
    pip install uvloop
    ```

* #### DXcam

    Required only if you're using the [**ScreenGear**](../../gears/screengear/overview/) API on Windows machines for better FPS performance. You can easily install it via pip:

    !!! fail "FYI, DXcam is **ONLY supported on Windows :fontawesome-brands-windows: Machines.**"

    ```sh
    pip install dxcam
    ```

&nbsp;

## Installation


??? danger "Installation command with `pip` has been changed in `v0.2.4`"

    The legacy `#!sh  pip install vidgear` command now installs critical bare-minimum dependencies only. Therefore in order to automatically install all the API specific dependencies as previous versions, use `#!sh  pip install vidgear[core]` command instead.

    === "`v0.2.4` and newer"

        ```sh
        # Install latest stable release with all Core dependencies
        pip install -U vidgear[core]
        ```

    === "Older"

        !!! fail "`[core]` keyword isn't available in versions older than `v0.2.4`"

        ```sh
        # Install older stable release with all Core dependencies
        pip install vidgear<0.2.4
        ```

    Similarly in your python project files like `setup.py` or `requirements.txt` or `setup.cfg`, use vidgear dependency as `#!sh  vidgear[core]>=0.2.4`  instead.

    !!! note "This change does not affects `#!sh pip install vidgear[asyncio]` command."



**Installation is as simple as:**

??? experiment "Installing vidgear with only selective dependencies"

    Starting with version `v0.2.2`, you can now run any VidGear API by installing only just specific dependencies required by the API in use(except for some Core dependencies). 

    This is useful when you want to manually review, select and install minimal API-specific dependencies on bare-minimum vidgear from scratch on your system:
    
    - Install bare-minimum vidgear as follows:

        
        === "`v0.2.4` and newer"

            ```sh
            # Install stable release with bare-minimum dependencies
            pip install -U vidgear
            ```

        === "Older"

            ```sh
            # Install stable without any dependencies
            pip install --no-deps vidgear<0.2.4
            ```

    - Then, you must install **Critical dependencies**(if not already):

        === "`v0.2.4` and newer"

            ```sh
            # Install opencv(only if not installed previously)
            pip install opencv-python 
            ```

        === "Older"

            ```sh
            # Install critical dependencies
            pip install cython, numpy, requests, tqdm, colorlog

            # Install opencv(only if not installed previously)
            pip install opencv-python 
            ```

    - Finally, manually install your **API-specific dependencies** as required by your API(in use):

        ```sh
        # Just copy-&-paste from table below
        pip install <API-specific dependencies>
        ```

        === "`v0.2.4` and newer"

            | APIs | Dependencies |
            |:---:|:---|
            | CamGear | `yt_dlp` |
            | PiGear | `picamera` |
            | VideoGear | *Based on CamGear or PiGear backend in use*  |
            | ScreenGear | `dxcam`, `mss`, `pyscreenshot`, `Pillow` |
            | WriteGear | **FFmpeg:** See [this doc ➶](../../gears/writegear/compression/advanced/ffmpeg_install/#ffmpeg-installation-instructions)  |
            | StreamGear | **FFmpeg:** See [this doc ➶](../../gears/streamgear/ffmpeg_install/#ffmpeg-installation-instructions) |
            | NetGear | `pyzmq`, `simplejpeg` |
            | WebGear | `starlette`, `jinja2`, `uvicorn`, `simplejpeg` |
            | WebGear_RTC | `aiortc`, `starlette`, `jinja2`, `uvicorn` |
            | NetGear_Async | `pyzmq`, `msgpack`, `msgpack_numpy`, `uvloop` |
            | Stabilizer Class | - |

        === "Older"

            | APIs | Dependencies |
            |:---:|:---|
            | CamGear | `pafy`, `yt_dlp`, `streamlink` |
            | PiGear | `picamera` |
            | VideoGear | *Based on CamGear or PiGear backend in use* |
            | ScreenGear | `dxcam`, `mss`, `pyscreenshot`, `Pillow` |
            | WriteGear | **FFmpeg:** See [this doc ➶](../../gears/writegear/compression/advanced/ffmpeg_install/#ffmpeg-installation-instructions)  |
            | StreamGear | **FFmpeg:** See [this doc ➶](../../gears/streamgear/ffmpeg_install/#ffmpeg-installation-instructions) |
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
    python -m pip install -U vidgear[core]

    # Or Install latest stable release with all Core & Asyncio dependencies
    python -m pip install -U vidgear[asyncio]
    ```

    And, If you don't have the privileges to the directory you're installing package. Then use `--user` flag, that makes pip install packages in your home directory instead:

    ```sh
    # Install latest stable release with all Core dependencies
    python -m pip install --upgrade --user vidgear[core]

    # Or Install latest stable release with all Core & Asyncio dependencies
    python -m pip install --upgrade --user vidgear[asyncio]
    ```

    Or, If you're using `py` as alias for installed python, then:

    ```sh
    # Install latest stable release with all Core dependencies
    py -m pip install --upgrade --user vidgear[core]

    # Or Install latest stable release with all Core & Asyncio dependencies
    py -m pip install --upgrade --user vidgear[asyncio]
    ```


```sh
# Install latest stable release with all Core dependencies
pip install -U vidgear[core]

# Or Install latest stable release with all Core & Asyncio dependencies
pip install -U vidgear[asyncio]
```

**And if you prefer to install VidGear directly from the repository:**

```sh
# Install latest stable release with all Core dependencies
pip install git+git://github.com/abhiTronix/vidgear@master#egg=vidgear[core]

# Or Install latest stable release with all Core & Asyncio dependencies
pip install git+git://github.com/abhiTronix/vidgear@master#egg=vidgear[asyncio]
```

**Or you can also download its wheel (`.whl`) package from our repository's [releases](https://github.com/abhiTronix/vidgear/releases) section, and thereby can be installed as follows:**

```sh
# Install latest stable release with all Core dependencies
pip install vidgear-0.3.1-py3-none-any.whl[core]

# Or Install latest stable release with all Core & Asyncio dependencies
pip install vidgear-0.3.1-py3-none-any.whl[asyncio]
```

&nbsp;

[^1]: :warning: The `ensurepip` module is missing/disabled on Ubuntu. Use `pip` method only.