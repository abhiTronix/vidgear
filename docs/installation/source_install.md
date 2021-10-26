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


> _Best option for trying latest patches(maybe experimental), forking for Pull Requests, or automatically installing all dependencies(with a few exceptions)._


## Prerequisites

When installing VidGear from source, FFmpeg and Aiortc are the only two API specific dependencies you need to install manually:

!!! question "What about rest of the dependencies?"

    Any other python dependencies _(Core/API specific)_ will be automatically installed based on your OS specifications.
    

??? alert "Upgrade your `pip`"

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

        Follow this dedicated [**FFmpeg Installation doc**](../../gears/writegear/compression/advanced/ffmpeg_install/) for its installation.


* #### Aiortc

    Required only if you're using the [**WebGear_RTC**](../../gears/webgear_rtc/overview/) API. You can easily install it via pip:

    ??? error "Microsoft Visual C++ 14.0 is required."
        
        Installing `aiortc` on windows may sometimes requires Microsoft Build Tools for Visual C++ libraries installed. You can easily fix this error by installing any **ONE** of these choices:

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

**If you want to just install and try out the checkout the latest beta [`testing`](https://github.com/abhiTronix/vidgear/tree/testing) branch , you can do so with the following command:**

!!! info "This can be useful if you want to provide feedback for a new feature or want to confirm if a bug you have encountered is fixed in the `testing` branch."

!!! warning "DO NOT clone or install `development` branch unless advised, as it is not tested with CI environments and possibly very unstable or unusable."

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
        
    * If you're using `py` as alias for installed python, then:

        ``` sh
        # clone the repository and get inside
        git clone https://github.com/abhiTronix/vidgear.git && cd vidgear

        # checkout the latest testing branch
        git checkout testing

        # install normally
        python -m pip install .

        # OR install with asyncio support
        python - m pip install .[asyncio]
        ```

??? experiment "Installing vidgear with only selective dependencies"

    Starting with version `v0.2.2`, you can now run any VidGear API by installing only just specific dependencies required by the API in use(except for some Core dependencies). 

    This is useful when you want to manually review, select and install minimal API-specific dependencies on bare-minimum vidgear from scratch on your system:
    
    - To install bare-minimum vidgear without any dependencies, use [`--no-deps`](https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-no-deps) pip flag as follows:

        ```sh
        # clone the repository and get inside
        git clone https://github.com/abhiTronix/vidgear.git && cd vidgear

        # checkout the latest testing branch
        git checkout testing

        # Install without any dependencies
        pip install --no-deps .
        ```

    - Then, you must install all **Core dependencies**:

        ```sh
        # Install core dependencies
        pip install cython, numpy, requests, tqdm, colorlog

        # Install opencv(only if not installed previously)
        pip install opencv-python 
        ```

    - Finally, manually install your **API-specific dependencies** as required by your API(in use):


        | APIs | Dependencies |
        |:---:|:---|
        | CamGear | `pafy`, `yt_dlp`, `streamlink` |
        | PiGear | `picamera` |
        | VideoGear | - |
        | ScreenGear | `mss`, `pyscreenshot`, `Pillow` |
        | WriteGear | **FFmpeg:** See [this doc ➶](https://abhitronix.github.io/vidgear/v0.2.2-dev/gears/writegear/compression/advanced/ffmpeg_install/#ffmpeg-installation-instructions)  |
        | StreamGear | **FFmpeg:** See [this doc ➶](https://abhitronix.github.io/vidgear/v0.2.2-dev/gears/streamgear/ffmpeg_install/#ffmpeg-installation-instructions) |
        | NetGear | `pyzmq`, `simplejpeg` |
        | WebGear | `starlette`, `jinja2`, `uvicorn`, `simplejpeg` |
        | WebGear_RTC | `aiortc`, `starlette`, `jinja2`, `uvicorn` |
        | NetGear_Async | `pyzmq`, `msgpack`, `msgpack_numpy`, `uvloop` |
                    
        ```sh
        # Just copy-&-paste from above table
        pip install <API-specific dependencies>
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


[^1]: The `ensurepip` module was added to the Python standard library in Python 3.4.
