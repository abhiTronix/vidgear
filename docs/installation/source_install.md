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

When installing VidGear from source, FFmpeg is the only API specific prerequisites you need to install manually:


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


&nbsp;


## Installation


**If you want to checkout the latest beta [`testing`](https://github.com/abhiTronix/vidgear/tree/testing) branch , you can do so with the following commands:**


!!! info "This can be useful if you want to provide feedback for a new feature or bug fix in the `testing` branch."

!!! danger "DO NOT clone or install any other branch other than `testing` unless advised, as it is not tested with CI environments and possibly very unstable or unusable."

??? experiment "Installing vidgear with only selective dependencies"

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
        | PiGear | `picamera` |
        | VideoGear | *Based on CamGear or PiGear backend in use*  |
        | ScreenGear | `mss`, `pyscreenshot`, `Pillow` |
        | WriteGear | **FFmpeg:** See [this doc ➶](https://abhitronix.github.io/vidgear/v0.2.2-dev/gears/writegear/compression/advanced/ffmpeg_install/#ffmpeg-installation-instructions)  |
        | StreamGear | **FFmpeg:** See [this doc ➶](https://abhitronix.github.io/vidgear/v0.2.2-dev/gears/streamgear/ffmpeg_install/#ffmpeg-installation-instructions) |
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
