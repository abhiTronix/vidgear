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

# FFmpeg Installation Instructions

<figure>
  <a href="http://ffmpeg.org/"><img src="../../../../../assets/images/ffmpeg.png" loading="lazy" alt="FFmpeg"  /></a>
</figure>

WriteGear must requires FFmpeg executables for its Compression capabilities in Compression Mode. You can following machine-specific instructions for its installation:


!!! error "In case WriteGear API fails to detect valid FFmpeg executables on your system _(even if Compression Mode is enabled)_, it automatically fallbacks to [Non-Compression Mode](../../../non_compression/overview/)."

&nbsp;

&nbsp;


## :material-linux: Linux FFmpeg Installation

The WriteGear API supports _Auto-Detection_ and _Manual Configuration_ methods on a Linux machine:

### A. Auto-Detection 

!!! quote "This is a recommended approach on Linux Machines"

If WriteGear API not receives any input from the user on [**`custom_ffmpeg`**](../../params/#custom_ffmpeg) parameter, then on Linux system, it tries to **auto-detects** the required FFmpeg installed binaries through validation test that employs `subprocess` python module. 

**Installation:** You can install easily install official FFmpeg according to your Linux Distro by following [this post ➶](https://www.tecmint.com/install-ffmpeg-in-linux/)


### B. Manual Configuration

* **Download:** You can also manually download the latest Linux Static Binaries(*based on your machine arch(x86/x64)*) from the link below:

    *Linux Static Binaries:* http://johnvansickle.com/ffmpeg/

* **Assignment:** Then, you can easily assign the custom path to the folder containing FFmpeg executables(`for e.g 'ffmpeg/bin'`)  or path of `ffmpeg` executable itself to the [**`custom_ffmpeg`**](../../params/#custom_ffmpeg) parameter in the WriteGear API.

    !!! warning "If binaries were not found at the manually specified path, WriteGear API will disable the Compression Mode!"

&nbsp;

&nbsp;

## :fontawesome-brands-windows: Windows FFmpeg Installation

The WriteGear API supports _Auto-Installation_ and _Manual Configuration_ methods on Windows systems.

### A. Auto-Installation

!!! quote "This is a recommended approach on Windows Machines"

If WriteGear API not receives any input from the user on [**`custom_ffmpeg`**](../../params/#custom_ffmpeg) parameter, then on Windows system WriteGear API **auto-generates** the required FFmpeg Static Binaries from a dedicated [**Github Server**](https://github.com/abhiTronix/FFmpeg-Builds) into the temporary directory _(for e.g. `C:\Temp`)_ of your machine.

!!! warning Important Information

    * The files downloaded to temporary directory _(for e.g. `C:\TEMP`)_, may get erased if your machine shutdowns/restarts.

    * You can also provide a custom save path for auto-downloading **FFmpeg Static Binaries** through [`-ffmpeg_download_path`](../../params/#output_params) parameter.

    * If binaries were found at the specified path, WriteGear automatically skips the auto-installation step.

    * ==If the required FFmpeg static binary fails to download, or extract, or validate during auto-installation, then, WriteGear API will **auto-disable** the Compression Mode and switches to [Non-Compression Mode](../../../non_compression/overview/)!==


### B. Manual Configuration

* **Download:** You can also manually download the latest Windows Static Binaries(*based on your machine arch(x86/x64)*) from the link below:
   
      *Windows Static Binaries:* https://ffmpeg.org/download.html#build-windows

*  **Assignment:** Then, you can easily assign the custom path to the folder containing FFmpeg executables(`for e.g 'C:/foo/Downloads/ffmpeg/bin'`) or path of `ffmpeg.exe` executable itself to the [**`custom_ffmpeg`**](../../params/#custom_ffmpeg) parameter in the WriteGear API.

    !!! warning "If binaries were not found at the manually specified path, WriteGear API will disable the Compression Mode!"


&nbsp;

&nbsp;

## :material-apple: MacOS FFmpeg Installation

The WriteGear API supports _Auto-Detection_ and _Manual Configuration_ methods on a macOS machine.

### A. Auto-Detection

!!! quote "This is a recommended approach on MacOS Machines"

If WriteGear API not receives any input from the user on [**`custom_ffmpeg`**](../../params/#custom_ffmpeg) parameter, then on macOS system, it tries to **auto-detects** the required FFmpeg installed binaries through validation test that employs `subprocess` python module.

**Installation:** You can easily install FFmpeg on your macOS machine by following [this tutorial ➶](https://trac.ffmpeg.org/wiki/CompilationGuide/macOS)

### B. Manual Configuration

* **Download:** You can also manually download the latest macOS Static Binaries(*only x64 Binaries*) from the link below:
  
    *MacOS Static Binaries:* http://johnvansickle.com/ffmpeg/

* **Assignment:** Then, you can easily assign the custom path to the folder containing FFmpeg executables(`for e.g 'ffmpeg/bin'`) or path of `ffmpeg` executable itself to the [**`custom_ffmpeg`**](../../params/#custom_ffmpeg) parameter in the WriteGear API.


    !!! warning "If binaries were not found at the manually specified path, WriteGear API will disable the Compression Mode!"

   
&nbsp;

