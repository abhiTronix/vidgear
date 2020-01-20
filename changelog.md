# CHANGELOG

## VidGear 0.1.7-dev

### New Features:

  * **WebGear Updates:**
    * Added a robust Live Video Streaming Server API that can transfer live video frames to any web browser on the network in real-time.
    * Implemented a flexible asyncio wrapper around Starlette ASGI application and can easily access its various components independently.
    * Added seamless access to various Starlette Response classes, Routing tables, Static Files, Templating engine(with Jinja2), etc.
    * Created a special internal wrapper around VideoGear API and gave access to all its properties.
    * Introduced new Auto-Generation Process to generate WebGear data files from the GitHub server and validate them automatically.
    * Added exclusive dictionary attributes to tweak WebGear performance, Route Tables and other internal properties.
    * Added new simple & elegant Bootstrap's Cover template, by @mdo for WebGear Server.
    * Implemented a new reducer function to reduce the frame size on-the-go.
    * Built `__main__.py` to directly run WebGear Server through the terminal.
    
  * **Logging Updates:**
    * Added new `logging handler` helper function.

### Updates/Improvements:
  
  * Add flake8 tests to Travis CLI to find undefined names. (@cclauss )
  * Removed PiGear from CLI completely  due to hardware emulation limitation.
  * Defined and isolated each API & test loggers and defined them globally to avoid copies. 
  * Limited Importing OpenCV API version check's scope to `helper.py` only.
  * Added case for incorrect `color_space` value in ScreenGear API.
  * Added `uvicorn` as an optional dependency to run WebGear as a live server.
  * Removed conflicting old system-wide logging formatter.
  * Added encapsulation for class functions and variables in WebGear API.
  * Improved and added `shutdown` function for safely stopping frame producer threads.
  * Replaced scratch progress bar with exclusive `TQDM` library's pbar.
  * Added various CLI tests for WebGear API.
  * Various Docs Updates.

### Breaking Updates / Improvements / Changes

  * :warning: Dropped support for Python 3.5 and below legacies.
  * Replaced python 3.5 matrices with python 3.8 matrices in Windows & macOS environment.

### Fixes:

  * `self` is an undefined name in `helper.py` context which will raise NameError. (@cclauss)
  * Fixed incorrect logic for pulling frames in ScreenGear API.
  * Added missing `camera_num` parameter to VideoGear.
  * Added OpenSSL's [SSL: CERTIFICATE_VERIFY_FAILED] bug workaround for macOS envs.
  * Removed videocapture benchmark tests for macOS envs.
  * Removed trivial `__main__.py` from codecov
  * Fixed various CLI tests for CamGear, NetGear & VideoGear API.



### Pull requests(PR) involved:

#88
#91
#95
#98

:warning: PyPi Release does NOT contain Tests and Scripts!

&nbsp; 

## VidGear 0.1.6

### New Features:
  * **NetGear API:**
    * Added powerful ZMQ Authentication & Data Encryption features for NetGear API:
      * Added exclusive `secure_mode` param for enabling it.
      * Added support for two most powerful `Stonehouse` & `Ironhouse` ZMQ security mechanisms.
      * Added smart auth-certificates/key generation and validation features
    * Implemented Robust Multi-Server support for NetGear API:
      * Enables Multiple Servers messaging support with a single client.
      * Added exclusive `multiserver_mode` param for enabling it.
      * Added support for `REQ/REP` &  `PUB/SUB` patterns for this mode.
      * Added ability to send additional data of any datatype along with the frame in realtime in this mode.
    * Introducing exclusive Bi-Directional Mode for bidirectional data transmission:
      * Added new `return_data` parameter to `recv()` function.
      * Added new `bidirectional_mode` attribute for enabling this mode.
      * Added support for `PAIR` & `REQ/REP` patterns for this mode
      * Added support for sending data of any python datatype.
      * Added support for `message` parameter for non-exclusive primary modes for this mode
    * Implemented compression support with on-the-fly flexible frame encoding for the Server-end:
      * Added initial support for `JPEG`, `PNG` & `BMP` encoding formats 
      * Added exclusive options attribute `compression_format` & `compression_param` to tweak this feature
      * Client-end will now decode frame automatically based on the encoding as well as support decoding flags
    * Added `force_terminate` attribute flag for handling force socket termination at the Server-end if there's latency in the network. 
    * Implemented new *Publish/Subscribe(`zmq.PUB/zmq.SUB`)* pattern for seamless Live Streaming in NetGear API.

  * **PiGear API:**
    * Added new threaded internal timing function for PiGear to handle any hardware failures/frozen threads.
    * PiGear will not exit safely with `SystemError` if Picamera ribbon cable is pulled out to save resources.
    * Added support for new user-defined `HWFAILURE_TIMEOUT` options attribute to alter timeout.

  * **VideoGear API:** 
    * Added `framerate` global variable and removed redundant function.
    * Added `CROP_N_ZOOM` attribute in Videogear API for supporting Crop and Zoom stabilizer feature.

  * **WriteGear API:** Added new `execute_ffmpeg_cmd` function to pass a custom command to its FFmpeg pipeline.

  * **Stabilizer class:** 
    * Added new Crop and Zoom feature.
      * Added `crop_n_zoom` param for enabling this feature.
    * Updated docs.

  * **CLI & Tests updates:**
    * Replaced python 3.5 matrices with latest python 3.8 matrices in Linux environment.
    * Added full support for **Codecov** in all CLI environments.
    * Updated OpenCV to v4.2.0-pre(master branch). 
    * Added various Netgear API tests.
    * Added initial Screengear API test.
    * More test RTSP feeds added with better error handling in CamGear network test.
    * Added tests for ZMQ authentication certificate generation.
    * Added badge and Minor doc updates.

  * Added VidGear's official native support for MacOS environments.
    

### Updates/Improvements:
  * Replace `print` logging commands with python's logging module completely.
  * Implemented encapsulation for class functions and variables on all gears.
  * Updated support for screen casting from multiple/all monitors in ScreenGear API.
  * Updated ScreenGear API to use *Threaded Queue Mode* by default, thereby removed redundant `THREADED_QUEUE_MODE` param.
  * Updated bash script path to download test dataset in `$TMPDIR` rather than `$HOME` directory for downloading testdata.
  * Implemented better error handling of colorspace in various videocapture APIs.
  * Updated bash scripts, Moved FFmpeg static binaries to `github.com`.
  * Updated bash scripts, Added additional flag to support un-secure apt sources.
  * CamGear API will now throw `RuntimeError` if source provided is invalid.
  * Updated threaded Queue mode in CamGear API for more robust performance.
  * Added new `camera_num` to support multiple Picameras.
  * Moved thread exceptions to the main thread and then re-raised.
  * Added alternate github mirror for FFmpeg static binaries auto-installation on windows oses.
  * Added `colorlog` python module for presentable colored logging.
  * Replaced `traceback` with `sys.exc_info`.
  * Overall APIs Code and Docs optimizations.
  * Updated Code Readability and Wiki Docs.
  * Updated ReadMe & Changelog with the latest changes.
  * Updated Travis CLI Tests with support for macOS environment.
  * Reformatted & implemented necessary MacOS related changes and dependencies in `travis.yml`.

### Breaking Updates / Improvements / Changes
  * :warning: Python 2.7 legacy support dropped completely.
  * :warning: Source-code Relicensed to Apache 2.0 License.
  * Python 3+ are only supported legacies for installing Vidgear v0.1.6 and above.
  * Python 2.7 and 3.4 legacies support dropped from VidGear CLI tests.

### Fixes
  * Reimplemented `Pub/Sub` pattern for smoother performance on various networks.
  * Fixed Assertion error in CamGear API during colorspace manipulation.
  * Fixed random freezing in `Secure Mode` and several related performance updates
  * Fixed `multiserver_mode` not working properly over some networks.
  * Fixed assigned Port address ignored bug (commit 073bca1).
  * Fixed several wrong definition bugs from NetGear API(commit 8f7153c).
  * Fixed unreliable dataset video URL(rehosted file on `github.com`).
  * Disabled `overwrite_cert` for client-end in NetGear API.
  * Disabled Universal Python wheel builds in `setup.cfg `file.
  * Removed duplicate code to import MSS(@BoboTiG) from ScreenGear API.
  * Eliminated unused redundant code blocks from library.
  * Fixed Code indentation in `setup.py` and updated new release information.
  * Fixed code definitions & Typos.
  * Fixed several bugs related to `secure_mode` & `multiserver_mode` Modes.
  * Fixed various macOS environment bugs.

### Pull requests(PR) involved:
  * PR #39
  * PR #42
  * PR #44
  * PR #52
  * PR #55
  * PR #62
  * PR #67
  * PR #72
  * PR #77
  * PR #78
  * PR #82
  * PR #84

:warning: PyPi Release does NOT contain Tests and Scripts!

&nbsp; 

## VidGear v0.1.5

### New Features:
  * Added new ScreenGear API, supports Live ScreenCasting.
  * Added new NetGear API, aids real-time frame transfer through messaging(ZmQ) over network.
  * Added new new Stabilizer Class, for minimum latency Video Stabilization with OpenCV.
  * Added Option to use VidGear API's standalone.
  * Added Option to use VideoGear API as internal wrapper around Stabilizer Class.
  * Added new parameter `stabilize` to VidGear API, to enable or disable Video Stabilization.
  * Added support for `**option` dict attributes to update VidGear's video stabilizer parameters directly. 
  * Added brand new VidGear logo and functional block diagram (`.svg`) in readme.md
  * Added new pictures and GIFs for improving readme.md readability 
  * Added new `contributing.md` and `changelog.md` for reference.
  * Added `collections.deque` import in Threaded Queue Mode for performance consideration
  * Added new `install_opencv.sh` bash scripts for Travis cli, to handle OpenCV installation.
  * Added new Project Issue & PR Templates
  * Added new Sponsor Button(`FUNDING.yml`)

### Updates/Improvements:
  * Updated New dependencies: `mss`, `pyzmq` and rejected redundant ones.
  * Revamped and refreshed look for `readme.md` and added new badges.
  * Updated Releases Documentation completely.
  * Updated CLI tests for new changes
  * Updated Code Documentation.
  * Updated bash scripts and removed redundant information
  * Updated `Youtube video` URL in tests
  * Completely Reformatted and Updated Wiki Docs with new changes.


### Breaking Updates / Improvements / Changes
  * Implemented experimental Threaded Queue Mode(_a.k.a Blocking Mode_) for fast, synchronized, error-free multi-threading.
  * Renamed bash script `pre-install.sh` to `prepare_dataset.sh` - downloads opensourced test datasets and static FFmpeg binaries for debugging.
  * Changed `script` folder location to `bash/script`.
  * `Python 3.4` removed from Travis CLI tests.

### Fixes
  * Temporarily fixed Travis CLI bug: Replaced `opencv-contrib-python` with OpenCV built from scratch as dependency.
  * Fixed CLI Timeout Bug: Disable Threaded Queue Mode for CLI Tests
  * Fixes `sys.stderr.close()` throws ValueError bug: Replaced `sys.close()` with `DEVNULL.close()`
  * Fixed Youtube Live Stream bug that return `NonType` frames in CamGear API.
  * Fixed `NoneType` frames bug in  PiGear class on initialization.
  * Fixed Wrong function definitions
  * Removed `/xe2` unicode bug from Stabilizer class.
  * Fixed `**output_params` _KeyError_ bug in WriteGear API
  * Fixed subprocess not closing properly on exit in WriteGear API.
  * Fixed bugs in ScreenGear: Non-negative `monitor` values
  * Fixed missing import, typos, wrong variable definitions
  * Removed redundant hack from `setup.py`
  * Fixed Minor YouTube playback Test CLI Bug 
  * Fixed new Twitter Intent
  * Fixed bug in bash script that not working properly due to changes at server end.

### Pull requests(PR) involved:
  * PR #17
  * PR #21
  * PR #22
  * PR #27
  * PR #31
  * PR #32
  * PR #33
  * PR #34

:warning: PyPi Release does NOT contain Tests and Scripts!

&nbsp; 

## VidGear v0.1.4

### New Features:
  * Added new WriteGear API: for enabling lossless video encoding and compression(built around FFmpeg and OpenCV Video Writer)
  * Added YouTube Mode for direct Video Pipelining from YouTube in CamGear API
  * Added new `y_tube` to access _YouTube Mode_ in CamGear API.
  * Added flexible Output file Compression control capabilities in compression-mode(WriteGear).
  * Added `-output_dimensions` special parameter to WriteGear API.
  * Added new `helper.py` to handle special helper functions.
  * Added feature to auto-download and configure FFmpeg Static binaries(if not found) on Windows platforms.
  * Added `-input_framerate` special parameter to WriteGear class to change/control output constant framerate in compression mode(WriteGear).
  * Added new Direct Video colorspace Conversion capabilities in CamGear and PiGear API.
  * Added new `framerate` class variable for CamGear API, to retrieve input framerate.
  * Added new parameter `backend` - changes the backend of CamGear's API
  * Added automatic required prerequisites installation ability, when installation from source.
  * Added Travis CLI Complete Integration for Linux-based Testing for VidGear.
  * Added and configured `travis.yml`
  * Added Appveyor CLI Complete Integration for Windows-based Testing in VidGear.
  * Added and configured new `appveyor.yml`
  * Added new bash script `pre-install.sh` to download opensourced test datasets and static FFmpeg binaries for debugging.
  * Added several new Tests(including Benchmarking Tests) for each API for testing with `pytest`.
  * Added license to code docs.
  * Added `Say Thank you!` badge to readme.

### Updates/Improvements:
  * Removed redundant dependencies
  * Updated `youtube-dl` as a dependency, as required by `pafy`'s backend.
  * Updated common VideoGear API with new parameter.
  * Update robust algorithm to auto-detect FFmpeg executables and test them, if failed, auto fallback to OpenCV's VideoWriter API. 
  * Improved system previously installed OpenCV detection in setup.py.
  * Updated setup.py with hack to remove bullets from pypi description. 
  * Updated Code Documentation
  * Reformatted & Modernized readme.md with new badges.
  * Reformatted and Updated Wiki Docs.

### Breaking Updates / Improvements / Changes
  * Bugs Patched: Removed unnecessary `-height` and `-width` parameter from CamGear API.
  * Replaced dependency `opencv-python` with `opencv-contrib-python` completely

### Fixes
  * Windows Cross-Platform fix: replaced dependency `os` with `platform` in setup.py.
  * Fixed Bug: Arises due to spaces in input `**options`/`**output_param` dictionary keys.
  * Fixed several wrong/missing variable & function definitions.
  * Fixed code uneven indentation.
  * Fixed several typos in docs.

### Pull requests(PR) involved:
  * PR #7
  * PR #8
  * PR #10
  * PR #12

:warning: PyPi Release does NOT contain Tests and Scripts!

&nbsp; 

## VidGear v0.1.3

### Fixes
  * Patched Major PiGear Bug: Incorrect import of PiRGBArray function in PiGear Class
  * Several fixes for backend `picamera` API handling during frame capture(PiGear)
  * Fixed missing frame variable initialization.
  * Fixed minor typos

### Pull requests(PR) Involved:
  * PR #6
  * PR #5

&nbsp; 

## VidGear v0.1.2

### New Features:
  * Added easy Source manipulation feature in CamGear API, to control features like `resolution, brightness, framerate etc.`
  * Added new `**option` parameter to CamGear API, provides the flexibility to manipulate input stream directly.
  * Added new parameters for Camgear API for time delay and logging.
  * Added new Logo to readme.md
  * Added new Wiki Documentation.

### Updates/Improvements:
  * Reformatted readme.md.
  * Updated Wiki Docs with new changes.


### Fixes:
  * Improved Error Handling in CamGear & PiGear API.
  * Fixed minor typos in docs.
    
### Pull requests(PR) Involved:

  * PR #4

&nbsp;

## VidGear v0.1.1

### New Features:
  * Release ViGear binaries on the Python Package Index (PyPI)
  * Added new and configured `setup.py` & `setup.cfg`

### Fixes:
  * Fixed PEP bugs: added and configured properly `__init__.py` in each folder 
  * Fixed PEP bugs: improved code Indentation
  * Fixed wrong imports: replaced `distutils.core` with `setuptools`
  * Fixed readme.md


&nbsp;

## VidGear v0.1.0

### New Features:
  * Initial Release
  * Converted [my `imutils` PR](https://github.com/jrosebr1/imutils/pull/105) into VidGear Python Project.
  * Renamed conventions and reformatted complete source-code from scratch.
  * Added support for both python 2.7 and 3 legacies
  * Added new multi-threaded CamGear, PiGear, and VideoGear APIs
  * Added multi-platform compatibility
  * Added robust & flexible control over the source in PiGear API.