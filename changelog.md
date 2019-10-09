# CHANGELOG

## VidGear 0.1.6-dev

### New Features:
  * Added powerful ZMQ Authentication & Data Encryption features for NetGear API:
    * Added exclusive `secure_mode` param for enabling it.
    * Added support for two most powerful `Stonehouse` & `Ironhouse` ZMQ security mechanisms.
    * Added smart auth-certificates/key generation and validation features.
  * Implemented Robust Multi-Server support for NetGear API.
    * Enables Multiple Servers messaging support with a single client.
    * Added exclusive `multiserver_mode` param for enabling it.
    * Added ability to send additional data of any datatype along with the frame in realtime in this mode.
  * Implemented new *Publish/Subscribe(`zmq.PUB/zmq.SUB`)* pattern for seamless Live Streaming.
  * Added VidGear's official native support for MacOS environment.

### Updates/Improvements:
  * Updated support for screen casting from all monitors in ScreenGear API.
  * Updated ScreenGear API to use *Threaded Queue Mode* by default, thereby removed redundant `THREADED_QUEUE_MODE` param.
  * Updated Tests bash scripts to use system-specific **Temp** directory instead of **Home** for downloading content.
  * Updated Wiki-Documentation with latest examples and Information.
  * Updated Travis CLI Tests with support for macOS environment
  * Reformatted & implemented necessary MacOS related changes and dependencies in `travis.yml`.

### Breaking Updates / Improvements / Changes
  * `Python 2.7` legacy support removed from CLI tests.
  * Newly implemented `secure_mode` will only support Python 3 and above legacies.

### Fixes
  * Fixed assigned Port address ignored bug (commit 073bca1)
  * Fixed several wrong definition bugs from NetGear API(commit 8f7153c).
  * Fixed unreliable dataset video URL(rehosted file on `github.com`)
  * Removed duplicate code to import MSS(@BoboTiG) from ScreenGear API.
  * Fixed Several bugs related to new `secure_mode` & `multiserver_mode` Modes.
  * Fixed various macOS environment bugs

### Pull requests(PR) involved:
  * PR #39
  * PR #42
  * PR #44
  * PR #52

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