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

# Release Notes

## v0.1.9-dev (in Progress)

### New Features

- **StreamGear API:**

    * [X] New API that automates transcoding workflow for generating Ultra-Low Latency, High-Quality, Dynamic & Adaptive Streaming Formats.
    * [X] Implemented multi-platform , standalone, highly extensible and flexible wrapper around FFmpeg for generating chunked-encoded media segments of the media, and easily accessing almost all of its parameters.
    * [X] API automatically transcodes videos/audio files & real-time frames into a sequence of multiple smaller chunks/segments and also creates a Manifest file.
    * [X] Added initial support for [MPEG-DASH](https://www.encoding.com/mpeg-dash/) _(Dynamic Adaptive Streaming over HTTP, ISO/IEC 23009-1)_.
    * [X] Constructed default behavior in StreamGear, for auto-creating a Primary Stream of same resolution and framerate as source.
    * [X] Added [TQDM](https://github.com/tqdm/tqdm) progress bar in non-debugged output for visual representation of internal processes.
    * [X] Implemented several internal methods for preprocessing FFmpeg and internal parameters for producing streams.
    * [X] Several standalone internal checks to ensure robust performance.
    * [X] New [`terminate()`](../bonus/reference/streamgear/#vidgear.gears.streamgear.StreamGear.terminate) function to terminate StremGear Safely.

    * [X] New StreamGear Dual Modes of Operation:
        + Implemented *Single-Source* and *Real-time Frames* like independent Transcoding Modes.
        + Linked `-video_source` attribute for activating these modes
        + **Single-Source Mode**, transcodes entire video/audio file _(as opposed to frames by frame)_ into a sequence of multiple smaller segments for streaming
        + **Real-time Frames Mode**, directly transcodes video-frames _(as opposed to a entire file)_, into a sequence of multiple smaller segments for streaming
        + Added separate functions, [`stream()`](../bonus/reference/streamgear/#vidgear.gears.streamgear.StreamGear.stream) for Real-time Frame Mode and [`transcode_source()`](../bonus/reference/streamgear/#vidgear.gears.streamgear.StreamGear.transcode_source) for Single-Source Mode for easy transcoding.
        + Included auto-colorspace detection and RGB Mode like features _(extracted from WriteGear)_, into StreamGear.  

    * [X] New StreamGear Parameters:
        + Developed several new parameters such as:
            + `output`: handles assets directory
            + `formats`: handles adaptive HTTP streaming format.
            + `custom_ffmpeg`: handles custom FFmpeg location.
            + `stream_params`: handles internal and FFmpeg parameter seamlessly.
            + `logging`: turns logging on or off.
        + New `stream_params` parameter allows us to exploit almost all FFmpeg parameters and flexibly change its internal settings, and seamlessly generating high-quality streams with its attributes:
            + `-streams` _(list of dictionaries)_ for building additional streams with `-resolution`, `-video_bitrate` & `-framerate` like sub-attributes.
            + `-audio` for specifying external audio.
            + `-video_source` for specifying Single-Source Mode source.
            + `-input_framerate` for handling input framerate in Real-time Frames Mode.
            + `-bpp` attribute for handling bits-per-pixels used to auto-calculate video-bitrate.
            + `-gop` to manually specify GOP length.
            + `-ffmpeg_download_path` to handle custom FFmpeg download path on windows.
            + `-clear_prev_assets` to remove any previous copies of SteamGear Assets.

    * [X] New StreamGear docs, MPEG-DASH demo, and recommended DASH players list:
        + Added new StreamGear docs, usage examples, parameters, references, new FAQs.
        + Added Several StreamGear usage examples w.r.t Mode of Operation.
        + Implemented [**Clappr**](https://github.com/clappr/clappr) based on [**Shaka-Player**](https://github.com/google/shaka-player), as Demo Player.
        + Added Adaptive-dimensional behavior for Demo-player, purely in css.
        + Hosted StreamGear generated DASH chunks on GitHub and served with `raw.githack.com`.
        + Introduced variable quality level-selector plugin for Clapper Player.
        + Provide various required javascripts and implemented additional functionality for player in `extra.js`.
        + Recommended tested Online, Command-line and GUI Adaptive Stream players.
        + Implemented separate FFmpeg installation doc for StreamGear API.
        + Reduced `rebufferingGoal` for faster response.

    * [X] New StreamGear CI tests:
        + Added IO and API initialization CI tests for its Modes.
        + Added various mode Streaming check CI tests.

- **NetGear_Async API:**
    * [X] Added new `send_terminate_signal` internal method.
    * [X] Added `WindowsSelectorEventLoopPolicy()` for windows 3.8+ envs.
    * [X] Moved Client auto-termination to separate method.
    * [X] Implemented graceful termination with `signal` API on UNIX machines.
    * [X] Added new `timeout` attribute for controlling Timeout in Connections.
    * [X] Added missing termination optimizer (`linger=0`) flag.
    * [X] Several ZMQ Optimizer Flags added to boost performance.

- **WriteGear API:**

    * [X] Added support for adding duplicate FFmpeg parameters to `output_params`:
        + Added new `-clones` attribute in `output_params` parameter for handing this behavior..
        + Support to pass FFmpeg parameters as list, while maintaining the exact order it was specified.
        + Built support for `zmq.REQ/zmq.REP` and `zmq.PUB/zmq.SUB` patterns in this mode.
        + Added new CI tests debugging this behavior.
        + Updated docs accordingly.
    * [X] Added support for Networks URLs in Compression Mode:
        + `output_filename` parameter supports Networks URLs in compression modes only
        + Added automated handling of non path/file Networks URLs as input.
        + Implemented new `is_valid_url` helper method to easily validate assigned URLs value.
        + Validates whether the given URL value has scheme/protocol supported by assigned/installed ffmpeg or not. 
        + WriteGear will throw `ValueError` if `-output_filename` is not supported.
        + Added related CI tests and docs.
    * [X] Added `disable_force_termination` attribute in WriteGear to disable force-termination.

- **NetGear API:**

    * [X] Added option to completely disable Native Frame-Compression:
        + Checks if any Incorrect/Invalid value is assigned on `compression_format` attribute.
        + Completely disables Native Frame-Compression.
        + Updated docs accordingly.

- **CamGear API:**
    * [X] Added new and robust regex for identifying YouTube URLs.
    * [X] Moved `youtube_url_validator` to Helper.

- **New `helper.py` methods:** 
    * [X] Added `validate_video` function to validate video_source.
    * [X] Added `extract_time` Extract time from give string value.
    * [X] Added `get_video_bitrate` to caliculate video birate from resolution, framerate, bits-per-pixels values.
    * [X] Added `delete_safe` to safely delete files of given extension.
    * [X] Added `validate_audio` to validate audio source.
    * [X] Added new Helper CI tests.
        + Added new `check_valid_mpd` function to test MPD files validity.
        + Added `mpegdash` library to CI requirements.

- **Deployed New Docs Upgrades:**
    * [X] Added new assets like _images, gifs, custom scripts, javascripts fonts etc._ for achieving better visual graphics in docs.
    * [X] Added `clappr.min.js`, `dash-shaka-playback.js`, `clappr-level-selector.min.js` third-party javascripts locally.
    * [X] Extended Overview docs Hyperlinks to include all major sub-pages _(such as Usage Examples, Reference, FAQs etc.)_.
    * [X] Replaced GIF with interactive MPEG-DASH Video Example in Stabilizer Docs. 
    * [X] Added new `pymdownx.keys` to replace `[Ctrl+C]/[⌘+C]` formats.
    * [X] Added new `custom.css` stylescripts variables for fluid animations in docs.
    * [X] Overridden announce bar and added donation button. 
    * [X] Lossless WEBP compressed all PNG assets for faster loading.
    * [X] Enabled lazy loading for GIFS for Performance Improvements.
    * [X] Reimplemented Admonitions contexts and added new ones.
    * [X] Added StreamGear and its different modes Docs Assets.
    * [X] Added patch for images & unicodes for PiP flavored markdown in `setup.py`.

- **Added `Request Info` and `Welcome` GitHub Apps to automate PR and issue workflow**
    * [X] Added new `config.yml` for customizations.
    * [X] Added various suitable configurations.

- Added new `-clones` attribute to handle FFmpeg parameter clones in StreamGear and WriteGear API.
- Added new Video-only and Audio-Only sources in bash script.
- Added new paths in bash script for storing StreamGear & WriteGear assets temporarily.

### Updates/Improvements

- [X] Added patch for `NotImplementedError` in NetGear_Async API on Windows 3.8+ envs.
- [X] Check for valid `output` file extension according to `format` selected in StreamGear.
- [X] Completed migration to `travis.com`.
- [X] Created new `temp_write` temp directory for WriteGear Assets in bash script.
- [X] Deleted old Redundant assets and added new ones.
- [X] Employed `isort` library to sort and group imports in Vidgear APIs.
- [X] Enabled exception for `list, tuple, int, float` in WriteGear API's `output_params` dict.
- [X] Enabled missing support for frame-compression in its primary Receive Mode.
- [X] Enforced pixel formats for streams.
- [X] Improved check for valid system path detection in WriteGear API.
- [X] Overrided `pytest-asyncio` fixture in NetGear_Async API.
- [X] Quoted Gear Headline for understanding each gear easily. 
- [X] Re-Positioned Gear's banner images in overview for better readability.
- [X] Reduced redundant try-except blocks in NetGear Async.
- [X] Reformatted and Simplified Docs context.
- [X] Reimplemented `return_testvideo_path` CI function with variable streams.
- [X] Reimplemented `skip_loop` in NetGear_Async to fix `asyncio.CancelledError`.
- [X] Reimplemented buggy audio handler in StreamGear.
- [X] Reimplemented images with `<figure>` and `<figurecaption>` like tags.
- [X] Removed Python < 3.8 condition from all CI tests.
- [X] Removed or Grouped redundant code for increasing codecov.
- [X] Removed redundant code and simplified algorithmic complexities in Gears.
- [X] Replaced `;nbsp` with `;thinsp` and `;emsp`.
- [X] Replaced `IOError` with more reliable `RuntimeError` in StreamGear Pipelines.
- [X] Replaced `del` with `pop` in dicts.
- [X] Replaced all Netgear CI tests with more reliable `try-except-final` blocks.
- [X] Replaced simple lists with `pymdownx.tasklist`.
- [X] Replaced subprocess `call()` with `run()` for better error handling in `execute_ffmpeg_cmd` function.
- [X] Resized over-sized docs images. 
- [X] Simplified `delete_safe` Helper function.
- [X] Simplified default audio-bitrate logic in StreamGear
- [X] Updated CI tests and cleared redundant code from NetGear_Async API.
- [X] Updated CI with new tests and Bumped Codecov.
- [X] Updated Issue and PR templates.
- [X] Updated Licenses for new files and shrink images dimensions.
- [X] Updated Missing Helpful tips and increased logging.
- [X] Updated PR guidelines for more clarity.
- [X] Updated WebGear examples addresses from `0.0.0.0` to `localhost`.
- [X] Updated WriteGear and StreamGear CI tests for not supporting temp directory.
- [X] Updated `README.md` and `changelog.md` with new changes.
- [X] Updated `check_output` and added `force_retrieve_stderr` support to `**kwargs` to extract `stderr` output even on FFmpeg  error.
- [X] Updated `dicts2args` to support internal repeated `coreX` FFmpeg parameters for StreamGear. 
- [X] Updated `mkdocs.yml`, `changelog.md` and `README.md` with latest changes.
- [X] Updated `validate_audio` Helper function will now retrieve audio-bitrate for validation.
- [X] Updated buggy `mpegdash` dependency with custom dev fork for Windows machines.
- [X] Updated core parameters for audio handling.
- [X] Updated logging for debugging selected eventloops in NetGear_Async API.
- [X] Updated termination linger to zero at Server's end.

### Breaking Updates/Changes

- [X] :warning: Changed Webgear API default address to `localhost` for cross-compatibility between different platforms.
- [X] In Netgear_Async API, `source` value can now be NoneType for a custom frame-generator at Server-end only.
- [X] Temp_(such as `/tmp` in linux)_ is now not a valid directory for WriteGear & StreamGear API outputs.
- [X] Moved vidgear docs assets _(i.e images, gifs, javascripts and stylescripts)_ to `override` directory.

### Bug-fixes

- [X] Added workaround for system path not handle correctly.
- [X] Fixed Bug: URL Audio format not being handled properly.
- [X] Fixed Critical Bug in NetGear_Async throwing `ValueError` with None-type Source.
- [X] Fixed Critical StreamGear Bug: FFmpeg pipeline terminating prematurely in Single-Source Mode.
- [X] Fixed Critical external audio handler bug: moved audio-input to input_parameters.
- [X] Fixed Frozen-threads bug in CI tests.
- [X] Fixed Mkdocs only accepting Relative paths.
- [X] Fixed OSError in WriteGear's compression mode.
- [X] Fixed StreamGear CI bugs for Windows and CI envs.
- [X] Fixed Typos and Indentation bugs in NetGear API.
- [X] Fixed ZMQ throwing error on termination if all max-tries exhausted.
- [X] Fixed `NameError` bug in NetGear API.
- [X] Fixed `NameError` bugs in StreamGear CI.
- [X] Fixed `NameError` in CI functions and tests.
- [X] Fixed `TimeoutError` bug in NetGear_Async CI tests.
- [X] Fixed `get_valid_ffmpeg_path` throwing `TypeError` with non-string values.
- [X] Fixed broken links in docs. 
- [X] Fixed critical duplicate logging bug.
- [X] Fixed default `gop` value not handle correctly.
- [X] Fixed handling of incorrect paths detection.
- [X] Fixed incorrect definitions in NetGear_Async.
- [X] Fixed left-over attribute bug in WriteGear.
- [X] Fixed logic and indentation bugs in CI tests.
- [X] Fixed logic for handling output parameters in WriteGear API.
- [X] Fixed missing definitions and logic bug in StreamGear.
- [X] Fixed missing import and incorrect CI definitions. 
- [X] Fixed missing source dimensions from `extract_resolutions` output in StreamGear API.
- [X] Fixed missing support for compression parameters in Multi-Clients Mode.
- [X] Fixed round off error in FPS.
- [X] Fixed several CI bugs and updated `extract_resolutions` method.
- [X] Fixed several bugs from CI Bidirectional Mode tests.
- [X] Fixed several typos in docs usage examples.
- [X] Fixed various `AttributeError` with wrong attribute names and definition in CI Helper functions.
- [X] Fixed wrong and missing definitions in docs.
- [X] Fixed wrong logic for extracting OpenCV frames.
- [X] Fixed wrong type bug in StreamGear API.
- [X] Fixed wrong type error bug in WriteGear API.
- [X] Fixed wrong variable assignments bug in WriteGear API.
- [X] Fixes to CLI tests and missing docs imports.
- [X] Many minor typos and wrong definitions.

### Pull Requests

* PR #129
* PR #130
* PR #155


&nbsp; 

&nbsp; 

## v0.1.8 (2020-06-12)

### New Features

- **Multiple Clients support in NetGear API:**

    * [X] Implemented support for handling any number of Clients simultaneously with a single Server in this mode.
    * [X] Added new `multiclient_mode` attribute for enabling this mode easily.
    * [X] Built support for `zmq.REQ/zmq.REP` and `zmq.PUB/zmq.SUB` patterns in this mode.
    * [X] Implemented ability to receive data from all Client(s) along with frames with `zmq.REQ/zmq.REP` pattern only.
    * [X] Updated related CI tests

- **Support for robust Lazy Pirate pattern(auto-reconnection) in NetGear API for both server and client ends:**

    * [X] Implemented a algorithm where NetGear rather than doing a blocking receive, will now:
        + Poll the socket and receive from it only when it's sure a reply has arrived.
        + Attempt to reconnect, if no reply has arrived within a timeout period.
        + Abandon the connection if there is still no reply after several requests.
    * [X] Implemented its default support for `REQ/REP` and `PAIR` messaging patterns internally.
    * [X] Added new `max_retries` and `request_timeout`(in seconds) for handling polling.
    * [X] Added `DONTWAIT` flag for interruption-free data receiving.
    * [X] Both Server and Client can now reconnect even after a premature termination.

- **Performance Updates for NetGear API:**

    * [X] Added default Frame Compression support for Bidirectional frame transmission in Bidirectional mode.
    * [X] Added support for `Reducer()` function in Helper.py to aid reducing frame-size on-the-go for more performance.
    * [X] Added small delay in `recv()` function at client's end to reduce system load. 
    * [X] Reworked and Optimized NetGear termination, and also removed/changed redundant definitions and flags.

- **Docs Migration to Mkdocs:**

    * [X] Implemented a beautiful, static documentation site based on [MkDocs](https://www.mkdocs.org/) which will then be hosted on GitHub Pages.
    * [X] Crafted base mkdocs with third-party elegant & simplistic [`mkdocs-material`](https://squidfunk.github.io/mkdocs-material/) theme.
    * [X] Implemented new `mkdocs.yml` for Mkdocs with relevant data.
    * [X] Added new `docs` folder to handle markdown pages and its assets.
    * [X] Added new Markdown pages(`.md`) to docs folder, which are carefully crafted documents - based on previous Wiki's docs, and some completely new additions.
    * [X] Added navigation under tabs for easily accessing each document.

    * [X] **New Assets:**
        + Added new assets like _gifs, images, custom scripts, favicons, site.webmanifest etc._ for bringing standard and quality to docs visual design.
        + Designed brand new logo and banner for VidGear Documents.
        + Deployed all assets under separate [*Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License*](https://creativecommons.org/licenses/by-nc-sa/4.0/).

    * [X] **Added Required Plugins and Extensions:**
        + Added support for all [pymarkdown-extensions](https://facelessuser.github.io/pymdown-extensions/).
        + Added support for some important `admonition`, `attr_list`, `codehilite`, `def_list`, `footnotes`, `meta`, and `toc` like Mkdocs extensions.
        + Enabled `search`, `minify` and `git-revision-date-localized` plugins support.
        + Added various VidGear's social links to yaml.
        + Added support for `en` _(English)_ language.

    * [X] **Auto-Build API Reference with `mkdocstrings:`**
        + Added support for [`mkdocstrings`](https://github.com/pawamoy/mkdocstrings) plugin for auto-building each VidGear's API references.
        + Added python handler for parsing python source-code to `mkdocstrings`.

    * [X] **Auto-Deploy Docs with Github Actions:**
        + Implemented Automated Docs Deployment on gh-pages through GitHub Actions workflow.
        + Added new workflow yaml with minimal configuration for automated docs deployment.
        + Added all required  python dependencies and environment for this workflow.
        + Added `master` branch on Ubuntu machine to build matrix.

### Updates/Improvements

- [X] Added in-built support for bidirectional frames(`NDarray`) transfer in Bidirectional mode.
- [X] Added support for User-Defined compression params in Bidirectional frames transfer.
- [X] Added workaround for `address already in use` bug at client's end.
- [X] Unified Bidirectional and Multi-Clients mode for client's return data transmission.
- [X] Replaced `ValueError` with more suitable `RuntimeError`.
- [X] Updated logging for better readability.
- [X] Added CI test for Multi-Clients mode.
- [X] Reformatted and grouped imports in VidGear.
- [X] Added `Reducer` Helper function CI test.
- [X] Added Reliability tests for both Server and Client end.
- [X] Disabled reliable reconnection for Multi-Clients mode.
- [X] Replaced `os.devnull` with suprocess's inbuilt function.
- [X] Updated README.md, Issue and PR templates with new information and updates.
- [X] Moved `changelog.md` to `/docs` and updated contribution guidelines.
- [X] Improved source-code docs for compatibility with `mkdocstrings`.
- [X] Added additional dependency `mkdocs-exclude`, for excluding files from Mkdocs builds.
- [X] Updated license and compressed images/diagrams.
- [X] Added new CI tests and Bumped Codecov.
- [X] Changed YouTube video URL for CI tests to Creative Commons(CC) video.
- [X] Removed redundant code.

### Breaking Updates/Changes

!!! danger "VidGear Docs moved to GitHub Pages, Now Available at https://abhitronix.github.io/vidgear."

- [X] Removed `filter` attribute from `options` parameter in NetGear API.
- [X] Removed `force_terminate` parameter support from NetGear API.
- [X] Disabled additional data of datatype `numpy.ndarray` for Server end in Bidirectional Mode.

### Bug-fixes

- [X] Fixed `'NoneType' object is not subscriptable` bug.
- [X] Fixed bugs related to delayed termination in NetGear API.
- [X] Reduced default `request_timeout` value to 4 and also lowered cut-off limit for the same.
- [X] Removed redundant ZMQ context termination and similar variables.
- [X] Added missing VidGear installation in workflow.
- [X] Excluded conflicting assets `README.md` from Mkdocs builds.
- [X] Fixed `pattern` value check bypassed if wrong value is assigned.
- [X] Fixed incorrect handling of additional data transferred in synchronous mode at both Server and Client end.
- [X] Replaced Netgear CI test with more reliable `try-except-final` blocks.
- [X] Updated termination linger to zero at Server's end.
- [X] Fixed `NameError` bug in NetGear API.
- [X] Fixed missing support for compression parameters in Multi-Clients Mode.
- [X] Fixed ZMQ throwing error on termination if all max-tries exhausted.
- [X] Enabled missing support for frame compression in its primary receive mode.
- [X] Fixed several bugs from CI Bidirectional Mode tests.
- [X] Removed or Grouped redundant code for increasing codecov.
- [X] Fixed Mkdocs only accepting Relative paths.
- [X] Fixed broken links in docs. 
- [X] Fixed round off error in FPS.
- [X] Many small typos and bugs fixes.

### Pull Requests

* PR #129
* PR #130


&nbsp; 

&nbsp; 

## v0.1.7 (2020-04-29)

### New Features

- **WebGear API:**
    * [X] Added a robust Live Video Server API that can transfer live video frames to any web browser on the network in real-time.
    * [X] Implemented a flexible asyncio wrapper around [`starlette`](https://www.starlette.io/) ASGI Application Server.
    * [X] Added seamless access to various starlette's Response classes, Routing tables, Static Files, Template engine(with Jinja2), etc.
    * [X] Added a special internal access to VideoGear API and all its parameters.
    * [X] Implemented a new Auto-Generation Work-flow to generate/download & thereby validate WebGear API data files from its GitHub server automatically.
    * [X] Added on-the-go dictionary parameter in WebGear to tweak performance, Route Tables and other internal properties easily.
    * [X] Added new simple & elegant default Bootstrap Cover Template for WebGear Server.
    * [X] Added `__main__.py` to directly run WebGear Server through the terminal.
    * [X] Added new gif and related docs for WebGear API.
    * [X] Added and Updated various CI tests for this API.


- **NetGear_Async API:** 
    * [X] Designed NetGear_Async asynchronous network API built upon ZeroMQ's asyncio API.
    * [X] Implemented support for state-of-the-art asyncio event loop [`uvloop`](https://github.com/MagicStack/uvloop) at its backend.
    * [X] Achieved Unmatchable high-speed and lag-free video streaming over the network with minimal resource constraint.
    * [X] Added exclusive internal wrapper around VideoGear API for this API.
    * [X] Implemented complete server-client handling and options to use variable protocols/patterns for this API.
    * [X] Implemented support for  all four ZeroMQ messaging patterns: i.e `zmq.PAIR`, `zmq.REQ/zmq.REP`, `zmq.PUB/zmq.SUB`, and `zmq.PUSH/zmq.PULL`.
    * [X] Implemented initial support for `tcp` and `ipc` protocols.
    * [X] Added new Coverage CI tests for NetGear_Async Network Gear.
    * [X] Added new Benchmark tests for benchmarking NetGear_Async against NetGear.

- **Asynchronous Enhancements:** 
    * [X] Added `asyncio` package to for handling asynchronous APIs.
    * [X] Moved WebGear API(webgear.py) to `asyncio` and created separate asyncio `helper.py` for it.
    * [X] Various Performance tweaks for Asyncio APIs with concurrency within a single thread.
    * [X] Moved `__main__.py` to asyncio for easier access to WebGear API through the terminal.
    * [X] Updated `setup.py` with new dependencies and separated asyncio dependencies.

- **General Enhancements:**
    
    * [X] Added new highly-precise Threaded FPS class for accurate benchmarking with `time.perf_counter` python module.
    * [X] Added a new [Gitter](https://gitter.im/vidgear/community) community channel.
    * [X] Added a new *Reducer* function to reduce the frame size on-the-go.
    * [X] Add *Flake8* tests to Travis CI to find undefined names. (PR by @cclauss)
    * [X] Added a new unified `logging handler` helper function for vidgear.

### Updates/Improvements
  
- [X] Re-implemented and simplified logic for NetGear Async server-end.
- [X] Added new dependencies for upcoming asyncio updates to `setup.py`.
- [X] Added `retry` function and replaced `wget` with `curl` for Linux test envs. 
- [X] Bumped OpenCV to latest `4.2.0-dev` for Linux test envs.
- [X] Updated YAML files to reflect new changes to different CI envs.
- [X] Separated each API logger with a common helper method to avoid multiple copies. 
- [X] Limited Importing OpenCV API version check's scope to `helper.py` only.
- [X] Implemented case for incorrect `color_space` value in ScreenGear API.
- [X] Removed old conflicting logging formatter with a common method and expanded logging.
- [X] Improved and added `shutdown` function for safely stopping frame producer threads in WebGear API.
- [X] Re-implemented and simplified all CI tests with maximum code-coverage in mind.
- [X] Replaced old `mkdir` function with new `mkdir_safe` helper function for creating directories safely.
- [X] Updated ReadMe.md with updated diagrams, gifs and information.
- [X] Improve, structured and Simplified the Contribution Guidelines.
- [X] Bundled CI requirements in a single command.(Suggested by @cclauss)
- [X] Replaced line endings CRLF with LF endings.
- [X] Added dos2unix for Travis OSX envs.
- [X] Bumped Codecov to maximum. 

### Breaking Updates/Changes

!!! warning "Dropped support for Python 3.5 and below legacies. (See [issue #99](https://github.com/abhiTronix/vidgear/issues/99))"

- [X] Dropped and replaced Python 3.5 matrices with new Python 3.8 matrices in all CI environments.
- [X] Implemented PEP-8 Styled [**Black**](https://github.com/psf/black) formatting throughout the source-code.
- [X] Limited protocols support to `tcp` and `ipc` only, in NetGear API.

### Bug-fixes
  
- [X] Fixed Major NetGear_Async bug where `__address` and `__port` are not set in async mode.(PR by @otter-in-a-suit) 
- [X] Fixed Major PiGear Color-space Conversion logic bug.
- [X] Workaround for `CAP_IMAGES` error in YouTube Mode.
- [X] Replaced incorrect `terminate()` with `join()` in PiGear.
- [X] Removed `uvloop` for windows as still [NOT yet supported](https://github.com/MagicStack/uvloop/issues/14).
- [X] Refactored Asynchronous Package name `async` to `asyncio`, since it is used as Keyword in python>=3.7 *(raises SyntaxError)*.
- [X] Fixed unfinished close of event loops bug in WebGear API.
- [X] Fixed NameError in helper.py.
- [X] Added fix for OpenCV installer failure on Linux test envs.
- [X] Fixed undefined NameError in `helper.py` context. (@cclauss)
- [X] Fixed incorrect logic while pulling frames from ScreenGear API.
- [X] Fixed missing functions in `__main__.py`.
- [X] Fixed Typos and definitions in docs.
- [X] Added missing `camera_num` parameter to VideoGear.
- [X] Added OpenSSL's [SSL: CERTIFICATE_VERIFY_FAILED] bug workaround for macOS envs.
- [X] Removed `download_url` meta from setup.py.
- [X] Removed PiGear from CI completely due to hardware emulation limitation.
- [X] Removed VideoCapture benchmark tests for macOS envs.
- [X] Removed trivial `__main__.py` from codecov.
- [X] Removed several redundant `try-catch` loops.
- [X] Renamed `youtube_url_validation` as `youtube_url_validator`.
- [X] Several minor wrong/duplicate variable definitions and various bugs fixed.
- [X] Fixed, Improved & removed many Redundant CI tests for various APIs.


### Pull Requests

* PR #88
* PR #91
* PR #93
* PR #95
* PR #98
* PR #101
* PR #114
* PR #118
* PR #124


&nbsp; 

&nbsp; 

## v0.1.6 (2020-01-01)

### New Features

- **NetGear API:**
    * [X] **Added powerful ZMQ Authentication & Data Encryption features for NetGear API:**
        + Added exclusive `secure_mode` param for enabling it.
        + Added support for two most powerful `Stonehouse` & `Ironhouse` ZMQ security mechanisms.
        + Added smart auth-certificates/key generation and validation features.
    * [X] **Implemented Robust Multi-Servers support for NetGear API:**
        + Enables Multiple Servers messaging support with a single client.
        + Added exclusive `multiserver_mode` param for enabling it.
        + Added support for `REQ/REP` &  `PUB/SUB` patterns for this mode.
        + Added ability to send additional data of any datatype along with the frame in realtime in this mode.
    * [X] **Introducing exclusive Bidirectional Mode for bidirectional data transmission:**
        + Added new `return_data` parameter to `recv()` function.
        + Added new `bidirectional_mode` attribute for enabling this mode.
        + Added support for `PAIR` & `REQ/REP` patterns for this mode
        + Added support for sending data of any python datatype.
        + Added support for `message` parameter for non-exclusive primary modes for this mode.
    * [X] **Implemented compression support with on-the-fly flexible frame encoding for the Server-end:**
        + Added initial support for `JPEG`, `PNG` & `BMP` encoding formats .
        + Added exclusive options attribute `compression_format` & `compression_param` to tweak this feature.
        + Client-end will now decode frame automatically based on the encoding as well as support decoding flags.
    * [X] Added `force_terminate` attribute flag for handling force socket termination at the Server-end if there's latency in the network. 
    * [X] Implemented new *Publish/Subscribe(`zmq.PUB/zmq.SUB`)* pattern for seamless Live Streaming in NetGear API.

- **PiGear API:**
    * [X] Added new threaded internal timing function for PiGear to handle any hardware failures/frozen threads.
    * [X] PiGear will not exit safely with `SystemError` if Picamera ribbon cable is pulled out to save resources.
    * [X] Added support for new user-defined `HWFAILURE_TIMEOUT` options attribute to alter timeout.

- **VideoGear API:** 
    * [X] Added `framerate` global variable and removed redundant function.
    * [X] Added `CROP_N_ZOOM` attribute in Videogear API for supporting Crop and Zoom stabilizer feature.

- **WriteGear API:** 

    * [X] Added new `execute_ffmpeg_cmd` function to pass a custom command to its FFmpeg pipeline.

- **Stabilizer class:** 
    * [X] Added new Crop and Zoom feature.
        + Added `crop_n_zoom` param for enabling this feature.
    * [X] Updated docs.

- **CI & Tests updates:**
    * [X] Replaced python 3.5 matrices with latest python 3.8 matrices in Linux environment.
    * [X] Added full support for **Codecov** in all CI environments.
    * [X] Updated OpenCV to v4.2.0-pre(master branch). 
    * [X] Added various Netgear API tests.
    * [X] Added initial Screengear API test.
    * [X] More test RTSP feeds added with better error handling in CamGear network test.
    * [X] Added tests for ZMQ authentication certificate generation.
    * [X] Added badge and Minor doc updates.

- [X] Added VidGear's official native support for MacOS environments.
    

### Updates/Improvements

- [X] Replace `print` logging commands with python's logging module completely.
- [X] Implemented encapsulation for class functions and variables on all gears.
- [X] Updated support for screen casting from multiple/all monitors in ScreenGear API.
- [X] Updated ScreenGear API to use *Threaded Queue Mode* by default, thereby removed redundant `THREADED_QUEUE_MODE` param.
- [X] Updated bash script path to download test dataset in `$TMPDIR` rather than `$HOME` directory for downloading testdata.
- [X] Implemented better error handling of colorspace in various videocapture APIs.
- [X] Updated bash scripts, Moved FFmpeg static binaries to `github.com`.
- [X] Updated bash scripts, Added additional flag to support un-secure apt sources.
- [X] CamGear API will now throw `RuntimeError` if source provided is invalid.
- [X] Updated threaded Queue mode in CamGear API for more robust performance.
- [X] Added new `camera_num` to support multiple Picameras.
- [X] Moved thread exceptions to the main thread and then re-raised.
- [X] Added alternate github mirror for FFmpeg static binaries auto-installation on windows oses.
- [X] Added `colorlog` python module for presentable colored logging.
- [X] Replaced `traceback` with `sys.exc_info`.
- [X] Overall APIs Code and Docs optimizations.
- [X] Updated Code Readability and Wiki Docs.
- [X] Updated ReadMe & Changelog with the latest changes.
- [X] Updated Travis CI Tests with support for macOS environment.
- [X] Reformatted & implemented necessary MacOS related changes and dependencies in `travis.yml`.

### Breaking Updates/Changes

!!! warning 
    * Python 2.7 legacy support dropped completely.
    * Source-code Relicensed to Apache 2.0 License.

- [X] Python 3+ are only supported legacies for installing v0.1.6 and above.
- [X] Python 2.7 and 3.4 legacies support dropped from CI tests.

### Bug-fixes

- [X] Reimplemented `Pub/Sub` pattern for smoother performance on various networks.
- [X] Fixed Assertion error in CamGear API during colorspace manipulation.
- [X] Fixed random freezing in `Secure Mode` and several related performance updates
- [X] Fixed `multiserver_mode` not working properly over some networks.
- [X] Fixed assigned Port address ignored bug (commit 073bca1).
- [X] Fixed several wrong definition bugs from NetGear API(commit 8f7153c).
- [X] Fixed unreliable dataset video URL(rehosted file on `github.com`).
- [X] Disabled `overwrite_cert` for client-end in NetGear API.
- [X] Disabled Universal Python wheel builds in `setup.cfg `file.
- [X] Removed duplicate code to import MSS(@BoboTiG) from ScreenGear API.
- [X] Eliminated unused redundant code blocks from library.
- [X] Fixed Code indentation in `setup.py` and updated new release information.
- [X] Fixed code definitions & Typos.
- [X] Fixed several bugs related to `secure_mode` & `multiserver_mode` Modes.
- [X] Fixed various macOS environment bugs.

### Pull Requests

- PR #39
- PR #42
- PR #44
- PR #52
- PR #55
- PR #62
- PR #67
- PR #72
- PR #77
- PR #78
- PR #82
- PR #84



&nbsp; 

&nbsp; 

## v0.1.5 (2019-07-24)

### New Features

- [X] Added new ScreenGear API, supports Live ScreenCasting.
- [X] Added new NetGear API, aids real-time frame transfer through messaging(ZmQ) over network.
- [X] Added new new Stabilizer Class, for minimum latency Video Stabilization with OpenCV.
- [X] Added Option to use API's standalone.
- [X] Added Option to use VideoGear API as internal wrapper around Stabilizer Class.
- [X] Added new parameter `stabilize` to API, to enable or disable Video Stabilization.
- [X] Added support for `**option` dict attributes to update VidGear's video stabilizer parameters directly. 
- [X] Added brand new logo and functional block diagram (`.svg`) in readme.md
- [X] Added new pictures and GIFs for improving readme.md readability 
- [X] Added new `contributing.md` and `changelog.md` for reference.
- [X] Added `collections.deque` import in Threaded Queue Mode for performance consideration
- [X] Added new `install_opencv.sh` bash scripts for Travis cli, to handle OpenCV installation.
- [X] Added new Project Issue & PR Templates
- [X] Added new Sponsor Button(`FUNDING.yml`)

### Updates/Improvements

- [X] Updated New dependencies: `mss`, `pyzmq` and rejected redundant ones.
- [X] Revamped and refreshed look for `readme.md` and added new badges.
- [X] Updated Releases Documentation completely.
- [X] Updated CI tests for new changes
- [X] Updated Code Documentation.
- [X] Updated bash scripts and removed redundant information
- [X] Updated `Youtube video` URL in tests
- [X] Completely Reformatted and Updated Wiki Docs with new changes.


### Breaking Updates/Changes

- [X] Implemented experimental Threaded Queue Mode(_a.k.a Blocking Mode_) for fast, synchronized, error-free multi-threading.
- [X] Renamed bash script `pre-install.sh` to `prepare_dataset.sh` - downloads opensourced test datasets and static FFmpeg binaries for debugging.
- [X] Changed `script` folder location to `bash/script`.
- [X] `Python 3.4` removed from Travis CI tests.

### Bug-fixes

- [X] Temporarily fixed Travis CI bug: Replaced `opencv-contrib-python` with OpenCV built from scratch as dependency.
- [X] Fixed CI Timeout Bug: Disable Threaded Queue Mode for CI Tests
- [X] Fixes** `sys.stderr.close()` throws ValueError bug: Replaced `sys.close()` with `DEVNULL.close()`
- [X] Fixed Youtube Live Stream bug that return `NonType` frames in CamGear API.
- [X] Fixed `NoneType` frames bug in  PiGear class on initialization.
- [X] Fixed Wrong function definitions
- [X] Removed `/xe2` unicode bug from Stabilizer class.
- [X] Fixed `**output_params` _KeyError_ bug in WriteGear API
- [X] Fixed subprocess not closing properly on exit in WriteGear API.
- [X] Fixed bugs in ScreenGear: Non-negative `monitor` values
- [X] Fixed missing import, typos, wrong variable definitions
- [X] Removed redundant hack from `setup.py`
- [X] Fixed Minor YouTube playback Test CI Bug 
- [X] Fixed new Twitter Intent
- [X] Fixed bug in bash script that not working properly due to changes at server end.

### Pull Requests

- PR #17
- PR #21
- PR #22
- PR #27
- PR #31
- PR #32
- PR #33
- PR #34



&nbsp; 

&nbsp; 

## v0.1.4 (2019-05-11)

### New Features

- [X] Added new WriteGear API: for enabling lossless video encoding and compression(built around FFmpeg and OpenCV Video Writer)
- [X] Added YouTube Mode for direct Video Pipelining from YouTube in CamGear API
- [X] Added new `y_tube` to access _YouTube Mode_ in CamGear API.
- [X] Added flexible Output file Compression control capabilities in compression-mode(WriteGear).
- [X] Added `-output_dimensions` special parameter to WriteGear API.
- [X] Added new `helper.py` to handle special helper functions.
- [X] Added feature to auto-download and configure FFmpeg Static binaries(if not found) on Windows platforms.
- [X] Added `-input_framerate` special parameter to WriteGear class to change/control output constant framerate in compression mode(WriteGear).
- [X] Added new Direct Video colorspace Conversion capabilities in CamGear and PiGear API.
- [X] Added new `framerate` class variable for CamGear API, to retrieve input framerate.
- [X] Added new parameter `backend` - changes the backend of CamGear's API
- [X] Added automatic required prerequisites installation ability, when installation from source.
- [X] Added Travis CI Complete Integration for Linux-based Testing for VidGear.
- [X] Added and configured `travis.yml`
- [X] Added Appveyor CI Complete Integration for Windows-based Testing in VidGear.
- [X] Added and configured new `appveyor.yml`
- [X] Added new bash script `pre-install.sh` to download opensourced test datasets and static FFmpeg binaries for debugging.
- [X] Added several new Tests(including Benchmarking Tests) for each API for testing with `pytest`.
- [X] Added license to code docs.
- [X] Added `Say Thank you!` badge to readme.

### Updates/Improvements

- [X] Removed redundant dependencies
- [X] Updated `youtube-dl` as a dependency, as required by `pafy`'s backend.
- [X] Updated common VideoGear API with new parameter.
- [X] Update robust algorithm to auto-detect FFmpeg executables and test them, if failed, auto fallback to OpenCV's VideoWriter API. 
- [X] Improved system previously installed OpenCV detection in setup.py.
- [X] Updated setup.py with hack to remove bullets from pypi description. 
- [X] Updated Code Documentation
- [X] Reformatted & Modernized readme.md with new badges.
- [X] Reformatted and Updated Wiki Docs.

### Breaking Updates/Changes

- [X] Bugs Patched: Removed unnecessary `-height` and `-width` parameter from CamGear API.
- [X] Replaced dependency `opencv-python` with `opencv-contrib-python` completely

### Bug-fixes

- [X] Windows Cross-Platform fix: replaced dependency `os` with `platform` in setup.py.
- [X] Fixed Bug: Arises due to spaces in input `**options`/`**output_param` dictionary keys.
- [X] Fixed several wrong/missing variable & function definitions.
- [X] Fixed code uneven indentation.
- [X] Fixed several typos in docs.

### Pull Requests

- PR #7
- PR #8
- PR #10
- PR #12



&nbsp; 

&nbsp; 

## v0.1.3 (2019-04-07)

### Bug-fixes

- [X] Patched Major PiGear Bug: Incorrect import of PiRGBArray function in PiGear Class
- [X] Several Fixes** for backend `picamera` API handling during frame capture(PiGear)
- [X] Fixed missing frame variable initialization.
- [X] Fixed minor typos

### Pull Requests

- PR #6
- PR #5

&nbsp; 

&nbsp; 

## v0.1.2 (2019-03-27)

### New Features

- [X] Added easy Source manipulation feature in CamGear API, to control features like `resolution, brightness, framerate etc.`
- [X] Added new `**option` parameter to CamGear API, provides the flexibility to manipulate input stream directly.
- [X] Added new parameters for Camgear API for time delay and logging.
- [X] Added new Logo to readme.md
- [X] Added new Wiki Documentation.

### Updates/Improvements

- [X] Reformatted readme.md.
- [X] Updated Wiki Docs with new changes.


### Bug-fixes

- [X] Improved Error Handling in CamGear & PiGear API.
- [X] Fixed minor typos in docs.
    
### Pull Requests

- PR #4

&nbsp;

## v0.1.1 (2019-03-24)

### New Features

- [X] Release ViGear binaries on the Python Package Index (PyPI)
- [X] Added new and configured `setup.py` & `setup.cfg`

### Bug-fixes

- [X] Fixed PEP bugs: added and configured properly `__init__.py` in each folder 
- [X] Fixed PEP bugs: improved code Indentation
- [X] Fixed wrong imports: replaced `distutils.core` with `setuptools`
- [X] Fixed readme.md


&nbsp;

## v0.1.0 (2019-03-17)

### New Features

- [X] Initial Release
- [X] Converted [my `imutils` PR](https://github.com/jrosebr1/imutils/pull/105) into Python Project.
- [X] Renamed conventions and reformatted complete source-code from scratch.
- [X] Added support for both python 2.7 and 3 legacies
- [X] Added new multi-threaded CamGear, PiGear, and VideoGear APIs
- [X] Added multi-platform compatibility
- [X] Added robust & flexible control over the source in PiGear API.
