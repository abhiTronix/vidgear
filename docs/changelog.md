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

# Release Notes

## v0.3.1 (2023-07-22)

??? tip "New Features"
    - [x] **WebGear:** 
        * Added an option to add a custom video endpoint path.
            + Users can now change the video endpoint path from `"/video"` to any alphanumeric string.
            + Added the `custom_video_endpoint` optional string attribute for this purpose.
            + Only alphanumeric strings with no spaces in between are allowed as its value.
    - [x] **ScreenGear:** 
        * Added `dxcam` support for Windows machines.
            + Implemented a complete end-to-end workflow for the `dxcam` backend.
            + `dxcam` is now the default backend for Windows machines when no backend is defined.
            + Added support for variable screen dimensions to capture an area from the screen.
            + Added the optional flag `dxcam_target_fps` to control the target fps in `dxcam`. Defaults to `0` (disabled).
            + RGB frames from `dxcam` are automatically converted into BGR.
            + For better performance, `video_mode` is enabled by default in `dxcam` backend.
            + Added necessary imports.
        * Added support for tuple values in the monitor parameter to specify device and output indexes as `(int[device_idx], int[output_idx])` in the `dxcam` backend only.
            + Default `int` index is also allowed as a value for selecting device index.
    - [x] **Helper**
        * Added multiple servers support for downloading assets.
            + Added GitHub server to the `generate_webdata` method to make it more robust for rate limits and other shortcomings.
            + Now, the `generate_webdata` method will retry a different server when one fails.
    - [x] **Setup.py**
        * Added `dxcam` dependency in `core` and `asyncio` extra requires.
        * Bumped version to `0.3.1`. 
    - [x] **Docs**
        * Added `dxcam` API specific prerequisites for ScreenGear API when installing on Windows via pip.
        * Added documentation for the `custom_video_endpoint` optional string attribute.
        * Added documentation for controlling Chunk size in HLS stream.
        * Added new hyperlinks for `dxcam` dependency.
    - [x] **CI**
        * Added a test case for `ndim==3` grayscale frames.
            + Added the `Custom_Grayscale_class` to generate `ndim==3` grayscale frames.
        * Added test cases for the `custom_video_endpoint` optional string attribute.


??? success "Updates/Improvements" 
    - [x] WebGear: 
        * Improved the conditions logic to check if non-empty values are assigned to optional parameters.
    - [x] WebGear_RTC: 
        * Improved the handling of the `format` parameter when constructing a `VideoFrame` from ndarray frames.
    - [x] ScreenGear: 
        * Enforced `dxcam` backend (if installed) when `monitor` is defined on Windows machines.
        * Refactored code blocks to ensure backward compatibility.
    - [x] Maintenance:
        * Cleaned up unused imports and code blocks.
        * Cleaned redundant code.
        * Improved logging.
        * Implemented short-circuiting.
        * Fixed comment typos.
        * Updated comments.
    - [x] Docs:
        * Updated ScreenGear API usage example docs, added new relevant information, updated requirements for `dxcam` support in Windows machines.
        * Refactored `monitor` and `backend` parameters docs of ScreenGear.
        * Updated class and class parameters descriptions in ScreenGear docs.
        * Updated a new description for ScreenGear API.
        * Updated Zenodo badge and the BibTeX entry.
        * Relocated some docs for a better context.
        * Removed ScreenGear name from Threaded Queue Mode doc.
        * Updated ScreenGear FAQs.
        * Updated changelog.md
    - [x] CI:
        * Updated the `test_webgear_rtc_custom_stream_class` method.
        * Updated the `test_webgear_options` method.
        * Updated the `test_webgear_routes` test to validate the new custom endpoint.
        * Increased code coverage by updating tests.


??? danger "Breaking Updates/Changes"
    - [ ] ScreenGear: 
        * Previously enforced threaded queue mode is now completely removed, resulting in a potential performance boost.
            + 💬 Reason: The IO is automatically blocked by the screen refresh rate, so adding the overhead of maintaining a separate queue is pointless.
        * Removed the `THREAD_TIMEOUT` optional flag.

??? bug "Bug-fixes"
    - [x] WebGear_RTC: 
        * Fixed a bug caused by PyAV's error when `ndim==3` grayscale frames are encountered. 
            + The API will now drop the third dimension if `ndim==3` grayscale frames are detected.
    - [x] ScreenGear:
        * Fixed backend not defined while logging.
    - [x] Setup.py:
        * Starting from version `8.0.0`, the python-mss library dropped support for Python `3.7`, so as a temporary measure, `mss` dependency has been pinned to version `7.0.1`.
    - [x] Docs:
        * Fixed context and added separate code for controlling chunk size in HLS and DASH streams in StreamGear docs.
        * Fixed naming conventions for the recently added DXcam backend in ScreenGear docs.
        * Fixed missing hyperlinks.
    - [x] CI:
        * Fixed m3u8 module failing to recognize Windows paths in ScreenGear tests.
        * Fixed a path bug by replacing the absolute file path with the decoded file content as a string in its `loads()` 


??? question "Pull Requests"
    * PR #367
    * PR #366
    * PR #365

&nbsp; 

&nbsp; 

## v0.3.0 (2023-01-26)

??? tip "New Features"
    - [x] **WriteGear:** 
        * Added support for user-defined and higher than 8-bit depth input frames pixel-format.
            + Added support for higher than 8-bit depth frames with datatypes of unsigned integer(`uint`) kind and element size `2`.
            + Added `dtype` parameter to internal `Preprocess` method for passing input frames datatype.
            + Implemented auto-calculation of input pixel-format based on number of channels in higher than 8-bit depth frames.
            + Added various known working pixel-formats(based on number of channels), supported by all prominent computer vision libraries.
            + Added support for up to 1-channel(`gray16-le/be`) to all the way up to 4-channels(`bgra64-le/be`) in input frames.
            + Added endianness little(`le`) or big(`be`) at the suffix of pixel-format based on byte-order of input frames datatypes.
            + Extended support for higher RGB 8-bit depth frames through RGB mode.
        * Added support for user-defined custom input pixel-format.
            + Added new `-input_pixfmt` attribute to `output_params` dictionary parameter for easily specifying custom input pixel-format.
            + Added newly implemented `get_supported_pixfmts` method import for verifying user-defined input pixel-format against Installed FFmpeg supported pixel-formats. Unsupported values will be discarded. 
            + Implemented runtime datatype validation check, such that all input frames must have same datatype.
        * Added support for Context Managers for proper handling of resources via `with` statement for allocating and releasing resources precisely. (Suggested by @sueskind)
            + Implement the `__enter__()` and `__exit__()` methods.
            + Added `__enter__` method that returns reference to the WriteGear Class.
            + Added `__exit__` method that automatically executes `close()` for performing the cleanup operations and handling exception gracefully. 
    - [x] **StreamGear:** 
        * Added support for Context Managers for proper handling of resources via `with` statement for allocating and releasing resources precisely. (Suggested by @sueskind)
            + Implement the `__enter__()` and `__exit__()` methods.
            + Added `__enter__` method that returns reference to the StreamGear Class.
            + Added `__exit__` method that automatically executes `close()` for performing the cleanup operations and handling exception gracefully. 
    - [x] **WebGear:**
        * Added way to completely disable Data-Files Auto-Generation WorkFlow.
            + Added new `skip_generate_webdata` boolean optional attribute(`False` by default) to completely disable Data-Files Auto-Generation WorkFlow.
            + This flag enables only `/video` route for disabled Data-Files Auto-Generation WorkFlow.
            + Implemented JSONResponse as placeholder response instead of Index, `404` and `500` HTML pages, when workflow is disabled. _(Note: Index HTML page will throw `404` status code.)_
            + Added necessary imports.
    - [x] **Helper:**
        * Added more robust implementation of validate_audio method.
            + Added new more robust regex pattern for extracting audio-samplerate.
            + Added new `validate_audio` method for calculating accurate bitrate(in kbps) from audio samplerate, channels, bit-depth values.
            + Implemented new patterns and logic for accurately extracting audio channels and bit-depth from given metadata.
        * Added support for Linux video device path _(such as `/dev/video0`)_.
    - [x] **Maintenance:** 
        * Logging current vidgear version when vidgear APIs are called, not at import.
            + Added `logcurr_vidgear_ver` helper function to facilitate logging current vidgear version, when called within a API.
            + Implemented `ver_is_logged` global variable in helper to log version only once, which can modifiable with `logcurr_vidgear_ver` method only. Followed recommendation given in official python docs: https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules
            + Current version can only be logged by VidGear APIs with the logging turned on _(i.e. `logging=True`)_.
    - [x] **Docs:**
        * Added new WriteGear Bonus Example:
            + Added "Using WriteGear's Compression Mode with `v4l2loopback` Virtual Cameras bonus python example.
            + Added related prerequisites and dependencies for creating `v4l2loopback` Virtual Cameras on Linux machines.
            + Added both With/Without-Audio cases for "Using WriteGear's Compression Mode for YouTube-Live Streaming".
        * Added `content.code.copy` and `content.tabs.link` features.
        * Added docs related to `skip_generate_webdata` optional attribute.
        * Added feedback features to mkdocs.yml.
        * Added `404.html` static template to `mkdocs.yml`.
    - [x] **CI:**
        * Added v4l2loopback support for testing `/dev/video0` device on Linux machines.
        * Added test cases for newer implementation of `validate_audio` method.
        * Added `test_skip_generate_webdata` to test `skip_generate_webdata` optional attribute.
        * Added tests for user-defined and higher than 8-bit depth input frames pixel-format.


??? success "Updates/Improvements" 
    - [x] WriteGear: 
        * Completely revamped code structure and comments.
            + Updated comments, description, and logging messages to more sensible and developer friendly.
            + Implemented operator short-circuiting to cleanup code as much as possible.
            + Renamed `startFFmpeg_Process` internal class method to `start_FFProcess`.
            + Renamed `Preprocess` internal class method to `PreprocessFFParams`.
            + Renamed `startCV_Process` internal class method to `start_CVProcess`.
            + Renamed `initiate` internal class parameter to `initiate_process`.
            + Renamed `force_termination` internal class parameter to `forced_termination`.
            + Enabled `output_params` parameters logging in both modes.
            + Improved `compression` and `logging` parameters boolean value handling.
            + Implemented `stdout` closing to cleanup pipeline before terminating.
    - [x] Helper:
        *  Updated `validate_audio` method with improved and more robust regex patterns for identifying audio bitrate in ay audio file.
    - [x] Setup.py:
        * Bumped version to `0.3.0`.
        * Replaced `>=` comparison operator with more flexible `~=`.
        * Replaced `distutils.version.LooseVersion` with `pkg_resources.parse_version`.
    - [x] Maintenance: 
        * Replaced depreciated `LooseVersion` with `parse_version`.
        * Updated `Retry` package to be imported from `requests.adapters`.
        * Moved terminal and python code text area to Question GitHub Form Schema.
        * Removed unnecessary imports.
        * Removed redundant code.
        * Improved logging messages.
        * Updated code comments.
        * Updated method descriptions.
        * Refactored code.
        * Increased coverage.
    - [x] Bash Script:
        * Updated FFmpeg Static Binaries links to latest date/version tag to `12-07-2022`.
        * Removed depreciated binaries download links and code.
    - [x] Docs:
        * Replaced all `raw.githubusercontent.com` GIF URLs with `user-images.githubusercontent.com`.
        * Reformatted `custom.css` and added missing comments.
        * Updated sponsor block.
        * Enabled Code Highlights.
        * Updated announcement bar.
        * Updated `changelog.md`.
        * Reduced `webgear_rtc.gif` size.
        * Updated Zenodo badge and the BibTeX entry.
    - [x] CI:
        * Added more flexible formats to `return_testvideo_path` function.
        * Updated `test_write` test for higher than 8-bit depth input frames pixel-format in WriteGear's Compression Mode.
        * Updated `actions/checkout` to `v3`.
        * Updated `actions/setup-python` to `v4`.
        * Updated `codecov/codecov-action` to `v3`.
        * Moved `test_colorspaces` test to CamGear tests.
        * Added deffcode library import.
      * Re-stuctured yaml code.

??? danger "Breaking Updates/Changes"
    - [ ] WriteGear: 
        * Renamed `output_filename` string parameter to `output`.
            + Since WriteGear API accepts all sorts of streams _(such as valid filename/path/URL)_ for encoding, thereby changing parameter name to `output` will be more true to its purpose.
            + Renaming `output_filename` to `output` in WriteGear API will also help user to not accidentally assume WriteGear supports only encoding of video files.
            + It matches the `output` parameter in StreamGear which basically does the same thing.
        * Renamed `cmd` parameter in `execute_ffmpeg_cmd()` class method to more sensible `command`.
        * `ValueError` will be raised if datatype of input frames mismatches Writegear API

??? bug "Bug-fixes"
    - [x] Camgear: 
        * Fixed `CamGear.read()` blocked unnecessarily.
            + 💬 When `THREADED_QUEUE_MODE` is enabled `CamGear.read()` blocks for an excessive duration when attempting to read past the end of a stream.
            + Added `None` frame to the queue at the end to signal we're done.
            + Added `terminate` Event check before continuing.
        * Fixed deadlock on exit.
            + 💬 The deadlock is due to `self.__queue.get(timeout=self.__thread_timeout)` line in `read()` method, which still waits for timeout(thread_timeout) to happen when main `update()` thread was already terminated on exit and queue was empty. Since there was no way to signal queue that stream is already ended, the blocking `queue.get()` keeps on waiting until timeout occurs.
            + The solution was to signal `queue.get()` that stream is already ended by putting `None` in queue on exiting the main `update()` thread.
    - [x] ScreenGear: 
        * Fixed `ScreenGear.read()` blocked during cold startup.
          + 💬 During startup, `ScreenGear.read()` doesn't checks if queue is empty before continuing.
    - [x] WriteGear: 
        * Fixed gstpipeline_mode not activating when wrongly assuming `output` value as valid path.
        * Fixed name 'compression' is not defined bug.
        * Fixed `AttributeError`.
    - [x] Helper:
        * Fixed `fltp` keyword in regex pattern causing non-ftlp streams to be not recognized.
        * Fixed response.headers returning `content-length` as Nonetype since it may not necessarily have the Content-Legth header set.
            + Reason: The response from gitlab.com  contains a Transfer-Encoding field as `'Transfer-Encoding': 'chunked'`, which means data is sent in a series of chunks, so the Content-Length header is emitted. More info: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Transfer-Encoding#Directives
        * Fixed Linux video device paths still not working. 
            + Moved `helper.py` logic to WriteGear and StreamGear APIs resp.
        * Fixed KeyError for empty metadata.
    - [x] Setup:
        * Pinned `pyzmq==24.0.1` working version.
        * Removed redundant patch for the issue.
    - [x] Maintaince:
        * Fixed missing pkg name `import_dependency_safe` functions calls.
    - [x] Bash Script: 
        * Fixed gstreamer installation.
    - [x] CI:
        * Fixed missing v4l2loopback apt dependency on Linux envs.
        * Added fix for RTCPeerConnection fails to create RTCDtlsTransport (Related issue: aiortc/aiortc#804)
            + Pinned `cryptography==38.0.4` in dependencies.
        * Pinned Linux image to `ubuntu-20.04` in github actions.
        * Fixed No module named 'httpx' bug.
            + Added `httpx` library import.
        * Fixed F821 undefined name bug.
        * Fixed Gstreamer bug.
    - [x] Docs:
        * Fixed hyperlinks to new GitHub's form schemas. 
        * Fixed non-rendering images in README.md 
            + Replaced all relative image/gifs paths with absolute URLs in README.md.
        * Fixed badges/shields#8671 badge issue in README.md
        * Fixed GitLab CDN links throwing blocked by CORS policy bug.
            + Replaced gitlab GitHack CDN links with with bitbucket.
        * Fixed DASH playback failing by setting the `maxAttempts` to Infinity.
        * Removed `x-sign` glow-text effect CSS.
        * Fixed several typos (suggested by @timgates42)
        * Fixed coverage badge.

??? question "Pull Requests"
    * PR #346
    * PR #348
    * PR #349
    * PR #350
    * PR #351

&nbsp; 

&nbsp; 

## v0.2.6 (2022-07-05)

??? tip "New Features"
    - [x] **Docs:**
        * Added new bonus example for RTSP/RTP Live-Streaming using WriteGear's Compression Mode.
        * Added "How to resolve zmq.error.ZMQError" FAQ for NetGear API.(PR by @iandol)
        * Added new ko-fi button to README.md
        * Added new contributors block to changelog.md
    - [x] **Maintenance:** 
        * Added new patterns to `.gitignore` to ignore pypi's `build` directory and `egg-info` files.
    - [x] **CI:**
        * Switched to new Issue GitHub's form schema using YAML
            + Added new `bug_report.yaml`.
            + Added new `question.yaml`.
            + Added new `proposal.yaml`.
            + Deleted depreciated markdown files.
            + Polished forms.

??? success "Updates/Improvements"  
    - [x] Setup.py:
        * Bumped version to `0.2.6`.
        * Updated logic operators and dependency.
            + Replaced `>=` comparsion operator with more flexible `~=`.
            + Replaced `distutils.version.LooseVersion` with `pkg_resources.parse_version`.
    - [x] Docs:
        * Updated Site Navigation.
            + Added new notices to inform users more effectively about bonus examples.
            + Added new `Bonus` section to navigation and moved suitable pages under it.
            + Updated headings and URLs.
        * Redesigned and Rewritten Donation and Contribution section to README.md
        * Updated Zenodo badge and Bibtex entry.
        * Updated Admonition Icon, FAQs and site-links.
        * Reformatted code and its comments.
        * Updated `changelog.md`.
    - [x] API:
        * Updated depreciated tostring() to tobytes(). `tostring` was renamed to `tobytes` for the purposes for clarity in Python 3.2. https://docs.python.org/3/library/array.html#array.array.tobytes
    - [x] CI:
        * Added more paths and files to skip commits.
    
??? danger "Breaking Updates/Changes"
    - [ ] `-input_framerate` parameter now accepts any positive value for WriteGear and StreamGear APIs.

??? bug "Bug-fixes"
    - [x] API:
        * Fixed `-input_framerate` less than 5 does not get used in WriteGear and StreamGear APIs.(PR by @freol35241)
    - [x] CamGear: Fixed Yt-dlp generated HTTP DASH Segments URLs not supported by OpenCV's VideoCapture(PR by @DynamiteC)
    - [x] StreamGear: 
        * Fixed `hls_segment_type` not working bug. (PR by @enarche-ahn)
        * Fixed critical logging parameter bug
            + Fixed debug logs even when `logging=False` in StreamGear's Real-time Mode. (patch suggested by @enarche-ahn)
            + Added length check to `-video_source` attribute to correctly infers it as empty(or invalid).
    - [x] CI:
        * Xfailed RTSP CamGear CI test.
        * Fixed pinned version syntax bug in docs_deployer workflow.
        * Fixed typos in Github forms and its context.
        * Added missing dependency.
    - [x] Docs:
        * Fixed jinja2 `3.1.0` or above breaks mkdocs.
            + `jinja2>=3.1.0` breaks mkdocs (mkdocs/mkdocs#2799), therefore pinned jinja2 version to `<3.1.0`.
        * Fixed support for new `mkdocstring` versions
            + Replaced rendering sub-value with options.
            + Removed pinned `mkdocstrings==0.17.0` version.
        * Fixed Netgear+Webgear bonus example code bugs.(PR by @iandol)
            + Added a missing import.
            + Removed `self.` typo.
            + Replaced the `return` value with `break` in the async as it triggers an error. 
        * Fixed external bug that causing "Home" tab to irresponsive randomly when accessed from other tabs.
        * Fixed indentation and spacing.
        * Fixed typos and updated context.
        * Removed dead code.

??? question "Pull Requests"
    * PR #288
    * PR #290
    * PR #293
    * PR #295
    * PR #307
    * PR #313
    * PR #320

??? new "New Contributors"
    * @iandol
    * @freol35241
    * @enarche-ahn
    * @DynamiteC

&nbsp; 

&nbsp; 

## v0.2.5 (2021-02-11)

??? tip "New Features"
    - [x] **WriteGear:** 
        * Add support for GStreamer pipeline in WriteGear API's Non-Compression mode:
            + Implemented GStreamer Pipeline Mode to accept GStreamer pipeline as string to its output_filename parameter.
            + Added new special `-gst_pipeline_mode` attribute for its output_params parameter.
            + This feature provides flexible way to directly write video frames into GStreamer Pipeline with controlled bitrate. 
            + Added new docs and updated existing docs with related changes.
        * Added new `-ffpreheaders` special attribute to WriteGear's options parameter:
            + This attribute is specifically required to set special FFmpeg parameters in Compression Mode that are present at the starting of command(such as `-re`).
            + This attribute only accepts **list** datatype as value.
            + Added related docs.
    - [x] **NetGear:** 
        * Added bidirectional data transfer support by extending Bidirectional mode support to exclusive Multi-Clients and Multi-Servers modes:
            + Users will now able to send data bidirectionally in both Multi-Clients and Multi-Servers exclusive modes.
            + Bidirectional mode will no longer disables automatically when Multi-Clients and Multi-Servers modes already enabled.
            + Added new docs and updated existing docs with related changes.
    - [x] **Maintenance:** 
        * ==Added official support for **Python-3.10** legacies.==
        * Added `float` value support to `THREAD_TIMEOUT` optional parameter.
        * Added info about dropped support for Python-3.6 legacies through announcement bar.
        * Added `config.md` file for Issue templates.
        * Added title to Issue templates.
    - [x] **Docs:**
        * Added new Code Annotations
        * Added new icons to headings.
        * Added Advanced VideoGear usage example with CamGear backend.

??? success "Updates/Improvements"  
    - [x] Setup.py:
        * Dropped support for Python-3.6 and below legacies.
        * Updated logging formatting.
        * Updated python_requires to `>=3.7`.
        * Bumped version to `0.2.5`.
    - [x] Helper:
        * Vidgear will now report current version on every run.
    - [x] Docs: 
        * Updated SSH tunneling docs context.
        * Excluded `docs` directory from CI envs.
        * Updated Zenodo badge and BibTeX entry.
        * Updated dark theme hue to `260`.
        * Updated Admonitions.
        * Additional warnings against pushing PR against VidGear's `testing` branch only.
        * Updated code comments.
    - [x] CI:
        * Removed support for Python-3.6 legacies from all workflows.
        * Updated NetGear's Exclusive Mode tests.
        * Added GStreamer Pipeline Mode tests.
    - [x] Maintenance: 
        * Updated Issue and PR templates.
        * Updated metadata.


??? danger "Breaking Updates/Changes"
    - [ ] **Dropped support for Python-3.6 legacies from vidgear.**

??? bug "Bug-fixes"
    - [x] NetGear: Fixed bidirectional mode overriding multi-clients mode's data.
    - [x] WriteGear: 
        * Fixed wrongly defined ffmpeg_preheaders.
        * Fixed condition logic bugs.
        * Fixed UnboundLocalError bug.
    - [x] Setup: Fixed uvicorn and aiortc dropped support for Python-3.6 legacies.
    - [x] CI: 
        * Fixed GitHub Actions interprets `3.10` as `3.1` if used without strings.
        * Fixed naming error in azure YAML.
    - [x] Docs:
        * Fixed codecov badge URL in README.md
        * Fixed hyperlinks in README.
        * Fixed indentation and spacing.
        * Fixed typos and updated context.
        * Removed dead code.
    - [x] Maintenance: 
        * Removed depreciated condition checks.

??? question "Pull Requests"
    * PR #283
    * PR #284


&nbsp; 

&nbsp; 

## v0.2.4 (2021-12-05)

??? tip "New Features"
    - [x] **CamGear:** 
        * Added a new YT_backend Internal Class with YT-DLP backend:
            + Implemented `YT_backend` a new CamGear's Internal YT-DLP backend class for extracting metadata from Streaming URLs.
            + Added support for pipeling (live) video-frames from all yt-dlp supported streaming sites: https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md#supported-sites
            + Implemented algorithm from scratch for auto-extracting resolution specific streamable URLs for pipelineing.
            + Implemented logic for auto-calculating `best` and `worst` resolutions.
            + Added new `ytv_metadata` global parameter to CamGear for accessing video's metadata(such as duration, title, description) on-the-go.
            + ⚠️ Playlists are still unsupported.
    - [x] **WebGear_RTC:** 
        * Implemented a new easy way of defining Custom Streaming Class with suitable source(such as OpenCV):
            + Added new `custom_stream` attribute with WebGear_RTC `options` parameter that allows you to easily define your own Custom Streaming Class with suitable source(such as OpenCV).
            + This implementation supports repeated Auto-Reconnection or Auto-Refresh out-of-the-box.
            + This implementation is more user-friendly and easy to integrate within complex APIs.
            + This implementation requires at-least `read()` and `stop()` methods implemented within Custom Streaming Class, otherwise WebGear_RTC will throw ValueError.
            + This implementation supports all vidgear's VideoCapture APIs readily as input.
    - [x] **Maintenance:**
        * Added new `.gitignore`  for specifying intentionally untracked files to ignore
            + Added more files entries to `.gitignore`.
        * Added new `.gitattributes` to manage how Git reads line endings.
            + Enabled `auto` default behavior, in case people don't have `core.autocrlf` set.
            + Enforced LF line-endings for selective files types.
            + Added Binary data files that specifies they are not text, and git should not try to change them.
            + Added Language aware diff headers.
            + Added Linguist language overrides.
    - [x] **Docs:**
        * Added bonus example to add real-time file audio encoding with VideoGear and Stabilizer class.
        * Added complete usage docs with new CamGear's Internal Class with YT-DLP backend.
        * Added instructions to extract video's metadata in CamGear.
        * Added donation link in page footer with bouncing heart animation through pure CSS.
        * Added info about critical changes in `v0.2.4` and above installation through new announcement bar.
        * Added related usage docs for new WebGear_RTC custom streaming class.
        * Added changes for upgrading mkdocs-material from `v7.x` to newer `v8.x`.
        * Added outdated version warning block.

??? success "Updates/Improvements"  
    - [x] CamGear:
        * Added `is_livestream` global YT_backend parameters.
        * Added default options for yt-dlp for extracting info_dict(metadata) of the video as a single JSON line.
        * Completely removed old logic for extracting streams using pafy.
        * Removed all dead code related to streamlink backend.
    - [x] Setup.py:
        * Moved all API specific dependencies to `extra_requires` under the name `"core"`. **[PR #268 by @zpapakipos]**
        * Added rule to replace GitHub heading links in description.
        * Updated `extra_require` dependencies.
        * Removed `streamlink` dependency.
        * Removed `pafy` dependency.
        * Removed `pyzmq` from latest_version group.
        * Updated SEO Keywords.
    - [x] Docs: 
        * Re-written `pip` and `source` installation docs. 
        * Added warning for using `-disable_force_termination` flag for short duration videos.
        * Added `permalink_title` entry to mkdocs.yml.
        * Updated CamGear parameters.
        * Updated Admonitions with related information.
        * Updated Functional Block Diagram(`gears_fbd.png`) image.
        * Updated installation instructions.
        * Updated Advanced examples using WebGear_RTC's custom streaming class.
        * Updated code highlighting.
        * Updated zenodo badge.
        * Updated BibTex for project citation.
        * Replaced incorrect API parameter docs.
        * Updated WebGear_RTC parameters.
    - [x] CI:
        * Updated CI tests for new WebGear_RTC custom streaming class.
        * Restored `test_stream_mode` CamGear test.
        * Updated Streaming Sites test links.
        * Added more tests cases.
    - [x] Maintenance: 
        * Updated spacing in logger formatting.
        * Renamed Asyncio Helper logger name.
        * Changed logging colors.
        * Updated logging messages.


??? danger "Breaking Updates/Changes"
    - [ ] Installation command with `pip` has been changed in `v0.2.4`:
        * The legacy `#!sh  pip install vidgear` command now installs critical bare-minimum dependencies only. Therefore in order to automatically install all the API specific dependencies as previous versions, use `#!sh  pip install vidgear[core]` command instead.
    - [ ] CamGear:
        * Removed `streamlink` backend support from `stream_mode` in favor of more reliable CamGear's Internal YT-DLP backend class for extracting metadata from Streaming URLs.
            + CamGear will raise `ValueError` if streaming site URL is unsupported by yt-dlp backend.
            + CamGear will raise `ValueError` if `yt-dlp` isn't installed and `stream_mode` is enabled.
        * Removed automatic enforcing of GStreamer backend for YouTube-livestreams and made it optional.
            + The CamGear will not raise ValueError if GStreamer support is missing in OpenCV backends.
    - [ ] WebGear_RTC:
        * Removed support for assigning Custom Media Server Class(inherited from aiortc's VideoStreamTrack) in WebGear_RTC through its `config` global parameter.
        * WebGear_RTC API will now throws ValueError if `source` parameter is NoneType as well as `custom_stream` attribute is undefined.
    - [ ] Helper: 
        * Removed `restore_levelnames` method.
        * Removed `youtube_url_validator` helper method.


??? bug "Bug-fixes"
    - [x] CamGear:
        * Fixed KeyError Bug for missing attributed in meta_data json in some streaming sites.
    - [x] Helper: 
        * Removed unused imports.
    - [x] Docs:
        * Removed slugify from mkdocs which was causing invalid hyperlinks in docs.
        * Fixed GitHub hyperlinks in README.md.
        * Fixed hyperlink in announcement bar.
        * Fixed content tabs failing to work.
        * Fixed line-endings and usage example code.
        * Removed any `pafy` and `streamlink` references.
        * Fixed context and typos.
    - [x] CI: 
        * Fixed NameError bugs in WebGear_RTC CI test.
    - [x] Maintenance: 
        * Removed dead logger code causing Python's Built-in logging module to hide logs.
        * Removed unused `logging` import.
        * Updated code comments.

??? question "Pull Requests"
    * PR #268
    * PR #272
    * PR #274

??? new "New Contributors"
    * @zpapakipos

&nbsp; 

&nbsp; 


## v0.2.3 (2021-10-27)

??? tip "New Features"
    - [x] **CamGear:** 
        * Added support for `4K` Streaming URLs.
    - [x] **Helper:** 
        * Implemented logging ColorFormatter string alignment.
            + Center aligned logging Level-name and Class-name.
            + Changed `%` formatting style with modern `{`.
            + Re-added `asctime` value to Formatter string.
            + Re-arranged parameter positions in Formatter string.
    - [x] **Maintenance:**
        * Added new `.gitignore`  for specifying intentionally untracked files to ignore
            + Added more files entries to `.gitignore`.
        * Added new `.gitattributes` to manage how Git reads line endings.
            + Enabled `auto` default behavior, in case people don't have `core.autocrlf` set.
            + Enforced LF line-endings for selective files types.
            + Added Binary data files that specifies they are not text, and git should not try to change them.
            + Added Language aware diff headers.
            + Added Linguist language overrides.
    - [x] **Docs:**
        * Added new ScreenGear with WebGear_RTC API bonus example.
        * Added support for `hl_lines` argument for highlighting specific code lines.
        * Added drop-shadow effects for its `slate` theme to improve visibility.

??? success "Updates/Improvements"  
    - [x] CamGear:
        * Replaced `youtube-dl` with `yt-dlp` as pafy backend for YouTube videos pipelining.
            + Implemented hack to trick pafy into assuming `yt-dlp` as `youtube-dl`.
            + Using `sys.modules` to present `yt-dlp` as `youtube-dl`.
            + `yt-dlp` python API functions exactly similar to `youtube-dl`.
            + Replaced `youtube-dl` dependency with `yt-dlp`.
            + Replaced `youtube-dl` imports with `yt-dlp`.
    - [x] StreamGear: 
        * Updated default `stream_count` internal dict key value to 1.
    - [x] Maintenance:
        * Introduced python short-circuiting for handling logging logic.
        * Enabled logging for `check_WriteAccess` method in WriteGear, StreamGear and NetGear APIs.
    - [x] Docs:
        * Added warning for ScreenGear outputting RGBA frames instead of default BGR frames with `mss` backend.
        * Added warnings for properly formatting `output_params` when assigning external audio-source in WriteGear.
        * Added depreciation notice for Python 3.6 legacies.
        * Restructured docs to make it more user-friendly.
        * Updated, Extended and Improved context.
        * Improved code comments.
        * Updated docs admonitions.
        * Updated `Zenodo` badge.
    - [x] CI: 
        * Migrated to new Codecov Uploader in Azure Pipelines.
            + Support for the Bash Uploader will be deprecated on February 1st, 2022. See: https://docs.codecov.com/docs/about-the-codecov-bash-uploader
            + Added commands for signature and SHASUM verification to ensure integrity of the Uploader before use.
            + Replaced related bash commands.
        * Replaced `env` with `export` in ci_linux.yml.
        * Replaced `bubkoo/needs-more-info@v1` with `wow-actions/needs-more-info@v1`.
        * Added codecov secret token through `env` variable. 
        * Added wildcard to skip CI tests for doc(`.md`) files.
        * Added `.md` files to Codecov ignore list.
        * Update vidgear's banner image.

??? danger "Breaking Updates/Changes"
    - [ ] `check_WriteAccess` will now return as invalid path if writing directory does not exists. This will effect output file handling in WriteGear and StreamGear APIs.

??? bug "Bug-fixes"
    - [x] StreamGear:
        * Fixed StreamGear Malformed URI Error with HLS Segments **[PR #243 by @Vboivin]**
            + Removed the extra `'%'` character from the naming convention for segment files.
            + Used `stream_count` internal dict variable to alter template for HLS segment filenames.
    - [x] WriteGear: 
        * Fixed bug in disable_force_termination logic which accidentally disables force termination.
    - [x] WebGear_RTC: 
        * Fixed `name 'VideoStreamTrack' is not defined` bug.
    - [x] Setup.py: 
        * Fixed `TypeError` bug.
        * Fixed invalid `latest_version` retrieval.
    - [x] Helper:
        * Fixed `check_WriteAccess` failing to recognize correct permission for writing the output file on windows platform. 
            + Implemented separate logic for `Windows` and `*nix` platforms.
            + Added new `stat` import.
            + Improved warnings and error handling.
            + Added logging parameter to `check_WriteAccess`.
        * Fixed bug in check_WriteAccess that throws `OSError` while handling URLs.
    - [x] Docs:
        * Fixed bugs in WriteGear's Compression Mode with Live Audio Input example.
        * Fixed "drop-shadow" property via `filter` function conflicting with sidecard button.
            + Added new CSS classes for image, admonitions and code highlight in dark theme.
        * Several internal and external webpage links typos fixed.
        * Fixed several language typos.
    - [x] CI: 
        * Fixed Azure Pipeline coverage upload bugs.
        * Fixed random errors in CamGear `stream_mode` test.
    - [x] Bash:
        * Removed the Windows carriage returns from the shell scripts to be able to execute them on Linux. 
    - [x] Fixed logging comments.

??? question "Pull Requests"
    * PR #249
    * PR #262

??? new "New Contributors"
    * @Vboivin

&nbsp; 

&nbsp; 


## v0.2.2 (2021-09-02)

??? tip "New Features"
    - [x] **StreamGear:** 
        * Native Support for Apple HLS Multi-Bitrate Streaming format:
            + Added support for new [Apple HLS](https://developer.apple.com/documentation/http_live_streaming) _(HTTP Live Streaming)_ HTTP streaming format in StreamGear.
            + Implemented default workflow for auto-generating primary HLS stream of same resolution and framerate as source.
            + Added HLS support in *Single-Source* and *Real-time Frames* Modes.
            + Implemented inherit support for `fmp4` and `mpegts` HLS segment types.
            + Added adequate default parameters required for trans-coding HLS streams.
            + Added native support for HLS live-streaming.
            + Added `"hls"` value to `format` parameter for easily selecting HLS format.
            + Added HLS support in `-streams` attribute for transcoding additional streams.
            + Added support for `.m3u8` and `.ts` extensions in `clear_prev_assets` workflow.
            + Added validity check for `.m3u8` extension in output when HLS format is used.
            + Separated DASH and HLS command handlers.
            + Created HLS format exclusive parameters.
            + Implemented `-hls_base_url` FFMpeg parameter support.
        * Added support for audio input from external device:
            + Implemented support for audio input from external device.
            + Users can now easily add audio device and decoder by formatting them as python list.
            + Modified `-audio` parameter to support `list` data type as value.
            + Modified `validate_audio` helper function to validate external audio devices.
        * Added `-seg_duration` to control segment duration.
    - [x] **NetGear:**
        * New SSH Tunneling Mode for remote connection:
            + New SSH Tunneling Mode for connecting ZMQ sockets across machines via SSH tunneling.
            + Added new `ssh_tunnel_mode` attribute to enable ssh tunneling at provide address at server end only.
            + Implemented new `check_open_port` helper method to validate availability of host at given open port.
            + Added new attributes `ssh_tunnel_keyfile` and `ssh_tunnel_pwd` to easily validate ssh connection.
            + Extended this feature to be compatible with bi-directional mode and auto-reconnection.
            + Disabled support for exclusive Multi-Server and Multi-Clients modes.
            + Implemented logic to automatically enable `paramiko` support if installed.
            + Reserved port-`47` for testing.
        * Additional colorspace support for input frames with Frame-Compression enabled:
            + Allowed to manually select colorspace on-the-fly with JPEG frame compression.
            + Updated `jpeg_compression` dict parameter to support colorspace string values.
            + Added all supported colorspace values by underline `simplejpeg` library.
            + Server enforced frame-compression colorspace on client(s).
            + Enable "BGR" colorspace by default.
            + Added Example for changing incoming frames colorspace with NetGear's Frame Compression.
            + Updated Frame Compression parameters in NetGear docs.
            + Updated existing CI tests to cover new frame compression functionality.
    - [x] **NetGear_Async:**
        * New exclusive Bidirectional Mode for bidirectional data transfer:
            + NetGear_Async's first-ever exclusive Bidirectional mode with pure asyncio implementation.
            + Bidirectional mode is only available with User-defined Custom Source(i.e. `source=None`)
            + Added support for `PAIR` & `REQ/REP` bidirectional patterns for this mode.
            + Added powerful `asyncio.Queues` for handling user data and frames in real-time.
            + Implemented new `transceive_data` method  to Transmit _(in Recieve mode)_ and Receive _(in Send mode)_ data in real-time.
            + Implemented `terminate_connection` internal asyncio method to safely terminate ZMQ connection and queues.
            + Added `msgpack` automatic compression encoding and decoding of data and frames in bidirectional mode.
            + Added support for `np.ndarray` video frames.
            + Added new `bidirectional_mode` attribute for enabling this mode.
            + Added 8-digit random alphanumeric id generator for each device.
            + NetGear_Async will throw `RuntimeError` if bidirectional mode is disabled at server or client but not both.
        * Added new `disable_confirmation` used to force disable termination confirmation from client in `terminate_connection`.
        * Added `task_done()` method after every `get()` call to gracefully terminate queues.
        * Added new `secrets` and `string` imports.
    - [x] **WebGear:** 
        * Updated JPEG Frame compression with `simplejpeg`:
            + Implemented JPEG compression algorithm for 4-5% performance boost at cost of minor loss in quality.
            + Utilized `encode_jpeg` and `decode_jpeg` methods to implement turbo-JPEG transcoding with `simplejpeg`.
            + Added new options to control JPEG frames *quality*, enable fastest *dct*, fast *upsampling*  to boost performance.
            + Added new `jpeg_compression`, `jpeg_compression_quality`, `jpeg_compression_fastdct`, `jpeg_compression_fastupsample` attributes.
            + Enabled fast dct by default with JPEG frames at `90%`.
            + Incremented default frame reduction to `25%`.
            + Implemented automated grayscale colorspace frames handling.
            + Updated old and added new usage examples.
            + Dropped support for depreciated attributes from WebGear and added new attributes.
        * Added new WebGear Theme: _(Checkout at https://github.com/abhiTronix/vidgear-vitals)_
            - Added responsive image scaling according to screen aspect ratios.
            - Added responsive text scaling.
            - Added rounded border and auto-center to image tag.
            - Added bootstrap css properties to implement auto-scaling.
            - Removed old `resize()` hack.
            - Improved text spacing and weight.
            - Integrated toggle full-screen to new implementation.
            - Hide Scrollbar both in WebGear_RTC and WebGear Themes.
            - Beautify files syntax and updated files checksum.
            - Refactor files and removed redundant code.
            - Bumped theme version to `v0.1.2`.
    - [x] **WebGear_RTC:**
        * Added native support for middlewares:
            + Added new global `middleware` variable for easily defining Middlewares as list.
            + Added validity check for Middlewares.
            + Added tests for middlewares support.
            + Added example for middlewares support.
            + Extended middlewares support to WebGear API too.
            + Added related imports.
        * Added new WebGear_RTC Theme:  _(Checkout at https://github.com/abhiTronix/vidgear-vitals)_
            + Implemented new responsive video scaling according to screen aspect ratios.
            + Added bootstrap CSS properties to implement auto-scaling.
            + Removed old `resize()` hack.
            + Beautify files syntax and updated files checksum.
            + Refactored files and removed redundant code.
            + Bumped theme version to `v0.1.2`
    - [x] **Helper:** 
        * New automated interpolation selection for gears:
            + Implemented `retrieve_best_interpolation` method to automatically select best available interpolation within OpenCV.
            + Added support for this method in WebGear, WebGear_RTC and Stabilizer Classes/APIs.
            + Added new CI tests for this feature.
        * Implemented `get_supported_demuxers` method to get list of supported demuxers.
    - [x] **CI:**
        * Added new `no-response` work-flow for stale issues.
        * Added new CI tests for SSH Tunneling Mode.
        * Added `paramiko` to CI dependencies.
        * Added support for `"hls"` format in existing CI tests.
        * Added new functions `check_valid_m3u8` and `extract_meta_video` for validating HLS files.
        * Added new `m3u8` dependency to CI workflows.
        * Added complete CI tests for NetGear_Async's new Bidirectional Mode:
            + Implemented new exclusive `Custom_Generator` class for testing bidirectional data dynamically on server-end.
            + Implemented new exclusive `client_dataframe_iterator` method for testing bidirectional data on client-end.
            + Implemented `test_netgear_async_options` and `test_netgear_async_bidirectionalmode` two new tests.
            + Added `timeout` value on server end in CI tests.
    - [x] **Setup.py:**
        * Added new `cython` and `msgpack` dependency.
        * Added `msgpack` and `msgpack_numpy` to auto-install latest.
    - [x] **BASH:** 
        * Added new `temp_m3u8` folder for generating M3U8 assets in CI tests.
    - [x] **Docs:**
        * Added docs for new Apple HLS StreamGear format:
            + Added StreamGear HLS transcoding examples for both StreamGear modes.
            + Updated StreamGear parameters to w.r.t new HLS configurations.
            + Added open-sourced *"Sintel" - project Durian Teaser Demo* with StreamGear's HLS stream using `Clappr` and raw.githack.com.
            + Added new HLS chunks at https://github.com/abhiTronix/vidgear-docs-additionals for StreamGear
            + Added support for HLS video in Clappr within `custom.js` using HlsjsPlayback plugin.
            + Added support for Video Thumbnail preview for HLS video in Clappr within `custom.js`
            + Added `hlsjs-playback.min.js` JS script and suitable configuration for HlsjsPlayback plugin.
            + Added custom labels for quality levels selector in `custom.js`.
            + Added new docs content related to new Apple HLS format.
            + Updated DASH chunk folder at https://github.com/abhiTronix/vidgear-docs-additionals.
            + Added example for audio input support from external device in StreamGear.
            + Added steps for using `-audio` attribute on different OS platforms in StreamGear.
        * Added usage examples for NetGear_Async's Bidirectional Mode:
            + Added new Usage examples and Reference doc for NetGear_Async's Bidirectional Mode.
            + Added new image asset for NetGear_Async's Bidirectional Mode.
            + Added NetGear_Async's `option` parameter reference.
            + Updated NetGear_Async definition in docs.
            + Changed font size for Helper methods.
            + Renamed `Bonus` section to `References` in `mkdocs.yml`.
        * Added Gitter sidecard embed widget:
            + Imported gitter-sidecar script to `main.html`.
            + Updated `custom.js` to set global window option.
            + Updated Sidecard UI in `custom.css`.
        * Added bonus examples to help section:
            + Implemented a curated list of more advanced examples with unusual configuration for each API.
        * Added several new contents and updated context.
        * Added support for search suggestions, search highlighting and search sharing _(i.e. deep linking)_
        * Added more content to docs to make it more user-friendly.
        * Added warning that JPEG Frame-Compression is disabled with Custom Source in WebGear.
        * Added steps for identifying and specifying sound card on different OS platforms in WriteGear.
        * Added Zenodo DOI badge and its reference in BibTex citations.
        * Added `extra.homepage` parameter, which allows for setting a dedicated URL for `site_url`.
        * Added `pymdownx.striphtml` plugin for stripping comments.
        * Added complete docs for SSH Tunneling Mode.
        * Added complete docs for NetGear's SSH Tunneling Mode.
        * Added `pip` upgrade related docs.
        * Added docs for installing vidgear with only selective dependencies
        * Added new `advance`/`experiment` admonition with new background color.
        * Added new icons SVGs for `advance` and `warning` admonition.
        * Added new usage example and related information.
        * Added new image assets for ssh tunneling example.
        * Added new admonitions
        * Added new FAQs.
          

??? success "Updates/Improvements"
    - [x] VidGear Core: 
        * New behavior to virtually isolate optional API specific dependencies by silencing `ImportError` on all VidGear's APIs import.
        * Implemented algorithm to cache all imports on startup but silence any `ImportError` on missing optional dependency.
        * Now `ImportError` will be raised only any certain API specific dependency is missing during given API's initialization.
        * New `import_dependency_safe` to imports specified dependency safely with `importlib` module.
        * Replaced all APIs imports with `import_dependency_safe`.
        * Added support for relative imports in `import_dependency_safe`.
        * Implemented `error` parameter to by default `ImportError` with a meaningful message if a dependency is missing, Otherwise if `error = log` a warning will be logged and on `error = silent` everything will be quit. But If a dependency is present, but older than specified, an error is raised if specified.
        * Implemented behavior that if a dependency is present, but older than `min_version` specified, an error is raised always.
        * Implemented `custom_message` to display custom message on error instead of default one.
        * Implemented separate `import_core_dependency` function to import and check for specified core dependency. 
        * `ImportError` will be raised immediately if core dependency not found.
    - [x] StreamGear: 
        * Replaced depreciated `-min_seg_duration` flag with `-seg_duration`.
        * Removed redundant `-re` flag from RTFM.
        * Improved Live-Streaming performance by disabling SegmentTimline
        * Improved DASH assets detection for removal by using filename prefixes.
    - [x] NetGear:
        * Replaced `np.newaxis` with `np.expand_dims`.
        * Replaced `random` module with `secrets` while generating system ID.
        * Update array indexing with `np.copy`.
    - [x] NetGear_Async:
        * Improved custom source handling.
        * Removed deprecated `loop` parameter from asyncio methods.
        * Re-implemented `skip_loop` parameter in `close()` method.
        * `run_until_complete` will not used if `skip_loop` is enabled.
        * `skip_loop` now will create asyncio task instead and will enable `disable_confirmation` by default.
        * Replaced `create_task` with `ensure_future` to ensure backward compatibility with python-3.6 legacies.
        * Simplified code for `transceive_data` method.
    - [x] WebGear_RTC: 
        * Improved handling of failed ICE connection.
        * Made `is_running` variable globally available for internal use.
    - [x] Helper: 
        * Added `4320p` resolution support to `dimensions_to_resolutions` method.
        * Implemented new `delete_file_safe` to safely delete files at given path.
        * Replaced `os.remove` calls with `delete_file_safe`.
        * Added support for filename prefixes in `delete_ext_safe` method.
        * Improved and simplified `create_blank_frame` functions frame channels detection.
        * Added `logging` parameter to capPropId function to forcefully discard any error(if required).
    - [x] Setup.py: 
        * Added patch for `numpy` dependency, `numpy` recently dropped support for python 3.6.x legacies. See https://github.com/numpy/numpy/releases/tag/v1.20.0
        * Removed version check on certain dependencies.
        * Re-added `aiortc` to auto-install latest version.
    - [x] Asyncio: 
        * Changed `asyncio.sleep` value to `0`.
            + The amount of time sleep is irrelevant; the only purpose await asyncio.sleep() serves is to force asyncio to suspend execution to the event loop, and give other tasks a chance to run. Also, `await asyncio.sleep(0)` will achieve the same effect. https://stackoverflow.com/a/55782965/10158117
    - [x] License: 
        * Dropped publication year range to avoid confusion. _(Signed and Approved by @abhiTronix)_
        * Updated Vidgear license's year of first publication of the work in accordance with US copyright notices defined by Title 17, Chapter 4(Visually perceptible copies): https://www.copyright.gov/title17/92chap4.html
        * Reflected changes in all copyright notices.
    - [x] CI: 
        * Updated macOS VM Image to latest in azure devops.
        * Updated VidGear Docs Deployer Workflow.
        * Updated WebGear_RTC CI tests.
        * Removed redundant code from CI tests.
        * Updated tests to increase coverage.
        * Enabled Helper tests for python 3.8+ legacies.
        * Enabled logging in `validate_video` method.
        * Added `-hls_base_url` to streamgear tests.
        * Update `mpegdash` dependency to `0.3.0-dev2` version in Appveyor.
        * Updated CI tests for new HLS support
        * Updated CI tests from scratch for new native HLS support in StreamGear.
        * Updated test patch for StreamGear.
        * Added exception for RunTimeErrors in NetGear CI tests.
        * Added more directories to Codecov ignore list.
        * Imported relative `logger_handler` for asyncio tests.
    - [x] Docs:
        * Re-positioned few docs comments at bottom for easier detection during stripping.
        * Updated to new extra `analytics` parameter in Material Mkdocs.
        * Updated dark theme to `dark orange`.
        * Changed fonts => text: `Muli` & code: `Fira Code`
        * Updated fonts to `Source Sans Pro`.
        * Updated `setup.py` update-link for modules.
        * Re-added missing StreamGear Code docs.
        * Several minor tweaks and typos fixed.
        * Updated `404.html` page.
        * Updated admonitions colors and beautified `custom.css`.
        * Replaced VideoGear & CamGear with OpenCV in CPU intensive examples.
        * Updated `mkdocs.yml` with new changes and URLs.
        * Moved FAQ examples to bonus examples.
        * Moved StreamGear primary modes to separate sections for better readability.
        * Implemented separate overview and usage example pages for StreamGear primary modes.
        * Improved StreamGear docs context and simplified language.
        * Renamed StreamGear `overview` page to `introduction`.
        * Re-written Threaded-Queue-Mode from scratch with elaborated functioning.
        * Replace *Paypal* with *Liberpay* in `FUNDING.yml`.
        * Updated FFmpeg Download links.
        * Reverted UI change in CSS.
        * Updated `changelog.md` and fixed clutter.
        * Updated `README.md` and `mkdocs.yml` with new additions
        * Updated context for CamGear example.
        * Restructured and added more content to docs.
        * Updated comments in source code.
        * Removed redundant data table tweaks from `custom.css`.
        * Re-aligned badges in README.md.
        * Beautify `custom.css`.
        * Updated `mkdocs.yml`.
        * Updated context and fixed typos.
        * Added missing helper methods in Reference.
        * Updated Admonitions.
        * Updates images assets.
        * Bumped CodeCov.
    - [x] Logging:
        * Improved logging level-names.
        * Updated logging messages.
    - [x] Minor tweaks to `needs-more-info` template.
    - [x] Updated issue templates and labels.
    - [x] Removed redundant imports.

??? danger "Breaking Updates/Changes"
    - [ ] Virtually isolated all API specific dependencies, Now `ImportError` for API-specific dependencies will be raised only when any of them is missing at API's initialization.
    - [ ] Renamed `delete_safe` to `delete_ext_safe`.
    - [ ] Dropped support for `frame_jpeg_quality`, `frame_jpeg_optimize`, `frame_jpeg_progressive` attributes from WebGear.

??? bug "Bug-fixes"
    - [x] CamGear:
        * Hot-fix for Live Camera Streams:
            + Added new event flag to keep check on stream read.
            + Implemented event wait for  `read()` to block it when source stream is busy.
            + Added and Linked `THREAD_TIMEOUT` with event wait timout.
            + Improved backward compatibility of new additions.
        * Enforced logging for YouTube live.
    - [x] NetGear: 
        * Fixed Bidirectional Video-Frame Transfer broken with frame-compression:
            + Fixed `return_data` interfering with return JSON-data in receive mode.
            + Fixed logic.
        * Fixed color-subsampling interfering with colorspace.
        * Patched external `simplejpeg` bug. Issue: https://gitlab.com/jfolz/simplejpeg/-/issues/11
            + Added `np.squeeze` to drop grayscale frame's 3rd dimension on Client's end.
        * Fixed bug that cause server end frame dimensions differ from client's end when frame compression enabled.
    - [X] NetGear_Async: 
        * Fixed bug related asyncio queue freezing on calling `join()`.
        * Fixed ZMQ connection bugs in bidirectional mode.
        * Fixed several critical bugs in event loop handling.
        * Fixed several bugs in bidirectional mode implementation.
        * Fixed missing socket termination in both server and client end.
        * Fixed `timeout` parameter logic.
        * Fixed typos in error messages.
    - [x] WebGear_RTC: 
        * Fixed stream freezes after web-page reloading:
            + Implemented new algorithm to continue stream even when webpage is reloaded.
            + Inherit and modified `next_timestamp` VideoStreamTrack method for generating accurate timestamps.
            + Implemented `reset_connections` callable to reset all peer connections and recreate Video-Server timestamps. (Implemented by @kpetrykin)
            + Added `close_connection` endpoint in JavaScript to inform server page refreshing.(Thanks to @kpetrykin)
            + Added exclusive reset connection node `/close_connection` in routes.
            + Added `reset()` method to Video-Server class for manually resetting timestamp clock.
            + Added `reset_enabled` flag to keep check on reloads.
            + Fixed premature webpage auto-reloading.
            + Added additional related imports.
        * Fixed web-page reloading bug after stream ended:
            + Disable webpage reload behavior handling for Live broadcasting.
            + Disable reload CI test on Windows machines due to random failures.
            + Improved handling of failed ICE connection.
        * Fixed Assertion error bug:
            + Source must raise MediaStreamError when stream ends instead of returning None-type.
    - [x] WebGear
        * Removed format specific OpenCV decoding and encoding support for WebGear.
    - [x] Helper: 
        * Regex bugs fixed:
            + New improved regex for discovering supported encoders in `get_supported_vencoders`.
            + Re-implemented check for extracting only valid output protocols in `is_valid_url`.
            + Minor tweaks for better regex compatibility.
        * Bugfix related to OpenCV import:
            + Bug fixed for OpenCV import comparison test failing with Legacy versions and throwing `ImportError`.
            + Replaced `packaging.parse_version` with more robust `distutils.version`.
        * Fixed bug with `create_blank_frame` that throws error with gray frames:
            + Implemented automatic output channel correction inside `create_blank_frame` function.
            + Extended automatic output channel correction support to asyncio package.
        * Implemented `RTSP` protocol validation as _demuxer_, since it's not a protocol but a demuxer.
        * Removed redundant `logger_handler`, `mkdir_safe`, `retrieve_best_interpolation`, `capPropId` helper functions from asyncio package. Relatively imported helper functions from non-asyncio package.
        * Removed unused `aiohttp` dependency.
        * Removed `asctime` formatting from logging.
    - [x] StreamGear: 
        * Fixed Multi-Bitrate HLS VOD streams:
            + Re-implemented complete workflow for Multi-Bitrate HLS VOD streams.
            + Extended support to both *Single-Source* and *Real-time Frames* Modes.
        * Fixed bugs with audio-video mapping.
        * Fixed master playlist not generating in output.
        * Fixed improper `-seg_duration` value resulting in broken pipeline.
        * Fixed expected aspect ratio not calculated correctly for additional streams.
        * Fixed stream not terminating when provided input from external audio device.
        * Fixed bugs related to external audio not mapped correctly in HLS format.
        * Fixed OPUS audio fragments not supported with MP4 video in HLS.
        * Fixed unsupported high audio bit-rate bug.
    - [x] Setup.py: 
        * Fixed `latest_version` returning incorrect version for some PYPI packages.
        * Removed `latest_version` variable support from `simplejpeg`.
        * Fixed `streamlink` only supporting requests==2.25.1 on Windows.
        * Removed all redundant dependencies like `colorama`, `aiofiles`, `aiohttp`.
        * Fixed typos in dependencies.
    - [x] Setup.cfg: 
        * Replaced dashes with underscores to remove warnings.
    - [x] CI:
        * Replaced buggy `starlette.TestClient` with `async-asgi-testclient` in WebGear_RTC
        * Removed `run()` method and replaced with pure asyncio implementation.
        * Added new `async-asgi-testclient` CI dependency.
        * Fixed `fake_picamera` class logger calling `vidgear` imports prematurely before importing `picamera` class in tests.
            + Implemented new `fake_picamera` class logger inherently with `logging` module.
            + Moved `sys.module` logic for faking to `init.py`.
            + Added `__init__.py` to ignore in Codecov.
        * Fixed event loop closing prematurely while reloading:
            + Internally disabled suspending event loop while reloading.
        * Event Policy Loop patcher added for WebGear_RTC tests.
        * Fixed `return_assets_path` path bug.
        * Fixed typo in `TimeoutError` exception import.
        * Fixed eventloop is already closed bug.
        * Fixed eventloop bugs in Helper CI tests.
        * Fixed several minor bugs related to new CI tests.
        * Fixed bug in PiGear tests. 
    - [x] Docs:
        * Fixed 404 page does not work outside the site root with mkdocs.
        * Fixed markdown files comments not stripped when converted to HTML.
        * Fixed missing heading in VideoGear.
        * Typos in links and code comments fixed.
        * Several minor tweaks and typos fixed.
        * Fixed improper URLs/Hyperlinks and related typos.
        * Fixed typos in usage examples.
        * Fixed redundant properties in CSS.
        * Fixed bugs in `mkdocs.yml`.
        * Fixed docs contexts and typos.
        * Fixed `stream.release()` missing in docs.
        * Fixed several typos in code comments.
        * Removed dead code from docs.
    - [x] Refactored Code and reduced redundancy.
    - [x] Fixed shutdown in `main.py`.
    - [x] Fixed logging comments.

??? question "Pull Requests"
    * PR #210
    * PR #215
    * PR #222
    * PR #223
    * PR #227
    * PR #231
    * PR #233
    * PR #237  
    * PR #239 
    * PR #243 

??? new "New Contributors"
    * @kpetrykin

&nbsp; 

&nbsp; 


## v0.2.1 (2021-04-25)

??? tip "New Features"
    - [x] **WebGear_RTC:**
        * A new API that is similar to WeGear API in all aspects but utilizes WebRTC standard instead of Motion JPEG for streaming.
        * Now it is possible to share data and perform teleconferencing peer-to-peer, without requiring that the user install plugins or any other third-party software.
        * Added a flexible backend for `aiortc` - a python library for Web Real-Time Communication (WebRTC).
        * Integrated all functionality and parameters of WebGear into WebGear_RTC API.
        * Implemented JSON Response with a WebRTC Peer Connection of Video Server.
        * Added a internal `RTC_VideoServer` server on WebGear_RTC, a inherit-class to aiortc's VideoStreamTrack API.
        * New Standalone UI Default theme v0.1.1 for WebGear_RTC from scratch without using 3rd-party assets. (by @abhiTronix)
        * New `custom.js` and `custom.css` for custom responsive behavior.
        * Added WebRTC support to `custom.js` and ensured compatibility with WebGear_RTC.
        * Added example support for ICE framework and STUN protocol like WebRTC features to `custom.js`.
        * Added `resize()` function to `custom.js` to automatically adjust `video` & `img` tags for smaller screens.
        * Added WebGear_RTC support in main.py for easy access through terminal using `--mode` flag.
        * Integrated all WebGear_RTC enhancements to WebGear Themes.
        * Added CI test for WebGear_RTC.
        * Added complete docs for WebGear_RTC API.
        * Added bare-minimum as well as advanced examples usage code.
        * Added new theme images.
        * Added Reference and FAQs.
    - [x] **CamGear API:**
        * New Improved Pure-Python Multiple-Threaded Implementation:
            + Optimized Threaded-Queue-Mode Performance. (PR by @bml1g12)
            + Replaced regular `queue.full` checks followed by sleep with implicit sleep with blocking `queue.put`.
            + Replaced regular `queue.empty` checks followed by queue.
            + Replaced `nowait_get` with a blocking `queue.get` natural empty check.
            + Up-to 2x performance boost than previous implementations. 
        * New `THREAD_TIMEOUT` attribute to prevent deadlocks:
            + Added support for `THREAD_TIMEOUT` attribute to its `options` parameter.
            + Updated CI Tests and docs.
    - [x] **WriteGear API:**
        * New more robust handling of default video-encoder in compression mode:
            + Implemented auto-switching of default video-encoder automatically based on availability.
            + API now selects Default encoder based on priority: `"libx264" > "libx265" > "libxvid" > "mpeg4"`.
            + Added `get_supported_vencoders` Helper method to enumerate Supported Video Encoders.
            + Added common handler for `-c:v` and `-vcodec` flags.
    - [x] **NetGear API:**
        * New Turbo-JPEG compression with simplejpeg
            + Implemented JPEG compression algorithm for 4-5% performance boost at cost of minor loss in quality.
            + Utilized `encode_jpeg` and `decode_jpeg` methods to implement turbo-JPEG transcoding with `simplejpeg`.
            + Added options to control JPEG frames quality, enable fastest dct, fast upsampling  to boost performance.
            + Added new `jpeg_compression`, `jpeg_compression_quality`, `jpeg_compression_fastdct`, `jpeg_compression_fastupsample` attributes.
            + Enabled fast dct by default with JPEG frames at 90%.
            + Added Docs for JPEG Frame Compression.
    - [x] **WebGear API:** 
        * New modular and flexible configuration for Custom Sources:
            + Implemented more convenient approach for handling custom source configuration.
            + Added new `config` global variable for this new behavior.
            + Now None-type `source` parameter value is allowed for defining own custom sources.
            + Added new Example case and Updates Docs for this feature.
            + Added new CI Tests.
        * New Browser UI Updates:
            + New Standalone UI Default theme v0.1.0 for browser (by @abhiTronix)
            + Completely rewritten theme from scratch with only local resources.
            + New `custom.js` and `custom.css` for custom responsive behavior.
            + New sample glow effect with css.
            + New sample click to full-screen behavior with javascript.
            + Removed all third-party theme dependencies.
            + Update links to new github server `abhiTronix/vidgear-vitals`
            + Updated docs with new theme's screenshots.
        * Added `enable_infinite_frames` attribute for enabling infinite frames.
        * Added New modular and flexible configuration for Custom Sources.
        * Bumped WebGear Theme Version to v0.1.1.
        * Updated Docs and CI tests.
    - [x] **ScreenGear API:**
        * Implemented Improved Pure-Python Multiple-Threaded like CamGear.
        * Added support for `THREAD_TIMEOUT` attribute to its `options` parameter.
    - [X] **StreamGear API:**
        * Enabled pseudo live-streaming flag `re` for live content.
    - [x] **Docs:**
        * Added new native docs versioning to mkdocs-material.
        * Added new examples and few visual tweaks.
        * Updated Stylesheet for versioning.
        * Added new DASH video chunks at https://github.com/abhiTronix/vidgear-docs-additionals for StreamGear and Stabilizer streams.
        * Added open-sourced "Tears of Steel" * project Mango Teaser video chunks.
        * Added open-sourced "Subspace Video Stabilization" http://web.cecs.pdx.edu/~fliu/project/subspace_stabilization/ video chunks.
        * Added support for DASH Video Thumbnail preview in Clappr within `custom.js`.
        * Added responsive clappr DASH player with bootstrap's `embed-responsive`.
        * Added new permalink icon and slugify to toc.
        * Added "back-to-top" button for easy navigation.
    - [x] **Helper:**
        * New GitHub Mirror with latest Auto-built FFmpeg Static Binaries:
            + Replaced new GitHub Mirror `abhiTronix/FFmpeg-Builds` in helper.py
            + New CI maintained Auto-built FFmpeg Static Binaries.
            + Removed all 3rd-party and old links for better compatibility and Open-Source reliability.
            + Updated Related CI tests.
        * Added auto-font-scaling for `create_blank_frame` method.
        * Added `c_name` parameter to `generate_webdata` and `download_webdata` to specify class.
        * A more robust Implementation of Downloading Artifacts:
            + Added a custom HTTP `TimeoutHTTPAdapter` Adapter with a default timeout for all HTTP calls based on [this GitHub comment]().
            + Implemented http client and the `send()` method to ensure that the default timeout is used if a timeout argument isn't provided.
            + Implemented Requests session`with` block to exit properly even if there are unhandled exceptions.
            + Add a retry strategy to custom `TimeoutHTTPAdapter` Adapter with max 3 retries and sleep(`backoff_factor=1`) between failed requests.
        * Added `create_blank_frame` method to create bland frames with suitable text.
    - [x] **[CI] Continuous Integration:**
        * Added new fake frame generated for fake `picamera` class with numpy.
        * Added new `create_bug` parameter to fake `picamera` class for emulating various artificial bugs.
        * Added float/int instance check on `time_delay` for camgear and pigear.
        * Added `EXIT_CODE` to new timeout implementation for pytests to upload codecov report when no timeout.
        * Added auxiliary classes to  fake `picamera` for facilitating the emulation.
        * Added new CI tests for PiGear Class for testing on all platforms.
        * Added `shutdown()` function to gracefully terminate WebGear_RTC API.
        * Added new `coreutils` brew dependency.
        * Added handler for variable check on exit and codecov upload.
        * Added `is_running` flag to WebGear_RTC to exit safely.    
    - [x] **Setup:**
        * New automated latest version retriever for packages:
            + Implemented new `latest_version` method to automatically retrieve latest version for packages.
            + Added Some Dependencies.
        * Added `simplejpeg` package for all platforms.

??? success "Updates/Improvements"
    - [x] Added exception for RunTimeErrors in NetGear CI tests.
    - [x] WriteGear: Critical file write access checking method:
        * Added new `check_WriteAccess` Helper method.
        * Implemented a new robust algorithm to check if given directory has write-access.
        * Removed old behavior which gives irregular results.
    - [x] Helper: Maintenance Updates
        * Added workaround for Python bug.
        * Added `safe_mkdir` to `check_WriteAccess` to automatically create non-existential parent folder in path.
        * Extended `check_WriteAccess` Patch to StreamGear.
        * Simplified `check_WriteAccess` to handle Windows envs easily.
        * Updated FFmpeg Static Download URL for WriteGear.
        * Implemented fallback option for auto-calculating bitrate from extracted audio sample-rate in `validate_audio` method.
    - [x] Docs: General UI Updates
        * Updated Meta tags for og site and twitter cards.
        * Replaced Custom dark theme toggle with mkdocs-material's official Color palette toggle
        * Added example for external audio input and creating segmented MP4 video in WriteGear FAQ.
        * Added example for YouTube streaming with WriteGear.
        * Removed custom `dark-material.js` and `header.html` files from theme.
        * Added blogpost link for detailed information on Stabilizer Working.
        * Updated `mkdocs.yml` and `custom.css` configuration.
        * Remove old hack to resize clappr DASH player with css.
        * Updated Admonitions.
        * Improved docs contexts.
        * Updated CSS for version-selector-button.
        * Adjusted files to match new themes.
        * Updated welcome-bot message for typos.
        * Removed redundant FAQs from NetGear Docs.
        * Updated Assets Images.
        * Updated spacing.
    - [x] CI:
        * Removed unused `github.ref` from yaml.
        * Updated OpenCV Bash Script for Linux envs.
        * Added `timeout-minutes` flag to github-actions workflow.
        * Added `timeout` flag to pytest.
        * Replaced Threaded Gears with OpenCV VideoCapture API.
        * Moved files and Removed redundant code.
        * Replaced grayscale frames with color frames for WebGear tests. 
        * Updated pytest timeout value to 15mins.
        * Removed `aiortc` automated install on Windows platform within setup.py.
        * Added new timeout logic to continue to run on external timeout for GitHub Actions Workflows.
        * Removed unreliable old timeout solution from WebGear_RTC.
        * Removed `timeout_decorator` and `asyncio_timeout` dependencies for CI.
        * Removed WebGear_RTC API exception from codecov.
        * Implemented new fake `picamera` class to CI utils for emulating RPi Camera-Module Real-time capabilities.
        * Implemented new `get_RTCPeer_payload` method to receive WebGear_RTC peer payload.
        * Removed PiGear from Codecov exceptions.
        * Disable Frame Compression in few NetGear tests failing on frame matching.
        * Updated NetGear CI  tests to support new attributes
        * Removed warnings and updated yaml
            + Added `pytest.ini` to address multiple warnings.
            + Updated azure workflow condition syntax.
        * Update `mike` settings for mkdocs versioning.
        * Updated codecov configurations.
        * Minor logging and docs updates.
        * Implemented pytest timeout for azure pipelines for macOS envs.
        * Added `aiortc` as external dependency in `appveyor.yml`.
        * Re-implemented WebGear_RTC improper offer-answer handshake in CI tests.
        * WebGear_RTC CI Updated with `VideoTransformTrack` to test stream play.
        * Implemented fake `AttributeError` for fake picamera class.
        * Updated PiGear CI tests to increment codecov.
        * Update Tests docs and other minor tweaks to increase overall coverage.
        * Enabled debugging and disabled exit 1 on error in azure pipeline.
        * Removed redundant benchmark tests.
    - [x] Helper: Added missing RTSP URL scheme to `is_valid_url` method.
    - [x] NetGear_Async: Added fix for uvloop only supporting python>=3.7 legacies.
    - [x] Extended WebGear's Video-Handler scope to `https`.
    - [x] CI: Remove all redundant 32-bit Tests from Appveyor:
        * Appveyor 32-bit Windows envs are actually running on 64-bit machines.
        * More information here: https://help.appveyor.com/discussions/questions/20637-is-it-possible-to-force-running-tests-on-both-32-bit-and-64-bit-windows
    - [x] Setup: Removed `latest_version` behavior from some packages.
    - [x] NetGear_Async: Revised logic for handling uvloop for all platforms and legacies.
    - [x] Setup: Updated logic to install uvloop-"v0.14.0" for python-3.6 legacies.
    - [x] Removed any redundant code from webgear.
    - [x] StreamGear:
        * Replaced Ordinary dict with Ordered Dict to use `move_to_end` method.
        * Moved external audio input to output parameters dict.
        * Added additional imports.
        * Updated docs to reflect changes.
    - [x] Numerous Updates to Readme and `mkdocs.yml`.
    - [x] Updated font to `FONT_HERSHEY_SCRIPT_COMPLEX` and enabled logging in create_blank_frame.
    - [x] Separated channels for downloading and storing theme files for WebGear and WebGear_RTC APIs.
    - [x] Removed `logging` condition to always inform user in a event of FFmpeg binary download failure.
    - [x] WebGear_RTC: 
        * Improved auto internal termination.
        * More Performance updates through `setCodecPreferences`.
        * Moved default Video RTC video launcher to `__offer`.
    - [x] NetGear_Async: Added timeout to client in CI tests.
    - [x] Reimplemented and updated `changelog.md`.
    - [x] Updated code comments.
    - [x] Setup: Updated keywords and classifiers.
    - [x] Bumped codecov.

??? danger "Breaking Updates/Changes"
    - [ ] WriteGear will automatically switch video encoder to default if specified encoder not found.
    - [ ] WriteGear will throw `RuntimeError` if no suitable default encoder found!
    - [ ] Removed format specific OpenCV decoding and encoding support for NetGear.
    - [ ] Dropped support for `compression_format`, `compression_param` attributes from NetGear.
    - [ ] Non-existent parent folder in `output_filename` value will no longer be considered as invalid in StreamGear and WriteGear APIs.
    - [ ] None-type `source` parameter value is allowed for WebGear and NetGear_Async for defining custom sources.

??? bug "Bug-fixes"
    - [x] CamGear: Fixed F821 undefined name 'queue' bug.
    - [x] NetGear_Async: Fixed `source` parameter missing `None` as default value.
    - [x] Fixed uvloops only supporting python>=3.7 in NetGear_Async.
    - [x] Helper:
        * Fixed Zombie processes in `check_output` method due a hidden bug in python. For reference: https://bugs.python.org/issue37380
        * Fixed regex in `validate_video` method.
    - [x] Docs: 
        * Invalid `site_url` bug patched in mkdocs.yml
        * Remove redundant mike theme support and its files.
        * Fixed video not centered when DASH video in fullscreen mode with clappr.
        * Fixed Incompatible new mkdocs-docs theme.
        * Fixed missing hyperlinks.
    - [x] CI: 
        * Fixed NetGear Address bug
        * Fixed bugs related to termination in WebGear_RTC.
        * Fixed random CI test failures and code cleanup.
        * Fixed string formating bug in Helper.py.
        * Fixed F821 undefined name bugs in WebGear_RTC tests.
        * NetGear_Async Tests fixes.
        * Fixed F821 undefined name bugs.
        * Fixed typo bugs in `main.py`.
        * Fixed Relative import bug in PiGear.
        * Fixed regex bug in warning filter.
        * Fixed WebGear_RTC frozen threads on exit.
        * Fixed bugs in codecov bash uploader setting for azure pipelines.
        * Fixed False-positive `picamera` import due to improper sys.module settings.
        * Fixed Frozen Threads on exit in WebGear_RTC API.
        * Fixed deploy error in `VidGear Docs Deployer` workflow
        * Fixed low timeout bug.
        * Fixed bugs in PiGear tests.
        * Patched F821 undefined name bug.
    - [x] StreamGear:
        * Fixed StreamGear throwing `Picture size 0x0 is invalid` bug with external audio.
        * Fixed default input framerate value getting discarded in Real-time Frame Mode.
        * Fixed internal list-formatting bug.
    - [x] Fixed E999 SyntaxError bug in `main.py`.
    - [x] Fixed Typo in bash script.
    - [x] Fixed WebGear freeze on reloading bug.
    - [x] Fixed anomalies in `install_opencv` bash script.
    - [x] Helper: Bug Fixed in `download_ffmpeg_binaries` method.
    - [x] Helper: Fixed OSError bug in `check_WriteAccess` method.
    - [x] Helper: Fixed Input Audio stream bitrate test failing to detect audio-bitrate in certain videos with `validate_audio` method.
    - [x] Fixed bugs in `requests` module's function arguments.
    - [x] Fixed None-type stream bug in WebGear.
    - [x] Fixed random crashes in WebGear.
    - [x] Fixed numerous CI test bugs.
    - [x] Fixed several typos.

??? question "Pull Requests"
    * PR #192
    * PR #196
    * PR #203
    * PR #206

??? new "New Contributors"
    * @bml1g12

&nbsp; 

&nbsp; 


## v0.2.0 (2021-01-01)

??? tip "New Features"
    - [x] **CamGear API:**
        * Support for various Live-Video-Streaming services:
            + Added seamless support for live video streaming sites like Twitch, LiveStream, Dailymotion etc.
            + Implemented flexible framework around `streamlink` python library with easy control over parameters and quality.
            + Stream Mode can now automatically detects whether `source` belong to YouTube or elsewhere, and handles it with appropriate API.
        * Re-implemented YouTube URLs Handler:
            + Re-implemented CamGear's YouTube URLs Handler completely from scratch.
            + New Robust Logic to flexibly handing video and video-audio streams.
            + Intelligent stream selector for selecting best possible stream compatible with OpenCV.
            + Added support for selecting stream qualities and parameters.
            + Implemented new `get_supported_quality` helper method for handling specified qualities
            + Fixed Live-Stream URLs not supported by OpenCV's Videocapture and its FFmpeg.
        * Added additional `STREAM_QUALITY` and `STREAM_PARAMS` attributes.
    - [x] **ScreenGear API:**
        * Multiple Backends Support:
            + Added new multiple backend support with new [`pyscreenshot`](https://github.com/ponty/pyscreenshot) python library.
            + Made `pyscreenshot` the default API for ScreenGear, replaces `mss`.
            + Added new `backend` parameter for this feature while retaining previous behavior.
            + Added native automated RGB to BGR conversion for default PIL backend.
            + Kept support for old `mss` for old compatibility and multi-screen support.
            + Added native dimensional support for multi-screen.
            + Added support all input from all multiple screens.
            + Updated ScreenGear Docs.
            + Updated ScreenGear CI tests.
    - [X] **StreamGear API:**
        * Changed default behaviour to support complete video transcoding.
        * Added `-livestream` attribute to support live-streaming.
        * Added additional parameters for `-livestream` attribute functionality.
        * Updated StreamGear Tests.
        * Updated StreamGear docs.
    - [x] **Stabilizer Class:** 
        * New Robust Error Handling with Blank Frames:
            + Elegantly handles all crashes due to Empty/Blank/Dark frames.
            + Stabilizer throws Warning with this new behavior instead of crashing.
            + Updated CI test for this feature.
    - [x] **Docs:**
        * Automated Docs Versioning:
            + Implemented Docs versioning through `mike` API.
            + Separate new workflow steps to handle different versions.
            + Updated docs deploy worflow to support `release` and `dev` builds.
            + Added automatic version extraction from github events.
            + Added `version-select.js` and `version-select.css` files.
        * Toggleable Dark-White Docs Support:
            + Toggle-button to easily switch dark, white and preferred theme.
            + New Updated Assets for dark backgrounds
            + New css, js files/content to implement this behavior.
            + New material icons for button.
            + Updated scheme to `slate` in `mkdocs.yml`.
        * New Theme and assets:
            + New `purple` theme with `dark-purple` accent color.
            + New images assets with updated transparent background.
            + Support for both dark and white theme.
            + Increased `rebufferingGoal` for dash videos.
            + New updated custom 404 page for docs.
        * Issue and PR automated-bots changes
            + New `need_info.yml` YAML Workflow.
            + New `needs-more-info.yml` Request-Info template.
            + Replaced Request-Info templates.
            + Improved PR and Issue welcome formatting.
        * Added custom HTML pages.
        * Added `show_root_heading` flag to disable headings in References.
        * Added new `inserAfter` function to version-select.js.
        * Adjusted hue for dark-theme for better contrast.
        * New usage examples and FAQs.
        * Added `gitmoji` for commits.
    - [x] **Continuous Integration:**
        * Maintenance Updates:
            + Added support for new `VIDGEAR_LOGFILE` environment variable in Travis CI.
            + Added missing CI tests.
            + Added logging for helper functions.
        * Azure-Pipeline workflow for MacOS envs
            + Added Azure-Pipeline Workflow for testing MacOS environment.
            + Added codecov support.
        * GitHub Actions workflow for Linux envs
            + Added GitHub Action work-flow for testing Linux environment.
        * New YAML to implement GitHub Action workflow for python 3.6, 3.7, 3,8 & 3.9 matrices.
        * Added Upload coverage to Codecov GitHub Action workflow.
        * New codecov-bash uploader for Azure Pipelines.
    - [x] **Logging:**
        * Added file support
            + Added `VIDGEAR_LOGFILE` environment variable to manually add file/dir path.
            + Reworked `logger_handler()` Helper methods (in asyncio too).
            + Added new formatter and Filehandler for handling logger files.
        * Added `restore_levelnames` auxiliary method for restoring logging levelnames.
    - [x] Added auto version extraction from package `version.py` in setup.py.

??? success "Updates/Improvements"
    - [x] Added missing Lazy-pirate auto-reconnection support for Multi-Servers and Multi-Clients Mode in NetGear API.
    - [x] Added new FFmpeg test path to Bash-Script and updated README broken links.
    - [x] Asset Cleanup:
        * Removed all third-party javascripts from projects.
        * Linked all third-party javascript directly.
        * Cleaned up necessary code from CSS and JS files.
        * Removed any copyrighted material or links.
    - [x] Rewritten Docs from scratch:
        * Improved complete docs formatting.
        * Simplified language for easier understanding.
        * Fixed `mkdocstrings` showing root headings.
        * Included all APIs methods to `mkdocstrings` docs.
        * Removed unnecessary information from docs.
        * Corrected Spelling and typos.
        * Fixed context and grammar.
        * Removed `motivation.md`.
        * Renamed many terms.
        * Fixed hyper-links.
        * Reformatted missing or improper information.
        * Fixed context and spellings in Docs files.
        * Simplified language for easy understanding.
        * Updated image sizes for better visibility.
    - [x] Bash Script: Updated to Latest OpenCV Binaries version and related changes
    - [x] Docs: Moved version-selector to header and changed default to alias.
    - [x] Docs: Updated `deploy_docs.yml` for releasing dev, stable, and release versions.
    - [x] Re-implemented overridden material theme.
    - [x] Updated docs with all new additions and examples.
    - [x] CamGear: CI Stream Mode test updated.
    - [x] Updated ReadMe.md badges.
    - [x] Updated CI tests. 
    - [x] Updated `setup.py` with new features.
    - [x] Updated `contributing.md` and `ReadMe.md`.
    - [x] Updated OpenCV version to `4.5.1-dev` in bash scripts
    - [x] Updated `changelog.md`.
    - [x] Moved WebGear API to Streaming Gears.
    - [x] Bumped Codecov.
    - [x] UI changes to version-select.js
    - [x] Docs: Retitle the versions and `mkdocs.yml` formatting updated.
    - [x] Docs: Version Selector UI reworked and other minor changes.

??? danger "Breaking Updates/Changes"
    - [ ] `y_tube` parameter renamed as `stream_mode` in CamGear API!
    - [ ] Removed Travis support and `travis.yml` deleted.

??? bug "Bug-fixes"
    - [x] Fixed StreamGear API Limited Segments Bug
    - [x] Fixed Missing links in docs and bump up version.
    - [x] CI: Fixed Appveyor need newer VM image to support Python 3.9.x matrix.
    - [x] ScreenGear BugFix: Fixed Error Handling and updated CI Tests.
    - [x] Fixed improper `mkdocs.yml` variables.
    - [x] Fixed GStreamer plugin support in bash scripts.
    - [x] Fixed typos in YAMLs and docs.
    - [x] Docs: Fixed Docs Deployer YAML bug for CI envs.
    - [x] Fixed wrong import in YAML.
    - [x] Fixed visible hyperlink on hover in dark-toggle button.
    - [x] Docs: Deployer YAML bug fixed.
    - [x] Docs YAML: issue jimporter/mike#33 patched and fixed `fetch-depth=0`.
    - [x] Docs: `version-select.js` bug fixed.
    - [x] Docs: UI Bugs Fixed.
    - [x] CI: Codecov bugfixes.
    - [x] Azure-Pipelines Codecov BugFixes.
    - [x] Fixed `version.json` not detecting properly in `version-select.js`.
    - [x] Fixed images not centered inside `<figure>` tag.
    - [x] Fixed Asset Colors.
    - [x] Fixed failing CI tests.
    - [x] Fixed Several logging bugs.

??? question "Pull Requests"
    * PR #164
    * PR #170
    * PR #173
    * PR #181
    * PR #183
    * PR #184 


&nbsp; 

&nbsp; 


## v0.1.9 (2020-08-31)

??? tip "New Features"
    - [x] **StreamGear API:**
        * New API that automates transcoding workflow for generating Ultra-Low Latency, High-Quality, Dynamic & Adaptive Streaming Formats.
        * Implemented multi-platform , standalone, highly extensible and flexible wrapper around FFmpeg for generating chunked-encoded media segments of the media, and easily accessing almost all of its parameters.
        * API automatically transcodes videos/audio files & real-time frames into a sequence of multiple smaller chunks/segments and also creates a Manifest file.
        * Added initial support for [MPEG-DASH](https://www.encoding.com/mpeg-dash/) _(Dynamic Adaptive Streaming over HTTP, ISO/IEC 23009-1)_.
        * Constructed default behavior in StreamGear, for auto-creating a Primary Stream of same resolution and framerate as source.
        * Added [TQDM](https://github.com/tqdm/tqdm) progress bar in non-debugged output for visual representation of internal processes.
        * Implemented several internal methods for preprocessing FFmpeg and internal parameters for producing streams.
        * Several standalone internal checks to ensure robust performance.
        * New `terminate()` function to terminate StremGear Safely.
        * New StreamGear Dual Modes of Operation:
            + Implemented *Single-Source* and *Real-time Frames* like independent Transcoding Modes.
            + Linked `-video_source` attribute for activating these modes
            + **Single-Source Mode**, transcodes entire video/audio file _(as opposed to frames by frame)_ into a sequence of multiple smaller segments for streaming
            + **Real-time Frames Mode**, directly transcodes video-frames _(as opposed to a entire file)_, into a sequence of multiple smaller segments for streaming
            + Added separate functions, `stream()` for Real-time Frame Mode and `transcode_source()` for Single-Source Mode for easy transcoding.
            + Included auto-colorspace detection and RGB Mode like features _(extracted from WriteGear)_, into StreamGear.  
        * New StreamGear Parameters:
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
        * New StreamGear docs, MPEG-DASH demo, and recommended DASH players list:
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
        * New StreamGear CI tests:
            + Added IO and API initialization CI tests for its Modes.
            + Added various mode Streaming check CI tests.
    - [x] **NetGear_Async API:**
        * Added new `send_terminate_signal` internal method.
        * Added `WindowsSelectorEventLoopPolicy()` for windows 3.8+ envs.
        * Moved Client auto-termination to separate method.
        * Implemented graceful termination with `signal` API on UNIX machines.
        * Added new `timeout` attribute for controlling Timeout in Connections.
        * Added missing termination optimizer (`linger=0`) flag.
        * Several ZMQ Optimizer Flags added to boost performance.
    - [x] **WriteGear API:**
        * Added support for adding duplicate FFmpeg parameters to `output_params`:
            + Added new `-clones` attribute in `output_params` parameter for handing this behavior..
            + Support to pass FFmpeg parameters as list, while maintaining the exact order it was specified.
            + Built support for `zmq.REQ/zmq.REP` and `zmq.PUB/zmq.SUB` patterns in this mode.
            + Added new CI tests debugging this behavior.
            + Updated docs accordingly.
        * Added support for Networks URLs in Compression Mode:
            + `output_filename` parameter supports Networks URLs in compression modes only
            + Added automated handling of non path/file Networks URLs as input.
            + Implemented new `is_valid_url` helper method to easily validate assigned URLs value.
            + Validates whether the given URL value has scheme/protocol supported by assigned/installed ffmpeg or not. 
            + WriteGear will throw `ValueError` if `-output_filename` is not supported.
            + Added related CI tests and docs.
        * Added `disable_force_termination` attribute in WriteGear to disable force-termination.
    - [x] **NetGear API:**
        * Added option to completely disable Native Frame-Compression:
            + Checks if any Incorrect/Invalid value is assigned on `compression_format` attribute.
            + Completely disables Native Frame-Compression.
            + Updated docs accordingly.
    - [x] **CamGear API:**
        * Added new and robust regex for identifying YouTube URLs.
        * Moved `youtube_url_validator` to Helper.
    - [x] **New `helper.py` methods:** 
        * Added `validate_video` function to validate video_source.
        * Added `extract_time` Extract time from give string value.
        * Added `get_video_bitrate` to calculate video birate from resolution, framerate, bits-per-pixels values.
        * Added `delete_safe` to safely delete files of given extension.
        * Added `validate_audio` to validate audio source.
        * Added new Helper CI tests.
            + Added new `check_valid_mpd` function to test MPD files validity.
            + Added `mpegdash` library to CI requirements.
    - [x] **Deployed New Docs Upgrades:**
        * Added new assets like _images, gifs, custom scripts, javascripts fonts etc._ for achieving better visual graphics in docs.
        * Added `clappr.min.js`, `dash-shaka-playback.js`, `clappr-level-selector.min.js` third-party javascripts locally.
        * Extended Overview docs Hyperlinks to include all major sub-pages _(such as Usage Examples, Reference, FAQs etc.)_.
        * Replaced GIF with interactive MPEG-DASH Video Example in Stabilizer Docs. 
        * Added new `pymdownx.keys` to replace `[Ctrl+C]/[⌘+C]` formats.
        * Added new `custom.css` stylescripts variables for fluid animations in docs.
        * Overridden announce bar and added donation button. 
        * Lossless WEBP compressed all PNG assets for faster loading.
        * Enabled lazy-loading for GIFS and Images for performance.
        * Reimplemented Admonitions contexts and added new ones.
        * Added StreamGear and its different modes Docs Assets.
        * Added patch for images & unicodes for PiP flavored markdown in `setup.py`.
    - [x] **Added `Request Info` and `Welcome` GitHub Apps to automate PR and issue workflow**
        * Added new `config.yml` for customizations.
        * Added various suitable configurations.
    - [x] Added new `-clones` attribute to handle FFmpeg parameter clones in StreamGear and WriteGear API.
    - [x] Added new Video-only and Audio-Only sources in bash script.
    - [x] Added new paths in bash script for storing StreamGear & WriteGear assets temporarily.

??? success "Updates/Improvements"
    - [x] Added patch for `NotImplementedError` in NetGear_Async API on Windows 3.8+ envs.
    - [x] Check for valid `output` file extension according to `format` selected in StreamGear.
    - [x] Completed migration to `travis.com`.
    - [x] Created new `temp_write` temp directory for WriteGear Assets in bash script.
    - [x] Deleted old Redundant assets and added new ones.
    - [x] Employed `isort` library to sort and group imports in Vidgear APIs.
    - [x] Enabled exception for `list, tuple, int, float` in WriteGear API's `output_params` dict.
    - [x] Enabled missing support for frame-compression in its primary Receive Mode.
    - [x] Enforced pixel formats for streams.
    - [x] Improved check for valid system path detection in WriteGear API.
    - [x] Overrided `pytest-asyncio` fixture in NetGear_Async API.
    - [x] Quoted Gear Headline for understanding each gear easily. 
    - [x] Re-Positioned Gear's banner images in overview for better readability.
    - [x] Reduced redundant try-except blocks in NetGear Async.
    - [x] Reformatted and Simplified Docs context.
    - [x] Reimplemented `return_testvideo_path` CI function with variable streams.
    - [x] Reimplemented `skip_loop` in NetGear_Async to fix `asyncio.CancelledError`.
    - [x] Reimplemented buggy audio handler in StreamGear.
    - [x] Reimplemented images with `<figure>` and `<figurecaption>` like tags.
    - [x] Removed Python < 3.8 condition from all CI tests.
    - [x] Removed or Grouped redundant code for increasing codecov.
    - [x] Removed redundant code and simplified algorithmic complexities in Gears.
    - [x] Replaced `;nbsp` with `;thinsp` and `;emsp`.
    - [x] Replaced `IOError` with more reliable `RuntimeError` in StreamGear Pipelines.
    - [x] Replaced `del` with `pop` in dicts.
    - [x] Replaced all Netgear CI tests with more reliable `try-except-final` blocks.
    - [x] Replaced simple lists with `pymdownx.tasklist`.
    - [x] Replaced subprocess `call()` with `run()` for better error handling in `execute_ffmpeg_cmd` function.
    - [x] Resized over-sized docs images. 
    - [x] Simplified `delete_safe` Helper function.
    - [x] Simplified default audio-bitrate logic in StreamGear
    - [x] Updated CI tests and cleared redundant code from NetGear_Async API.
    - [x] Updated CI with new tests and Bumped Codecov.
    - [x] Updated Issue and PR templates.
    - [x] Updated Licenses for new files and shrink images dimensions.
    - [x] Updated Missing Helpful tips and increased logging.
    - [x] Updated PR guidelines for more clarity.
    - [x] Updated WebGear examples addresses from `0.0.0.0` to `localhost`.
    - [x] Updated WriteGear and StreamGear CI tests for not supporting temp directory.
    - [x] Updated `README.md` and `changelog.md` with new changes.
    - [x] Updated `check_output` and added `force_retrieve_stderr` support to `**kwargs` to extract `stderr` output even on FFmpeg  error.
    - [x] Updated `dicts2args` to support internal repeated `coreX` FFmpeg parameters for StreamGear. 
    - [x] Updated `mkdocs.yml`, `changelog.md` and `README.md` with latest changes.
    - [x] Updated `validate_audio` Helper function will now retrieve audio-bitrate for validation.
    - [x] Updated buggy `mpegdash` dependency with custom dev fork for Windows machines.
    - [x] Updated core parameters for audio handling.
    - [x] Updated logging for debugging selected eventloops in NetGear_Async API.
    - [x] Updated termination linger to zero at Server's end.

??? danger "Breaking Updates/Changes"
    - [ ] Changed Webgear API default address to `localhost` for cross-compatibility between different platforms.
    - [ ] In Netgear_Async API, `source` value can now be NoneType for a custom frame-generator at Server-end only.
    - [ ] Temp _(such as `/tmp` in linux)_ is now not a valid directory for WriteGear & StreamGear API outputs.
    - [ ] Moved vidgear docs assets _(i.e images, gifs, javascripts and stylescripts)_ to `override` directory.

??? bug "Bug-fixes"
    - [x] Added workaround for system path not handle correctly.
    - [x] Fixed Bug: URL Audio format not being handled properly.
    - [x] Fixed Critical Bug in NetGear_Async throwing `ValueError` with None-type Source.
    - [x] Fixed Critical StreamGear Bug: FFmpeg pipeline terminating prematurely in Single-Source Mode.
    - [x] Fixed Critical external audio handler bug: moved audio-input to input_parameters.
    - [x] Fixed Frozen-threads bug in CI tests.
    - [x] Fixed Mkdocs only accepting Relative paths.
    - [x] Fixed OSError in WriteGear's compression mode.
    - [x] Fixed StreamGear CI bugs for Windows and CI envs.
    - [x] Fixed Typos and Indentation bugs in NetGear API.
    - [x] Fixed ZMQ throwing error on termination if all max-tries exhausted.
    - [x] Fixed `NameError` bug in NetGear API and CI tests.
    - [x] Fixed `TimeoutError` bug in NetGear_Async CI tests.
    - [x] Fixed `get_valid_ffmpeg_path` throwing `TypeError` with non-string values.
    - [x] Fixed broken links in docs. 
    - [x] Fixed critical duplicate logging bug.
    - [x] Fixed default `gop` value not handle correctly.
    - [x] Fixed handling of incorrect paths detection.
    - [x] Fixed incorrect definitions in NetGear_Async.
    - [x] Fixed left-over attribute bug in WriteGear.
    - [x] Fixed logic and indentation bugs in CI tests.
    - [x] Fixed logic for handling output parameters in WriteGear API.
    - [x] Fixed missing definitions and logic bug in StreamGear.
    - [x] Fixed missing import and incorrect CI definitions. 
    - [x] Fixed missing source dimensions from `extract_resolutions` output in StreamGear API.
    - [x] Fixed missing support for compression parameters in Multi-Clients Mode.
    - [x] Fixed round off error in FPS.
    - [x] Fixed several CI bugs and updated `extract_resolutions` method.
    - [x] Fixed several bugs from CI Bidirectional Mode tests.
    - [x] Fixed several typos in docs usage examples.
    - [x] Fixed various `AttributeError` with wrong attribute names and definition in CI Helper functions.
    - [x] Fixed wrong and missing definitions in docs.
    - [x] Fixed wrong logic for extracting OpenCV frames.
    - [x] Fixed wrong type bug in StreamGear API.
    - [x] Fixed wrong type error bug in WriteGear API.
    - [x] Fixed wrong variable assignments bug in WriteGear API.
    - [x] Fixes to CLI tests and missing docs imports.
    - [x] Many minor typos and wrong definitions.

??? question "Pull Requests"
    * PR #129
    * PR #130
    * PR #155


&nbsp; 

&nbsp; 

## v0.1.8 (2020-06-12)

??? tip "New Features"
    - [x] **NetGear API:**
        * Multiple Clients support:
            + Implemented support for handling any number of Clients simultaneously with a single Server in this mode.
            + Added new `multiclient_mode` attribute for enabling this mode easily.
            + Built support for `zmq.REQ/zmq.REP` and `zmq.PUB/zmq.SUB` patterns in this mode.
            + Implemented ability to receive data from all Client(s) along with frames with `zmq.REQ/zmq.REP` pattern only.
            + Updated related CI tests
        * Support for robust Lazy Pirate pattern(auto-reconnection) in NetGear API for both server and client ends:
            + Implemented a algorithm where NetGear rather than doing a blocking receive, will now:
                + Poll the socket and receive from it only when it's sure a reply has arrived.
                + Attempt to reconnect, if no reply has arrived within a timeout period.
                + Abandon the connection if there is still no reply after several requests.
            + Implemented its default support for `REQ/REP` and `PAIR` messaging patterns internally.
            + Added new `max_retries` and `request_timeout`(in seconds) for handling polling.
            + Added `DONTWAIT` flag for interruption-free data receiving.
            + Both Server and Client can now reconnect even after a premature termination.
        * Performance Updates:
            + Added default Frame Compression support for Bidirectional frame transmission in Bidirectional mode.
            + Added support for `Reducer()` function in Helper.py to aid reducing frame-size on-the-go for more performance.
            + Added small delay in `recv()` function at client's end to reduce system load. 
            + Reworked and Optimized NetGear termination, and also removed/changed redundant definitions and flags.
    - [x] **Docs: Migration to Mkdocs**
        * Implemented a beautiful, static documentation site based on [MkDocs](https://www.mkdocs.org/) which will then be hosted on GitHub Pages.
        * Crafted base mkdocs with third-party elegant & simplistic [`mkdocs-material`](https://squidfunk.github.io/mkdocs-material/) theme.
        * Implemented new `mkdocs.yml` for Mkdocs with relevant data.
        * Added new `docs` folder to handle markdown pages and its assets.
        * Added new Markdown pages(`.md`) to docs folder, which are carefully crafted documents - [x] based on previous Wiki's docs, and some completely new additions.
        * Added navigation under tabs for easily accessing each document.
        * New Assets:
            + Added new assets like _gifs, images, custom scripts, favicons, site.webmanifest etc._ for bringing standard and quality to docs visual design.
            + Designed brand new logo and banner for VidGear Documents.
            + Deployed all assets under separate [*Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License*](https://creativecommons.org/licenses/by-nc-sa/4.0/).
        * Added Required Plugins and Extensions:
            + Added support for all [pymarkdown-extensions](https://facelessuser.github.io/pymdown-extensions/).
            + Added support for some important `admonition`, `attr_list`, `codehilite`, `def_list`, `footnotes`, `meta`, and `toc` like Mkdocs extensions.
            + Enabled `search`, `minify` and `git-revision-date-localized` plugins support.
            + Added various VidGear's social links to yaml.
            + Added support for `en` _(English)_ language.
        * Auto-Build API Reference with `mkdocstrings:`
            + Added support for [`mkdocstrings`](https://github.com/pawamoy/mkdocstrings) plugin for auto-building each VidGear's API references.
            + Added python handler for parsing python source-code to `mkdocstrings`.
        * Auto-Deploy Docs with GitHub Actions:
            + Implemented Automated Docs Deployment on gh-pages through GitHub Actions workflow.
            + Added new workflow yaml with minimal configuration for automated docs deployment.
            + Added all required  python dependencies and environment for this workflow.
            + Added `master` branch on Ubuntu machine to build matrix.

??? success "Updates/Improvements"
    - [x] Added in-built support for bidirectional frames(`NDarray`) transfer in Bidirectional mode.
    - [x] Added support for User-Defined compression params in Bidirectional frames transfer.
    - [x] Added workaround for `address already in use` bug at client's end.
    - [x] Unified Bidirectional and Multi-Clients mode for client's return data transmission.
    - [x] Replaced `ValueError` with more suitable `RuntimeError`.
    - [x] Updated logging for better readability.
    - [x] Added CI test for Multi-Clients mode.
    - [x] Reformatted and grouped imports in VidGear.
    - [x] Added `Reducer` Helper function CI test.
    - [x] Added Reliability tests for both Server and Client end.
    - [x] Disabled reliable reconnection for Multi-Clients mode.
    - [x] Replaced `os.devnull` with suprocess's inbuilt function.
    - [x] Updated README.md, Issue and PR templates with new information and updates.
    - [x] Moved `changelog.md` to `/docs` and updated contribution guidelines.
    - [x] Improved source-code docs for compatibility with `mkdocstrings`.
    - [x] Added additional dependency `mkdocs-exclude`, for excluding files from Mkdocs builds.
    - [x] Updated license and compressed images/diagrams.
    - [x] Added new CI tests and Bumped Codecov.
    - [x] Changed YouTube video URL for CI tests to Creative Commons(CC) video.
    - [x] Removed redundant code.

??? danger "Breaking Updates/Changes"
    - [ ] VidGear Docs moved to GitHub Pages, Now Available at https://abhitronix.github.io/vidgear.
    - [ ] Removed `filter` attribute from `options` parameter in NetGear API.
    - [ ] Removed `force_terminate` parameter support from NetGear API.
    - [ ] Disabled additional data of datatype `numpy.ndarray` for Server end in Bidirectional Mode.

??? bug "Bug-fixes"
    - [x] Fixed `'NoneType' object is not subscriptable` bug.
    - [x] Fixed bugs related to delayed termination in NetGear API.
    - [x] Reduced default `request_timeout` value to 4 and also lowered cut-off limit for the same.
    - [x] Removed redundant ZMQ context termination and similar variables.
    - [x] Added missing VidGear installation in workflow.
    - [x] Excluded conflicting assets `README.md` from Mkdocs builds.
    - [x] Fixed `pattern` value check bypassed if wrong value is assigned.
    - [x] Fixed incorrect handling of additional data transferred in synchronous mode at both Server and Client end.
    - [x] Replaced Netgear CI test with more reliable `try-except-final` blocks.
    - [x] Updated termination linger to zero at Server's end.
    - [x] Fixed `NameError` bug in NetGear API.
    - [x] Fixed missing support for compression parameters in Multi-Clients Mode.
    - [x] Fixed ZMQ throwing error on termination if all max-tries exhausted.
    - [x] Enabled missing support for frame compression in its primary receive mode.
    - [x] Fixed several bugs from CI Bidirectional Mode tests.
    - [x] Removed or Grouped redundant code for increasing codecov.
    - [x] Fixed Mkdocs only accepting Relative paths.
    - [x] Fixed broken links in docs. 
    - [x] Fixed round off error in FPS.
    - [x] Many small typos and bugs fixes.

??? question "Pull Requests"
    * PR #129
    * PR #130


&nbsp; 

&nbsp; 

## v0.1.7 (2020-04-29)

??? tip "New Features"
    - [x] **WebGear API:**
        * Added a robust Live Video Server API that can transfer live video frames to any web browser on the network in real-time.
        * Implemented a flexible asyncio wrapper around [`starlette`](https://www.starlette.io/) ASGI Application Server.
        * Added seamless access to various starlette's Response classes, Routing tables, Static Files, Template engine(with Jinja2), etc.
        * Added a special internal access to VideoGear API and all its parameters.
        * Implemented a new Auto-Generation Work-flow to generate/download & thereby validate WebGear API data files from its GitHub server automatically.
        * Added on-the-go dictionary parameter in WebGear to tweak performance, Route Tables and other internal properties easily.
        * Added new simple & elegant default Bootstrap Cover Template for WebGear Server.
        * Added `__main__.py` to directly run WebGear Server through the terminal.
        * Added new gif and related docs for WebGear API.
        * Added and Updated various CI tests for this API.
    - [x] **NetGear_Async API:** 
        * Designed NetGear_Async asynchronous network API built upon ZeroMQ's asyncio API.
        * Implemented support for state-of-the-art asyncio event loop [`uvloop`](https://github.com/MagicStack/uvloop) at its backend.
        * Achieved Unmatchable high-speed and lag-free video streaming over the network with minimal resource constraint.
        * Added exclusive internal wrapper around VideoGear API for this API.
        * Implemented complete server-client handling and options to use variable protocols/patterns for this API.
        * Implemented support for  all four ZeroMQ messaging patterns: i.e `zmq.PAIR`, `zmq.REQ/zmq.REP`, `zmq.PUB/zmq.SUB`, and `zmq.PUSH/zmq.PULL`.
        * Implemented initial support for `tcp` and `ipc` protocols.
        * Added new Coverage CI tests for NetGear_Async Network Gear.
        * Added new Benchmark tests for benchmarking NetGear_Async against NetGear.
    - [x] **Asynchronous Enhancements:** 
        * Added `asyncio` package to for handling asynchronous APIs.
        * Moved WebGear API(webgear.py) to `asyncio` and created separate asyncio `helper.py` for it.
        * Various Performance tweaks for Asyncio APIs with concurrency within a single thread.
        * Moved `__main__.py` to asyncio for easier access to WebGear API through the terminal.
        * Updated `setup.py` with new dependencies and separated asyncio dependencies.
    - [x] **General Enhancements:**
        * Added new highly-precise Threaded FPS class for accurate benchmarking with `time.perf_counter` python module.
        * Added a new [Gitter](https://gitter.im/vidgear/community) community channel.
        * Added a new *Reducer* function to reduce the frame size on-the-go.
        * Add *Flake8* tests to Travis CI to find undefined names. (PR by @cclauss)
        * Added a new unified `logging handler` helper function for vidgear.

??? success "Updates/Improvements"
    - [x] Re-implemented and simplified logic for NetGear Async server-end.
    - [x] Added new dependencies for upcoming asyncio updates to `setup.py`.
    - [x] Added `retry` function and replaced `wget` with `curl` for Linux test envs. 
    - [x] Bumped OpenCV to latest `4.2.0-dev` for Linux test envs.
    - [x] Updated YAML files to reflect new changes to different CI envs.
    - [x] Separated each API logger with a common helper method to avoid multiple copies. 
    - [x] Limited Importing OpenCV API version check's scope to `helper.py` only.
    - [x] Implemented case for incorrect `color_space` value in ScreenGear API.
    - [x] Removed old conflicting logging formatter with a common method and expanded logging.
    - [x] Improved and added `shutdown` function for safely stopping frame producer threads in WebGear API.
    - [x] Re-implemented and simplified all CI tests with maximum code-coverage in mind.
    - [x] Replaced old `mkdir` function with new `mkdir_safe` helper function for creating directories safely.
    - [x] Updated ReadMe.md with updated diagrams, gifs and information.
    - [x] Improve, structured and Simplified the Contribution Guidelines.
    - [x] Bundled CI requirements in a single command.(Suggested by @cclauss)
    - [x] Replaced line endings CRLF with LF endings.
    - [x] Added dos2unix for Travis OSX envs.
    - [x] Bumped Codecov to maximum. 

??? danger "Breaking Updates/Changes"
    - [ ] **Dropped support for Python 3.5 and below legacies. (See [issue #99](https://github.com/abhiTronix/vidgear/issues/99))**
    - [ ] Dropped and replaced Python 3.5 matrices with new Python 3.8 matrices in all CI environments.
    - [ ] Implemented PEP-8 Styled [**Black**](https://github.com/psf/black) formatting throughout the source-code.
    - [ ] Limited protocols support to `tcp` and `ipc` only, in NetGear API.

??? bug "Bug-fixes"
    - [x] Fixed Major NetGear_Async bug where `__address` and `__port` are not set in async mode.(PR by @otter-in-a-suit) 
    - [x] Fixed Major PiGear Color-space Conversion logic bug.
    - [x] Workaround for `CAP_IMAGES` error in YouTube Mode.
    - [x] Replaced incorrect `terminate()` with `join()` in PiGear.
    - [x] Removed `uvloop` for windows as still [NOT yet supported](https://github.com/MagicStack/uvloop/issues/14).
    - [x] Refactored Asynchronous Package name `async` to `asyncio`, since it is used as Keyword in python>=3.7 *(raises SyntaxError)*.
    - [x] Fixed unfinished close of event loops bug in WebGear API.
    - [x] Fixed NameError in helper.py.
    - [x] Added fix for OpenCV installer failure on Linux test envs.
    - [x] Fixed undefined NameError in `helper.py` context. (@cclauss)
    - [x] Fixed incorrect logic while pulling frames from ScreenGear API.
    - [x] Fixed missing functions in `__main__.py`.
    - [x] Fixed Typos and definitions in docs.
    - [x] Added missing `camera_num` parameter to VideoGear.
    - [x] Added OpenSSL's [SSL: CERTIFICATE_VERIFY_FAILED] bug workaround for macOS envs.
    - [x] Removed `download_url` meta from setup.py.
    - [x] Removed PiGear from CI completely due to hardware emulation limitation.
    - [x] Removed VideoCapture benchmark tests for macOS envs.
    - [x] Removed trivial `__main__.py` from codecov.
    - [x] Removed several redundant `try-catch` loops.
    - [x] Renamed `youtube_url_validation` as `youtube_url_validator`.
    - [x] Several minor wrong/duplicate variable definitions and various bugs fixed.
    - [x] Fixed, Improved & removed many Redundant CI tests for various APIs.


??? question "Pull Requests"
    * PR #88
    * PR #91
    * PR #93
    * PR #95
    * PR #98
    * PR #101
    * PR #114
    * PR #118
    * PR #124

??? new "New Contributors"
    * @cclauss
    * @chollinger93


&nbsp; 

&nbsp; 

## v0.1.6 (2020-01-01)

??? tip "New Features"
    - [x] **NetGear API:**
        * Added powerful ZMQ Authentication & Data Encryption features for NetGear API:
            + Added exclusive `secure_mode` param for enabling it.
            + Added support for two most powerful `Stonehouse` & `Ironhouse` ZMQ security mechanisms.
            + Added smart auth-certificates/key generation and validation features.
        * Implemented Robust Multi-Servers support for NetGear API:
            + Enables Multiple Servers messaging support with a single client.
            + Added exclusive `multiserver_mode` param for enabling it.
            + Added support for `REQ/REP` &  `PUB/SUB` patterns for this mode.
            + Added ability to send additional data of any datatype along with the frame in realtime in this mode.
        * Introducing exclusive Bidirectional Mode for bidirectional data transmission:
            + Added new `return_data` parameter to `recv()` function.
            + Added new `bidirectional_mode` attribute for enabling this mode.
            + Added support for `PAIR` & `REQ/REP` patterns for this mode
            + Added support for sending data of any python datatype.
            + Added support for `message` parameter for non-exclusive primary modes for this mode.
        * Implemented compression support with on-the-fly flexible frame encoding for the Server-end:
            + Added initial support for `JPEG`, `PNG` & `BMP` encoding formats .
            + Added exclusive options attribute `compression_format` & `compression_param` to tweak this feature.
            + Client-end will now decode frame automatically based on the encoding as well as support decoding flags.
        * Added `force_terminate` attribute flag for handling force socket termination at the Server-end if there's latency in the network. 
        * Implemented new *Publish/Subscribe(`zmq.PUB/zmq.SUB`)* pattern for seamless Live Streaming in NetGear API.
    - [x] **PiGear API:**
        * Added new threaded internal timing function for PiGear to handle any hardware failures/frozen threads.
        * PiGear will not exit safely with `SystemError` if Picamera ribbon cable is pulled out to save resources.
        * Added support for new user-defined `HWFAILURE_TIMEOUT` options attribute to alter timeout.
    - [x] **VideoGear API:** 
        * Added `framerate` global variable and removed redundant function.
        * Added `CROP_N_ZOOM` attribute in Videogear API for supporting Crop and Zoom stabilizer feature.
    - [x] **WriteGear API:** 
        * Added new `execute_ffmpeg_cmd` function to pass a custom command to its FFmpeg pipeline.
    - [x] **Stabilizer class:** 
        * Added new Crop and Zoom feature.
            + Added `crop_n_zoom` param for enabling this feature.
        * Updated docs.
    - [x] **CI & Tests updates:**
        * Replaced python 3.5 matrices with latest python 3.8 matrices in Linux environment.
        * Added full support for **Codecov** in all CI environments.
        * Updated OpenCV to v4.2.0-pre(master branch). 
        * Added various Netgear API tests.
        * Added initial Screengear API test.
        * More test RTSP feeds added with better error handling in CamGear network test.
        * Added tests for ZMQ authentication certificate generation.
        * Added badge and Minor doc updates.
    - [x] Added VidGear's official native support for MacOS environments.
    

??? success "Updates/Improvements"
    - [x] Replace `print` logging commands with python's logging module completely.
    - [x] Implemented encapsulation for class functions and variables on all gears.
    - [x] Updated support for screen casting from multiple/all monitors in ScreenGear API.
    - [x] Updated ScreenGear API to use *Threaded Queue Mode* by default, thereby removed redundant `THREADED_QUEUE_MODE` param.
    - [x] Updated bash script path to download test dataset in `$TMPDIR` rather than `$HOME` directory for downloading testdata.
    - [x] Implemented better error handling of colorspace in various videocapture APIs.
    - [x] Updated bash scripts, Moved FFmpeg static binaries to `github.com`.
    - [x] Updated bash scripts, Added additional flag to support un-secure apt sources.
    - [x] CamGear API will now throw `RuntimeError` if source provided is invalid.
    - [x] Updated threaded Queue mode in CamGear API for more robust performance.
    - [x] Added new `camera_num` to support multiple Picameras.
    - [x] Moved thread exceptions to the main thread and then re-raised.
    - [x] Added alternate github mirror for FFmpeg static binaries auto-installation on windows oses.
    - [x] Added `colorlog` python module for presentable colored logging.
    - [x] Replaced `traceback` with `sys.exc_info`.
    - [x] Overall APIs Code and Docs optimizations.
    - [x] Updated Code Readability and Wiki Docs.
    - [x] Updated ReadMe & Changelog with the latest changes.
    - [x] Updated Travis CI Tests with support for macOS environment.
    - [x] Reformatted & implemented necessary MacOS related changes and dependencies in `travis.yml`.

??? danger "Breaking Updates/Changes"
    - [ ] Python 2.7 legacy support dropped completely.
    - [ ] Source-code Relicensed to Apache 2.0 License.
    - [ ] Python 3+ are only supported legacies for installing v0.1.6 and above.
    - [ ] Python 2.7 and 3.4 legacies support dropped from CI tests.

??? bug "Bug-fixes"
    - [x] Reimplemented `Pub/Sub` pattern for smoother performance on various networks.
    - [x] Fixed Assertion error in CamGear API during colorspace manipulation.
    - [x] Fixed random freezing in `Secure Mode` and several related performance updates
    - [x] Fixed `multiserver_mode` not working properly over some networks.
    - [x] Fixed assigned Port address ignored bug (commit 073bca1).
    - [x] Fixed several wrong definition bugs from NetGear API(commit 8f7153c).
    - [x] Fixed unreliable dataset video URL(rehosted file on `github.com`).
    - [x] Disabled `overwrite_cert` for client-end in NetGear API.
    - [x] Disabled Universal Python wheel builds in `setup.cfg `file.
    - [x] Removed duplicate code to import MSS(@BoboTiG) from ScreenGear API.
    - [x] Eliminated unused redundant code blocks from library.
    - [x] Fixed Code indentation in `setup.py` and updated new release information.
    - [x] Fixed code definitions & Typos.
    - [x] Fixed several bugs related to `secure_mode` & `multiserver_mode` Modes.
    - [x] Fixed various macOS environment bugs.

??? question "Pull Requests"
    - [x] PR #39
    - [x] PR #42
    - [x] PR #44
    - [x] PR #52
    - [x] PR #55
    - [x] PR #62
    - [x] PR #67
    - [x] PR #72
    - [x] PR #77
    - [x] PR #78
    - [x] PR #82
    - [x] PR #84

??? new "New Contributors"
    * @BoboTiG

&nbsp; 

&nbsp; 

## v0.1.5 (2019-07-24)

??? tip "New Features"
    - [x] Added new ScreenGear API, supports Live ScreenCasting.
    - [x] Added new NetGear API, aids real-time frame transfer through messaging(ZmQ) over network.
    - [x] Added new new Stabilizer Class, for minimum latency Video Stabilization with OpenCV.
    - [x] Added Option to use API's standalone.
    - [x] Added Option to use VideoGear API as internal wrapper around Stabilizer Class.
    - [x] Added new parameter `stabilize` to API, to enable or disable Video Stabilization.
    - [x] Added support for `**option` dict attributes to update VidGear's video stabilizer parameters directly. 
    - [x] Added brand new logo and functional block diagram (`.svg`) in readme.md
    - [x] Added new pictures and GIFs for improving readme.md readability 
    - [x] Added new `contributing.md` and `changelog.md` for reference.
    - [x] Added `collections.deque` import in Threaded Queue Mode for performance consideration
    - [x] Added new `install_opencv.sh` bash scripts for Travis cli, to handle OpenCV installation.
    - [x] Added new Project Issue & PR Templates
    - [x] Added new Sponsor Button(`FUNDING.yml`)

??? success "Updates/Improvements"
    - [x] Updated New dependencies: `mss`, `pyzmq` and rejected redundant ones.
    - [x] Revamped and refreshed look for `readme.md` and added new badges.
    - [x] Updated Releases Documentation completely.
    - [x] Updated CI tests for new changes
    - [x] Updated Code Documentation.
    - [x] Updated bash scripts and removed redundant information
    - [x] Updated `Youtube video` URL in tests
    - [x] Completely Reformatted and Updated Wiki Docs with new changes.

??? danger "Breaking Updates/Changes"
    - [ ] Implemented experimental Threaded Queue Mode(_a.k.a Blocking Mode_) for fast, synchronized, error-free multi-threading.
    - [ ] Renamed bash script `pre-install.sh` to `prepare_dataset.sh` - [x] downloads opensourced test datasets and static FFmpeg binaries for debugging.
    - [ ] Changed `script` folder location to `bash/script`.
    - [ ] `Python 3.4` removed from Travis CI tests.

??? bug "Bug-fixes"
    - [x] Temporarily fixed Travis CI bug: Replaced `opencv-contrib-python` with OpenCV built from scratch as dependency.
    - [x] Fixed CI Timeout Bug: Disable Threaded Queue Mode for CI Tests
    - [x] Fixes** `sys.stderr.close()` throws ValueError bug: Replaced `sys.close()` with `DEVNULL.close()`
    - [x] Fixed Youtube Live Stream bug that return `NonType` frames in CamGear API.
    - [x] Fixed `NoneType` frames bug in  PiGear class on initialization.
    - [x] Fixed Wrong function definitions
    - [x] Removed `/xe2` unicode bug from Stabilizer class.
    - [x] Fixed `**output_params` _KeyError_ bug in WriteGear API
    - [x] Fixed subprocess not closing properly on exit in WriteGear API.
    - [x] Fixed bugs in ScreenGear: Non-negative `monitor` values
    - [x] Fixed missing import, typos, wrong variable definitions
    - [x] Removed redundant hack from `setup.py`
    - [x] Fixed Minor YouTube playback Test CI Bug 
    - [x] Fixed new Twitter Intent
    - [x] Fixed bug in bash script that not working properly due to changes at server end.

??? question "Pull Requests"
    - [x] PR #17
    - [x] PR #21
    - [x] PR #22
    - [x] PR #27
    - [x] PR #31
    - [x] PR #32
    - [x] PR #33
    - [x] PR #34

&nbsp; 

&nbsp; 

## v0.1.4 (2019-05-11)

??? tip "New Features"
    - [x] Added new WriteGear API: for enabling lossless video encoding and compression(built around FFmpeg and OpenCV Video Writer)
    - [x] Added YouTube Mode for direct Video Pipelining from YouTube in CamGear API
    - [x] Added new `y_tube` to access _YouTube Mode_ in CamGear API.
    - [x] Added flexible Output file Compression control capabilities in compression-mode(WriteGear).
    - [x] Added `-output_dimensions` special parameter to WriteGear API.
    - [x] Added new `helper.py` to handle special helper functions.
    - [x] Added feature to auto-download and configure FFmpeg Static binaries(if not found) on Windows platforms.
    - [x] Added `-input_framerate` special parameter to WriteGear class to change/control output constant framerate in compression mode(WriteGear).
    - [x] Added new Direct Video colorspace Conversion capabilities in CamGear and PiGear API.
    - [x] Added new `framerate` class variable for CamGear API, to retrieve input framerate.
    - [x] Added new parameter `backend` - [x] changes the backend of CamGear's API
    - [x] Added automatic required prerequisites installation ability, when installation from source.
    - [x] Added Travis CI Complete Integration for Linux-based Testing for VidGear.
    - [x] Added and configured `travis.yml`
    - [x] Added Appveyor CI Complete Integration for Windows-based Testing in VidGear.
    - [x] Added and configured new `appveyor.yml`
    - [x] Added new bash script `pre-install.sh` to download opensourced test datasets and static FFmpeg binaries for debugging.
    - [x] Added several new Tests(including Benchmarking Tests) for each API for testing with `pytest`.
    - [x] Added license to code docs.
    - [x] Added `Say Thank you!` badge to `Readme.md`.

??? success "Updates/Improvements"
    - [x] Removed redundant dependencies
    - [x] Updated `youtube-dl` as a dependency, as required by `pafy`'s backend.
    - [x] Updated common VideoGear API with new parameter.
    - [x] Update robust algorithm to auto-detect FFmpeg executables and test them, if failed, auto fallback to OpenCV's VideoWriter API. 
    - [x] Improved system previously installed OpenCV detection in setup.py.
    - [x] Updated setup.py with hack to remove bullets from pypi description. 
    - [x] Updated Code Documentation
    - [x] Reformatted & Modernized readme.md with new badges.
    - [x] Reformatted and Updated Wiki Docs.

??? danger "Breaking Updates/Changes"
    - [ ] Removed `-height` and `-width` parameter from CamGear API.
    - [ ] Replaced dependency `opencv-python` with `opencv-contrib-python` completely

??? bug "Bug-fixes"
    - [x] Windows Cross-Platform fix: replaced dependency `os` with `platform` in setup.py.
    - [x] Fixed Bug: Arises due to spaces in input `**options`/`**output_param` dictionary keys.
    - [x] Fixed several wrong/missing variable & function definitions.
    - [x] Fixed code uneven indentation.
    - [x] Fixed several typos in docs.

??? question "Pull Requests"
    - [x] PR #7
    - [x] PR #8
    - [x] PR #10
    - [x] PR #12



&nbsp; 

&nbsp; 

## v0.1.3 (2019-04-07)

??? bug "Bug-fixes"
    - [x] Patched Major PiGear Bug: Incorrect import of PiRGBArray function in PiGear Class
    - [x] Several Fixes for backend `picamera` API handling during frame capture(PiGear)
    - [x] Fixed missing frame variable initialization.
    - [x] Fixed minor typos

??? question "Pull Requests"
    - [x] PR #6
    - [x] PR #5

&nbsp; 

&nbsp; 

## v0.1.2 (2019-03-27)

??? tip "New Features"
    - [x] Added easy Source manipulation feature in CamGear API, to control features like `resolution, brightness, framerate etc.`
    - [x] Added new `**option` parameter to CamGear API, provides the flexibility to manipulate input stream directly.
    - [x] Added new parameters for Camgear API for time delay and logging.
    - [x] Added new Logo to readme.md
    - [x] Added new Wiki Documentation.

??? success "Updates/Improvements"
    - [x] Reformatted readme.md.
    - [x] Updated Wiki Docs with new changes.


??? bug "Bug-fixes"
    - [x] Improved Error Handling in CamGear & PiGear API.
    - [x] Fixed minor typos in docs.

??? question "Pull Requests"
    - [x] PR #4

&nbsp;

&nbsp;


## v0.1.1 (2019-03-24)

??? tip "New Features"
    - [x] Release ViGear binaries on the Python Package Index (PyPI)
    - [x] Added new and configured `setup.py` & `setup.cfg`

??? bug "Bug-fixes"
    - [x] Fixed PEP bugs: added and configured properly `__init__.py` in each folder 
    - [x] Fixed PEP bugs: improved code Indentation
    - [x] Fixed wrong imports: replaced `distutils.core` with `setuptools`
    - [x] Fixed readme.md

&nbsp;

&nbsp;

## v0.1.0 (2019-03-17)

??? tip "New Features"
    - [x] Initial Release
    - [x] Converted [my `imutils` PR](https://github.com/jrosebr1/imutils/pull/105) into Python Project.
    - [x] Renamed conventions and reformatted complete source-code from scratch.
    - [x] Added support for both python 2.7 and 3 legacies
    - [x] Added new multi-threaded CamGear, PiGear, and VideoGear APIs
    - [x] Added multi-platform compatibility
    - [x] Added robust & flexible control over the source in PiGear API.