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

## v0.2.2 (In Progress)

??? tip "New Features"
    - [x] **NetGear:**
        * New SSH Tunneling Mode for connecting ZMQ sockets across machines via SSH tunneling.
        * Added new `ssh_tunnel_mode` attribute to enable ssh tunneling at provide address at server end only.
        * Implemented new `check_open_port` helper method to validate availability of host at given open port.
        * Added new attributes `ssh_tunnel_keyfile` and `ssh_tunnel_pwd` to easily validate ssh connection.
        * Extended this feature to be compatible with bi-directional mode and auto-reconnection.
        * Initially disabled support for exclusive Multi-Server and Multi-Clients modes.
        * Implemented logic to automatically enable `paramiko` support if installed.
        * Reserved port-47 for testing.
    - [x] **WebGear_RTC:**
        * Added native support for middlewares.
        * Added new global `middleware` variable for easily defining Middlewares as list.
        * Added validity check for Middlewares.
        * Added tests for middlewares support.
        * Added example for middlewares support.
        * Added related imports.
    - [x] **CI:**
         * Added new `no-response` work-flow for stale issues.
         * Added NetGear CI Tests
         * Added new CI tests for SSH Tunneling Mode.
         * Added "paramiko" to CI dependencies.

    - [x] **Docs:**
         * Added Zenodo DOI badge and its reference in BibTex citations.
         * Added `pymdownx.striphtml` plugin for stripping comments.
         * Added complete docs for SSH Tunneling Mode.
         * Added complete docs for NetGear's SSH Tunneling Mode.
         * Added new usage example and related information.
         * Added new image assets for ssh tunneling example.
         * New admonitions and beautified css
          

??? success "Updates/Improvements"
    - [x] Added exception for RunTimeErrors in NetGear CI tests.
    - [x] Extended Middlewares support to WebGear API too.
    - [x] Docs:
        * Added `extra.homepage` parameter, which allows for setting a dedicated URL for `site_url`.
        * Re-positioned few docs comments at bottom for easier detection during stripping.
        * Updated dark theme to `dark orange`.
        * Updated fonts to `Source Sans Pro`.
        * Fixed missing heading in VideoGear.
        * Update setup.py update link for assets.
        * Added missing StreamGear Code docs.
        * Several minor tweaks and typos fixed.
        * Updated 404 page and workflow.
        * Updated README.md and mkdocs.yml  with new additions.
        * Re-written Threaded-Queue-Mode from scratch with elaborated functioning.
        * Replace Paypal with Liberpay in FUNDING.yml
        * Updated FFmpeg Download links.
        * Restructured docs.
        * Updated mkdocs.yml.
    - [x] Helper: 
        * Implemented new `delete_file_safe` to safely delete files at given path.
        * Replaced `os.remove` calls with `delete_file_safe`.
    - [x] CI: 
        * Updated VidGear Docs Deployer Workflow
        * Updated test
    - [x] Updated issue templates and labels.

??? danger "Breaking Updates/Changes"
    - [ ] Renamed `delete_safe` to `delete_ext_safe`.


??? bug "Bug-fixes"
    - [x] Critical Bugfix related to OpenCV Binaries import.
        * Bug fixed for OpenCV import comparsion test failing with Legacy versions and throwing ImportError.
        * Replaced `packaging.parse_version` with more robust `distutils.version`.
        * Removed redundant imports.
    - [x] Setup: 
        * Removed `latest_version` variable support from `simplejpeg`.
        * Fixed minor typos in dependencies.
    - [x] Setup_cfg: Replaced dashes with underscores to remove warnings.
    - [x] Docs:
        * Fixed 404 page does not work outside the site root with mkdocs.
        * Fixed markdown files comments not stripped when converted to HTML.
        * Fixed typos


??? question "Pull Requests"
    * PR #210
    * PR #215


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
    - [x] Helper: Added missing RSTP URL scheme to `is_valid_url` method.
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
    - [ ] :warning: WriteGear will automatically switch video encoder to default if specified encoder not found.
    - [ ] :warning: WriteGear will throw `RuntimeError` if no suitable default encoder found!
    - [ ] :warning: Removed format specific OpenCV decoding and encoding support for NetGear.
    - [ ] :warning: Dropped support for `compression_format`, `compression_param` attributes from NetGear.
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
    - [ ] :warning: `y_tube` parameter renamed as `stream_mode` in CamGear API!
    - [ ] :warning: Removed Travis support and `travis.yml` deleted.

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
    - [ ] :warning: Changed Webgear API default address to `localhost` for cross-compatibility between different platforms.
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
    - [ ] :warning: VidGear Docs moved to GitHub Pages, Now Available at https://abhitronix.github.io/vidgear.
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
    - [ ] :warning: **Dropped support for Python 3.5 and below legacies. (See [issue #99](https://github.com/abhiTronix/vidgear/issues/99))**
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
    - [ ] :warning: Python 2.7 legacy support dropped completely.
    - [ ] :warning: Source-code Relicensed to Apache 2.0 License.
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