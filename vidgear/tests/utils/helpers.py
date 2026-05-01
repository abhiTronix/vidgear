"""
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
"""

# Shared test helper utilities for the vidgear test-suite.
# Import these functions directly in test modules as needed.
# Do NOT import from conftest.py — conftest is reserved for pytest fixtures/hooks.

import os
import platform
import tempfile


def get_testing_dir():
    """
    Returns the root testing directory: ``<system-tmpdir>/testing_dir``.

    This mirrors the ``$TMPFOLDER`` variable set in
    ``scripts/bash/prepare_dataset.sh`` and is the single source of truth
    for all temporary test artefacts produced by the vidgear test-suite.
    """
    return os.path.join(tempfile.gettempdir(), "testing_dir")


def return_static_ffmpeg():
    """
    Returns the system-specific path to the downloaded FFmpeg static binary
    located under ``<testing_dir>/Downloads/FFmpeg_static/``.
    """
    testing_dir = get_testing_dir()
    if platform.system() == "Windows":
        path = os.path.join(
            testing_dir, "Downloads/FFmpeg_static/ffmpeg/bin/ffmpeg.exe"
        )
    elif platform.system() == "Darwin":
        path = os.path.join(
            testing_dir, "Downloads/FFmpeg_static/ffmpeg/bin/ffmpeg"
        )
    else:
        path = os.path.join(
            testing_dir, "Downloads/FFmpeg_static/ffmpeg/ffmpeg"
        )
    return os.path.abspath(path)


def return_testvideo_path(fmt="av"):
    """
    Returns the path to a test video file located under
    ``<testing_dir>/Downloads/Test_videos/``.

    Parameters
    ----------
    fmt : str, optional
        One of ``"av"`` (audio+video, default), ``"vo"`` (video-only),
        or ``"ao"`` (audio-only).
    """
    supported_fmts = {
        "av": "BigBuckBunny_4sec.mp4",
        "vo": "BigBuckBunny_4sec_VO.mp4",
        "ao": "BigBuckBunny_4sec_AO.aac",
    }
    req_fmt = fmt if (fmt in supported_fmts) else "av"
    path = os.path.join(
        get_testing_dir(), "Downloads", "Test_videos", supported_fmts[req_fmt]
    )
    return os.path.abspath(path)
