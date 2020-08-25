"""
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
"""
# import libraries
import logging as log
import os
import platform
import re
import subprocess
import tempfile
import cv2
import pytest

from os.path import expanduser
from vidgear.gears import CamGear, StreamGear
from mpegdash.parser import MPEGDASHParser

# define test logger
logger = log.getLogger("Test_init")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


# define machine os
_windows = True if os.name == "nt" else False


def return_static_ffmpeg():
    """
    returns system specific FFmpeg static path
    """
    path = ""
    if platform.system() == "Windows":
        path += os.path.join(
            tempfile.gettempdir(), "Downloads/FFmpeg_static/ffmpeg/bin/ffmpeg.exe"
        )
    elif platform.system() == "Darwin":
        path += os.path.join(
            tempfile.gettempdir(), "Downloads/FFmpeg_static/ffmpeg/bin/ffmpeg"
        )
    else:
        path += os.path.join(
            tempfile.gettempdir(), "Downloads/FFmpeg_static/ffmpeg/ffmpeg"
        )
    return os.path.abspath(path)


def return_testvideo_path():
    """
    returns Test video path
    """
    path = "{}/Downloads/Test_videos/BigBuckBunny_4sec.mp4".format(
        tempfile.gettempdir()
    )
    return os.path.abspath(path)


def check_valid_mpd(file="", exp_reps=1):
    """
    checks if given file is a valid MPD(MPEG-DASH Manifest file)
    """
    if not file or not os.path.isfile(file):
        return False
    try:
        mpd = MPEGDASHParser.parse(file)
        all_reprs = []
        for period in mpd.periods:
            for adapt_set in period.adaptation_sets:
                for rep in adapt_set.representations:
                    all_reprs.append(rep)
    except Exception as e:
        logger.error(str(e))
        return False
    return True if (len(all_reprs) >= exp_reps) else False


def getframe():
    """
    returns empty numpy frame/array of dimensions: (500,800,3)
    """
    return np.zeros([500, 800, 3], dtype=np.uint8)


def return_static_ffmpeg():
    """
    returns system specific FFmpeg static path
    """
    path = ""
    if platform.system() == "Windows":
        path += os.path.join(
            tempfile.gettempdir(), "Downloads/FFmpeg_static/ffmpeg/bin/ffmpeg.exe"
        )
    elif platform.system() == "Darwin":
        path += os.path.join(
            tempfile.gettempdir(), "Downloads/FFmpeg_static/ffmpeg/bin/ffmpeg"
        )
    else:
        path += os.path.join(
            tempfile.gettempdir(), "Downloads/FFmpeg_static/ffmpeg/ffmpeg"
        )
    return os.path.abspath(path)


def return_testvideo_path():
    """
    returns Test video path
    """
    path = "{}/Downloads/Test_videos/BigBuckBunny_4sec.mp4".format(
        tempfile.gettempdir()
    )
    return os.path.abspath(path)


@pytest.mark.xfail(raises=RuntimeError)
@pytest.mark.parametrize("c_ffmpeg", [return_static_ffmpeg(), "wrong_path"])
def test_custom_ffmpeg(c_ffmpeg):
    """
    Testing custom FFmpeg for StreamGear 
    """
    StreamGear(output="output.mpd", custom_ffmpeg=c_ffmpeg, logging=True)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize("format", ["dash", "mash", 1234, None])
def test_formats(format):
    """
    Testing different formats for StreamGear 
    """
    StreamGear(output="output.mpd", format=format, logging=True)


@pytest.mark.parametrize(
    "output", [None, "output.mpd", os.path.join(expanduser("~"), "test_mpd")]
)
def test_outputs(output):
    """
    Testing different output for StreamGear 
    """
    stream_params = {"-clear_prev_assets": True} if not (output is None) else {}
    try:
        StreamGear(output=output, logging=True, **stream_params)
    except Exception as e:
        if output is None:
            pytest.xfail(str(e))
        else:
            pytest.fail(str(e))
