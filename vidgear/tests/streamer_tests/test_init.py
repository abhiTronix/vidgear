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
import tempfile
import pytest

from os.path import expanduser
from vidgear.gears import StreamGear
from vidgear.gears.helper import logger_handler

# define test logger
logger = log.getLogger("Test_init")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


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


@pytest.mark.xfail(raises=RuntimeError)
@pytest.mark.parametrize("c_ffmpeg", [return_static_ffmpeg(), "wrong_path", 1234])
def test_custom_ffmpeg(c_ffmpeg):
    """
    Testing custom FFmpeg for StreamGear 
    """
    StreamGear(output="output.mpd", custom_ffmpeg=c_ffmpeg, logging=True)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize("format", ["dash", "mash", "unknown", 1234, None])
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
    stream_params = (
        {"-clear_prev_assets": True}
        if (output and output == "output.mpd")
        else {"-clear_prev_assets": "invalid"}
    )
    try:
        StreamGear(output=output, logging=True, **stream_params)
    except Exception as e:
        if output is None:
            pytest.xfail(str(e))
        else:
            pytest.fail(str(e))
