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

# import the necessary packages

import logging as log
import os
import platform
import tempfile
from os.path import expanduser

import pytest

from vidgear.gears import StreamGear
from vidgear.gears.helper import logger_handler
from vidgear.tests.utils.helpers import get_testing_dir, return_static_ffmpeg

# define test logger
logger = log.getLogger("Test_init")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)





@pytest.mark.xfail(raises=RuntimeError)
@pytest.mark.parametrize("c_ffmpeg", [return_static_ffmpeg(), "wrong_path", 1234])
def test_custom_ffmpeg(c_ffmpeg):
    """
    Testing custom FFmpeg for StreamGear
    """
    streamer = StreamGear(output=os.path.join(get_testing_dir(), "output.mpd"), custom_ffmpeg=c_ffmpeg, logging=True)
    streamer.close()


@pytest.mark.xfail(raises=(AssertionError, ValueError))
@pytest.mark.parametrize("format", ["hls", "mash", 1234, None])
def test_formats(format):
    """
    Testing different formats for StreamGear
    """
    streamer = StreamGear(output=os.path.join(get_testing_dir(), "output.mpd"), format=format, logging=True)
    streamer.close()


@pytest.mark.parametrize(
    "output",
    [None, os.path.join(get_testing_dir(), "output.mpd"), os.path.join(get_testing_dir(), "output.m3u8")],
)
def test_outputs(output):
    """
    Testing different output for StreamGear
    """
    _m3u8 = os.path.join(get_testing_dir(), "output.m3u8")
    stream_params = (
        {"-clear_prev_assets": True}
        if (output and not output.endswith("m3u8"))
        else {"-clear_prev_assets": "invalid"}
    )
    try:
        streamer = StreamGear(
            output=output,
            format="hls" if output == _m3u8 else "dash",
            logging=True,
            **stream_params
        )
        streamer.close()
    except Exception as e:
        if output is None or (isinstance(output, str) and output.endswith("m3u8")):
            pytest.xfail(str(e))
        else:
            pytest.fail(str(e))
