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

import os, platform
import pytest
import tempfile
from vidgear.gears import CamGear
from vidgear.gears.helper import logger_handler
from .fps import FPS
import logging as log

logger = log.getLogger("Test_benchmark_playback")
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


def return_testvideo(level=0):
    """
    returns test H264 videos path with increasing Video quality(resolution & bitrate) with Level-0(Lowest ~HD 2Mbps) and Level-5(Highest ~4k UHD 120mpbs)
    """
    Levels = [
        "BigBuckBunny.mp4",
        "50_mbps_hd_h264.mkv",
        "90_mbps_hd_hevc_10bit.mkv",
        "120_mbps_4k_uhd_h264.mkv",
    ]
    path = "{}/Downloads/Test_videos/{}".format(tempfile.gettempdir(), Levels[level])
    return os.path.abspath(path)


def playback(level):
    """
    tests CamGear API's playback capabilities
    """
    options = {"THREADED_QUEUE_MODE": False}
    stream = CamGear(source=level, **options).start()
    fps = FPS().start()
    while True:
        frame = stream.read()
        if frame is None:
            break
        fps.update()
        logger.info("approx. FPS: {:.2f}".format(fps.fps()))
    stream.stop()


@pytest.mark.skipif(platform.system() == "Darwin", reason="Too Slow!")
@pytest.mark.parametrize(
    "level",
    [
        return_testvideo(0),
        return_testvideo(1),
        return_testvideo(2),
        return_testvideo(3),
    ],
)
def test_benchmark(level):
    """
    Benchmarks low to extreme 4k video playback capabilities of CamGear API
    """
    try:
        playback(level)
    except Exception as e:
        logger.exception(str(e))
