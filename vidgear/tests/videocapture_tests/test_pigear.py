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
# import the necessary packages

import os
import cv2
import time
import numpy as np
import pytest
import logging as log
import platform

# Faking
import sys
from ..utils import fake_picamera

sys.modules["picamera"] = fake_picamera.picamera
sys.modules["picamera.array"] = fake_picamera.picamera.array

from vidgear.gears.helper import logger_handler

# define test logger
logger = log.getLogger("Test_pigear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


@pytest.mark.skipif((platform.system() != "Linux"), reason="Not Implemented")
def test_pigear_playback():
    """
    Tests PiGear's playback capabilities
    """
    try:
        from vidgear.gears import PiGear

        # open pi video stream with default parameters
        stream = PiGear(logging=True, colorspace="COLOR_BGR2GRAY").start()
        # playback
        i = 0
        while i < 10:
            frame = stream.read()
            if frame is None:
                break
            i += 1
        # clean resources
        stream.stop()
    except Exception as e:
        if isinstance(e, ImportError):
            logger.exception(e)
        else:
            pytest.fail(str(e))


test_data = [
    ("invalid", None, "", "invalid", {}, None, AssertionError),
    (-1, "invalid", "", 0.1, {}, None, AssertionError),
    (1, None, "invalid", 0.1, {}, None, AssertionError),
    (0, (640, 480), 60, 0, {"HWFAILURE_TIMEOUT": 15.0}, None, ValueError),
    (0, (640, 480), 60, 0, {}, "COLOR_BGR2INVALID", None),
    (0, (640, 480), 60, 0, {"create_bug": True}, "None", RuntimeError),
    (0, (640, 480), 60, 0, {"create_bug": "fail"}, "None", RuntimeError),
    (0, (640, 480), 60, 0, {"unknown_attr": "fail"}, "None", AttributeError),
    (
        0,
        (640, 480),
        60,
        0,
        {"HWFAILURE_TIMEOUT": 1.5, "create_bug": 10},
        "COLOR_BGR2GRAY",
        SystemError,
    ),
]


@pytest.mark.skipif((platform.system() != "Linux"), reason="Not Implemented")
@pytest.mark.parametrize(
    "camera_num, resolution, framerate, time_delay, options, colorspace, exception_type",
    test_data,
)
def test_pigear_parameters(
    camera_num, resolution, framerate, time_delay, options, colorspace, exception_type
):
    """
    Tests PiGear's options and colorspace.
    """
    try:
        from vidgear.gears import PiGear

        # open pi video stream with default parameters
        stream = PiGear(
            camera_num=camera_num,
            resolution=resolution,
            framerate=framerate,
            logging=True,
            time_delay=time_delay,
            **options
        ).start()
        # playback
        i = 0
        while i < 20:
            frame = stream.read()
            if frame is None:
                break
            time.sleep(0.1)
            if i == 10:
                if colorspace == "COLOR_BGR2INVALID":
                    # test wrong colorspace value
                    stream.color_space = 1546755
                else:
                    # test invalid colorspace value
                    stream.color_space = "red"
            i += 1
        # clean resources
        stream.stop()
    except Exception as e:
        if not (exception_type is None) and isinstance(e, exception_type):
            logger.exception(e)
        else:
            pytest.fail(str(e))
