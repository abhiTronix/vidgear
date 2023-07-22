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

import pytest
import logging as log
import platform

from vidgear.gears import ScreenGear
from vidgear.gears.helper import logger_handler

# define test logger
logger = log.getLogger("Test_screengear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


test_data = [
    (1, "", {}, None),
    (
        1,
        "mss",
        {"top": 40, "left": 0, "width": 100, "height": 100},
        "COLOR_BGR2INVALID",
    ),
    (None, "mss", {}, "COLOR_BGR2GRAY"),
    (
        None,
        "jil",
        {"top": 40, "left": 0, "width": 100, "height": 100},
        "COLOR_BGR2INVALID",
    ),
    (3, "", {}, None),
]


@pytest.mark.skipif((platform.system() == "Windows"), reason="Buggy!")
@pytest.mark.parametrize("monitor, backend, options, colorspace", test_data)
def test_screengear(monitor, backend, options, colorspace):
    """
    Tests ScreenGear's playback capabilities with custom defined dimensions -> passes if fails with ScreenShotError
    """
    try:
        # define dimensions of screen w.r.t to given monitor to be captured
        # Open Live Screencast on current monitor
        stream = ScreenGear(
            monitor=monitor,
            backend=backend,
            logging=True,
            colorspace=colorspace,
            **options
        ).start()
        # playback
        i = 0
        while i < 20:
            frame = stream.read()
            if frame is None:
                break
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
        if monitor == 3 or backend == "jil" or platform.system() == "Linux":
            logger.exception(e)
        else:
            pytest.fail(str(e))
