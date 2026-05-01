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
import sys
import tempfile

from vidgear.tests.utils.helpers import return_testvideo_path

import pytest

from vidgear.gears import VideoGear
from vidgear.gears.helper import Backend, logger_handler

# define test logger
logger = log.getLogger("Test_videogear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


# define machine os
_windows = (os.name == "nt")





@pytest.mark.skipif((platform.system() != "Linux"), reason="Not Implemented")
def test_PiGear_import():
    """
    Testing VideoGear Import -> assign to fail when PiGear class is imported
    via the new `api=Backend.PIGEAR` selector.
    """
    # cleanup environment

    try:
        del sys.modules["picamera"]
        del sys.modules["picamera.array"]
    except KeyError:
        pass

    try:
        stream = VideoGear(api=Backend.PIGEAR, logging=True).start()
        stream.stop()
    except Exception as e:
        if isinstance(e, ImportError):
            pytest.xfail(str(e))
        else:
            pytest.fail(str(e))


@pytest.mark.skipif((platform.system() != "Linux"), reason="Not Implemented")
def test_enablePiCamera_deprecation():
    """
    Legacy `enablePiCamera=True` must still route to PiGear and emit a
    DeprecationWarning.
    """
    try:
        del sys.modules["picamera"]
        del sys.modules["picamera.array"]
    except KeyError:
        pass

    try:
        with pytest.warns(DeprecationWarning):
            stream = VideoGear(enablePiCamera=True, logging=True).start()
        stream.stop()
    except Exception as e:
        if isinstance(e, ImportError):
            pytest.xfail(str(e))
        else:
            pytest.fail(str(e))


def test_invalid_api_type():
    """
    Non-`Backend` value passed to `api` must raise TypeError.
    """
    with pytest.raises(TypeError):
        VideoGear(api="camgear", logging=True)


def test_camgear_default_backend():
    """
    Default backend (CAMGEAR) playback smoke test.
    """
    try:
        stream = VideoGear(source=return_testvideo_path(), logging=True).start()
        framerate = stream.framerate
        while True:
            frame = stream.read()
            if frame is None:
                break
        stream.stop()
        assert framerate > 0
    except Exception as e:
        pytest.fail(str(e))


def test_camgear_explicit_backend():
    """
    Explicit `api=Backend.CAMGEAR` must behave identically to default.
    """
    try:
        stream = VideoGear(
            api=Backend.CAMGEAR, source=return_testvideo_path(), logging=True
        ).start()
        while True:
            frame = stream.read()
            if frame is None:
                break
        stream.stop()
    except Exception as e:
        pytest.fail(str(e))


def test_ffgear_backend():
    """
    `api=Backend.FFGEAR` must route to FFGear (skipped if deffcode missing).
    """
    pytest.importorskip("deffcode", reason="`deffcode` is required for FFGear backend")
    try:
        stream = VideoGear(
            api=Backend.FFGEAR,
            source=return_testvideo_path(),
            frame_format="bgr24",
            logging=True,
        ).start()
        # FFGear has no `framerate` attr; VideoGear should fall back to 0.0
        assert stream.framerate == 0.0
        frame = stream.read()
        assert frame is not None
        stream.stop()
    except Exception as e:
        pytest.fail(str(e))


# Video credit: http://www.liushuaicheng.org/CVPR2014/index.html
test_data = [
    (
        "https://gitlab.com/abhiTronix/Imbakup/-/raw/master/Images/example4_train_input.mp4",
        {
            "SMOOTHING_RADIUS": 5,
            "BORDER_SIZE": 10,
            "BORDER_TYPE": "replicate",
            "CROP_N_ZOOM": True,
        },
    ),
    (
        "https://gitlab.com/abhiTronix/Imbakup/-/raw/master/Images/example_empty_train_input.mp4",
        {
            "SMOOTHING_RADIUS": 5,
            "BORDER_SIZE": 15,
            "BORDER_TYPE": "reflect",
        },
    ),
    (
        "https://gitlab.com/abhiTronix/Imbakup/-/raw/master/Images/example4_train_input.mp4",
        {
            "SMOOTHING_RADIUS": "5",
            "BORDER_SIZE": "15",
            "BORDER_TYPE": ["reflect"],
            "CROP_N_ZOOM": "yes",
        },
    ),
    (return_testvideo_path(), {"BORDER_TYPE": "im_wrong"}),
]


@pytest.mark.parametrize("source, options", test_data)
def test_video_stablization(source, options):
    """
    Testing VideoGear's Video Stablization playback capabilities
    """
    try:
        # open stream
        stab_stream = VideoGear(
            source=source, stabilize=True, logging=True, **options
        ).start()
        framerate = stab_stream.framerate
        # playback
        while True:
            frame = stab_stream.read()  # read stabilized frames
            if frame is None:
                break
        # clean resources
        stab_stream.stop()
        logger.debug("Input Framerate: {}".format(framerate))
        assert framerate > 0
    except Exception as e:
        pytest.fail(str(e))
