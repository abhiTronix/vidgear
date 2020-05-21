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
# import libraries
import logging as log
import os
import platform
import shutil
import tempfile
import cv2
import numpy as np
import pytest
import requests

from os.path import expanduser
from vidgear.gears.asyncio.helper import generate_webdata, validate_webdata
from vidgear.gears.helper import (
    check_output, reducer
    download_ffmpeg_binaries,
    generate_auth_certificates,
    get_valid_ffmpeg_path,
    logger_handler,
    validate_ffmpeg,
)

# define test logger
logger = log.getLogger("Test_helper")
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


def getframe():
    """
    returns empty numpy frame/array of dimensions: (500,800,3)
    """
    return np.zeros([500, 800, 3], dtype=np.uint8)


def test_ffmpeg_static_installation():
    """
    Test to ensure successful FFmpeg static Installation on Windows
    """
    startpath = os.path.abspath(
        os.path.join(tempfile.gettempdir(), "Downloads/FFmpeg_static")
    )
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = " " * 4 * (level)
        logger.debug("{}{}/".format(indent, os.path.basename(root)))
        subindent = " " * 4 * (level + 1)
        for f in files:
            logger.debug("{}{}".format(subindent, f))


test_data = [
    (
        "wrong_test_path",
        ("win64" if platform.machine().endswith("64") else "win32") if _windows else "",
    ),
    (
        tempfile.gettempdir(),
        ("win64" if platform.machine().endswith("64") else "win32") if _windows else "",
    ),
    (tempfile.gettempdir(), "wrong_bit"),
]


@pytest.mark.parametrize("paths, os_bit", test_data)
def test_ffmpeg_binaries_download(paths, os_bit):
    """
    Testing Static FFmpeg auto-download on Windows OS
    """
    file_path = ""
    try:
        file_path = download_ffmpeg_binaries(
            path=paths, os_windows=_windows, os_bit=os_bit
        )
        if file_path:
            assert os.path.isfile(file_path), "FFmpeg download failed!"
            shutil.rmtree(os.path.abspath(os.path.join(file_path, "../..")))
    except Exception as e:
        if paths == "wrong_test_path" or os_bit == "wrong_bit":
            pass
        elif isinstance(e, requests.exceptions.Timeout):
            logger.exceptions(str(e))
        else:
            pytest.fail(str(e))


@pytest.mark.parametrize("paths", ["wrong_test_path", return_static_ffmpeg()])
def test_validate_ffmpeg(paths):
    """
    Testing downloaded FFmpeg Static binaries validation on Windows OS
    """
    try:
        output = validate_ffmpeg(paths, logging=True)
        if paths != "wrong_test_path":
            assert bool(output), "Validation Test failed at path: {}".format(paths)
    except Exception as e:
        if paths == "wrong_test_path":
            pass
        else:
            pytest.fail(str(e))


test_data = [
    ("", "", True),
    ("wrong_test_path", "", False),
    ("", "wrong_test_path", False),
    ("", tempfile.gettempdir(), True),
    (return_static_ffmpeg(), "", True),
    (os.path.dirname(return_static_ffmpeg()), "", True),
]


@pytest.mark.parametrize("paths, ffmpeg_download_paths, results", test_data)
def test_get_valid_ffmpeg_path(paths, ffmpeg_download_paths, results):
    """
    Testing FFmpeg excutables validation and correction:
    """
    try:
        output = get_valid_ffmpeg_path(
            custom_ffmpeg=paths,
            is_windows=_windows,
            ffmpeg_download_path=ffmpeg_download_paths,
            logging=True,
        )
        if not (
            paths == "wrong_test_path" or ffmpeg_download_paths == "wrong_test_path"
        ):
            assert (
                bool(output) == results
            ), "FFmpeg excutables validation and correction Test failed at path: {} and FFmpeg ffmpeg_download_paths: {}".format(
                paths, ffmpeg_download_paths
            )
    except Exception as e:
        if paths == "wrong_test_path" or ffmpeg_download_paths == "wrong_test_path":
            pass
        elif isinstance(e, requests.exceptions.Timeout):
            logger.exceptions(str(e))
        else:
            pytest.fail(str(e))


test_data = [
    (os.path.join(expanduser("~"), ".vidgear"), False, True),
    ("test_folder", False, True),
    (tempfile.gettempdir(), False, True),
    (tempfile.gettempdir(), True, True),
]


@pytest.mark.parametrize("paths, overwrite_cert, results", test_data)
def test_generate_auth_certificates(paths, overwrite_cert, results):
    """
    Testing auto-Generation and auto-validation of CURVE ZMQ keys/certificates 
    """
    try:
        if overwrite_cert:
            logger.warning(
                "Overwriting ZMQ Authentication certificates over previous ones!"
            )
        output = generate_auth_certificates(
            paths, overwrite=overwrite_cert, logging=True
        )
        assert bool(output) == results
    except Exception as e:
        pytest.fail(str(e))


test_data = [
    (expanduser("~"), False, True),
    (os.path.join(expanduser("~"), ".vidgear"), True, True),
    ("test_folder", False, True),
]


@pytest.mark.parametrize("paths, overwrite_default, results", test_data)
def test_generate_webdata(paths, overwrite_default, results):
    """
    Testing auto-Generation and auto-validation of WebGear data files 
    """
    try:
        output = generate_webdata(
            paths, overwrite_default=overwrite_default, logging=True
        )
        assert bool(output) == results
    except Exception as e:
        if isinstance(e, requests.exceptions.Timeout):
            logger.exceptions(str(e))
        else:
            pytest.fail(str(e))


@pytest.mark.xfail(raises=Exception)
def test_validate_webdata():
    """
    Testing validation function of WebGear API
    """
    validate_webdata(
        os.path.join(expanduser("~"), ".vidgear"),
        files=["im_not_a_file1", "im_not_a_file2", "im_not_a_file3"],
        logging=True,
    )


@pytest.mark.xfail(raises=Exception)
def test_check_output():
    """
    Testing validation function of WebGear API
    """
    check_output(["ffmpeg", "-Vv"])


@pytest.mark.parametrize(
    "frame , percentage, result",
    [(getframe(), 85, True), (None, 80, False), (getframe(), 95, False)],
)
def test_reducer(frame, percentage, result):
    """
    Testing frame size reducer function 
    """
    if not (frame is None):
        org_size = frame.shape[:2]
    try:
        reduced_frame = reducer(frame, percentage)
        logger.debug(reduced_frame.shape)
        assert not (reduced_frame is None)
        reduced_frame_size = reduced_frame.shape[:2]
        assert (
            100 * reduced_frame_size[0] // (100 - percentage) == org_size[0]
        )  # cross-check width
        assert (
            100 * reduced_frame_size[1] // (100 - percentage) == org_size[1]
        )  # cross-check height
    except Exception as e:
        if isinstance(e, ValueError) and not (result):
            pass
        else:
            pytest.fail(str(e))
