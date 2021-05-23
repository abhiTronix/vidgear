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
import numpy as np
import pytest
import shutil
import logging as log
import platform
import requests
import tempfile
from os.path import expanduser
from mpegdash.parser import MPEGDASHParser

from vidgear.gears.helper import (
    reducer,
    dict2Args,
    mkdir_safe,
    delete_ext_safe,
    check_output,
    extract_time,
    create_blank_frame,
    is_valid_url,
    logger_handler,
    validate_audio,
    validate_video,
    validate_ffmpeg,
    get_video_bitrate,
    restore_levelnames,
    get_valid_ffmpeg_path,
    download_ffmpeg_binaries,
    check_gstreamer_support,
    generate_auth_certificates,
    get_supported_resolution,
    dimensions_to_resolutions,
)
from vidgear.gears.asyncio.helper import generate_webdata, validate_webdata

# define test logger
logger = log.getLogger("Test_helper")
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


def return_testvideo_path(fmt="av"):
    """
    returns Test video path
    """
    supported_fmts = {
        "av": "BigBuckBunny_4sec.mp4",
        "vo": "BigBuckBunny_4sec_VO.mp4",
        "ao": "BigBuckBunny_4sec_AO.mp4",
    }
    req_fmt = fmt if (fmt in supported_fmts) else "av"
    path = "{}/Downloads/Test_videos/{}".format(
        tempfile.gettempdir(), supported_fmts[req_fmt]
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


test_data = [
    {"-thread_queue_size": "512", "-f": "alsa", "-clones": 24},
    {
        "-thread_queue_size": "512",
        "-f": "alsa",
        "-clones": ["-map", "0:v:0", "-map", "1:a?"],
        "-ac": "1",
        "-ar": "48000",
        "-i": "plughw:CARD=CAMERA,DEV=0",
    },
    {
        "-thread_queue_size": "512",
        "-f": "alsa",
        "-ac": "1",
        "-ar": "48000",
        "-i": "plughw:CARD=CAMERA,DEV=0",
    },
]


@pytest.mark.parametrize("dictionary", test_data)
def test_dict2Args(dictionary):
    """
    Testing dict2Args helper function.
    """
    result = dict2Args(dictionary)
    if result and isinstance(result, list):
        logger.debug("dict2Args converted Arguments are: {}".format(result))
    else:
        pytest.fail("Failed to complete this test!")


test_data = [
    (
        os.path.join(tempfile.gettempdir(), "temp_ffmpeg"),
        "win32" if _windows else "",
    ),
    (
        os.path.join(tempfile.gettempdir(), "temp_ffmpeg"),
        "win64" if _windows else "",
    ),
    ("wrong_test_path", "wrong_bit"),
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
            logger.debug("FFmpeg Binary path: {}".format(file_path))
            assert os.path.isfile(file_path), "FFmpeg download failed!"
            shutil.rmtree(os.path.abspath(os.path.join(file_path, "../..")))
    except Exception as e:
        if paths == "wrong_test_path" or os_bit == "wrong_bit":
            pass
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
    ("", os.path.join(tempfile.gettempdir(), "temp_ffmpeg"), True),
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
    (os.path.join(tempfile.gettempdir(), "temp_ffmpeg"), False, True),
    (os.path.join(tempfile.gettempdir(), "temp_ffmpeg"), True, True),
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


@pytest.mark.parametrize(
    "URL, result",
    [
        ("rtmp://live.twitch.tv/", True),
        (None, False),
        ("unknown://invalid.com/", False),
    ],
)
def test_is_valid_url(URL, result):
    """
    Testing is_valid_url function
    """
    try:
        result_url = is_valid_url(return_static_ffmpeg(), url=URL, logging=True)
        assert result_url == result, "URL validity test Failed!"
    except Exception as e:
        pytest.fail(str(e))


@pytest.mark.parametrize(
    "path, result",
    [
        (return_testvideo_path(), True),
        (None, False),
    ],
)
def test_validate_video(path, result):
    """
    Testing validate_video function
    """
    try:
        results = validate_video(return_static_ffmpeg(), video_path=path)
        if result:
            assert not (results is None), "Video path validity test Failed!"
    except Exception as e:
        pytest.fail(str(e))


@pytest.mark.parametrize(
    "path, result",
    [
        (return_testvideo_path(), True),
        (return_testvideo_path(fmt="vo"), False),
        (None, False),
    ],
)
def test_validate_audio(path, result):
    """
    Testing validate_audio function
    """
    try:
        results = validate_audio(return_static_ffmpeg(), file_path=path)
        if result:
            assert results, "Audio path validity test Failed!"
    except Exception as e:
        pytest.fail(str(e))


@pytest.mark.parametrize(
    "frame , text",
    [(getframe(), "ok"), (None, ""), (getframe(), 123)],
)
def test_create_blank_frame(frame, text):
    """
    Testing frame size reducer function
    """
    try:
        text_frame = create_blank_frame(frame=frame, text=text)
        logger.debug(text_frame.shape)
        assert not (text_frame is None)
    except Exception as e:
        if not (frame is None):
            pytest.fail(str(e))


@pytest.mark.parametrize(
    "value, result",
    [
        ("Duration: 00:00:08.44, start: 0.000000, bitrate: 804 kb/s", 8),
        ("Duration: 00:07:08 , start: 0.000000, bitrate: 804 kb/s", 428),
        ("", False),
    ],
)
def test_extract_time(value, result):
    """
    Testing extract_time function
    """
    try:
        results = extract_time(value)
        assert results == result, "Extract time function test Failed!"
    except Exception as e:
        pytest.fail(str(e))


@pytest.mark.parametrize(
    "value, result",
    [
        (["256x144", "1280x720", "3840x2160"], ["144p", "720p", "2160p"]),
        (["480p", "1920x1080"], ["480p", "1080p"]),
        ("invalid", []),
    ],
)
def test_dimensions_to_resolutions(value, result):
    """
    Testing dimensions_to_resolutions function
    """
    try:
        results = dimensions_to_resolutions(value)
        assert results == result, "dimensions_to_resolutions function Failed!"
    except Exception as e:
        pytest.fail(str(e))


@pytest.mark.parametrize(
    "value, result",
    [
        ("360P", "360p"),
        ("720p", "720p"),
        ("invalid", "best"),
    ],
)
def test_get_supported_resolution(value, result):
    """
    Testing get_supported_resolution function
    """
    try:
        results = get_supported_resolution(value, logging=True)
        assert results == result, "get_supported_resolution function Failed!"
    except Exception as e:
        pytest.fail(str(e))


def test_get_video_bitrate():
    """
    Testing get_video_bitrate function
    """
    try:
        get_video_bitrate(640, 480, 60.0, 0.1)
    except Exception as e:
        pytest.fail(str(e))


@pytest.mark.skipif(platform.system() in ["Darwin", "Windows"], reason="Not supported")
def test_check_gstreamer_support():
    """
    Testing check_gstreamer_support function
    """
    try:
        assert check_gstreamer_support(), "Test check_gstreamer_support failed!"
    except Exception as e:
        pytest.fail(str(e))


@pytest.mark.parametrize(
    "ext, result",
    [
        ([".m4s", ".mpd"], True),
        ([], False),
    ],
)
def test_delete_ext_safe(ext, result):
    """
    Testing delete_ext_safe function
    """
    try:
        path = os.path.join(expanduser("~"), "test_mpd")
        if ext:
            mkdir_safe(path, logging=True)
            # re-create directory for more coverage
            mkdir_safe(path, logging=True)
            mpd_file_path = os.path.join(path, "dash_test.mpd")
            from vidgear.gears import StreamGear

            stream_params = {
                "-video_source": return_testvideo_path(),
            }
            streamer = StreamGear(output=mpd_file_path, **stream_params)
            streamer.transcode_source()
            streamer.terminate()
            assert check_valid_mpd(mpd_file_path)
        delete_ext_safe(path, ext, logging=True)
        assert not os.listdir(path), "`delete_ext_safe` Test failed!"
        # cleanup
        if os.path.isdir(path):
            shutil.rmtree(path)
    except Exception as e:
        if result:
            pytest.fail(str(e))
