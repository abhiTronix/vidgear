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
import re
import cv2
import pytest
import logging as log
import platform
import tempfile
import subprocess
from six import string_types

from vidgear.gears import CamGear, WriteGear
from vidgear.gears.helper import capPropId, check_output, logger_handler

# define test logger
logger = log.getLogger("Test_commpression_mode")
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


def return_testvideo_path():
    """
    returns Test video path
    """
    path = "{}/Downloads/Test_videos/BigBuckBunny_4sec.mp4".format(
        tempfile.gettempdir()
    )
    return os.path.abspath(path)


def getFrameRate(path):
    """
    Returns framerate of video(at path provided) using FFmpeg
    """
    process = subprocess.Popen(
        [return_static_ffmpeg(), "-i", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    stdout, _ = process.communicate()
    output = stdout.decode()
    match_dict = re.search(r"\s(?P<fps>[\d\.]+?)\stbr", output).groupdict()
    return float(match_dict["fps"])


@pytest.mark.xfail(raises=AssertionError)
@pytest.mark.parametrize("c_ffmpeg", [return_static_ffmpeg(), "wrong_path"])
def test_input_framerate(c_ffmpeg):
    """
    Testing "-input_framerate" parameter provided by WriteGear(in Compression Mode)
    """
    stream = cv2.VideoCapture(return_testvideo_path())  # Open stream
    test_video_framerate = stream.get(cv2.CAP_PROP_FPS)
    output_params = (
        {"-input_framerate": test_video_framerate}
        if (c_ffmpeg != "wrong_path")
        else {"-input_framerate": "wrong_input"}
    )
    writer = WriteGear(
        output_filename="Output_tif.mp4",
        custom_ffmpeg=c_ffmpeg,
        logging=True,
        **output_params
    )  # Define writer
    while True:
        (grabbed, frame) = stream.read()
        if not grabbed:
            break
        writer.write(frame)
    stream.release()
    writer.close()
    output_video_framerate = getFrameRate(os.path.abspath("Output_tif.mp4"))
    assert test_video_framerate == output_video_framerate
    os.remove(os.path.abspath("Output_tif.mp4"))


@pytest.mark.xfail(raises=AssertionError)
@pytest.mark.parametrize(
    "conversion", ["COLOR_BGR2GRAY", "COLOR_BGR2INVALID", "COLOR_BGR2BGRA"]
)
def test_write(conversion):
    """
    Testing WriteGear Compression-Mode(FFmpeg) Writer capabilties in different colorspace with CamGearAPI.
    """
    # Open stream
    stream = CamGear(
        source=return_testvideo_path(), colorspace=conversion, logging=True
    ).start()
    writer = WriteGear(
        output_filename="Output_tw.mp4", custom_ffmpeg=return_static_ffmpeg()
    )  # Define writer
    while True:
        frame = stream.read()
        # check if frame is None
        if frame is None:
            # if True break the infinite loop
            break
        if conversion == "COLOR_BGR2RGBA":
            writer.write(frame, rgb_mode=True)
        elif conversion == "COLOR_BGR2INVALID":
            # test invalid color_space value
            stream.color_space = "wrong_colorspace"
            conversion = "COLOR_BGR2INVALID2"
            writer.write(frame)
        elif conversion == "COLOR_BGR2INVALID2":
            # test wrong color_space value
            stream.color_space = 1546755
            conversion = ""
            writer.write(frame)
        else:
            writer.write(frame)
    stream.stop()
    writer.close()
    basepath, _ = os.path.split(return_static_ffmpeg())
    ffprobe_path = os.path.join(
        basepath, "ffprobe.exe" if os.name == "nt" else "ffprobe"
    )
    result = check_output(
        [
            ffprobe_path,
            "-v",
            "error",
            "-count_frames",
            "-i",
            os.path.abspath("Output_tw.mp4"),
        ]
    )
    if result:
        if not isinstance(result, string_types):
            result = result.decode()
        logger.debug("Result: {}".format(result))
        for i in ["Error", "Invalid", "error", "invalid"]:
            assert not (i in result)
    os.remove(os.path.abspath("Output_tw.mp4"))


@pytest.mark.xfail(raises=AssertionError)
def test_output_dimensions():
    """
    Testing "-output_dimensions" special parameter provided by WriteGear(in Compression Mode)
    """
    dimensions = (640, 480)
    stream = cv2.VideoCapture(return_testvideo_path())
    output_params = {}
    if platform.system() == "Windows":
        output_params = {
            "-output_dimensions": dimensions,
            "-ffmpeg_download_path": tempfile.gettempdir(),
        }
    else:
        output_params = {"-output_dimensions": dimensions}
    writer = WriteGear(
        output_filename="Output_tod.mp4",
        custom_ffmpeg=return_static_ffmpeg(),
        logging=True,
        **output_params
    )  # Define writer
    while True:
        (grabbed, frame) = stream.read()
        if not grabbed:
            break
        writer.write(frame)
    stream.release()
    writer.close()

    output = cv2.VideoCapture(os.path.abspath("Output_tod.mp4"))
    output_dim = (
        output.get(cv2.CAP_PROP_FRAME_WIDTH),
        output.get(cv2.CAP_PROP_FRAME_HEIGHT),
    )
    assert output_dim[0] == 640 and output_dim[1] == 480
    output.release()

    os.remove(os.path.abspath("Output_tod.mp4"))


test_data_class = [
    ("", "", {}, False),
    ("Output1.mp4", "", {}, True),
    (os.path.join(tempfile.gettempdir(), "temp_write"), "", {}, True),
    ("Output2.mp4", "", {"-vcodec": "libx264", "-crf": 0, "-preset": "fast"}, True),
    (
        "Output3.mp4",
        return_static_ffmpeg(),
        {"-c:v": "libx265", "-vcodec": "libx264", "-crf": 0, "-preset": "veryfast"},
        True,
    ),
]


@pytest.mark.parametrize("f_name, c_ffmpeg, output_params, result", test_data_class)
def test_WriteGear_compression(f_name, c_ffmpeg, output_params, result):
    """
    Testing WriteGear Compression-Mode(FFmpeg) with different parameters
    """
    try:
        stream = cv2.VideoCapture(return_testvideo_path())  # Open stream
        writer = WriteGear(
            output_filename=f_name, compression_mode=True, **output_params
        )
        while True:
            (grabbed, frame) = stream.read()
            if not grabbed:
                break
            writer.write(frame)
        stream.release()
        writer.close()
        if f_name and os.path.isfile(os.path.abspath(f_name)):
            os.remove(os.path.abspath(f_name))
    except Exception as e:
        if result:
            pytest.fail(str(e))


@pytest.mark.parametrize(
    "ffmpeg_command_to_save_audio, logging, output_params",
    [
        (
            [
                "-y",
                "-i",
                return_testvideo_path(),
                "-vn",
                "-acodec",
                "copy",
                "input_audio.aac",
            ],
            False,
            {"-i": None, "-disable_force_termination": True},
        ),
        (None, True, {"-i": None, "-disable_force_termination": "OK"}),
        ([], False, {"-i": None}),
        (
            ["wrong_input"],
            True,
            {"-ffmpeg_download_path": 53}
            if (platform.system() == "Windows")
            else {"-disable_force_termination": "OK"},
        ),
        (["wrong_input"], True, {"-disable_force_termination": True}),
    ],
)
def test_WriteGear_customFFmpeg(ffmpeg_command_to_save_audio, logging, output_params):
    """
    Testing WriteGear Compression-Mode(FFmpeg) custom FFmpeg Pipeline by seperating audio from video
    """
    writer = None
    try:
        # define writer
        writer = WriteGear(
            output_filename="Output.mp4",
            compression_mode=(True if ffmpeg_command_to_save_audio else False),
            logging=logging,
            **output_params
        )  # Define writer

        # execute FFmpeg command
        writer.execute_ffmpeg_cmd(ffmpeg_command_to_save_audio)

        # assert audio file is created successfully
        if ffmpeg_command_to_save_audio and isinstance(
            ffmpeg_command_to_save_audio, list
        ):
            assert os.path.isfile("input_audio.aac")
    except Exception as e:
        if isinstance(e, AssertionError):
            pytest.fail(str(e))
        elif isinstance(e, ValueError):
            pytest.xfail("Test Passed!")
        else:
            logger.exception(str(e))
    finally:
        if not (writer is None):
            writer.close()
