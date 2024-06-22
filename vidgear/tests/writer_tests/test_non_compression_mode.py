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

import os
import cv2
import pytest
import logging as log
import platform
import tempfile

from six import string_types
from vidgear.gears import WriteGear
from vidgear.gears.helper import capPropId, check_output, logger_handler

# define test logger
logger = log.getLogger("Test_non_commpression_mode")
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
    returns Test Video path
    """
    path = "{}/Downloads/Test_videos/BigBuckBunny_4sec.mp4".format(
        tempfile.gettempdir()
    )
    return os.path.abspath(path)


def remove_file_safe(path):
    """
    Remove file safely
    """
    try:
        if path and os.path.isfile(os.path.abspath(path)):
            os.remove(path)
    except Exception as e:
        logger.exception(e)


@pytest.mark.xfail(raises=AssertionError)
@pytest.mark.parametrize("conversion", ["COLOR_BGR2GRAY", "COLOR_BGR2YUV"])
def test_write(conversion):
    """
    Testing VidGear Non-Compression(OpenCV) Mode Writer
    """
    stream = cv2.VideoCapture(return_testvideo_path())
    writer = WriteGear(output="Output_twc.avi", compression_mode=False)  # Define writer
    while True:
        (grabbed, frame) = stream.read()
        # read frames
        # check if frame empty
        if not grabbed:
            # if True break the infinite loop
            break
        if conversion:
            frame = cv2.cvtColor(frame, capPropId(conversion))
        writer.write(frame)
    stream.release()
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
            os.path.abspath("Output_twc.avi"),
        ]
    )
    if result:
        if not isinstance(result, string_types):
            result = result.decode()
        logger.debug("Result: {}".format(result))
        for i in ["Error", "Invalid", "error", "invalid"]:
            assert not (i in result)
    remove_file_safe("Output_twc.avi")


test_data_class = [
    ("", {"-gst_pipeline_mode": "invalid"}, False),
    (os.path.join(tempfile.gettempdir(), "temp_write"), {}, True),
    (
        "Output_twc.mp4",
        {
            "-fourcc": "DIVX",
            "-fps": 25,
            "-backend": "CAP_FFMPEG",
            "-disable_ffmpeg_window": True,
            "-color": True,
            "-gst_pipeline_mode": False,
        },
        True,
    ),
    (
        "Output_twc.avi",
        {
            "-fourcc": ["NULL"],
            "-backend": "INVALID",
            "-unknown": "INVALID",
            "-fps": -11,
        },
        False,
    ),
    (
        "appsrc ! videoconvert ! avenc_mpeg4 bitrate=100000 ! mp4mux ! filesink location=foo.mp4",
        (
            {"-gst_pipeline_mode": True}
            if platform.system() == "Linux"
            else {"-gst_pipeline_mode": "invalid"}
        ),
        True if platform.system() == "Linux" else False,
    ),
]


@pytest.mark.parametrize("f_name, output_params, result", test_data_class)
def test_WriteGear_compression(f_name, output_params, result):
    """
    Testing VidGear Non-Compression(OpenCV) Mode with different parameters
    """
    try:
        stream = cv2.VideoCapture(return_testvideo_path())
        with WriteGear(
            output=f_name, compression_mode=False, logging=True, **output_params
        ) as writer:
            while True:
                (grabbed, frame) = stream.read()
                if not grabbed:
                    break
                writer.write(frame)
            stream.release()
        remove_file_safe(
            "foo.html"
            if "-gst_pipeline_mode" in output_params
            and output_params["-gst_pipeline_mode"] == True
            else f_name
        )
    except Exception as e:
        if result:
            pytest.fail(str(e))
        else:
            pytest.xfail(str(e))
