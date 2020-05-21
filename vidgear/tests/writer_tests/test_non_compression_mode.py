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
import tempfile
import cv2
import pytest
from six import string_types

from vidgear.gears import WriteGear
from vidgear.gears.helper import capPropId, check_output, logger_handler

# define test logger
logger = log.getLogger("Test_non_commpression_mode")
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
	returns Test Video path
	"""
    path = "{}/Downloads/Test_videos/BigBuckBunny_4sec.mp4".format(
        tempfile.gettempdir()
    )
    return os.path.abspath(path)


@pytest.mark.xfail(raises=AssertionError)
@pytest.mark.parametrize(
    "conversion", ["COLOR_BGR2GRAY", "COLOR_BGR2YUV"]
)
def test_write(conversion):
    """
	Testing VidGear Non-Compression(OpenCV) Mode Writer
	"""
    stream = cv2.VideoCapture(return_testvideo_path())
    writer = WriteGear(
        output_filename="Output_twc.avi", compression_mode=False
    )  # Define writer
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
    os.remove(os.path.abspath("Output_twc.avi"))


test_data_class = [
    ("", {}, False),
    (tempfile.gettempdir(), {}, True),
    (
        "Output_twc.mp4",
        {"-fourcc": "DIVX", "-fps": 25, "-backend": "CAP_FFMPEG", "-color": True},
        True,
    ),
    (
        "Output_twc.avi",
        {"-fourcc": "NULL", "-backend": "INVALID"},
        False,
    )
]


@pytest.mark.parametrize("f_name, output_params, result", test_data_class)
def test_WriteGear_compression(f_name, output_params, result):
    """
	Testing VidGear Non-Compression(OpenCV) Mode with different parameters
	"""
    try:
        stream = cv2.VideoCapture(return_testvideo_path())
        writer = WriteGear(
            output_filename=f_name,
            compression_mode=False,
            logging=True,
            **output_params
        )
        while True:
            (grabbed, frame) = stream.read()
            if not grabbed:
                break
            writer.write(frame)
        stream.release()
        writer.close()
        if f_name and f_name != tempfile.gettempdir():
            os.remove(os.path.abspath(f_name))
    except Exception as e:
        if result:
            pytest.fail(str(e))
