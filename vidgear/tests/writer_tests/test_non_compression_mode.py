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
import platform

import cv2
import pytest

from vidgear.gears import WriteGear
from vidgear.gears.helper import capPropId, check_gstreamer_support, delete_file_safe
from vidgear.tests.utils.helpers import (
    decode_frames,
    get_testing_dir,
    return_testvideo_path,
)


@pytest.mark.parametrize("conversion", ["COLOR_BGR2GRAY", "COLOR_BGR2YUV"])
def test_write(tmp_path, conversion):
    """
    Testing VidGear Non-Compression(OpenCV) Mode Writer (grayscale frames included)
    """
    stream = cv2.VideoCapture(return_testvideo_path())
    output_path = str(tmp_path / "Output_twc.avi")
    writer = WriteGear(output=output_path, compression_mode=False)  # Define writer
    count = 0
    while True:
        (grabbed, frame) = stream.read()
        if not grabbed:
            break
        writer.write(cv2.cvtColor(frame, capPropId(conversion)))
        count += 1
    stream.release()
    writer.close()
    assert len(decode_frames(output_path)) == count


test_data_class = [
    ("", {"-gst_pipeline_mode": "invalid"}, False),
    (os.path.join(get_testing_dir(), "temp_write"), {}, True),
    (
        os.path.join(get_testing_dir(), "Output_twc.mp4"),
        {
            "-fourcc": "DIVX",
            "-fps": 29.97,
            "-backend": "CAP_FFMPEG",
            "-disable_ffmpeg_window": True,
            "-color": True,
            "-gst_pipeline_mode": False,
        },
        True,
    ),
    (
        os.path.join(get_testing_dir(), "Output_twc.avi"),
        {
            "-fourcc": ["NULL"],
            "-backend": "INVALID",
            "-unknown": "INVALID",
            "-fps": -11,
        },
        False,
    ),
    (
        "appsrc ! videoconvert ! avenc_mpeg4 bitrate=100000 ! mp4mux ! filesink location={}".format(
            os.path.join(get_testing_dir(), "foo.mp4")
        ),
        (
            {"-gst_pipeline_mode": True}
            if platform.system() == "Linux"
            else {"-gst_pipeline_mode": "invalid"}
        ),
        platform.system() == "Linux" and check_gstreamer_support(),
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
        delete_file_safe(
            os.path.join(get_testing_dir(), "foo.mp4")
            if output_params.get("-gst_pipeline_mode")
            else f_name
        )
    except Exception as e:
        if result:
            pytest.fail(str(e))
        else:
            pytest.xfail(str(e))
