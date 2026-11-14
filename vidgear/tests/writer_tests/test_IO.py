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
import pathlib
import platform

import numpy as np
import pytest

from vidgear.gears import WriteGear
from vidgear.tests.utils.helpers import probe_video, return_static_ffmpeg


def test_failedextension(tmp_path):
    """
    IO Test - made to fail with filename with wrong extention
    """
    input_data = np.zeros((480, 640, 3), np.uint8)
    # 'garbage' extension does not exist
    writer = WriteGear(output=str(tmp_path / "garbage.garbage"), logging=True)
    with pytest.raises(ValueError, match="pipe broke"):
        for _ in range(100):
            writer.write(input_data)
    writer.close()


@pytest.mark.parametrize("size", [[(480, 640, 5)], [(480, 640, 1), (480, 640, 3)]])
def test_failedchannels(tmp_path, size):
    """
    IO Test - made to fail with invalid channel lengths
    """
    with (
        WriteGear(output=str(tmp_path / "output.mp4"), logging=True) as writer,
        pytest.raises(ValueError, match="channels"),
    ):
        for shape in size:
            writer.write(np.zeros(shape, np.uint8))


@pytest.mark.parametrize("compression_mode", [True, False])
def test_fail_framedimension(tmp_path, compression_mode):
    """
    IO Test - made to fail with multiple frame dimension
    """
    with WriteGear(
        output=str(tmp_path / "output.mp4"),
        compression_mode=compression_mode,
        logging=True,
    ) as writer:
        writer.write(None)
        writer.write(np.zeros((480, 640, 3), np.uint8))
        with pytest.raises(ValueError, match="same size"):
            writer.write(np.zeros((580, 640, 3), np.uint8))


@pytest.mark.parametrize(
    "compression_mode, path, valid",
    [
        (True, "output.mp4", True),
        (
            True,
            "/dev/video0",
            platform.system() == "Linux" and pathlib.Path("/dev/video0").is_char_device(),
        ),
        (True, "rtmp://live.twitch.tv/", True),
        (True, "unknown://invalid.com/", False),
        (True, os.path.join("missing_dir", "output.mp4"), False),
        (False, "output.mp4", True),
        (False, "rtmp://live.twitch.tv/", False),
    ],
)
def test_paths(tmp_path, compression_mode, path, valid):
    """
    Paths Test - Test various paths/urls supported by WriteGear.
    """
    if "://" not in path and not path.startswith("/dev"):
        path = str(tmp_path / path)
    if valid:
        WriteGear(output=path, compression_mode=compression_mode, logging=True).close()
    else:
        with pytest.raises(ValueError, match="not supported"):
            WriteGear(output=path, compression_mode=compression_mode, logging=True)


@pytest.mark.parametrize("v_codec", ["unknown", "mpeg4", "copy"])
def test_invalid_encoder(tmp_path, v_codec):
    """
    Unknown encoders must raise, valid ones (and `copy`) must be used as is
    """
    output = str(tmp_path / "output.avi")
    frame = np.zeros((48, 64, 3), np.uint8)
    with WriteGear(
        output=output,
        custom_ffmpeg=return_static_ffmpeg(),
        logging=True,
        **{"-vcodec": v_codec},
    ) as writer:
        if v_codec == "unknown":
            with pytest.raises(ValueError, match="unknown"):
                writer.write(frame)
            return
        for _ in range(5):
            writer.write(frame)
    assert probe_video(output)["approx_video_nframes"] == 5
