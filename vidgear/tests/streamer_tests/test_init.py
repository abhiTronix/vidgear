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

import pytest

from vidgear.gears import StreamGear
from vidgear.tests.utils.helpers import return_static_ffmpeg


def test_custom_ffmpeg(tmp_path):
    """
    Testing valid custom FFmpeg for StreamGear
    """
    StreamGear(output=str(tmp_path / "output.mpd"), custom_ffmpeg=return_static_ffmpeg()).close()


@pytest.mark.parametrize("c_ffmpeg", ["wrong_path", 1234])
def test_custom_ffmpeg_invalid(c_ffmpeg, tmp_path):
    """
    Testing invalid custom FFmpeg for StreamGear raises
    """
    with pytest.raises(RuntimeError):
        StreamGear(output=str(tmp_path / "output.mpd"), custom_ffmpeg=c_ffmpeg)


@pytest.mark.parametrize("format, ext", [("dash", "mpd"), ("hls", "m3u8"), ("HLS ", "m3u8")])
def test_formats(format, ext, tmp_path):
    """
    Testing supported formats for StreamGear
    """
    StreamGear(output=str(tmp_path / "output.{}".format(ext)), format=format).close()


@pytest.mark.parametrize(
    "format, output",
    [
        ("hls", "output.mpd"),  # extension mismatch
        ("mash", "output.mpd"),  # typo
        (1234, "output.mpd"),
        (None, "output.mpd"),
        ("dash", None),
        ("dash", "temp_mpd"),  # no extension dot
        ("dash", "missing_dir/output.mpd"),  # parent must exist
        ("dash", "unknown://invalid.com/output.mpd"),  # unsupported URL scheme
    ],
)
def test_invalid_outputs(format, output, tmp_path):
    """
    Testing invalid formats/outputs for StreamGear raise
    """
    with pytest.raises(ValueError):
        StreamGear(
            output=str(tmp_path / output) if output and "://" not in output else output,
            format=format,
        )


def test_clear_prev_assets(tmp_path):
    """
    Testing `-clear_prev_assets` removes StreamGear's own assets only, even for a new manifest name
    """
    stale = ["init-stream2.m4s", "chunk-stream0-00001.m4s", "chunk-stream1-001.ts", "stream_0.m3u8", "init_0.mp4"]
    user = ["other.mpd", "notes.txt", "video.mp4"]
    for name in stale + user:
        (tmp_path / name).touch()
    StreamGear(output=str(tmp_path / "new.mpd"), **{"-clear_prev_assets": True}).close()
    assert sorted(os.listdir(tmp_path)) == sorted(user)
