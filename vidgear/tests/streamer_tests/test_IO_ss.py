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
import subprocess

import pytest

from vidgear.gears import StreamGear
from vidgear.tests.utils.helpers import return_testvideo_path


@pytest.mark.parametrize("output", ["garbage.garbage", "output.m3u8"])
def test_failedextension(output, tmp_path):
    """
    IO Test - made to fail with filename with wrong extension
    """
    with pytest.raises(ValueError, match="extension"):
        StreamGear(output=str(tmp_path / output), **{"-video_source": return_testvideo_path()})


@pytest.mark.parametrize(
    "stream_params",
    [
        {"-video_source": "garbage.garbage"},  # missing file
        {"-video_source": 1234},
        {"-video_source": "unknown://invalid.com/input.mp4"},
        {"-video_source": return_testvideo_path(fmt="ao")},  # no video stream
        {"-video_source": return_testvideo_path(), "-audio": "/no/such/audio.aac"},
        {"-video_source": return_testvideo_path(), "-audio": return_testvideo_path(fmt="vo")},  # no audio
        {"-audio": ["-f", "lavfi", "-i", "invalid_source"]},
        {"-audio": 1234},
    ],
)
def test_invalid_sources(stream_params, tmp_path):
    """
    IO Test - invalid `-video_source`/`-audio` raise, instead of silently switching mode or dropping audio
    """
    with pytest.raises(ValueError):
        StreamGear(output=str(tmp_path / "output.mpd"), **stream_params)


@pytest.mark.parametrize(
    "path, format",
    [
        ("rtmp://live.twitch.tv/output.mpd", "dash"),
        ("rtmp://live.twitch.tv/output.m3u8", "hls"),
    ],
)
def test_paths_ss(path, format):
    """
    Paths Test - Test URL outputs supported by StreamGear.
    """
    stream_params = {
        "-video_source": return_testvideo_path(),
        "-ffmpeg_download_path": 12345,
        "-livestream": "invalid",
    }
    StreamGear(output=path, format=format, **stream_params).close()


def test_method_call_ss(tmp_path):
    """
    Method calling Test - Made to fail by calling method in the wrong context.
    """
    streamer = StreamGear(output=str(tmp_path / "output.mpd"), **{"-video_source": return_testvideo_path()})
    with pytest.raises(RuntimeError):
        streamer.stream("garbage.garbage")
    streamer.close()


@pytest.mark.parametrize("format", ["dash", "hls"])
def test_invalid_params_ss(format, tmp_path):
    """
    Method calling Test - FFmpeg failure raises `CalledProcessError`.
    """
    streamer = StreamGear(
        output=str(tmp_path / "output{}".format(".mpd" if format == "dash" else ".m3u8")),
        format=format,
        **{"-video_source": return_testvideo_path(), "-vcodec": "unknown"}
    )
    with pytest.raises(subprocess.CalledProcessError):
        streamer.transcode_source()
    streamer.close()
