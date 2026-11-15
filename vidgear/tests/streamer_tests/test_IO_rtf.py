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
import warnings

import numpy as np
import pytest

from vidgear.gears import StreamGear


@pytest.mark.parametrize("sizes", [[(48, 64, 5)], [(48, 64, 1), (48, 64, 3)], [(48, 64, 3), (58, 64, 3)]])
def test_failed_frames(sizes, tmp_path):
    """
    IO Test - made to fail with invalid channels, or frames of different channels/size
    """
    streamer = StreamGear(str(tmp_path / "output.mpd"))
    with pytest.raises(ValueError):
        streamer.stream(None)
        for size in sizes:
            streamer.stream(np.zeros(size, np.uint8))
    streamer.close()


@pytest.mark.parametrize("dtype", [np.float32, np.uint16, np.int16])
def test_failed_dtype(dtype, tmp_path):
    """
    IO Test - non `uint8` frames are rejected instead of being written as garbage
    """
    with StreamGear(str(tmp_path / "output.mpd")) as streamer, pytest.raises(ValueError, match="uint8"):
        streamer.stream(np.zeros((48, 64, 3), dtype))


def test_stream_after_close(tmp_path):
    """
    IO Test - `stream()` after `close()` raises
    """
    streamer = StreamGear(str(tmp_path / "output.mpd"))
    streamer.stream(np.zeros((48, 64, 3), np.uint8))
    streamer.close()
    with pytest.raises(RuntimeError, match="closed"):
        streamer.stream(np.zeros((48, 64, 3), np.uint8))


def test_method_call_rtf(tmp_path):
    """
    Method calling Test - Made to fail by calling method in the wrong context.
    """
    streamer = StreamGear(output=str(tmp_path / "output.mpd"))
    with pytest.raises(RuntimeError):
        streamer.transcode_source()
    streamer.close()


@pytest.mark.parametrize("format", ["dash", "hls"])
def test_invalid_params_rtf(format, tmp_path):
    """
    Invalid parameter Failure Test - FFmpeg dies and `stream()` reports its last words
    """
    streamer = StreamGear(
        output=str(tmp_path / "output{}".format(".mpd" if format == "dash" else ".m3u8")),
        format=format,
        **{"-vcodec": "unknown", "-livestream": "invalid"}
    )
    with pytest.raises(ValueError, match="returncode"):
        for _ in range(500):
            streamer.stream(np.zeros((48, 64, 3), np.uint8))
    streamer.close()
    assert streamer.returncode not in (None, 0)


def test_no_deprecation_warning(tmp_path):
    """
    `stream()` warns about `rgb_mode` only when it is passed (keyword or positional)
    """
    frame = np.zeros((48, 64, 3), np.uint8)
    with StreamGear(str(tmp_path / "output.mpd")) as streamer:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            streamer.stream(frame)
        with pytest.warns(DeprecationWarning):
            streamer.stream(frame, rgb_mode=False)
        with pytest.warns(DeprecationWarning):
            streamer.stream(frame, False)
