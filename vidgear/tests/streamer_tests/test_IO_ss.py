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
import numpy as np
import pytest
import tempfile
import subprocess

from vidgear.gears import StreamGear


def return_testvideo_path():
    """
    returns Test video path
    """
    path = "{}/Downloads/Test_videos/BigBuckBunny_4sec.mp4".format(
        tempfile.gettempdir()
    )
    return os.path.abspath(path)


@pytest.mark.xfail(raises=(AssertionError, ValueError))
@pytest.mark.parametrize("output", ["garbage.garbage", "output.m3u8"])
def test_failedextension(output):
    """
    IO Test - made to fail with filename with wrong extension
    """
    stream_params = {"-video_source": return_testvideo_path()}
    streamer = StreamGear(output=output, logging=True, **stream_params)
    streamer.transcode_source()
    streamer.terminate()


def test_failedextensionsource():
    """
    IO Test - made to fail with filename with wrong extension for source
    """
    with pytest.raises(RuntimeError):
        # 'garbage' extension does not exist
        stream_params = {"-video_source": "garbage.garbage"}
        streamer = StreamGear(output="output.mpd", logging=True, **stream_params)
        streamer.transcode_source()
        streamer.terminate()


@pytest.mark.parametrize(
    "path, format",
    [
        ("rtmp://live.twitch.tv/output.mpd", "dash"),
        ("rtmp://live.twitch.tv/output.m3u8", "hls"),
        ("unknown://invalid.com/output.mpd", "dash"),
    ],
)
def test_paths_ss(path, format):
    """
    Paths Test - Test various paths/urls supported by StreamGear.
    """
    streamer = None
    try:
        stream_params = {"-video_source": return_testvideo_path()}
        streamer = StreamGear(output=path, format=format, logging=True, **stream_params)
    except Exception as e:
        if isinstance(e, ValueError):
            pytest.xfail("Test Passed!")
        else:
            pytest.fail(str(e))
    finally:
        if not streamer is None:
            streamer.terminate()


@pytest.mark.xfail(raises=RuntimeError)
def test_method_call_ss():
    """
    Method calling Test - Made to fail by calling method in the wrong context.
    """
    stream_params = {"-video_source": return_testvideo_path()}
    streamer = StreamGear(output="output.mpd", logging=True, **stream_params)
    streamer.stream("garbage.garbage")
    streamer.terminate()


@pytest.mark.xfail(raises=(AttributeError, RuntimeError))
def test_method_call_ss():
    """
    Method calling Test - Made to fail by calling method in the wrong context.
    """
    stream_params = {"-video_source": return_testvideo_path()}
    streamer = StreamGear(output="output.mpd", logging=True, **stream_params)
    streamer.stream("garbage.garbage")
    streamer.terminate()


@pytest.mark.xfail(raises=subprocess.CalledProcessError)
@pytest.mark.parametrize("format", ["dash", "hls"])
def test_invalid_params_ss(format):
    """
    Method calling Test - Made to fail by calling method in the wrong context.
    """
    stream_params = {"-video_source": return_testvideo_path(), "-vcodec": "unknown"}
    streamer = StreamGear(
        output="output{}".format(".mpd" if format == "dash" else ".m3u8"),
        format=format,
        logging=True,
        **stream_params
    )
    streamer.transcode_source()
    streamer.terminate()
