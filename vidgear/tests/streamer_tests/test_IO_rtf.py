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

import numpy as np
import pytest

from vidgear.gears import StreamGear


def test_assertfailedstream():
    """
    IO Test - made to fail with Wrong Output file path
    """
    with pytest.raises(ValueError):
        # wrong folder path does not exist
        streamer = StreamGear(output="wrong_path/output.mpd", logging=True)
        streamer.terminate()


def test_failedchannels():
    """
    IO Test - made to fail with invalid channel length
    """
    np.random.seed(0)
    # generate random data for 10 frames
    random_data = np.random.random(size=(480, 640, 5)) * 255
    input_data = random_data.astype(np.uint8)

    # 'garbage' extension does not exist
    with pytest.raises(ValueError):
        streamer = StreamGear("output.mpd", logging=True)
        streamer.stream(input_data)
        streamer.terminate()


def test_fail_framedimension():
    """
    IO Test - made to fail with multiple frame dimension
    """
    np.random.seed(0)
    # generate random data for 10 frames
    random_data1 = np.random.random(size=(480, 640, 3)) * 255
    input_data1 = random_data1.astype(np.uint8)

    np.random.seed(0)
    random_data2 = np.random.random(size=(580, 640, 3)) * 255
    input_data2 = random_data2.astype(np.uint8)

    streamer = None
    try:
        streamer = StreamGear(output="output.mpd")
        streamer.stream(None)
        streamer.stream(input_data1)
        streamer.stream(input_data2)
    except Exception as e:
        if isinstance(e, ValueError):
            pytest.xfail("Test Passed!")
        else:
            pytest.fail(str(e))
    finally:
        if not streamer is None:
            streamer.terminate()


@pytest.mark.xfail(raises=RuntimeError)
def test_method_call_rtf():
    """
    Method calling Test - Made to fail by calling method in the wrong context.
    """
    stream_params = {"-video_source": 1234}  # for CI testing only
    streamer = StreamGear(output="output.mpd", logging=True, **stream_params)
    streamer.transcode_source()
    streamer.terminate()


def test_invalid_params_rtf():
    """
    Invalid parameter Failure Test - Made to fail by calling invalid parameters
    """
    np.random.seed(0)
    # generate random data for 10 frames
    random_data = np.random.random(size=(480, 640, 3)) * 255
    input_data = random_data.astype(np.uint8)

    stream_params = {"-vcodec": "unknown"}
    streamer = StreamGear(output="output.mpd", logging=True, **stream_params)
    with pytest.raises(ValueError):
        streamer.stream(input_data)
        streamer.stream(input_data)
    streamer.terminate()
