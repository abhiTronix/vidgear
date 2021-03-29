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

from vidgear.gears import WriteGear


def test_failedextension():
    """
    IO Test - made to fail with filename with wrong extention
    """
    np.random.seed(0)
    # generate random data for 10 frames
    random_data = np.random.random(size=(10, 480, 640, 3)) * 255
    input_data = random_data.astype(np.uint8)

    # 'garbage' extension does not exist
    with pytest.raises(ValueError):
        writer = WriteGear("garbage.garbage", logging=True)
        writer.write(input_data)
        writer.close()


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize("size", [(480, 640, 5), [(480, 640, 1), (480, 640, 3)]])
def test_failedchannels(size):
    """
    IO Test - made to fail with invalid channel lengths
    """
    np.random.seed(0)
    if len(size) > 1:
        random_data_1 = np.random.random(size=size[0]) * 255
        input_data_ch1 = random_data_1.astype(np.uint8)
        random_data_2 = np.random.random(size=size[1]) * 255
        input_data_ch3 = random_data_2.astype(np.uint8)
        writer = WriteGear("output.mp4", compression_mode=True)
        writer.write(input_data_ch1)
        writer.write(input_data_ch3)
        writer.close()
    else:
        random_data = np.random.random(size=size) * 255
        input_data = random_data.astype(np.uint8)
        writer = WriteGear("output.mp4", compression_mode=True, logging=True)
        writer.write(input_data)
        writer.close()


@pytest.mark.parametrize("compression_mode", [True, False])
def test_fail_framedimension(compression_mode):
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

    writer = None
    try:
        writer = WriteGear("output.mp4", compression_mode=compression_mode, logging=True)
        writer.write(None)
        writer.write(input_data1)
        writer.write(input_data2)
    except Exception as e:
        if isinstance(e, ValueError):
            pytest.xfail("Test Passed!")
        else:
            pytest.fail(str(e))
    finally:
        if not writer is None:
            writer.close()


@pytest.mark.parametrize(
    "compression_mode, path",
    [
        (True, "output.mp4"),
        (True, "rtmp://live.twitch.tv/"),
        (True, "unknown://invalid.com/"),
        (False, "output.mp4"),
        (False, "rtmp://live.twitch.tv/"),
    ],
)
def test_paths(compression_mode, path):
    """
    Paths Test - Test various paths/urls supported by WriteGear.
    """
    writer = None
    try:
        writer = WriteGear(path, compression_mode=compression_mode, logging=True)
    except Exception as e:
        if isinstance(e, ValueError):
            pytest.xfail("Test Passed!")
        else:
            pytest.fail(str(e))
    finally:
        if not writer is None:
            writer.close()


def test_invalid_encoder():
    """
    Invalid encoder Failure Test
    """
    np.random.seed(0)
    # generate random data for 10 frames
    random_data = np.random.random(size=(480, 640, 3)) * 255
    input_data = random_data.astype(np.uint8)
    try:
        output_params = {"-vcodec": "unknown"}
        writer = WriteGear(
            "output.mp4", compression_mode=True, logging=True, **output_params
        )
        writer.write(input_data)
        writer.write(input_data)
        writer.close()
    except Exception as e:
        pytest.fail(str(e))
