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

import asyncio
import logging as log

import cv2
import numpy as np
import pytest

from vidgear.gears.asyncio.helper import (
    create_blank_frame,
    h264_access_units,
    reducer,
)
from vidgear.gears.helper import logger_handler, retrieve_best_interpolation

# define test logger
logger = log.getLogger("Test_Asyncio_Helper")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.SelectorEventLoop()
    yield loop
    loop.close()


def getframe():
    """
    returns empty numpy frame/array of dimensions: (500,800,3)
    """
    return (np.random.standard_normal([500, 800, 3]) * 255).astype(np.uint8)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "frame , percentage, interpolation, result",
    [
        (getframe(), 85, cv2.INTER_AREA, True),
        (None, 80, cv2.INTER_AREA, False),
        (getframe(), 95, cv2.INTER_AREA, False),
        (getframe(), 80, "invalid", False),
        (getframe(), 80, 797, False),
    ],
)
async def test_reducer_asyncio(frame, percentage, interpolation, result):
    """
    Testing frame size reducer function
    """
    if frame is not None:
        org_size = frame.shape[:2]
    try:
        reduced_frame = await reducer(frame, percentage, interpolation)
        logger.debug(reduced_frame.shape)
        assert reduced_frame is not None
        reduced_frame_size = reduced_frame.shape[:2]
        assert (
            100 * reduced_frame_size[0] // (100 - percentage) == org_size[0]
        )  # cross-check width
        assert (
            100 * reduced_frame_size[1] // (100 - percentage) == org_size[1]
        )  # cross-check height
    except Exception as e:
        if not (result):
            pytest.xfail(str(e))
        else:
            pytest.fail(str(e))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "frame , text",
    [
        (getframe(), "ok"),
        (cv2.cvtColor(getframe(), cv2.COLOR_BGR2BGRA), "ok"),
        (None, ""),
        (getframe(), 123),
    ],
)
async def test_create_blank_frame_asyncio(frame, text):
    """
    Testing create_blank_frame function
    """
    try:
        text_frame = create_blank_frame(frame=frame, text=text, logging=True)
        logger.debug(text_frame.shape)
        assert text_frame is not None
    except Exception as e:
        if frame is not None:
            pytest.fail(str(e))


@pytest.mark.parametrize(
    "interpolations",
    [
        "invalid",
        ["invalid", "invalid2", "INTER_LANCZOS4"],
        ["INTER_NEAREST_EXACT", "INTER_LINEAR_EXACT", "INTER_LANCZOS4"],
    ],
)
def test_retrieve_best_interpolation(interpolations):
    """
    Testing retrieve_best_interpolation method
    """
    try:
        output = retrieve_best_interpolation(interpolations)
        if interpolations != "invalid":
            assert output, "Test failed"
    except Exception as e:
        pytest.fail(str(e))


def nal(nal_type, first_mb=True, payload=b"\xaa\xbb", long_code=True):
    """
    returns Annex-B NAL unit; slices carry `first_mb_in_slice == 0` flag in MSB of first payload byte
    """
    body = (b"\x80" if first_mb else b"\x00") + payload if nal_type in (1, 5) else payload
    return (b"\x00\x00\x00\x01" if long_code else b"\x00\x00\x01") + bytes([0x60 | nal_type]) + body


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_size", [1, 3, 7, 65536])
async def test_h264_access_units(chunk_size):
    """
    Testing H.264 Annex-B access-unit assembly: keyframe bundling, multi-slice pictures, chunk boundaries & EOF flush
    """
    keyframe = nal(7) + nal(8) + nal(6, long_code=False) + nal(5) + nal(5, first_mb=False)
    delta1 = nal(1) + nal(1, first_mb=False, long_code=False)
    delta2 = nal(1)
    reader = asyncio.StreamReader()
    reader.feed_data(keyframe + delta1 + delta2)
    reader.feed_eof()
    units = [u async for u in h264_access_units(reader, chunk_size=chunk_size)]
    assert units == [keyframe, delta1, delta2]
