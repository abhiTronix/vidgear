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
import cv2
import sys
import queue
import platform
import numpy as np
import pytest
import asyncio
import functools
import logging as log
import tempfile

from vidgear.gears.asyncio import NetGear_Async
from vidgear.gears.asyncio.helper import logger_handler

# define test logger
logger = log.getLogger("Test_NetGear_Async")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


def return_testvideo_path():
    """
    returns Test Video path
    """
    path = "{}/Downloads/Test_videos/BigBuckBunny_4sec.mp4".format(
        tempfile.gettempdir()
    )
    return os.path.abspath(path)


# Create a async frame generator as custom source
async def custom_frame_generator():
    # Open video stream
    stream = cv2.VideoCapture(return_testvideo_path())
    # loop over stream until its terminated
    while True:
        # read frames
        (grabbed, frame) = stream.read()
        # check if frame empty
        if not grabbed:
            break
        # yield frame
        yield frame
        # sleep for sometime
        await asyncio.sleep(0.000001)
    # close stream
    stream.release()


# Create a async function where you want to show/manipulate your received frames
async def client_iterator(client):
    # loop over Client's Asynchronous Frame Generator
    async for frame in client.recv_generator():
        # test frame validity
        assert not (frame is None or np.shape(frame) == ()), "Failed Test"
        # await before continuing
        await asyncio.sleep(0.000001)


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.SelectorEventLoop()
    yield loop
    loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pattern",
    [0, 2, 3, 4],
)
async def test_netgear_async_playback(pattern):
    try:
        # define and launch Client with `receive_mode = True`
        client = NetGear_Async(
            logging=True, pattern=pattern, receive_mode=True, timeout=7.0
        ).launch()
        options_gear = {"THREAD_TIMEOUT": 60}
        server = NetGear_Async(
            source=return_testvideo_path(),
            pattern=pattern,
            logging=True,
            **options_gear
        ).launch()
        # gather and run tasks
        input_coroutines = [server.task, client_iterator(client)]
        res = await asyncio.gather(*input_coroutines, return_exceptions=True)
    except Exception as e:
        if isinstance(e, queue.Empty):
            pytest.fail(str(e))
    finally:
        server.close(skip_loop=True)
        client.close(skip_loop=True)


test_data_class = [
    (None, False),
    (custom_frame_generator(), True),
    ([], False),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("generator, result", test_data_class)
async def test_netgear_async_custom_server_generator(generator, result):
    try:
        server = NetGear_Async(protocol="udp", logging=True)  # invalid protocol
        server.config["generator"] = generator
        server.launch()
        # define and launch Client with `receive_mode = True` and timeout = 5.0
        client = NetGear_Async(logging=True, receive_mode=True, timeout=5.0).launch()
        # gather and run tasks
        input_coroutines = [server.task, client_iterator(client)]
        res = await asyncio.gather(*input_coroutines, return_exceptions=True)
    except Exception as e:
        if result:
            pytest.fail(str(e))
    finally:
        if result:
            server.close(skip_loop=True)
            client.close(skip_loop=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("address, port", [("172.31.11.15.77", "5555"), (None, "5555")])
async def test_netgear_async_addresses(address, port):
    try:
        # define and launch Client with `receive_mode = True`
        client = NetGear_Async(
            address=address, port=port, logging=True, timeout=5.0, receive_mode=True
        ).launch()
        if address is None:
            options_gear = {"THREAD_TIMEOUT": 60}
            server = NetGear_Async(
                source=return_testvideo_path(),
                address=address,
                port=port,
                logging=True,
                **options_gear
            ).launch()
            # gather and run tasks
            input_coroutines = [server.task, client_iterator(client)]
            await asyncio.gather(*input_coroutines, return_exceptions=True)
        else:
            await asyncio.ensure_future(client_iterator(client))
    except Exception as e:
        if address == "172.31.11.15.77" or isinstance(e, queue.Empty):
            logger.exception(str(e))
        else:
            pytest.fail(str(e))
    finally:
        if address is None:
            server.close(skip_loop=True)
        client.close(skip_loop=True)


@pytest.mark.asyncio
@pytest.mark.xfail(raises=ValueError)
async def test_netgear_async_recv_generator():
    # define and launch server
    server = NetGear_Async(source=return_testvideo_path(), logging=True)
    async for frame in server.recv_generator():
        logger.error("Failed")
    server.close(skip_loop=True)
