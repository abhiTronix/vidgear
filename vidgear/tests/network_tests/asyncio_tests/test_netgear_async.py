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
import sys
import numpy as np
import pytest
import asyncio
import logging as log
import tempfile

from vidgear.gears import VideoGear
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
    stream = VideoGear(source=return_testvideo_path()).start()
    # loop over stream until its terminated
    while True:
        # read frames
        frame = stream.read()
        # check if frame empty
        if frame is None:
            break
        # yield frame
        yield frame
        # sleep for sometime
        await asyncio.sleep(0.000001)
    # close stream
    stream.stop()


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
            logging=True, pattern=pattern, receive_mode=True
        ).launch()
        server = NetGear_Async(
            source=return_testvideo_path(), pattern=pattern, logging=True
        ).launch()
        # gather and run tasks
        input_coroutines = [server.task, client_iterator(client)]
        res = await asyncio.gather(*input_coroutines, return_exceptions=True)
    except Exception as e:
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
        server = NetGear_Async(
            source=return_testvideo_path(), protocol="udp", logging=True
        )  # invalid protocol
        if generator:
            server.config["generator"] = generator
        else:
            server.config = ["Invalid"]
        server.launch()
        # define and launch Client with `receive_mode = True` and timeout = 12.0
        client = NetGear_Async(logging=True, timeout=12.0, receive_mode=True).launch()
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
@pytest.mark.parametrize("address, port", [("www.idk.com", "5555"), (None, "5555")])
async def test_netgear_async_addresses(address, port):
    try:
        # define and launch Client with `receive_mode = True`
        client = NetGear_Async(
            address=address, port=port, logging=True, receive_mode=True
        ).launch()
        if address is None:
            server = NetGear_Async(
                source=return_testvideo_path(), address=address, port=port, logging=True
            ).launch()
            # gather and run tasks
            input_coroutines = [server.task, client_iterator(client)]
            await asyncio.gather(*input_coroutines, return_exceptions=True)
        else:
            await asyncio.ensure_future(client_iterator(client))
    except Exception as e:
        if address == "www.idk.com":
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
