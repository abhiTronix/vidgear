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
import pytest
import asyncio
import logging as log
import requests
import tempfile
from starlette.routing import Route
from starlette.responses import PlainTextResponse
from starlette.testclient import TestClient

from vidgear.gears.asyncio import WebGear
from vidgear.gears.asyncio.helper import logger_handler

# define test logger
logger = log.getLogger("Test_webgear")
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


def hello_webpage(request):
    """
    returns PlainTextResponse callback for hello world webpage
    """
    return PlainTextResponse("Hello, world!")


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


test_data = [
    (return_testvideo_path(), True, None, 0),
    (return_testvideo_path(), False, "COLOR_BGR2HSV", 10),
]


@pytest.mark.parametrize("source, stabilize, colorspace, time_delay", test_data)
def test_webgear_class(source, stabilize, colorspace, time_delay):
    """
    Test for various WebGear API parameters
    """
    try:
        web = WebGear(
            source=source,
            stabilize=stabilize,
            colorspace=colorspace,
            time_delay=time_delay,
            logging=True,
        )
        client = TestClient(web(), raise_server_exceptions=True)
        response = client.get("/")
        assert response.status_code == 200
        response_404 = client.get("/test")
        assert response_404.status_code == 404
        web.shutdown()
    except Exception as e:
        pytest.fail(str(e))


test_data = [
    {
        "frame_size_reduction": 47,
        "frame_jpeg_quality": 88,
        "frame_jpeg_optimize": True,
        "frame_jpeg_progressive": False,
        "overwrite_default_files": "invalid_value",
        "custom_data_location": True,
    },
    {
        "frame_size_reduction": "invalid_value",
        "frame_jpeg_quality": "invalid_value",
        "frame_jpeg_optimize": "invalid_value",
        "frame_jpeg_progressive": "invalid_value",
        "overwrite_default_files": True,
        "custom_data_location": "im_wrong",
    },
    {"custom_data_location": tempfile.gettempdir()},
]


@pytest.mark.parametrize("options", test_data)
def test_webgear_options(options):
    """
    Test for various WebGear API internal options
    """
    try:
        web = WebGear(source=return_testvideo_path(), logging=True, **options)
        client = TestClient(web(), raise_server_exceptions=True)
        response = client.get("/")
        assert response.status_code == 200
        response_video = client.get("/video")
        assert response_video.status_code == 200
        web.shutdown()
    except Exception as e:
        if isinstance(e, AssertionError):
            logger.exception(str(e))
        elif isinstance(e, requests.exceptions.Timeout):
            logger.exceptions(str(e))
        else:
            pytest.fail(str(e))


test_data_class = [
    (None, False),
    (custom_frame_generator(), True),
    ([], False),
]


@pytest.mark.parametrize("generator, result", test_data_class)
async def test_webgear_custom_server_generator(generator, result):
    """
    Test for WebGear API's custom source
    """
    try:
        web = WebGear(logging=True)
        web.config["generator"] = generator
        client = TestClient(web(), raise_server_exceptions=True)
        response_video = client.get("/video")
        assert response_video.status_code == 200
        web.shutdown()
    except Exception as e:
        if result:
            pytest.fail(str(e))


def test_webgear_routes():
    """
    Test for WebGear API's custom routes
    """
    try:
        # add various performance tweaks as usual
        options = {
            "frame_size_reduction": 40,
            "frame_jpeg_quality": 80,
            "frame_jpeg_optimize": True,
            "frame_jpeg_progressive": False,
        }
        # initialize WebGear app
        web = WebGear(source=return_testvideo_path(), logging=True, **options)

        # modify route to point our rendered webpage
        web.routes.append(Route("/hello", endpoint=hello_webpage))

        # test
        client = TestClient(web(), raise_server_exceptions=True)
        response = client.get("/")
        assert response.status_code == 200
        response_hello = client.get("/hello")
        assert response_hello.status_code == 200
        web.shutdown()
    except Exception as e:
        pytest.fail(str(e))


@pytest.mark.xfail(raises=RuntimeError)
def test_webgear_routes_validity():
    # initialize WebGear app
    web = WebGear(source=return_testvideo_path(), logging=True)
    # modify route
    web.routes.clear()
    # test
    client = TestClient(web(), raise_server_exceptions=True)
    # shutdown
    web.shutdown()
