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
import pytest
import asyncio
import logging as log
import requests
import tempfile
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
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
    stream = cv2.VideoCapture(return_testvideo_path())
    # loop over stream until its terminated
    while True:
        # read frames
        (grabbed, frame) = stream.read()
        # check if frame empty
        if not grabbed:
            break
        # handle JPEG encoding
        encodedImage = cv2.imencode(".jpg", frame)[1].tobytes()
        # yield frame in byte format
        yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" + encodedImage + b"\r\n")
        await asyncio.sleep(0)
    # close stream
    stream.release()


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


@pytest.mark.parametrize(
    "options",
    [
        {
            "jpeg_compression_colorspace": "invalid",
            "jpeg_compression_quality": 5,
            "custom_data_location": True,
            "jpeg_compression_fastdct": "invalid",
            "jpeg_compression_fastupsample": "invalid",
            "frame_size_reduction": "invalid",
            "overwrite_default_files": "invalid",
            "enable_infinite_frames": "invalid",
        },
        {
            "jpeg_compression_colorspace": " gray  ",
            "jpeg_compression_quality": 50,
            "jpeg_compression_fastdct": True,
            "jpeg_compression_fastupsample": True,
            "overwrite_default_files": True,
            "enable_infinite_frames": False,
            "custom_data_location": tempfile.gettempdir(),
        },
        {
            "jpeg_compression_quality": 55.55,
            "jpeg_compression_fastdct": True,
            "jpeg_compression_fastupsample": True,
            "custom_data_location": "im_wrong",
        },
        {
            "enable_infinite_frames": True,
            "custom_data_location": return_testvideo_path(),
        },
    ],
)
def test_webgear_options(options):
    """
    Test for various WebGear API internal options
    """
    try:
        colorspace = (
            "COLOR_BGR2GRAY"
            if "jpeg_compression_colorspace" in options
            and isinstance(options["jpeg_compression_colorspace"], str)
            and options["jpeg_compression_colorspace"].strip().upper() == "GRAY"
            else None
        )
        web = WebGear(
            source=return_testvideo_path(),
            colorspace=colorspace,
            logging=True,
            **options
        )
        client = TestClient(web(), raise_server_exceptions=True)
        response = client.get("/")
        assert response.status_code == 200
        response_video = client.get("/video")
        assert response_video.status_code == 200
        web.shutdown()
    except Exception as e:
        if isinstance(e, AssertionError) or isinstance(e, os.access):
            pytest.xfail(str(e))
        elif isinstance(e, requests.exceptions.Timeout):
            logger.exceptions(str(e))
        else:
            pytest.fail(str(e))


test_data_class = [
    (None, False),
    (custom_frame_generator, True),
    ([], False),
]


@pytest.mark.parametrize("generator, result", test_data_class)
def test_webgear_custom_server_generator(generator, result):
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


test_data_class = [
    (None, False),
    ([Middleware(CORSMiddleware, allow_origins=["*"])], True),
    ([Route("/hello", endpoint=hello_webpage)], False),  # invalid value
]


@pytest.mark.parametrize("middleware, result", test_data_class)
def test_webgear_custom_middleware(middleware, result):
    """
    Test for WebGear API's custom middleware
    """
    try:
        web = WebGear(source=return_testvideo_path(), logging=True)
        web.middleware = middleware
        client = TestClient(web(), raise_server_exceptions=True)
        response = client.get("/")
        assert response.status_code == 200
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
            "jpeg_compression_quality": 80,
            "jpeg_compression_fastdct": True,
            "jpeg_compression_fastupsample": False,
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
