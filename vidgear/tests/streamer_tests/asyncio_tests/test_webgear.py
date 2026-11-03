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
import os
import tempfile
import warnings

import cv2
import pytest
import requests
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route, WebSocketRoute
from starlette.testclient import TestClient

from vidgear.gears.asyncio import WebGear
from vidgear.gears.helper import Backend, logger_handler
from vidgear.tests.utils.helpers import get_testing_dir, return_testvideo_path

# define test logger
logger = log.getLogger("Test_webgear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)





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
        {"frame_size_reduction": 90},  # out of range (reducer needs < 90): reverts to default
        {
            "jpeg_compression_colorspace": "invalid",
            "jpeg_compression_quality": 5,
            "custom_data_location": True,
            "skip_generate_webdata": "invalid",
            "jpeg_compression_fastdct": "invalid",
            "jpeg_compression_fastupsample": "invalid",
            "frame_size_reduction": "invalid",
            "overwrite_default_files": "invalid",
            "enable_infinite_frames": "invalid",
            "custom_video_endpoint": "x#x",
        },
        {
            "jpeg_compression_colorspace": " gray  ",
            "jpeg_compression_quality": 50,
            "skip_generate_webdata": "invalid",
            "jpeg_compression_fastdct": True,
            "jpeg_compression_fastupsample": True,
            "overwrite_default_files": True,
            "enable_infinite_frames": False,
            "custom_data_location": get_testing_dir(),
            "custom_video_endpoint": "x x",
        },
        {
            "jpeg_compression_quality": 55.55,
            "jpeg_compression_fastdct": True,
            "skip_generate_webdata": "invalid",
            "jpeg_compression_fastupsample": True,
            "custom_data_location": "im_wrong",
        },
        {
            "enable_infinite_frames": True,
            "custom_data_location": return_testvideo_path(),
            "custom_video_endpoint": " xyz ",
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
        response_video = client.get(
            "/xyz"
            if "custom_video_endpoint" in options
            and options["custom_video_endpoint"] == " xyz "
            else "/video"
        )
        assert response_video.status_code == 200
        web.shutdown()
    except Exception as e:
        if isinstance(e, (AssertionError, PermissionError)):
            pytest.xfail(str(e))
        elif isinstance(e, requests.exceptions.Timeout):
            logger.error(str(e))
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
            "custom_video_endpoint": "x1y1z",
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
        # test new default endpoint
        response_xyz = client.get("/x1y1z")
        assert response_xyz.status_code == 200
        # test old default endpoint
        response_video = client.get("/video")
        assert response_video.status_code == 404
        web.shutdown()
    except Exception as e:
        pytest.fail(str(e))


def test_skip_generate_webdata():
    """
    Test for `skip_generate_webdata` optional attribute
    """
    try:
        # add various performance tweaks as usual
        options = {
            "frame_size_reduction": 40,
            "jpeg_compression_quality": 80,
            "jpeg_compression_fastdct": True,
            "jpeg_compression_fastupsample": False,
            "skip_generate_webdata": True,
        }
        web = WebGear(source=return_testvideo_path(), logging=True, **options)
        client = TestClient(web(), raise_server_exceptions=True)
        response = client.get("/")
        assert "detail" in response.json()
        response_404 = client.get("/test")
        assert "detail" in response_404.json()
        assert response_404.status_code == 404
        response_video = client.get("/video")
        assert response_video.status_code == 200
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
    TestClient(web(), raise_server_exceptions=True)
    # shutdown
    web.shutdown()


def test_webgear_ffgear_backend():
    """
    Test WebGear API with FFGear backend
    """
    try:
        options = {
            "frame_size_reduction": 40,
            "jpeg_compression_quality": 80,
        }
        web = WebGear(
            api=Backend.FFGEAR,
            source=return_testvideo_path(),
            logging=True,
            **options,
        )
        client = TestClient(web(), raise_server_exceptions=True)
        response = client.get("/")
        assert response.status_code == 200
        response_video = client.get("/video")
        assert response_video.status_code == 200
        web.shutdown()
    except Exception as e:
        pytest.fail(str(e))


def nal_types(unit):
    """
    returns H.264 NAL unit types inside Annex-B access unit
    """
    return [
        unit[i + 3] & 0x1F
        for i in range(len(unit) - 3)
        if unit[i : i + 3] == b"\x00\x00\x01"
    ]


@pytest.mark.parametrize(
    "options",
    [
        {},  # H.264 by default
        {
            "streaming_mode": " H264 ",
            "h264_options": {"-crf": 30, "-g": 10},
            "frame_size_reduction": 35,  # odd frame dimensions get cropped
            "custom_video_endpoint": "xyz",
        },
    ],
)
def test_webgear_h264_streaming(options):
    """
    Test WebGear API's low-latency H.264 WebSocket streaming
    """
    try:
        web = WebGear(source=return_testvideo_path(), logging=True, **options)
        endpoint = "/xyz" if "custom_video_endpoint" in options else "/video"
        client = TestClient(web(), raise_server_exceptions=True)
        response = client.get("/")
        assert response.status_code == 200
        assert 'data-mode="h264"' in response.text
        with client.websocket_connect("/ws{}".format(endpoint)) as ws:
            keyframe = ws.receive_bytes()
            delta = ws.receive_bytes()
        # MJPEG route stays available as fallback (drains source, so tested last)
        assert client.get(endpoint).status_code == 200
        assert keyframe.startswith(b"\x00\x00\x00\x01")
        assert {7, 8, 5} <= set(nal_types(keyframe))
        assert set(nal_types(delta)) == {1}
        web.shutdown()
    except Exception as e:
        pytest.fail(str(e))


@pytest.mark.parametrize(
    "source, options",
    [
        (return_testvideo_path(), {"streaming_mode": "mjpeg", "h264_options": "invalid"}),
        (None, {}),  # custom generators stay MJPEG
        (None, {"streaming_mode": "h264"}),
        (return_testvideo_path(), {"streaming_mode": "invalid", "skip_generate_webdata": True}),
    ],
)
def test_webgear_h264_fallback(source, options):
    """
    Test WebGear API's H.264 streaming fallback & routes
    """
    web = WebGear(source=source, logging=True, **options)
    if source is None:
        web.config["generator"] = custom_frame_generator
    ws_routes = [r.path for r in web.routes if isinstance(r, WebSocketRoute)]
    if options.get("skip_generate_webdata"):
        assert ws_routes == ["/ws/video"]
    else:
        assert not ws_routes
        client = TestClient(web(), raise_server_exceptions=True)
        assert 'data-mode="mjpeg"' in client.get("/").text
    web.shutdown()


def test_webgear_enablePiCamera_deprecated():
    """
    Test that `enablePiCamera` triggers DeprecationWarning in WebGear
    """
    try:
        with pytest.warns(DeprecationWarning, match="enablePiCamera"):
            web = WebGear(
                enablePiCamera=False,
                source=return_testvideo_path(),
                logging=True,
            )
        web.shutdown()
    except Exception as e:
        pytest.fail(str(e))
