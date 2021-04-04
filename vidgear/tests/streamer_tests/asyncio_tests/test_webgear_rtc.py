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
import cv2
import pytest
import asyncio
import logging as log
import requests
import tempfile
from starlette.routing import Route
from starlette.responses import PlainTextResponse
from starlette.testclient import TestClient

from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

from vidgear.gears.asyncio import WebGear_RTC
from vidgear.gears.asyncio.helper import logger_handler

# define test logger
logger = log.getLogger("Test_webgear_rtc")
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


class Custom_RTCServer(VideoStreamTrack):
    """
    Custom WebGear_RTC Server and inherit-class to aiortc's VideoStreamTrack API.
    """

    def __init__(self, source=None):

        # don't forget this line!
        super().__init__()

        # initialize global params
        self.stream = cv2.VideoCapture(source)

    async def recv(self):
        """
        A coroutine function that yields `av.frame.Frame`.
        """
        # get next timestamp
        pts, time_base = await self.next_timestamp()

        # read video frame
        (grabbed, frame) = self.stream.read()

        # if NoneType
        if not grabbed:
            return None

        # contruct `av.frame.Frame` from `numpy.nd.array`
        av_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        av_frame.pts = pts
        av_frame.time_base = time_base

        # return `av.frame.Frame`
        return av_frame

    def terminate(self):
        """
        Gracefully terminates VideoGear stream
        """
        # terminate
        if not (self.stream is None):
            self.stream.release()
            self.stream = None


class Invalid_Custom_RTCServer_1(VideoStreamTrack):
    """
    Custom Invalid WebGear_RTC Server
    """

    def __init__(self, source=None):

        # don't forget this line!
        super().__init__()

        # initialize global params
        self.stream = cv2.VideoCapture(source)
        self.stream.release()

    async def recv(self):
        """
        A coroutine function that yields `av.frame.Frame`.
        """
        # get next timestamp
        pts, time_base = await self.next_timestamp()

        # read video frame
        (grabbed, frame) = self.stream.read()

        # if NoneType
        if not grabbed:
            return None

        # contruct `av.frame.Frame` from `numpy.nd.array`
        av_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        av_frame.pts = pts
        av_frame.time_base = time_base

        # return `av.frame.Frame`
        return av_frame


class Invalid_Custom_RTCServer_2:
    """
    Custom Invalid WebGear_RTC Server
    """

    def __init__(self, source=None):

        # don't forget this line!
        super().__init__()


test_data = [
    (return_testvideo_path(), True, None, 0),
    (return_testvideo_path(), False, "COLOR_BGR2HSV", 3),
]


@pytest.mark.parametrize("source, stabilize, colorspace, time_delay", test_data)
def test_webgear_rtc_class(source, stabilize, colorspace, time_delay):
    """
    Test for various WebGear_RTC API parameters
    """
    try:
        web = WebGear_RTC(
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
        "overwrite_default_files": "invalid_value",
        "enable_infinite_frames": "invalid_value",
        "custom_data_location": True,
    },
    {
        "frame_size_reduction": "invalid_value",
        "overwrite_default_files": True,
        "enable_infinite_frames": False,
        "custom_data_location": "im_wrong",
    },
    {"custom_data_location": tempfile.gettempdir()},
]


@pytest.mark.parametrize("options", test_data)
def test_webgear_rtc_options(options):
    """
    Test for various WebGear_RTC API internal options
    """
    try:
        web = WebGear_RTC(source=return_testvideo_path(), logging=True, **options)
        client = TestClient(web(), raise_server_exceptions=True)
        response = client.get("/")
        assert response.status_code == 200
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
    (Custom_RTCServer(source=return_testvideo_path()), True),
    ([], False),
    (Invalid_Custom_RTCServer_1(source=return_testvideo_path()), False),
    (Invalid_Custom_RTCServer_2(source=return_testvideo_path()), False),
]


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize("server, result", test_data_class)
def test_webgear_rtc_custom_server_generator(server, result):
    """
    Test for WebGear_RTC API's custom source
    """
    web = WebGear_RTC(logging=True)
    web.config["server"] = server
    client = TestClient(web(), raise_server_exceptions=True)
    web.shutdown()


def test_webgear_rtc_routes():
    """
    Test for WebGear_RTC API's custom routes
    """
    try:
        # add various performance tweaks as usual
        options = {
            "frame_size_reduction": 40,
        }
        # initialize WebGear_RTC app
        web = WebGear_RTC(source=return_testvideo_path(), logging=True, **options)

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
def test_webgear_rtc_routes_validity():
    # initialize WebGear_RTC app
    web = WebGear_RTC(source=return_testvideo_path(), logging=True)
    # modify route
    web.routes.clear()
    # test
    client = TestClient(web(), raise_server_exceptions=True)
    web.shutdown()
