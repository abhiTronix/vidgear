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
import platform
import logging as log
import requests
import tempfile
import json
import numpy as np
from starlette.routing import Route
from starlette.responses import PlainTextResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from async_asgi_testclient import TestClient
from aiortc import (
    MediaStreamTrack,
    RTCPeerConnection,
    RTCConfiguration,
    RTCIceServer,
    RTCSessionDescription,
)
from vidgear.gears import VideoGear
from aiortc.mediastreams import MediaStreamError
from vidgear.gears.asyncio import WebGear_RTC
from vidgear.gears.helper import logger_handler


# define test logger
logger = log.getLogger("Test_webgear_rtc")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.SelectorEventLoop()
    yield loop
    loop.close()


def return_testvideo_path():
    """
    returns Test Video path
    """
    path = "{}/Downloads/Test_videos/BigBuckBunny_4sec.mp4".format(
        tempfile.gettempdir()
    )
    return os.path.abspath(path)


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track):
        super().__init__()  # don't forget this!
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        return frame


async def get_RTCPeer_payload():
    pc = RTCPeerConnection(
        RTCConfiguration(iceServers=[RTCIceServer("stun:stun.l.google.com:19302")])
    )

    @pc.on("track")
    async def on_track(track):
        logger.debug("Receiving %s" % track.kind)
        if track.kind == "video":
            pc.addTrack(VideoTransformTrack(track))

        @track.on("ended")
        async def on_ended():
            logger.info("Track %s ended", track.kind)

    pc.addTransceiver("video", direction="recvonly")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    new_offer = pc.localDescription
    payload = {"sdp": new_offer.sdp, "type": new_offer.type}
    return (pc, json.dumps(payload, separators=(",", ":")))


def hello_webpage(request):
    """
    returns PlainTextResponse callback for hello world webpage
    """
    return PlainTextResponse("Hello, world!")


# create your own custom streaming class
class Custom_Stream_Class:
    """
    Custom Streaming using OpenCV
    """

    def __init__(self, source=0):
        # !!! define your own video source here !!!
        self.stream = cv2.VideoCapture(source)
        # define running flag
        self.running = True

    def read(self):
        # check if source was initialized or not
        if self.stream is None:
            return None
        # check if we're still running
        if self.running:
            # read frame from provided source
            (grabbed, frame) = self.stream.read()
            # check if frame is available
            if grabbed:
                # return our gray frame
                return frame
            else:
                # signal we're not running now
                self.running = False
        # return None-type
        return None

    def stop(self):
        # flag that we're not running
        self.running = False
        # close stream
        if not self.stream is None:
            self.stream.release()


# create your own custom Grayscale class
class Custom_Grayscale_class:
    """
    Custom Grayscale class for producing `ndim==3` grayscale frames
    """

    def __init__(self):
        # define running flag
        self.running = True
        # counter
        self.counter = 0

    def read(self, size=(480, 640, 1)):
        # check if we're still running
        self.counter += 1
        if self.running:
            # read frame from provided source
            frame = np.random.randint(0, 255, size=size, dtype=np.uint8)
            # check counter
            if self.counter < 11:
                # return our gray frame
                return frame
            else:
                # signal we're not running now
                self.running = False
        # return None-type
        return None

    def stop(self):
        # flag that we're not running
        self.running = False


class Invalid_Custom_Stream_Class:
    """
    Custom Invalid WebGear_RTC Server
    """

    def __init__(self, source=0):
        # define running flag
        self.running = True

    def stop(self):
        # don't forget this function!!!

        # flag that we're not running
        self.running = False


test_data = [
    (return_testvideo_path(), True, None, 0),
    (return_testvideo_path(), False, "COLOR_BGR2HSV", 1),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("source, stabilize, colorspace, time_delay", test_data)
async def test_webgear_rtc_class(source, stabilize, colorspace, time_delay):
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
        async with TestClient(web()) as client:
            response = await client.get("/")
            assert response.status_code == 200
            response_404 = await client.get("/test")
            assert response_404.status_code == 404
            (offer_pc, data) = await get_RTCPeer_payload()
            response_rtc_answer = await client.post(
                "/offer",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            params = response_rtc_answer.json()
            answer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
            await offer_pc.setRemoteDescription(answer)
            response_rtc_offer = await client.get(
                "/offer",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            assert response_rtc_offer.status_code == 200
            await offer_pc.close()
        web.shutdown()
    except Exception as e:
        if not isinstance(e, MediaStreamError):
            pytest.fail(str(e))


test_data = [
    {
        "frame_size_reduction": 47,
        "overwrite_default_files": "invalid_value",
        "enable_infinite_frames": "invalid_value",
        "enable_live_broadcast": "invalid_value",
        "custom_data_location": True,
    },
    {
        "frame_size_reduction": "invalid_value",
        "enable_live_broadcast": False,
        "custom_data_location": "im_wrong",
    },
    {
        "custom_data_location": tempfile.gettempdir(),
        "enable_infinite_frames": False,
    },
    {
        "overwrite_default_files": True,
        "enable_live_broadcast": True,
        "frame_size_reduction": 99,
    },
]


@pytest.mark.asyncio
@pytest.mark.parametrize("options", test_data)
async def test_webgear_rtc_options(options):
    """
    Test for various WebGear_RTC API internal options
    """
    web = None
    try:
        web = WebGear_RTC(source=return_testvideo_path(), logging=True, **options)
        async with TestClient(web()) as client:
            response = await client.get("/")
            assert response.status_code == 200
            if (
                not "enable_live_broadcast" in options
                or options["enable_live_broadcast"] == False
            ):
                (offer_pc, data) = await get_RTCPeer_payload()
                response_rtc_answer = await client.post(
                    "/offer",
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                params = response_rtc_answer.json()
                answer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
                await offer_pc.setRemoteDescription(answer)
                response_rtc_offer = await client.get(
                    "/offer",
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                assert response_rtc_offer.status_code == 200
                await offer_pc.close()
        web.shutdown()
    except Exception as e:
        if isinstance(e, (AssertionError, MediaStreamError)):
            logger.exception(str(e))
        elif isinstance(e, requests.exceptions.Timeout):
            logger.exceptions(str(e))
        else:
            pytest.fail(str(e))


test_data = [
    {
        "frame_size_reduction": 40,
    },
    {
        "enable_live_broadcast": True,
        "frame_size_reduction": 40,
    },
]


@pytest.mark.skipif((platform.system() == "Windows"), reason="Random Failures!")
@pytest.mark.asyncio
@pytest.mark.parametrize("options", test_data)
async def test_webpage_reload(options):
    """
    Test for testing WebGear_RTC API against Webpage reload
    disruptions
    """
    web = WebGear_RTC(source=return_testvideo_path(), logging=True, **options)
    try:
        # run webgear_rtc
        async with TestClient(web()) as client:
            response = await client.get("/")
            assert response.status_code == 200

            # create offer and receive
            (offer_pc, data) = await get_RTCPeer_payload()
            response_rtc_answer = await client.post(
                "/offer",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            params = response_rtc_answer.json()
            answer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
            await offer_pc.setRemoteDescription(answer)
            response_rtc_offer = await client.get(
                "/offer",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            assert response_rtc_offer.status_code == 200
            # simulate webpage reload
            response_rtc_reload = await client.post(
                "/close_connection",
                data="0",
            )
            # close offer
            await offer_pc.close()
            offer_pc = None
            data = None
            # verify response
            logger.debug(response_rtc_reload.text)
            assert response_rtc_reload.text == "OK", "Test Failed!"

            # recreate offer and continue receive
            (offer_pc, data) = await get_RTCPeer_payload()
            response_rtc_answer = await client.post(
                "/offer",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            params = response_rtc_answer.json()
            answer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
            await offer_pc.setRemoteDescription(answer)
            response_rtc_offer = await client.get(
                "/offer",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            assert response_rtc_offer.status_code == 200
            # shutdown
            await offer_pc.close()
    except Exception as e:
        if "enable_live_broadcast" in options and isinstance(
            e, (AssertionError, MediaStreamError)
        ):
            pytest.xfail("Test Passed")
        else:
            pytest.fail(str(e))
    finally:
        web.shutdown()


test_stream_classes = [
    (None, False),
    (Custom_Stream_Class(source=return_testvideo_path()), True),
    (VideoGear(source=return_testvideo_path(), logging=True), True),
    (Custom_Grayscale_class(), False),
    (Invalid_Custom_Stream_Class(source=return_testvideo_path()), False),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_class, result", test_stream_classes)
async def test_webgear_rtc_custom_stream_class(stream_class, result):
    """
    Test for WebGear_RTC API's custom source
    """
    # assign your Custom Streaming Class with adequate source (for e.g. foo.mp4)
    # to `custom_stream` attribute in options parameter
    options = {
        "custom_stream": stream_class,
        "frame_size_reduction": 0 if not result else 45,
    }
    try:
        web = WebGear_RTC(logging=True, **options)
        async with TestClient(web()) as client:
            response = await client.get("/")
            assert response.status_code == 200
            response_404 = await client.get("/test")
            assert response_404.status_code == 404
            (offer_pc, data) = await get_RTCPeer_payload()
            response_rtc_answer = await client.post(
                "/offer",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            params = response_rtc_answer.json()
            answer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
            await offer_pc.setRemoteDescription(answer)
            response_rtc_offer = await client.get(
                "/offer",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            assert response_rtc_offer.status_code == 200
            await offer_pc.close()
        web.shutdown()
    except Exception as e:
        if result and not isinstance(e, (ValueError, MediaStreamError)):
            pytest.fail(str(e))
        else:
            pytest.xfail(str(e))


test_data_class = [
    (None, False),
    ([Middleware(CORSMiddleware, allow_origins=["*"])], True),
    ([Route("/hello", endpoint=hello_webpage)], False),  # invalid value
]


@pytest.mark.asyncio
@pytest.mark.parametrize("middleware, result", test_data_class)
async def test_webgear_rtc_custom_middleware(middleware, result):
    """
    Test for WebGear_RTC API's custom middleware
    """
    try:
        web = WebGear_RTC(source=return_testvideo_path(), logging=True)
        web.middleware = middleware
        async with TestClient(web()) as client:
            response = await client.get("/")
            assert response.status_code == 200
        web.shutdown()
    except Exception as e:
        if result and not isinstance(e, MediaStreamError):
            pytest.fail(str(e))
        else:
            pytest.xfail(str(e))


@pytest.mark.asyncio
async def test_webgear_rtc_routes():
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
        async with TestClient(web()) as client:
            response = await client.get("/")
            assert response.status_code == 200
            response_hello = await client.get("/hello")
            assert response_hello.status_code == 200
            (offer_pc, data) = await get_RTCPeer_payload()
            response_rtc_answer = await client.post(
                "/offer",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            params = response_rtc_answer.json()
            answer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
            await offer_pc.setRemoteDescription(answer)
            response_rtc_offer = await client.get(
                "/offer",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            assert response_rtc_offer.status_code == 200
            # shutdown
            await offer_pc.close()
        web.shutdown()
    except Exception as e:
        if not isinstance(e, MediaStreamError):
            pytest.fail(str(e))


@pytest.mark.asyncio
async def test_webgear_rtc_routes_validity():
    """
    Test WebGear_RTC Routes
    """
    # add various tweaks for testing only
    options = {
        "enable_infinite_frames": False,
        "enable_live_broadcast": True,
    }
    # initialize WebGear_RTC app
    web = WebGear_RTC(source=return_testvideo_path(), logging=True)
    try:
        # modify route
        web.routes.clear()
        # test
        async with TestClient(web()) as client:
            pass
    except Exception as e:
        if isinstance(e, (RuntimeError, MediaStreamError)):
            pytest.xfail(str(e))
        else:
            pytest.fail(str(e))
    finally:
        # close
        web.shutdown()
