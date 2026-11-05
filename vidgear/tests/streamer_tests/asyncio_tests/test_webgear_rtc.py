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
import time

import cv2
import numpy as np
import pytest
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from httpx2 import ASGITransport, AsyncClient
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from vidgear.gears import VideoGear
from vidgear.gears.asyncio import WebGear_RTC
from vidgear.gears.helper import Backend, logger_handler
from vidgear.tests.utils.helpers import get_testing_dir, return_testvideo_path

# define test logger
logger = log.getLogger("Test_webgear_rtc")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

pytestmark = pytest.mark.asyncio(loop_scope="module")


class Peer:
    """
    Minimal receive-only WebRTC client (no STUN, loopback host candidates only), which collects received frames.
    """

    def __init__(self):
        self.pc = RTCPeerConnection()
        self.frames = []
        self.ended = asyncio.Event()
        self.id = None

        @self.pc.on("track")
        def on_track(track):
            async def pull():
                try:
                    while True:
                        self.frames.append(await track.recv())
                except MediaStreamError:
                    self.ended.set()

            self.task = asyncio.ensure_future(pull())

    async def offer(self, kind="video"):
        self.pc.addTransceiver(kind, direction="recvonly")
        await self.pc.setLocalDescription(await self.pc.createOffer())
        return {"sdp": self.pc.localDescription.sdp, "type": "offer"}

    async def connect(self, client):
        response = await client.post("/offer", json=await self.offer())
        assert response.status_code == 200, response.text
        answer = response.json()
        self.id = answer["id"]
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )
        return answer

    async def wait_frames(self, count, timeout=30):
        deadline = time.time() + timeout
        while len(self.frames) < count:
            assert time.time() < deadline, "received only {} frames".format(
                len(self.frames)
            )
            await asyncio.sleep(0.1)
        return self.frames


async def wait_for(predicate, timeout=30):
    deadline = time.time() + timeout
    while not predicate():
        assert time.time() < deadline, "timed out"
        await asyncio.sleep(0.1)


def client_for(web):
    return AsyncClient(transport=ASGITransport(app=web()), base_url="http://testserver")


def peers_of(web):
    return web._WebGear_RTC__pcs


def server_of(web):
    return web._WebGear_RTC__default_rtc_server


class Synthetic:
    """
    Custom stream yielding `count` frames of given shape at `framerate` (infinite if `count` is None).
    """

    def __init__(self, shape=(480, 640, 3), count=None, framerate=30):
        self.shape, self.count, self.framerate = shape, count, framerate
        self.read_count = 0

    def read(self):
        if self.count is not None and self.read_count >= self.count:
            return None
        self.read_count += 1
        return np.full(self.shape, (self.read_count * 7) % 255, dtype=np.uint8)

    def stop(self):
        self.count = 0


# create your own custom streaming class
class Custom_Stream_Class:
    """
    Custom Streaming using OpenCV
    """

    def __init__(self, source=0):
        self.stream = cv2.VideoCapture(source)
        self.running = True

    def read(self):
        if self.stream is None:
            return None
        if self.running:
            (grabbed, frame) = self.stream.read()
            if grabbed:
                return frame
            self.running = False
        return None

    def stop(self):
        self.running = False
        if self.stream is not None:
            self.stream.release()


class Invalid_Custom_Stream_Class:
    """
    Custom Invalid WebGear_RTC Server, without `read()`
    """

    def stop(self):
        pass


def hello_webpage(request):
    return PlainTextResponse("Hello, world!")


@pytest.mark.parametrize(
    "stabilize, colorspace, time_delay",
    [(True, None, 0), (False, "COLOR_BGR2HSV", 1)],
)
async def test_webgear_rtc_class(stabilize, colorspace, time_delay):
    """
    Test for various WebGear_RTC API parameters, with real frames received
    """
    web = WebGear_RTC(
        source=return_testvideo_path(),
        stabilize=stabilize,
        colorspace=colorspace,
        time_delay=time_delay,
        logging=True,
    )
    async with client_for(web) as client:
        assert (await client.get("/")).status_code == 200
        assert (await client.get("/test")).status_code == 404
        peer = Peer()
        await peer.connect(client)
        await peer.wait_frames(5)
        await peer.pc.close()
    web.shutdown()


def test_webgear_rtc_no_source():
    with pytest.raises(ValueError):
        WebGear_RTC(source=None, logging=True)


@pytest.mark.parametrize(
    "options",
    [
        {
            "frame_size_reduction": 47,
            "overwrite_default_files": "invalid_value",
            "enable_infinite_frames": "invalid_value",
            "enable_live_broadcast": "invalid_value",
            "custom_data_location": True,
            "codec": "H265",
            "ice_servers": "stun:invalid",
        },
        {
            "frame_size_reduction": "invalid_value",
            "enable_live_broadcast": False,
            "custom_data_location": "im_wrong",
            "ice_servers": [{"bad_key": 1}],
        },
        {
            "custom_data_location": get_testing_dir(),
            "enable_infinite_frames": False,
            "frame_size_reduction": True,
        },
        {
            "overwrite_default_files": True,
            "enable_live_broadcast": True,
            "frame_size_reduction": 99,
            "codec": "vp8",
            "ice_servers": [],
        },
    ],
)
async def test_webgear_rtc_options(options):
    """
    Test for various WebGear_RTC API internal options
    """
    try:
        web = WebGear_RTC(source=return_testvideo_path(), logging=True, **options)
    except AssertionError as e:
        # invalid `custom_data_location` path
        assert options.get("custom_data_location") == "im_wrong", str(e)
        return
    async with client_for(web) as client:
        assert (await client.get("/")).status_code == 200
        peer = Peer()
        await peer.connect(client)
        frames = await peer.wait_frames(3)
        if options.get("frame_size_reduction") == 47:
            # 47% of 1280x720 = 678x381, cropped to even dimensions
            assert (frames[-1].width, frames[-1].height) == (678, 380)
        await peer.pc.close()
    web.shutdown()


@pytest.mark.parametrize("live", [False, True])
async def test_multiple_peers_get_every_frame(live):
    """
    Every peer receives full framerate (no frame splitting between peers)
    """
    stream = Synthetic(framerate=30)
    web = WebGear_RTC(
        custom_stream=stream,
        frame_size_reduction=0,
        enable_live_broadcast=live,
        logging=True,
    )
    async with client_for(web) as client:
        peers = [Peer(), Peer()]
        for peer in peers:
            await peer.connect(client)
        for peer in peers:
            await peer.wait_frames(1)
        counts = [len(p.frames) for p in peers]
        reads = stream.read_count
        await asyncio.sleep(3)
        received = [len(p.frames) - c for p, c in zip(peers, counts, strict=True)]
        read = stream.read_count - reads
        logger.debug("source read {} frames, peers received {}".format(read, received))
        # each peer gets ~every source frame, not half of them
        assert read >= 60
        assert all(r >= 0.75 * read for r in received), (read, received)
        for peer in peers:
            await peer.pc.close()
    web.shutdown()


@pytest.mark.parametrize("codec", ["H264", "VP8"])
async def test_odd_dimensions(codec):
    """
    Odd sized frames are cropped to even dimensions, so H.264 gets frames too
    """
    web = WebGear_RTC(
        custom_stream=Synthetic(shape=(481, 641, 3)),
        frame_size_reduction=0,
        codec=codec,
        logging=True,
    )
    async with client_for(web) as client:
        peer = Peer()
        answer = await peer.connect(client)
        assert "{}/90000".format(codec) in answer["sdp"]
        other = "VP8" if codec == "H264" else "H264"
        assert "{}/90000".format(other) not in answer["sdp"]
        frames = await peer.wait_frames(5)
        assert (frames[-1].width, frames[-1].height) == (640, 480)
        await peer.pc.close()
    web.shutdown()


@pytest.mark.parametrize(
    "shape", [(480, 640, 1), (480, 640), (480, 640, 4), (480, 640, 5)]
)
async def test_frame_channels(shape):
    """
    Grayscale & BGRA frames stream; unsupported channels end stream cleanly
    """
    web = WebGear_RTC(custom_stream=Synthetic(shape=shape), logging=True)
    async with client_for(web) as client:
        peer = Peer()
        await peer.connect(client)
        if shape[-1] == 5:
            server = server_of(web)
            await wait_for(lambda: not server.is_running)
            # server terminated stream, new offers are refused
            assert (await client.post("/offer", json=await Peer().offer())).status_code == 503
        else:
            await peer.wait_frames(3)
        await peer.pc.close()
    web.shutdown()


async def test_invalid_offers():
    """
    `/offer` rejects invalid requests with 4xx, instead of 500s
    """
    web = WebGear_RTC(custom_stream=Synthetic(), logging=True)
    async with client_for(web) as client:
        assert (await client.get("/offer")).status_code == 405
        for body in ["not json", "{}", "[]", '{"sdp": 1, "type": "offer"}']:
            response = await client.post("/offer", content=body)
            assert response.status_code == 400, body
        valid = await Peer().offer()
        for payload in [
            {"sdp": valid["sdp"], "type": "answer"},
            {"sdp": valid["sdp"], "type": "bogus"},
            {"sdp": "v=0\r\nbroken", "type": "offer"},
            await Peer().offer(kind="audio"),
        ]:
            response = await client.post("/offer", json=payload)
            assert response.status_code == 400, payload
            assert "error" in response.json()
        # nothing leaked, video-server not launched
        assert not peers_of(web)
    web.shutdown()


async def test_close_connection_and_reconnect():
    """
    `/close_connection` closes only the peer with given id, and page reload reconnects
    """
    web = WebGear_RTC(custom_stream=Synthetic(), logging=True)
    async with client_for(web) as client:
        first, second = Peer(), Peer()
        await first.connect(client)
        await second.connect(client)
        await first.wait_frames(3)
        # legacy/unknown requests can't drop other peers
        for body in ["1", "0", "{}", '{"id": "unknown"}', '{"id": []}', "bad"]:
            response = await client.post("/close_connection", content=body)
            assert response.text == "DISABLED", body
        assert len(peers_of(web)) == 2
        # close own peer (page reload)
        response = await client.post("/close_connection", json={"id": first.id})
        assert response.text == "OK"
        assert list(peers_of(web)) == [second.id]
        await first.pc.close()
        # reconnect after reload
        reloaded = Peer()
        await reloaded.connect(client)
        await reloaded.wait_frames(3)
        before = len(second.frames)
        await second.wait_frames(before + 3)
        for peer in (second, reloaded):
            await peer.pc.close()
    web.shutdown()


async def test_peers_pruned_and_source_paused():
    """
    Server drops closed peers by itself, and pauses source when no peers are left
    """
    stream = Synthetic()
    web = WebGear_RTC(custom_stream=stream, logging=True)
    async with client_for(web) as client:
        peer = Peer()
        await peer.connect(client)
        await peer.wait_frames(3)
        await peer.pc.close()
        await wait_for(lambda: not peers_of(web))
        await asyncio.sleep(0.5)
        paused_at = stream.read_count
        await asyncio.sleep(1)
        assert stream.read_count <= paused_at + 1
        # resumes for next peer, without bursting frames
        peer = Peer()
        await peer.connect(client)
        await peer.wait_frames(3)
        await peer.pc.close()
    web.shutdown()


@pytest.mark.parametrize("infinite", [False, True])
async def test_stream_end(infinite):
    """
    Ended stream refuses new peers with 503, unless infinite frames are enabled
    """
    web = WebGear_RTC(
        custom_stream=Synthetic(count=10),
        enable_infinite_frames=infinite,
        logging=True,
    )
    async with client_for(web) as client:
        peer = Peer()
        await peer.connect(client)
        if infinite:
            # keeps sending blank "No Input" frames
            await peer.wait_frames(20)
            await Peer().connect(client)
        else:
            server = server_of(web)
            await wait_for(lambda: not server.is_running)
            await peer.wait_frames(1)
            assert len(peer.frames) <= 11  # 10 frames + "The End" frame
            response = await client.post("/offer", json=await Peer().offer())
            assert response.status_code == 503
        await peer.pc.close()
    web.shutdown()
    # offers after shutdown are refused too
    async with client_for(web) as client:
        response = await client.post("/offer", json=await Peer().offer())
        assert response.status_code == 503


async def test_lifespan_closes_peers():
    """
    App lifespan shutdown closes all peers & video-server
    """
    web = WebGear_RTC(custom_stream=Synthetic(), logging=True)
    app = web()
    async with app.router.lifespan_context(app), AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        peer = Peer()
        await peer.connect(client)
        await peer.wait_frames(3)
    assert not peers_of(web)
    await asyncio.wait_for(peer.ended.wait(), 30)
    await peer.pc.close()


async def test_ice_servers_on_page(tmp_path):
    """
    `ice_servers` are passed to homepage template context
    """
    # default theme files, with an index template that prints its context
    WebGear_RTC(custom_stream=Synthetic(), custom_data_location=str(tmp_path)).shutdown()
    index = tmp_path / ".vidgear" / "webgear_rtc" / "templates" / "index.html"
    index.write_text("{{ ice_servers | tojson }}")
    servers = [{"urls": "stun:stun.example.org:3478"}]
    web = WebGear_RTC(
        custom_stream=Synthetic(),
        custom_data_location=str(tmp_path),
        ice_servers=servers,
        logging=True,
    )
    async with client_for(web) as client:
        response = await client.get("/")
        assert response.status_code == 200
        assert response.json() == servers
        assert (await client.get("/static/js/custom.js")).status_code == 200
    web.shutdown()


@pytest.mark.parametrize(
    "stream_class",
    [
        Custom_Stream_Class(source=return_testvideo_path()),
        VideoGear(
            source=return_testvideo_path(), colorspace="COLOR_BGR2BGRA", logging=True
        ),
    ],
)
async def test_webgear_rtc_custom_stream_class(stream_class):
    """
    Test for WebGear_RTC API's custom source
    """
    web = WebGear_RTC(custom_stream=stream_class, frame_size_reduction=45, logging=True)
    async with client_for(web) as client:
        peer = Peer()
        await peer.connect(client)
        await peer.wait_frames(3)
        await peer.pc.close()
    web.shutdown()


@pytest.mark.parametrize("stream_class", [None, Invalid_Custom_Stream_Class()])
def test_webgear_rtc_invalid_custom_stream(stream_class):
    with pytest.raises(ValueError):
        WebGear_RTC(custom_stream=stream_class, logging=True)


@pytest.mark.parametrize(
    "middleware, result",
    [
        (None, False),
        ([Middleware(CORSMiddleware, allow_origins=["*"])], True),
        ([Route("/hello", endpoint=hello_webpage)], False),  # invalid value
    ],
)
async def test_webgear_rtc_custom_middleware(middleware, result):
    """
    Test for WebGear_RTC API's custom middleware
    """
    web = WebGear_RTC(source=return_testvideo_path(), logging=True)
    web.middleware = middleware
    try:
        if not result:
            with pytest.raises((AssertionError, RuntimeError)):
                web()
            return
        async with client_for(web) as client:
            assert (await client.get("/")).status_code == 200
    finally:
        web.shutdown()


async def test_webgear_rtc_routes():
    """
    Test for WebGear_RTC API's custom routes
    """
    web = WebGear_RTC(source=return_testvideo_path(), logging=True)
    web.routes.append(Route("/hello", endpoint=hello_webpage))
    async with client_for(web) as client:
        assert (await client.get("/")).status_code == 200
        assert (await client.get("/hello")).status_code == 200
        peer = Peer()
        await peer.connect(client)
        await peer.wait_frames(3)
        await peer.pc.close()
    web.shutdown()


def test_webgear_rtc_routes_validity():
    """
    Test WebGear_RTC Routes
    """
    web = WebGear_RTC(source=return_testvideo_path(), logging=True)
    web.routes.clear()
    with pytest.raises(RuntimeError):
        web()
    web.shutdown()


@pytest.mark.parametrize("frame_format", ["bgr24", "rgb24", "gray"])
async def test_webgear_rtc_ffgear_backend(frame_format):
    """
    Test WebGear_RTC API with FFGear backend
    """
    web = WebGear_RTC(
        api=Backend.FFGEAR,
        source=return_testvideo_path(),
        frame_format=frame_format,
        frame_size_reduction=40,
        logging=True,
    )
    async with client_for(web) as client:
        peer = Peer()
        await peer.connect(client)
        await peer.wait_frames(3)
        await peer.pc.close()
    web.shutdown()


def test_webgear_rtc_enablePiCamera_deprecated():
    """
    Test that `enablePiCamera` triggers DeprecationWarning in WebGear_RTC
    """
    with pytest.warns(DeprecationWarning, match="enablePiCamera"):
        web = WebGear_RTC(
            enablePiCamera=False,
            source=return_testvideo_path(),
            logging=True,
        )
    web.shutdown()


def test_webgear_rtc_invalid_api():
    with pytest.raises(TypeError):
        WebGear_RTC(api="invalid", source=return_testvideo_path())
