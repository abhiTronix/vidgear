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
import pickle
import time
import warnings

import cv2
import msgspec
import numpy as np
import pytest
import zmq
import zmq.asyncio

import vidgear.gears.asyncio.netgear_async as netgear_async_module
from vidgear.gears.asyncio import NetGear_Async
from vidgear.gears.helper import Backend, logger_handler
from vidgear.tests.utils.helpers import get_free_port, return_testvideo_path

# define test logger
logger = log.getLogger("Test_NetGear_Async")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

# every test must finish, and the 3.16 asyncio removals must not creep back
pytestmark = [
    pytest.mark.timeout(120),
    pytest.mark.filterwarnings("error::DeprecationWarning:vidgear"),
]

_windows = os.name == "nt"


def video_frames(limit=None):
    """
    Frames of the test video, read directly with OpenCV.
    """
    stream = cv2.VideoCapture(return_testvideo_path())
    frames = []
    while limit is None or len(frames) < limit:
        grabbed, frame = stream.read()
        if not grabbed:
            break
        frames.append(frame)
    stream.release()
    assert frames, "Test video is missing!"
    return frames


def small_frame(seed=0):
    return np.random.default_rng(seed).integers(0, 255, (48, 64, 3), dtype=np.uint8)


async def frame_generator(frames):
    for frame in frames:
        yield frame
        await asyncio.sleep(0)


async def collect(client, limit=None):
    """
    Receives frames (or data-frame tuples) until the server terminates.
    """
    received = []
    async for item in client.recv_generator():
        received.append(item)
        if limit and len(received) == limit:
            break
    return received


async def run(server, client, limit=None):
    """
    Runs server and client to completion, then closes both.
    """
    try:
        client_task = asyncio.ensure_future(collect(client, limit))
        await asyncio.sleep(0)
        await asyncio.wait_for(server.task, 60)
        await server.aclose()
        return await asyncio.wait_for(client_task, 30)
    finally:
        await server.aclose()
        await client.aclose()


@pytest.mark.parametrize("pattern", [0, 1, 3])
@pytest.mark.asyncio
async def test_netgear_async_playback(pattern):
    """
    Frames arrive exactly, in order, and writeable
    """
    port = get_free_port()
    client = NetGear_Async(port=port, pattern=pattern, receive_mode=True, timeout=7.0)
    server = NetGear_Async(
        port=port,
        source=return_testvideo_path(),
        pattern=pattern,
        timeout=7.0,
        logging=True,
        THREAD_TIMEOUT=60,
    ).launch()
    received = await run(server, client)
    expected = video_frames()
    assert len(received) == len(expected)
    for got, sent in zip(received, expected, strict=True):
        assert got.flags.writeable
        assert np.array_equal(got, sent)


@pytest.mark.asyncio
async def test_netgear_async_pubsub():
    """
    PUB/SUB delivers frames; termination is best-effort (HWM 1), so the client may time out instead
    """
    port = get_free_port()
    frames = [small_frame(i) for i in range(5)]

    async def slow_frames():
        await asyncio.sleep(0.5)  # let the subscriber join
        for frame in frames:
            yield frame
            await asyncio.sleep(0.01)

    client = NetGear_Async(port=port, pattern=2, receive_mode=True, timeout=3.0)
    server = NetGear_Async(port=port, pattern=2, timeout=3.0)
    server.config["generator"] = slow_frames()
    server.launch()
    received = []

    async def subscriber():
        try:
            async for frame in client.recv_generator():
                received.append(frame)
        except TimeoutError:
            pass

    try:
        sub = asyncio.ensure_future(subscriber())
        await asyncio.wait_for(server.task, 30)
        await server.aclose()
        await asyncio.wait_for(sub, 10)
    finally:
        await server.aclose()
        await client.aclose()
    assert received
    assert all(any(np.array_equal(r, f) for f in frames) for r in received)


@pytest.mark.asyncio
async def test_netgear_async_invalid_pattern_protocol():
    """
    Invalid pattern/protocol values fall back to PAIR/tcp
    """
    port = get_free_port()
    frames = [small_frame(i) for i in range(3)]
    client = NetGear_Async(port=port, pattern=4, protocol="udp", receive_mode=True)
    server = NetGear_Async(port=port, pattern="x", protocol="udp")
    server.config["generator"] = frame_generator(frames)
    server.launch()
    received = await run(server, client)
    assert len(received) == 3


@pytest.mark.parametrize("generator", [None, [], "invalid"])
@pytest.mark.asyncio
async def test_netgear_async_custom_server_generator_invalid(generator):
    """
    Only async generators are accepted
    """
    server = NetGear_Async(port=get_free_port(), timeout=5.0)
    server.config["generator"] = generator
    server.launch()
    try:
        with pytest.raises(ValueError):
            await server.task
    finally:
        await server.aclose()


@pytest.mark.asyncio
async def test_netgear_async_custom_server_generator_frame_types():
    """
    Non-ndarray frames are rejected
    """
    async def bad():
        yield "not a frame"

    server = NetGear_Async(port=get_free_port(), timeout=5.0)
    server.config["generator"] = bad()
    server.launch()
    try:
        with pytest.raises(ValueError):
            await server.task
    finally:
        await server.aclose()


_BI_VALUES = [
    "Hi",
    0,
    False,
    [],
    "",
    {1: "apple"},
    [1, "string", ["list"]],
    np.int64(7),
    None,
]
_BI_EXPECTED = ["Hi", 0, False, [], "", {1: "apple"}, [1, "string", ["list"]], 7, None]


@pytest.mark.parametrize("pattern", [0, 1])
@pytest.mark.asyncio
async def test_netgear_async_bidirectional_mode(pattern):
    """
    Bi-mode data round-trips both ways, falsy values and types preserved
    """
    port = get_free_port()
    frame = small_frame()
    server = NetGear_Async(port=port, pattern=pattern, timeout=5.0, bidirectional_mode=True)
    client = NetGear_Async(
        port=port, pattern=pattern, receive_mode=True, timeout=5.0, bidirectional_mode=True
    )
    from_client = []

    async def server_gen():
        for value in [*_BI_VALUES, "last"]:
            yield (value, frame)
            from_client.append(await server.transceive_data())

    server.config["generator"] = server_gen()
    server.launch()
    from_server = []

    async def client_loop():
        async for data, recvd in client.recv_generator():
            assert np.array_equal(recvd, frame)
            from_server.append(data)
            i = len(from_server) - 1
            reply = _BI_VALUES[i] if i < len(_BI_VALUES) else None
            await client.transceive_data(data=reply)

    try:
        task = asyncio.ensure_future(client_loop())
        await asyncio.wait_for(server.task, 30)
        await server.aclose()
        await asyncio.wait_for(task, 10)
    finally:
        await server.aclose()
        await client.aclose()
    assert from_server == [*_BI_EXPECTED, "last"]
    # replies go out with the next frame, so the first one is None
    assert from_client == [None, *_BI_EXPECTED]


@pytest.mark.asyncio
async def test_netgear_async_bidirectional_mode_ndarray():
    """
    Bi-mode frames returned by the client
    """
    port = get_free_port()
    frame = small_frame()
    target = small_frame(1)
    server = NetGear_Async(port=port, timeout=5.0, bidirectional_mode=True)
    client = NetGear_Async(port=port, receive_mode=True, timeout=5.0, bidirectional_mode=True)
    from_client = []

    async def server_gen():
        for _ in range(3):
            yield (None, frame)
            from_client.append(await server.transceive_data())

    server.config["generator"] = server_gen()
    server.launch()

    async def client_loop():
        async for _data, _frame in client.recv_generator():
            await client.transceive_data(data=target)

    try:
        task = asyncio.ensure_future(client_loop())
        await asyncio.wait_for(server.task, 30)
        await server.aclose()
        await asyncio.wait_for(task, 10)
    finally:
        await server.aclose()
        await client.aclose()
    assert from_client[0] is None
    assert all(np.array_equal(r, target) for r in from_client[1:])


@pytest.mark.asyncio
async def test_netgear_async_bidirectional_mismatch():
    """
    Bi-mode enabled on one end only raises at the client
    """
    port = get_free_port()
    server = NetGear_Async(port=port, timeout=3.0, bidirectional_mode=True)
    client = NetGear_Async(port=port, receive_mode=True, timeout=3.0)

    async def server_gen():
        while True:
            yield ("data", small_frame())

    server.config["generator"] = server_gen()
    server.launch()
    try:
        with pytest.raises(RuntimeError):
            await collect(client)
    finally:
        await server.aclose()
        await client.aclose()


async def _send_raw(port, parts):
    """
    Sends a raw multipart message to a NetGear_Async client and returns what its generator raises.
    """
    client = NetGear_Async(port=port, receive_mode=True, timeout=3.0)
    ctx = zmq.asyncio.Context()
    peer = ctx.socket(zmq.PAIR)
    peer.connect("tcp://localhost:{}".format(port))
    try:
        gen = client.recv_generator()
        pending = asyncio.ensure_future(gen.__anext__())
        await peer.send_multipart(parts)
        with pytest.raises(ValueError) as exc:
            await pending
        return str(exc.value)
    finally:
        peer.close(linger=0)
        ctx.term()
        await client.aclose()


@pytest.mark.asyncio
async def test_netgear_async_rejects_pickle_payload():
    """
    msgpack_numpy's pickle payload must not execute, and is rejected
    """
    executed = []

    class Exploit:
        def __reduce__(self):
            return (executed.append, ("pwned",))

    # legacy msgpack_numpy encoding of an object array
    evil = msgspec.msgpack.encode(
        {"nd": True, "kind": b"O", "type": "|O", "shape": [1], "data": pickle.dumps(Exploit())}
    )
    header = msgspec.msgpack.encode({"terminate": False, "bi_mode": False, "data": ""})
    message = await _send_raw(get_free_port(), [header, evil])
    assert "same vidgear version" in message
    assert not executed


@pytest.mark.parametrize(
    "header, payload, error",
    [
        ({"v": 2, "dtype": "|O", "shape": [1]}, b"12345678", "object/void dtype"),
        ({"v": 2, "dtype": "|V8", "shape": [1]}, b"12345678", "object/void dtype"),
        ({"v": 2, "dtype": "<u1", "shape": [480, 640, 3]}, b"short", "cannot reshape"),
        ({"v": 2, "dtype": "bogus", "shape": [1]}, b"1", "Invalid dtype"),
        ({"v": 2, "dtype": 5, "shape": [1]}, b"1", "Expected `str | null`"),
        ({"v": 2}, None, "without a frame"),
        ({"v": 1, "dtype": "<u1", "shape": [1]}, b"1", "wire version"),
    ],
)
@pytest.mark.asyncio
async def test_netgear_async_malformed_messages(header, payload, error):
    """
    Hostile or malformed messages raise a clean ValueError
    """
    parts = [msgspec.msgpack.encode(header)] + ([payload] if payload else [])
    assert error in await _send_raw(get_free_port(), parts)


@pytest.mark.parametrize("receive_mode", [False, True])
@pytest.mark.asyncio
async def test_netgear_async_close_before_launch(receive_mode):
    """
    Closing a never-launched gear works, and releases its context
    """
    gear = NetGear_Async(port=get_free_port(), receive_mode=receive_mode)
    await gear.aclose()
    assert gear._NetGear_Async__msg_context.closed


@pytest.mark.asyncio
async def test_netgear_async_server_close_without_client():
    """
    Server close is bounded by `timeout` when no client is around
    """
    server = NetGear_Async(port=get_free_port(), timeout=2.0)
    server.config["generator"] = frame_generator([small_frame()])
    server.launch()
    await asyncio.sleep(0.2)
    start = time.monotonic()
    await server.aclose()
    assert time.monotonic() - start < 6
    assert server.task.done()


@pytest.mark.asyncio
async def test_netgear_async_context_manager():
    """
    `async with` closes the gear
    """
    async with NetGear_Async(port=get_free_port(), receive_mode=True) as client:
        pass
    assert client._NetGear_Async__msg_context.closed


def test_netgear_async_asyncio_run():
    """
    Works inside `asyncio.run()` with `await aclose()`, without deprecated asyncio APIs
    """

    async def main():
        port = get_free_port()
        frames = [small_frame(i) for i in range(3)]
        client = NetGear_Async(port=port, receive_mode=True, timeout=5.0)
        server = NetGear_Async(port=port, timeout=5.0)
        server.config["generator"] = frame_generator(frames)
        server.launch()
        return await run(server, client)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert len(asyncio.run(main())) == 3


def test_netgear_async_sync_close():
    """
    Documented sync usage: own loop, `run_until_complete`, then `close()`
    """
    port = get_free_port()
    client = NetGear_Async(port=port, receive_mode=True, timeout=5.0)
    server = NetGear_Async(port=port, timeout=5.0)
    server.config["generator"] = frame_generator([small_frame(i) for i in range(3)])
    server.launch()
    received = []

    async def main():
        task = asyncio.ensure_future(collect(client))
        await server.task
        await server.aclose()
        received.extend(await task)

    server.loop.run_until_complete(main())
    client.close()
    server.close()
    assert len(received) == 3
    assert server.loop.is_closed()


@pytest.mark.parametrize(
    "pattern, options, source, raises",
    [
        (0, {"bidirectional_mode": True}, None, False),
        (2, {"bidirectional_mode": False}, None, False),
        (0, {"bidirectional_mode": False}, "source", False),
        (1, {"bidirectional_mode": "invalid"}, None, False),
        (2, {"bidirectional_mode": True}, None, True),
        (3, {"bidirectional_mode": True}, None, True),
        (0, {"bidirectional_mode": True}, "source", True),
    ],
)
@pytest.mark.asyncio
async def test_netgear_async_options(pattern, options, source, raises):
    """
    `bidirectional_mode` validation
    """
    kwargs = dict(
        pattern=pattern,
        source=return_testvideo_path() if source else None,
        port=get_free_port(),
        **options,
    )
    if raises:
        with pytest.raises(ValueError):
            NetGear_Async(**kwargs)
    else:
        await NetGear_Async(**kwargs).aclose()


@pytest.mark.asyncio
async def test_netgear_async_timeout_validation():
    """
    Invalid timeouts fall back to the default
    """
    for value in (0, -1, "x"):
        gear = NetGear_Async(port=get_free_port(), receive_mode=True, timeout=value)
        assert gear._NetGear_Async__timeout == 15.0
        await gear.aclose()


@pytest.mark.asyncio
async def test_netgear_async_recv_generator_send_mode():
    """
    `recv_generator()` raises in send mode
    """
    server = NetGear_Async(port=get_free_port())
    try:
        with pytest.raises(ValueError):
            async for _frame in server.recv_generator():
                pass
    finally:
        await server.aclose()


@pytest.mark.asyncio
async def test_netgear_async_launch_receive_noop():
    """
    `launch()` in receive mode has no side effects
    """
    client = NetGear_Async(port=get_free_port(), receive_mode=True)
    assert client.launch() is client
    assert client.task is None
    await client.aclose()


@pytest.mark.asyncio
async def test_netgear_async_invalid_address():
    """
    Binding an invalid address raises when the generator starts
    """
    client = NetGear_Async(address="172.31.11.15.77", port=get_free_port(), receive_mode=True)
    try:
        with pytest.raises(RuntimeError):
            await collect(client)
    finally:
        await client.aclose()


@pytest.mark.skipif(_windows, reason="ipc needs UNIX domain sockets")
@pytest.mark.parametrize("explicit_path", [False, True])
@pytest.mark.asyncio
async def test_netgear_async_ipc(explicit_path, tmp_path):
    """
    `ipc` works with default and explicit paths
    """
    address = str(tmp_path / "netgear_async.ipc") if explicit_path else None
    port = get_free_port()
    client = NetGear_Async(address=address, port=port, protocol="ipc", receive_mode=True)
    server = NetGear_Async(address=address, port=port, protocol="ipc")
    server.config["generator"] = frame_generator([small_frame()])
    server.launch()
    assert len(await run(server, client)) == 1


@pytest.mark.asyncio
async def test_netgear_async_pigear_without_source(monkeypatch):
    """
    `api=Backend.PIGEAR` builds the stream without a `source`
    """
    calls = []

    class FakeVideoGear:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def stop(self):
            pass

    monkeypatch.setattr(netgear_async_module, "VideoGear", FakeVideoGear)
    server = NetGear_Async(port=get_free_port(), api=Backend.PIGEAR)
    assert calls and calls[0]["api"] == Backend.PIGEAR
    assert server.config["generator"] is not None
    await server.aclose()


@pytest.mark.asyncio
async def test_netgear_async_enablePiCamera_deprecated():
    """
    `enablePiCamera=False` must route to CamGear and emit a DeprecationWarning.
    """
    with pytest.warns(DeprecationWarning, match="enablePiCamera"):
        server = NetGear_Async(
            enablePiCamera=False,
            source=return_testvideo_path(),
            port=get_free_port(),
        )
    await server.aclose()


@pytest.mark.asyncio
async def test_netgear_async_ffgear_backend():
    """
    `api=Backend.FFGEAR` must route through FFGear and transfer frames successfully.
    """
    pytest.importorskip("deffcode", reason="`deffcode` is required for FFGear backend")
    port = get_free_port()
    client = NetGear_Async(port=port, receive_mode=True, timeout=7.0)
    server = NetGear_Async(
        api=Backend.FFGEAR,
        source=return_testvideo_path(),
        port=port,
        timeout=7.0,
    ).launch()
    received = await run(server, client)
    assert received and received[0].shape == video_frames(1)[0].shape
