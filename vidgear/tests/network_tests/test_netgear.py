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

import json
import logging as log
import os
import socket
import stat
import time
from contextlib import contextmanager

import cv2
import numpy as np
import pytest
import simplejpeg
import zmq

import vidgear.gears.netgear as netgear_module
from vidgear.gears import NetGear
from vidgear.gears.helper import logger_handler
from vidgear.tests.utils.helpers import get_free_port, return_testvideo_path

# define test logger
logger = log.getLogger("Test_netgear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

# every test must finish, and a crashed receive thread must fail the test
pytestmark = [
    pytest.mark.timeout(120),
    pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning"),
]

_windows = os.name == "nt"


@pytest.fixture(scope="module")
def video_frame():
    """
    A real 720p BGR frame from the test video.
    """
    stream = cv2.VideoCapture(return_testvideo_path())
    grabbed, frame = stream.read()
    stream.release()
    assert grabbed, "Test video is missing!"
    return frame


@pytest.fixture
def port():
    return get_free_port()


def small_frame(seed=0):
    return np.random.default_rng(seed).integers(0, 255, (48, 64, 3), dtype=np.uint8)


def lan_ip():
    """
    Host's LAN IP, or None if unavailable.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
    except OSError:
        return None
    return None if ip.startswith("127.") else ip


@contextmanager
def netgear_pair(port, pattern=0, address=None, client_options=None, **options):
    """
    Yields a connected (server, client), closing both afterwards.
    """
    client = server = None
    try:
        client = NetGear(
            address=address,
            port=port,
            pattern=pattern,
            receive_mode=True,
            logging=True,
            **(client_options or options),
        )
        server = NetGear(address=address, port=port, pattern=pattern, logging=True, **options)
        # PUB/SUB: let the subscriber join before publishing
        pattern == 2 and time.sleep(0.5)
        yield server, client
    finally:
        server is not None and server.close()
        client is not None and client.close()


def assert_similar(a, b, tolerance=10.0):
    """
    Approximate equality for JPEG-compressed frames.
    """
    assert a.shape == b.shape
    assert np.abs(a.astype(np.int16) - b.astype(np.int16)).mean() < tolerance


@pytest.mark.parametrize("address", [None, "127.0.0.1"])
def test_playback(address, port):
    """
    Tests NetGear Bare-minimum network playback capabilities (default JPEG compression)
    """
    stream = cv2.VideoCapture(return_testvideo_path())
    count = 0
    try:
        with netgear_pair(port, address=address) as (server, client):
            while True:
                (grabbed, frame_server) = stream.read()
                if not grabbed:
                    break
                server.send(frame_server)
                assert_similar(frame_server, client.recv())
                count += 1
    finally:
        stream.release()
    assert count > 0


def test_invalid_address(port):
    """
    Binding an invalid address must raise
    """
    with pytest.raises(RuntimeError):
        NetGear(address="172.31.11.15.77", port=port, receive_mode=True)


@pytest.mark.parametrize("receive_mode", [True, False])
def test_primary_mode(receive_mode, port):
    """
    `send()` in receive mode and `recv()` in send mode must raise
    """
    conn = NetGear(port=port, receive_mode=receive_mode)
    try:
        with pytest.raises(ValueError):
            conn.send(small_frame()) if receive_mode else conn.recv()
    finally:
        conn.close()


@pytest.mark.parametrize("pattern", [0, 1, 2, 3])
def test_patterns(pattern, port, video_frame):
    """
    Testing NetGear different messaging patterns with exact (uncompressed) frames
    """
    options = {"flag": 0, "copy": True, "track": False, "jpeg_compression": False}
    with netgear_pair(port, pattern=pattern, **options) as (server, client):
        server.send(video_frame)
        frame_client = client.recv(return_data=[1, 2, 3] if pattern >= 2 else None)
        assert np.array_equal(video_frame, frame_client)


@pytest.mark.parametrize("pattern, protocol", [(4, None), (-1, "udp"), ("x", "tcp")])
def test_invalid_pattern_protocol(pattern, protocol, port):
    """
    Invalid pattern/protocol values fall back to PAIR/tcp
    """
    frame = small_frame()
    client = NetGear(port=port, pattern=pattern, protocol=protocol, receive_mode=True)
    server = NetGear(port=port, pattern=pattern, protocol=protocol, jpeg_compression=False)
    try:
        server.send(frame)
        assert np.array_equal(client.recv(), frame)
    finally:
        server.close()
        client.close()


@pytest.mark.parametrize(
    "options_server",
    [
        {"jpeg_compression": "invalid", "jpeg_compression_quality": 5},
        {
            "jpeg_compression": " gray  ",
            "jpeg_compression_quality": 50,
            "jpeg_compression_fastdct": True,
            "jpeg_compression_fastupsample": True,
        },
        {
            "jpeg_compression": True,
            "jpeg_compression_quality": 55.55,
            "jpeg_compression_fastdct": False,
            "jpeg_compression_fastupsample": True,
            "jpeg_compression_colorsubsampling": "420",
        },
        {"jpeg_compression": "RGB", "jpeg_compression_colorsubsampling": "invalid"},
    ],
)
def test_compression(options_server, port, video_frame):
    """
    Testing NetGear's real-time frame compression capabilities
    """
    gray = str(options_server["jpeg_compression"]).strip().upper() == "GRAY"
    frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY) if gray else video_frame
    with netgear_pair(port, client_options={}, **options_server) as (server, client):
        for _ in range(3):
            server.send(frame)
            frame_client = client.recv()
            assert frame_client.ndim == frame.ndim
            assert_similar(frame, frame_client, tolerance=15.0)


@pytest.mark.parametrize(
    "frame, colorspace",
    [
        (np.random.default_rng(0).integers(0, 255, (48, 64), dtype=np.uint8), "BGR"),
        (np.random.default_rng(0).random((48, 64, 3), dtype=np.float32), "BGR"),
        (np.random.default_rng(0).integers(0, 255, (48, 64, 4), dtype=np.uint8), "BGR"),
        (np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8), "GRAY"),
    ],
)
def test_compression_fallback(frame, colorspace, port):
    """
    Frames JPEG can't encode with the selected colorspace are sent raw instead of failing
    """
    with netgear_pair(port, client_options={}, jpeg_compression=colorspace) as (
        server,
        client,
    ):
        server.send(frame)
        assert np.array_equal(client.recv(), frame)


def test_compression_roi(port, video_frame):
    """
    Row-strided ROI crops are JPEG-encoded directly
    """
    roi = video_frame[100:400, 200:600]
    assert not roi.flags["C_CONTIGUOUS"]
    with netgear_pair(port) as (server, client):
        server.send(roi)
        assert_similar(roi, client.recv())


def _raw_peer(port):
    """
    A raw PAIR socket posing as a NetGear sender.
    """
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PAIR)
    sock.connect("tcp://localhost:{}".format(port))
    return ctx, sock


def _jpeg_header():
    return json.dumps(
        {
            "terminate_flag": False,
            "compression": {"dct": True, "ups": False, "colorspace": "BGR"},
            "message": None,
            "pattern": "0",
            "dtype": "",
            "shape": "",
        }
    ).encode()


def _bomb():
    """
    A tiny JPEG whose SOF header claims 8000x8000 pixels.
    """
    jpeg = bytearray(simplejpeg.encode_jpeg(small_frame()))
    sof = jpeg.index(b"\xff\xc0")
    jpeg[sof + 5 : sof + 9] = (8000).to_bytes(2, "big") * 2
    return bytes(jpeg)


@pytest.mark.parametrize("bad_payload", [b"not a jpeg", _bomb()])
def test_invalid_jpeg_dropped(bad_payload, port):
    """
    A corrupt or oversized JPEG costs one frame, not the stream
    """
    client = NetGear(port=port, receive_mode=True, logging=True)
    ctx, peer = _raw_peer(port)
    try:
        peer.send_multipart([_jpeg_header(), bad_payload])
        peer.send_multipart([_jpeg_header(), simplejpeg.encode_jpeg(small_frame())])
        assert client.recv().shape == (48, 64, 3)
        peer.send_json({"terminate_flag": True})
        assert client.recv() is None
    finally:
        client.close()
        peer.close(linger=0)
        ctx.term()


def test_receive_thread_error_raised(port):
    """
    An unexpected error in the receive thread is re-raised by `recv()` instead of hanging
    """
    client = NetGear(port=port, receive_mode=True)
    ctx, peer = _raw_peer(port)
    try:
        peer.send(b"not json")
        with pytest.raises(RuntimeError):
            client.recv()
    finally:
        client.close()
        peer.close(linger=0)
        ctx.term()


def _secure_addresses():
    ip = lan_ip()
    return [None, "127.0.0.1"] + ([ip] if ip else [])


@pytest.mark.parametrize("address", _secure_addresses())
@pytest.mark.parametrize("pattern", [0, 1])
@pytest.mark.parametrize("security_mech", [1, 2])
def test_secure_mode(address, pattern, security_mech, port, tmp_path, video_frame):
    """
    Testing NetGear's Secure Mode with default and explicit addresses
    """
    options = {
        "secure_mode": security_mech,
        "custom_cert_location": str(tmp_path),
        "jpeg_compression": False,
    }
    with netgear_pair(
        port,
        pattern=pattern,
        address=address,
        client_options=options,
        **options,
    ) as (server, client):
        server.send(video_frame)
        assert np.array_equal(video_frame, client.recv())


def test_secure_mode_overwrite_cert(port, tmp_path, video_frame):
    """
    Server regenerates keys with `overwrite_cert`, client then loads the new ones
    """
    options = {"secure_mode": 2, "custom_cert_location": str(tmp_path)}
    server = NetGear(port=port, overwrite_cert=True, **options)
    client = NetGear(port=port, receive_mode=True, overwrite_cert=True, **options)
    try:
        server.send(video_frame)
        assert_similar(video_frame, client.recv())
    finally:
        server.close()
        client.close()


def test_secure_mode_allowed_ips(port, tmp_path):
    """
    `allowed_ips` whitelist rejects other peers
    """
    options = {
        "secure_mode": 1,
        "custom_cert_location": str(tmp_path),
        "request_timeout": 4,
        "max_retries": 1,
    }
    with netgear_pair(
        port, client_options=dict(options, allowed_ips=["10.9.9.9"]), **options
    ) as (server, _client), pytest.raises(RuntimeError):
        server.send(small_frame())


@pytest.mark.skipif(_windows, reason="POSIX permissions")
def test_secure_mode_key_permissions(port, tmp_path):
    """
    CURVE secret keys are owner-only
    """
    NetGear(port=port, secure_mode=1, custom_cert_location=str(tmp_path)).close()
    secret_dir = tmp_path / ".vidgear" / "keys" / "private_keys"
    assert stat.S_IMODE(secret_dir.stat().st_mode) == 0o700
    for key in secret_dir.iterdir():
        assert stat.S_IMODE(key.stat().st_mode) == 0o600


def test_secure_mode_reconnect_keeps_curve(port, tmp_path):
    """
    Lazy-Pirate reconnect keeps CURVE enabled
    """
    server = NetGear(
        port=port, pattern=1, secure_mode=1, custom_cert_location=str(tmp_path)
    )
    try:
        assert server.send(small_frame()) is None  # no client: reconnects
        assert server._NetGear__msg_socket.mechanism == zmq.CURVE
    finally:
        server.close()


@pytest.mark.parametrize(
    "pattern, target_data, expected, options",
    [
        (0, [1, "string", ["list"]], [1, "string", ["list"]], {"jpeg_compression": False}),
        (1, {1: "apple", 2: "cat"}, {"1": "apple", "2": "cat"}, {"jpeg_compression": False}),
        (1, (1, "a"), [1, "a"], {"jpeg_compression": True}),
        (0, 0, 0, {"jpeg_compression": False}),
    ],
)
def test_bidirectional_mode(pattern, target_data, expected, options, port, video_frame):
    """
    Testing NetGear's Bidirectional Mode with JSON data
    """
    with netgear_pair(port, pattern=pattern, bidirectional_mode=True, **options) as (
        server,
        client,
    ):
        server.send(video_frame, message=target_data)
        server_data, frame_client = client.recv(return_data=target_data)
        # replies go out with the next frame
        client_data = server.send(video_frame, message=target_data)
        assert server_data == expected
        assert client_data == expected
        if not options["jpeg_compression"]:
            assert np.array_equal(video_frame, frame_client)


@pytest.mark.parametrize("colorspace", [False, "GRAY"])
def test_bidirectional_mode_ndarray(colorspace, port, video_frame):
    """
    Bidirectional Mode returning frames
    """
    target = video_frame if not colorspace else cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    with netgear_pair(
        port, pattern=1, bidirectional_mode=True, jpeg_compression=colorspace
    ) as (server, client):
        server.send(target)
        client.recv(return_data=target)
        client_data = server.send(target)
        if colorspace:
            assert_similar(target, client_data)
        else:
            assert np.array_equal(target, client_data)


def test_message_json_only(port):
    """
    `message` must be JSON-serializable
    """
    with netgear_pair(port) as (server, _client), pytest.raises(TypeError):
        server.send(small_frame(), message=b"bytes")


@pytest.mark.parametrize(
    "pattern, options",
    [
        (1, {"multiserver_mode": True, "multiclient_mode": True}),
        (0, {"multiserver_mode": True}),
        (0, {"multiclient_mode": True}),
        (2, {"bidirectional_mode": True}),
        (3, {"bidirectional_mode": True}),
        (2, {"multiserver_mode": True, "ssh_tunnel_mode": "new@sdf.org"}),
    ],
)
def test_invalid_modes(pattern, options, monkeypatch):
    """
    Invalid exclusive mode combinations must raise
    """
    monkeypatch.setattr(netgear_module, "check_open_port", lambda *a, **k: True)
    with pytest.raises(ValueError):
        NetGear(pattern=pattern, port=["5556"], **options)


@pytest.mark.parametrize(
    "server_ports, client_ports, options",
    [
        (None, "5555", {"multiserver_mode": True}),
        ("5555", None, {"multiclient_mode": True}),
    ],
)
def test_ports(server_ports, client_ports, options):
    """
    Wrong port values in exclusive modes must raise
    """
    with pytest.raises(ValueError):
        if server_ports:
            NetGear(pattern=1, port=server_ports, **options)
        else:
            NetGear(port=client_ports, pattern=1, receive_mode=True, **options)


@pytest.mark.parametrize("pattern", [1, 2, 3])
def test_multiserver_mode(pattern, video_frame):
    """
    Testing NetGear's Multi-Server Mode with three unique servers
    """
    ports = [get_free_port() for _ in range(3)]
    options = {"jpeg_compression": False, "multiserver_mode": True, "receive_timeout": 10}
    client = NetGear(port=ports, pattern=pattern, receive_mode=True, logging=True, **options)
    servers = [NetGear(pattern=pattern, port=pt, logging=True, **options) for pt in ports]
    pattern == 2 and time.sleep(0.5)
    try:
        received = {}
        for srv, msg in zip(servers, [None, "", "hello"], strict=True):
            srv.send(video_frame, message=msg)
            # always a 3-tuple, whatever the message
            port, message, frame = client.recv()
            assert message == msg
            received[port] = frame
        assert sorted(received) == sorted(ports)
        for frame in received.values():
            assert np.array_equal(video_frame, frame)
        # all servers terminate => end of stream
        for srv in servers:
            srv.close()
        servers = []
        assert client.recv() is None
    finally:
        for srv in servers:
            srv.close()
        client.close()


@pytest.mark.parametrize("pattern", [1, 2, 3])
def test_multiclient_mode(pattern):
    """
    Testing NetGear's Multi-Client Mode with three unique clients
    """
    ports = [get_free_port() for _ in range(3)]
    options = {"multiclient_mode": True, "jpeg_compression": False}
    if pattern == 1:
        options["bidirectional_mode"] = True
    else:
        options["receive_timeout"] = 10
    clients = [
        NetGear(port=pt, pattern=pattern, receive_mode=True, logging=True, **options)
        for pt in ports
    ]
    server = NetGear(pattern=pattern, port=ports, logging=True, **options)
    pattern == 2 and time.sleep(0.5)
    try:
        frames = [small_frame(i) for i in range(6)]
        if pattern == 1:
            # REQ round-robins, one frame per client per cycle
            for frame in frames[:3]:
                server.send(frame, message="data")
            got = []
            for i, client in enumerate(clients):
                message, frame = client.recv(return_data="client-{}".format(i))
                assert message == "data"
                got.append(frame.tobytes())
            assert sorted(got) == sorted(f.tobytes() for f in frames[:3])
            # replies (sent with the next frame) carry each client's port and data
            replies = dict(server.send(frame) for frame in frames[3:])
            assert sorted(replies) == sorted(ports)
            assert sorted(replies.values()) == ["client-0", "client-1", "client-2"]
        elif pattern == 2:
            # PUB broadcasts every frame to every client
            server.send(frames[0])
            for client in clients:
                assert np.array_equal(client.recv(), frames[0])
        else:
            # PUSH round-robins: frames are partitioned between clients
            time.sleep(0.5)
            for frame in frames:
                server.send(frame)
            got = [client.recv() for client in clients for _ in range(2)]
            assert sorted(f.tobytes() for f in got) == sorted(f.tobytes() for f in frames)
        if pattern >= 2:
            # every client gets the termination flag
            server.close()
            server = None
            for client in clients:
                assert client.recv() is None
    finally:
        server is not None and server.close()
        for client in clients:
            client.close()


@pytest.mark.parametrize(
    "pattern, options",
    [
        (1, {"max_retries": 1, "request_timeout": 4}),
        (1, {"max_retries": 1, "request_timeout": 4, "multiserver_mode": True}),
        (2, {"subscriber_timeout": 2}),
        (3, {"receive_timeout": 2}),
    ],
)
def test_client_reliability(pattern, options, port):
    """
    A receiver without server gives up and returns None
    """
    client = NetGear(
        pattern=pattern,
        port=[port] if "multiserver_mode" in options else port,
        receive_mode=True,
        logging=True,
        **options,
    )
    try:
        assert client.recv() is None
    finally:
        client.close()


@pytest.mark.parametrize(
    "options",
    [
        {"max_retries": 2, "request_timeout": 4},
        {"max_retries": 2, "request_timeout": 4, "bidirectional_mode": True},
        {"max_retries": 2, "request_timeout": 4, "multiclient_mode": True},
    ],
)
def test_server_reliability(options, port):
    """
    A sender without client reconnects, then raises
    """
    server = NetGear(
        pattern=1,
        port=[port] if "multiclient_mode" in options else port,
        logging=True,
        **options,
    )
    try:
        assert server.send(small_frame()) is None
        with pytest.raises(RuntimeError):
            server.send(small_frame())
    finally:
        server.close()


def test_retry_budget_resets(port):
    """
    Retries are restored after a successful exchange
    """
    server = NetGear(port=port, pattern=1, max_retries=2, request_timeout=4)
    client = None
    try:
        assert server.send(small_frame()) is None  # 1 retry left
        client = NetGear(port=port, pattern=1, receive_mode=True)
        server.send(small_frame())  # success, budget back to 2
        assert client.recv() is not None
        client.close(kill=True)
        client = None
        assert server.send(small_frame()) is None  # still alive
    finally:
        server.close()
        client is not None and client.close()


def test_max_retries_zero_rejected():
    """
    `max_retries=0` is invalid and ignored
    """
    server = NetGear(pattern=1, max_retries=0)
    try:
        assert server._NetGear__max_retries == 3
    finally:
        server.close()


def test_ssh_tunnel_endpoint(monkeypatch):
    """
    SSH ports are validated, not mangled
    """
    checked, tunnels = [], []
    monkeypatch.setattr(
        netgear_module, "check_open_port", lambda ip, port: checked.append((ip, port)) or True
    )
    monkeypatch.setattr(
        netgear_module.ssh,
        "tunnel_connection",
        lambda sock, addr, server, **kw: tunnels.append((addr, server)),
    )
    server = NetGear(address="10.0.0.5", port="5454", ssh_tunnel_mode=" user@host:4722 ")
    server.close()
    assert checked == [("host", 4722)]
    assert tunnels == [("tcp://10.0.0.5:5454", "user@host:4722")]
    with pytest.raises(ValueError):
        NetGear(protocol="ipc", ssh_tunnel_mode="user@host:4722")


@pytest.mark.skipif(_windows, reason="ipc needs UNIX domain sockets")
@pytest.mark.parametrize("explicit_path", [False, True])
def test_ipc(explicit_path, port):
    """
    `ipc` works with default and explicit paths
    """
    # macOS tmp_path exceeds the 103-char `sun_path` limit
    address = f"/tmp/netgear_{port}.ipc" if explicit_path else None
    client = NetGear(address=address, port=port, protocol="ipc", receive_mode=True)
    server = NetGear(address=address, port=port, protocol="ipc", jpeg_compression=False)
    try:
        frame = small_frame()
        server.send(frame)
        assert np.array_equal(client.recv(), frame)
    finally:
        server.close()
        client.close()


@pytest.mark.parametrize("pattern", [2, 3])
def test_close_returns_promptly(pattern, port):
    """
    `close()` never hangs without a peer
    """
    for receive_mode in (True, False):
        conn = NetGear(pattern=pattern, port=port, receive_mode=receive_mode)
        start = time.monotonic()
        conn.close()
        assert time.monotonic() - start < 3


def test_pattern3_send_without_client_raises(port):
    """
    PUSH `send()` raises once its queue is full and no client takes frames
    """
    server = NetGear(pattern=3, port=port, send_hwm=1, request_timeout=4)
    try:
        with pytest.raises(RuntimeError):
            for _ in range(5):
                server.send(small_frame())
    finally:
        server.close()


@pytest.mark.parametrize("pattern", [0, 1, 3])
def test_send_none_terminates(pattern, port):
    """
    `send(None)` signals end of stream
    """
    with netgear_pair(port, pattern=pattern) as (server, client):
        server.send(small_frame())
        assert client.recv() is not None
        assert server.send(None) is None
        assert client.recv() is None


@pytest.mark.parametrize("pattern", [0, 1, 2, 3])
def test_end_of_stream_delivers_all(pattern, port):
    """
    Frames queued at the receiver are returned before termination
    """
    frames = [small_frame(i) for i in range(30)]
    with netgear_pair(port, pattern=pattern, jpeg_compression=False) as (server, client):
        for frame in frames:
            server.send(frame)
            time.sleep(0.005)
        server.close()
        received = []
        while (frame := client.recv()) is not None:
            received.append(frame)
    assert len(received) == len(frames)
    assert all(np.array_equal(a, b) for a, b in zip(received, frames, strict=True))


def test_kill_keeps_other_instances():
    """
    Tests `close(kill=True)` on one NetGear doesn't break another NetGear's sockets in the same process
    """
    doomed = NetGear(port=get_free_port(), receive_mode=True)
    port = get_free_port()
    client = NetGear(port=port, receive_mode=True)
    server = NetGear(port=port)
    try:
        doomed.close(kill=True)
        frame = np.zeros((48, 64, 3), np.uint8)
        server.send(frame)
        assert client.recv().shape == frame.shape
    finally:
        server.close()
        client.close()
