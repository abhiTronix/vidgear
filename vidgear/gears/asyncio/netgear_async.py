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
import contextlib
import inspect
import logging as log
import os
import platform
import secrets
import string
import warnings
from collections.abc import AsyncGenerator
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

# import helper packages
from ..helper import (
    Backend,
    import_dependency_safe,
    load_curve_keys,
    logcurr_vidgear_ver,
    logger_handler,
    zmq_endpoint,
)

# import additional API(s)
from ..videogear import VideoGear

# safe import critical Class modules
zmq = import_dependency_safe("zmq", pkg_name="pyzmq", error="silent", min_version="4.0")
if zmq is not None:
    import zmq.asyncio
    from zmq.auth import CURVE_ALLOW_ANY
    from zmq.auth.thread import ThreadAuthenticator
msgspec = import_dependency_safe("msgspec", error="silent", min_version="0.20.0")
uvloop = import_dependency_safe("uvloop", error="silent")

# define logger
logger = log.getLogger("NetGear_Async")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

# wire format version, bumped on incompatible changes
_WIRE_VERSION = 2

if msgspec is not None:

    class _Header(msgspec.Struct, omit_defaults=True):
        """
        Header (first part) of every NetGear_Async message. A raw frame buffer follows as the second part
        when `dtype` and `shape` are set.
        """

        v: int  # wire version
        terminate: bool = False
        bi_mode: bool = False
        data: Any = None  # user data (msgpack-serializable)
        dtype: str | None = None
        shape: tuple[int, ...] | None = None

    def _enc_hook(obj: Any) -> Any:
        # numpy scalars inside user data
        if isinstance(obj, np.generic):
            return obj.item()
        raise NotImplementedError(
            "Objects of type {} are not supported".format(type(obj).__name__)
        )

    _encoder = msgspec.msgpack.Encoder(enc_hook=_enc_hook)
    _decoder = msgspec.msgpack.Decoder(_Header)


def _pack(array: NDArray | None = None, **fields: Any) -> list:
    """
    Builds a multipart message: header, plus the raw C-contiguous array buffer if given.
    """
    if array is None:
        return [_encoder.encode(_Header(v=_WIRE_VERSION, **fields))]
    array = np.ascontiguousarray(array)
    header = _Header(v=_WIRE_VERSION, dtype=array.dtype.str, shape=array.shape, **fields)
    return [_encoder.encode(header), array]


def _unpack(parts: list) -> tuple[Any, NDArray | None]:
    """
    Validates and decodes a multipart message from an untrusted peer.

    **Returns:** a tuple of header and the (writeable) array, or `None` if no array was sent.
    """
    try:
        header = _decoder.decode(memoryview(parts[0]))
    except msgspec.DecodeError as e:
        raise ValueError(
            "[NetGear_Async:ERROR] :: Invalid message received ({}). Both ends must run the same vidgear version!".format(
                e
            )
        ) from e
    if header.v != _WIRE_VERSION:
        raise ValueError(
            "[NetGear_Async:ERROR] :: Peer uses wire version `{}`, expected `{}`. Both ends must run the same vidgear version!".format(
                header.v, _WIRE_VERSION
            )
        )
    if len(parts) == 1:
        return header, None
    if header.dtype is None or header.shape is None:
        raise ValueError("[NetGear_Async:ERROR] :: Frame received without dtype/shape!")
    try:
        dtype = np.dtype(header.dtype)
    except TypeError as e:
        raise ValueError(
            "[NetGear_Async:ERROR] :: Invalid dtype `{}` received!".format(header.dtype)
        ) from e
    if dtype.hasobject or dtype.kind == "V":
        raise ValueError("[NetGear_Async:ERROR] :: Refusing object/void dtype from peer!")
    # `reshape` raises `ValueError` if the buffer size doesn't match `shape`
    return header, np.frombuffer(parts[1], dtype=dtype).reshape(header.shape)


class NetGear_Async:
    """
    NetGear_Async is an asyncio port of NetGear API, built on `zmq.asyncio`. It provides complete server-client handling with various
    options to use variable protocols/patterns similar to NetGear, but lacks in term of flexibility as it supports only a few NetGear's Exclusive Modes.

    When NetGear_Async creates its own event loop, it uses uvloop _(if installed, not on Windows)_. NetGear_Async allows us to define
    our custom Server as source to transform frames easily before sending them across the network.

    NetGear_Async now supports additional **bidirectional data transmission** between receiver(client) and sender(server) while transferring frames.
    Users can easily build complex applications such as like _Real-Time Video Chat_ in just few lines of code.

    In addition to all this, NetGear_Async API also provides internal wrapper around VideoGear, which itself provides internal access to CamGear, PiGear and FFGear APIs, thereby
    granting it exclusive power for transferring frames incoming from any source to the network.

    NetGear_Async as of now supports four ZeroMQ messaging patterns:

    - `zmq.PAIR` _(ZMQ Pair Pattern)_
    - `zmq.REQ/zmq.REP` _(ZMQ Request/Reply Pattern)_
    - `zmq.PUB/zmq.SUB` _(ZMQ Publish/Subscribe Pattern)_
    - `zmq.PUSH/zmq.PULL` _(ZMQ Push/Pull Pattern)_

    Whereas supported protocol are: `tcp` and `ipc`.
    """

    def __init__(
        self,
        # NetGear_Async parameters
        address: str | None = None,
        port: str | None = None,
        protocol: str = "tcp",
        pattern: int = 0,
        receive_mode: bool = False,
        timeout: int | float = 15.0,
        # VideoGear parameters
        api: Backend = Backend.CAMGEAR,
        stabilize: bool = False,
        # PiGear parameters
        camera_num: int = 0,
        resolution: tuple[int, int] = (640, 480),
        framerate: int | float = 25,
        # CamGear/FFGear parameters
        source: Any = None,
        stream_mode: bool = False,
        backend: int = 0,
        # FFGear parameters
        source_demuxer: str | None = None,
        frame_format: str = "bgr24",
        custom_ffmpeg: str = "",
        # common parameters
        colorspace: str | None = None,
        time_delay: int = 0,
        logging: bool = False,
        # deprecated
        enablePiCamera: bool | None = None,
        **options: Any,
    ):
        """
        This constructor method initializes the object state and attributes of the NetGear_Async class.

        Parameters:
            address (str): sets the receiver's network address (or `ipc` path).
            port (str): sets the valid Network Port of the Server/Client.
            protocol (str): sets the valid messaging protocol between Server/Client.
            pattern (int): sets the supported messaging pattern(flow of communication) between Server/Client.
            receive_mode (bool): select the NetGear_Async's Mode of operation.
            timeout (int/float): maximum waiting time (in sec, `> 0`) for messages, replies and the `close()` handshake. Default: `15.0`.
            api (Backend): selects the capture backend. Accepted values are `Backend.CAMGEAR` _(default)_,
                `Backend.PIGEAR`, and `Backend.FFGEAR`. Raises `TypeError` if an invalid value is given.
            stabilize (bool): enable access to Stabilizer Class for stabilizing frames.
            camera_num (int): [PiGear only] selects the camera module index. Must be `>= 0`.
            resolution (tuple): [PiGear only] sets `(width, height)` of the source. Default: `(640, 480)`.
            framerate (int/float): [PiGear only] sets the framerate of the source. Default: `25`.
            source (Any): [CamGear/FFGear] defines the source for the input stream (device index,
                filepath, network URL, or image-sequence glob). Default: `None` _(custom generator, unless `api=Backend.PIGEAR`)_.
            stream_mode (bool): [CamGear/FFGear] enables Stream-Mode for `yt_dlp`-backed streaming URLs.
            backend (int): [CamGear only] selects the OpenCV VideoCapture backend (e.g. `cv2.CAP_DSHOW`).
            source_demuxer (str): [FFGear only] specifies the FFmpeg demuxer for the source
                (e.g. `"v4l2"`, `"dshow"`, `"avfoundation"`). Default: `None` (auto-detect).
            frame_format (str): [FFGear only] specifies the pixel layout for decoded frames
                (any FFmpeg-supported pix_fmt, e.g. `"bgr24"`, `"gray"`, `"yuv420p"`). Default: `"bgr24"`.
            custom_ffmpeg (str): [FFGear only] path to a custom FFmpeg executable. Default: `""` (use PATH).
            colorspace (str): [CamGear/PiGear only] selects the colorspace of the input stream.
            time_delay (int): [CamGear/PiGear only] time delay (in seconds) before reading frames.
            logging (bool): enables/disables logging. Default: `False`.
            enablePiCamera (bool): **DEPRECATED** — use `api=Backend.PIGEAR` instead. Will be removed
                in a future release.
            options (dict): additional tweak parameters forwarded to the selected backend gear,
                NetGear_Async internals, and/or the Stabilizer class.
        """
        # enable logging if specified
        self.__logging = logging if isinstance(logging, bool) else False

        # print current version
        logcurr_vidgear_ver(logging=self.__logging)

        # handle deprecated `enablePiCamera`
        if enablePiCamera is not None:
            warnings.warn(
                "`enablePiCamera` is deprecated in NetGear_Async; use `api=Backend.PIGEAR` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            api = Backend.PIGEAR if enablePiCamera else Backend.CAMGEAR

        # raise error(s) for critical Class imports
        import_dependency_safe(
            "zmq" if zmq is None else "", min_version="4.0", pkg_name="pyzmq"
        )
        import_dependency_safe("msgspec" if msgspec is None else "", min_version="0.20.0")

        # define valid messaging patterns => `0`: PAIR, `1`:(REQ, REP), `2`:(PUB, SUB), `3`:(PUSH, PULL)
        valid_messaging_patterns = {
            0: (zmq.PAIR, zmq.PAIR),
            1: (zmq.REQ, zmq.REP),
            2: (zmq.PUB, zmq.SUB),
            3: (zmq.PUSH, zmq.PULL),
        }

        # check whether user-defined messaging pattern is valid
        if not (isinstance(pattern, int) and pattern in valid_messaging_patterns):
            logger.warning("Invalid pattern `{}`. Defaulting to `zmq.PAIR`!".format(pattern))
            pattern = 0
        self.__msg_pattern = pattern
        self.__pattern = valid_messaging_patterns[pattern]

        # check  whether user-defined messaging protocol is valid
        if protocol not in ["tcp", "ipc"]:
            logger.warning("Invalid protocol `{}`. Defaulting to `tcp`!".format(protocol))
            protocol = "tcp"

        # initialize Termination flag
        self.__terminate = False
        # initialize and assign `Receive Mode`
        self.__receive_mode = receive_mode
        # initialize stream handler
        self.__stream = None
        # initialize Messaging Socket
        self.__msg_socket = None
        # initialize NetGear_Async's configuration dictionary
        self.config = {}
        # server task handler
        self.task = None
        # define Bidirectional mode
        self.__bi_mode = False  # handles Bidirectional mode state

        # assign timeout
        if isinstance(timeout, (int, float)) and timeout > 0:
            self.__timeout = float(timeout)
        else:
            logger.warning("Invalid `timeout` value `{}`. Defaulting to `15.0`!".format(timeout))
            self.__timeout = 15.0

        # generate 8-digit random system id
        self.__id = "".join(
            secrets.choice(string.ascii_uppercase + string.digits) for i in range(8)
        )

        # Handle user-defined options dictionary values
        # reformat dictionary
        options = {str(k).strip(): v for k, v in options.items()}
        # handle bidirectional mode
        bi_mode = options.pop("bidirectional_mode", False)
        if bi_mode is True:
            if pattern >= 2:
                raise ValueError(
                    "[NetGear_Async:ERROR] :: `{}` pattern is not valid when Bidirectional Mode is enabled. Kindly refer Docs for more Information!".format(
                        pattern
                    )
                )
            if source is not None:
                raise ValueError(
                    "[NetGear_Async:ERROR] :: Custom source must be used when Bidirectional Mode is enabled. Kindly refer Docs for more Information!"
                )
            self.__bi_mode = True
            self.__logging and logger.debug(
                "Bidirectional Data Transmission is enabled for this connection!"
            )
        elif bi_mode is not False:
            logger.error("`bidirectional_mode` value is invalid!")

        # define messaging asynchronous Context
        self.__msg_context = zmq.asyncio.Context()

        # handle secure mode => `0`: Grasslands (off), `1`: StoneHouse, `2`: IronHouse
        secure_mode = options.pop("secure_mode", 0)
        cert_location = options.pop("custom_cert_location", "") or os.path.join(
            os.path.expanduser("~"), ".vidgear"
        )
        overwrite_cert = options.pop("overwrite_cert", False) is True and not receive_mode
        self.__curve_keys = None  # (secret, public, server_public) CURVE keys
        self.__z_auth = None
        if secure_mode in (1, 2):
            if not zmq.has("curve"):
                raise RuntimeError(
                    "[NetGear_Async:ERROR] :: Secure Mode requires libzmq built with CURVE support!"
                )
            self.__curve_keys, public_keys_dir = load_curve_keys(
                cert_location, receive_mode, overwrite=overwrite_cert, logging=self.__logging
            )
            if receive_mode:
                # the receiver binds and is the CURVE server: ZAP runs in a thread on a sync
                # shadow of the messaging context (IronHouse also authenticates the sender's key)
                self.__z_auth = ThreadAuthenticator(
                    zmq.Context.shadow(self.__msg_context.underlying)
                )
                self.__z_auth.start()
                self.__z_auth.configure_curve(
                    domain="*",
                    location=public_keys_dir if secure_mode == 2 else CURVE_ALLOW_ANY,
                )
        elif secure_mode != 0:
            logger.warning("Invalid `secure_mode` value `{}`. Disabling it!".format(secure_mode))

        # use the running event loop, otherwise own a new one
        try:
            self.loop = asyncio.get_running_loop()
            self.__owns_loop = False
        except RuntimeError:
            self.loop = (
                uvloop.new_event_loop()
                if uvloop is not None and platform.system() != "Windows"
                else asyncio.new_event_loop()
            )
            self.__owns_loop = True
        self.__logging and logger.debug(
            "Using `{}` event loop for this process.".format(self.loop.__class__.__name__)
        )

        # the receiver binds, the sender connects
        self.__endpoint = zmq_endpoint(
            protocol, address, "5555" if port is None else port, bind=receive_mode
        )

        if not receive_mode:
            # Handle video source
            if source is None and api != Backend.PIGEAR:
                self.config = {"generator": None}
                self.__logging and logger.debug(
                    "No source given, assign a custom generator to `config['generator']`."
                )
            else:
                # define stream with necessary params
                self.__stream = VideoGear(
                    api=api,
                    stabilize=stabilize,
                    source=source,
                    camera_num=camera_num,
                    stream_mode=stream_mode,
                    backend=backend,
                    source_demuxer=source_demuxer,
                    frame_format=frame_format,
                    custom_ffmpeg=custom_ffmpeg,
                    colorspace=colorspace,
                    resolution=resolution,
                    framerate=framerate,
                    logging=logging,
                    time_delay=time_delay,
                    **options,
                )
                # define default frame generator in configuration
                self.config = {"generator": self.__frame_generator()}

        # create asyncio queue if bidirectional mode activated
        self.__queue = asyncio.Queue() if self.__bi_mode else None

    def launch(self) -> Self:
        """
        Launches the Server handler task in Send Mode _(no-op in Receive Mode, use `recv_generator()` instead)_.
        """
        if not self.__receive_mode:
            self.__logging and logger.debug(
                "Creating NetGear_Async asynchronous server handler!"
            )
            # create task for Server Handler
            self.task = self.loop.create_task(self.__server_handler())
        # return instance
        return self

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def __make_socket(self):
        """
        Creates the messaging socket, and binds (receiver) or connects (sender) it.
        """
        sock = self.__msg_context.socket(
            self.__pattern[1 if self.__receive_mode else 0]
        )
        if self.__msg_pattern == 1 and not self.__receive_mode:
            sock.REQ_RELAXED = True
            sock.REQ_CORRELATE = True
        if self.__msg_pattern == 2:
            sock.set_hwm(1)
            self.__receive_mode and sock.setsockopt(zmq.SUBSCRIBE, b"")
        if self.__curve_keys is not None:
            secret, public, server_public = self.__curve_keys
            sock.curve_secretkey = secret
            sock.curve_publickey = public
            if self.__receive_mode:
                sock.curve_server = True
            else:
                sock.curve_serverkey = server_public
        try:
            if self.__receive_mode:
                sock.bind(self.__endpoint)
            else:
                sock.connect(self.__endpoint)
        except Exception as e:
            sock.close(linger=0)
            logger.exception(str(e))
            raise RuntimeError(
                "[NetGear_Async:ERROR] :: Failed to {} address: {} and pattern: {}{}!".format(
                    "bind" if self.__receive_mode else "connect",
                    self.__endpoint,
                    self.__msg_pattern,
                    " and Bidirectional Mode enabled" if self.__bi_mode else "",
                )
            ) from e
        self.__logging and logger.debug(
            "Successfully {} to address: {} with pattern: {}.".format(
                "bound" if self.__receive_mode else "connected",
                self.__endpoint,
                self.__msg_pattern,
            )
        )
        return sock

    async def __recv(self) -> list:
        """
        Receives a multipart message within the timeout.
        """
        async with asyncio.timeout(self.__timeout):
            while True:
                try:
                    return await self.__msg_socket.recv_multipart(copy=False)
                except zmq.Again:
                    # pyzmq raises on a spurious wakeup (seen right after a CURVE handshake)
                    continue

    async def __server_handler(self):
        """
        Handles various Server-end processes/tasks.
        """
        # validate assigned frame generator in NetGear_Async configuration
        if not (isinstance(self.config, dict) and "generator" in self.config):
            raise RuntimeError(
                "[NetGear_Async:ERROR] :: Assigned NetGear_Async configuration is invalid!"
            )
        if not inspect.isasyncgen(self.config["generator"]):
            raise ValueError(
                "[NetGear_Async:ERROR] :: Invalid configuration. Assigned generator must be a asynchronous generator function/method only!"
            )

        # define our messaging socket
        self.__msg_socket = self.__make_socket()
        self.__logging and logger.info("Send Mode is successfully activated and ready to send data!")

        # loop over our Asynchronous frame generator
        async for dataframe in self.config["generator"]:
            if self.__bi_mode:
                if not (isinstance(dataframe, tuple) and len(dataframe) == 2):
                    raise ValueError(
                        "[NetGear_Async:ERROR] :: Send Mode only accepts tuple(data, frame) as input in Bidirectional Mode. Kindly refer vidgear docs!"
                    )
                (data, frame) = dataframe
                if isinstance(data, np.ndarray):
                    logger.warning(
                        "Skipped unsupported `data` of datatype: {}!".format(
                            type(data).__name__
                        )
                    )
                    data = None
            else:
                frame, data = dataframe, None
            if not isinstance(frame, np.ndarray):
                raise ValueError(
                    "[NetGear_Async:ERROR] :: Invalid frame of type `{}` received from the generator!".format(
                        type(frame).__name__
                    )
                )

            # send header and raw frame buffer as one multipart message
            await self.__msg_socket.send_multipart(
                _pack(frame, bi_mode=self.__bi_mode, data=data), copy=False
            )

            # synchronous patterns get a reply for every frame
            if self.__msg_pattern < 2:
                header, array = _unpack(await self.__recv())
                if self.__bi_mode:
                    await self.__queue.put(array if array is not None else header.data)
                else:
                    self.__logging and logger.debug(header.data)

    async def recv_generator(self) -> AsyncGenerator[Any, None]:
        """
        A default Asynchronous Frame Generator for NetGear_Async's Receiver-end.

        **Yields:** writeable frames _(numpy.ndarray)_, or `(data, frame)` tuples in Bidirectional Mode.
        """
        # check whether `receive mode` is activated
        if not (self.__receive_mode):
            # raise Value error and exit
            self.__terminate = True
            raise ValueError(
                "[NetGear_Async:ERROR] :: `recv_generator()` function cannot be accessed while `receive_mode` is disabled. Kindly refer vidgear docs!"
            )

        # initialize, define and bind messaging socket
        self.__msg_socket = self.__make_socket()
        self.__logging and logger.info("Receive Mode is activated successfully!")

        # loop until terminated
        while not self.__terminate:
            header, frame = _unpack(await self.__recv())
            # terminate if exit` flag received from server
            if header.terminate:
                # send confirmation message to server if bidirectional patterns
                if self.__msg_pattern < 2:
                    await self.__msg_socket.send_multipart(_pack(terminate=True))
                self.__logging and logger.info("Termination signal received from server!")
                self.__terminate = True
                break
            if frame is None:
                raise ValueError("[NetGear_Async:ERROR] :: Message received without a frame!")

            # check if bidirectional patterns
            if self.__msg_pattern < 2:
                if self.__bi_mode and header.bi_mode:
                    return_data = (
                        self.__queue.get_nowait() if not self.__queue.empty() else None
                    )
                    reply = (
                        _pack(return_data)
                        if isinstance(return_data, np.ndarray)
                        else _pack(data=return_data)
                    )
                    await self.__msg_socket.send_multipart(reply, copy=False)
                elif self.__bi_mode or header.bi_mode:
                    # raise error if bidirectional mode is disabled at server or client but not both
                    raise RuntimeError(
                        "[NetGear_Async:ERROR] :: Invalid configuration! Bidirectional Mode is not activate on {} end.".format(
                            "server" if self.__bi_mode else "client"
                        )
                    )
                else:
                    # otherwise just send confirmation message to server
                    await self.__msg_socket.send_multipart(
                        _pack(data="Data received on client: {} !".format(self.__id))
                    )
            # yield received tuple(data-frame) if bidirectional mode or else just frame
            yield (header.data, frame) if self.__bi_mode else frame
            # sleep for sometime
            await asyncio.sleep(0)

    async def __frame_generator(self):
        """
        Returns a default frame-generator for NetGear_Async's Server Handler.
        """
        # start stream
        self.__stream.start()
        # loop over stream until its terminated
        while not self.__terminate:
            # read frames off the event loop (blocking call)
            frame = await asyncio.to_thread(self.__stream.read)
            # break if NoneType
            if frame is None:
                break
            # yield frame
            yield frame

    async def transceive_data(self, data: Any = None) -> Any:
        """
        Bidirectional Mode exclusive method to Transmit data _(in Receive mode)_ and Receive data _(in Send mode)_.

        Parameters:
            data (any): inputs msgpack-serializable data _(or a `numpy.ndarray` frame)_ for sending back to Server.
        """
        recvd_data = None
        if not self.__terminate:
            if self.__bi_mode:
                if self.__receive_mode:
                    await self.__queue.put(data)
                else:
                    if not self.__queue.empty():
                        recvd_data = await self.__queue.get()
                        self.__queue.task_done()
            else:
                logger.error(
                    "`transceive_data()` function cannot be used when Bidirectional Mode is disabled."
                )
        return recvd_data

    async def aclose(self, disable_confirmation: bool = False) -> None:
        """
        Terminates all NetGear_Async Asynchronous processes gracefully, bounded by `timeout`.

        Parameters:
            disable_confirmation (bool): don't wait for the Client's termination confirmation in bidirectional patterns.
        """
        self.__logging and logger.debug(
            "Terminating various {} Processes. Please wait.".format(
                "Receive Mode" if self.__receive_mode else "Send Mode"
            )
        )
        # indicate that process should be terminated
        self.__terminate = True
        if not self.__receive_mode:
            # stop the server task if still running
            if self.task is not None and not self.task.done():
                self.task.cancel()
                with contextlib.suppress(BaseException):
                    await self.task
            # terminate stream
            self.__stream is not None and self.__stream.stop()

        sock = self.__msg_socket
        delivered = False  # whether frames/terminate flag may still be queued
        if sock is not None and not sock.closed:
            try:
                if not self.__receive_mode:
                    # signal `exit` flag for termination!
                    await asyncio.wait_for(
                        sock.send_multipart(_pack(terminate=True)), timeout=self.__timeout
                    )
                    delivered = True
                    # check for confirmation in bidirectional patterns
                    if self.__msg_pattern < 2 and not disable_confirmation:
                        await asyncio.wait_for(sock.recv_multipart(), timeout=self.__timeout)
                        self.__logging and logger.debug("Client confirmed termination.")
            except (TimeoutError, zmq.ZMQError) as e:
                self.__logging and logger.debug(
                    "Termination not confirmed by Client: {}".format(type(e).__name__)
                )
            finally:
                # give queued messages a bounded time to flush
                sock.close(linger=int(self.__timeout * 1000) if delivered else 0)

        # empty queue in bidirectional mode
        while self.__queue is not None and not self.__queue.empty():
            self.__queue.get_nowait()

        self.__z_auth is not None and self.__z_auth.stop()
        # release ZMQ context, waiting (off the loop) for lingering messages
        if not self.__msg_context.closed:
            await asyncio.to_thread(self.__msg_context.term)
        self.__logging and logger.info(
            "{} successfully terminated!".format(
                "Receive Mode" if self.__receive_mode else "Send Mode"
            )
        )

    def close(self, skip_loop: bool = False) -> asyncio.Task | None:
        """
        Terminates all NetGear_Async Asynchronous processes gracefully. Inside a running event loop prefer `await aclose()`.

        Parameters:
            skip_loop (bool): don't close the event loop owned by NetGear_Async, and skip the termination confirmation.

        **Returns:** the scheduled termination task when called inside a running event loop, otherwise `None`.
        """
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is not None:
            # can't block a running loop, schedule termination instead
            self.__close_task = running_loop.create_task(
                self.aclose(disable_confirmation=skip_loop)
            )
            return self.__close_task
        self.loop.run_until_complete(self.aclose(disable_confirmation=skip_loop))
        if not skip_loop and self.__owns_loop:
            self.loop.close()
        return None
