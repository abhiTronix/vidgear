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
from __future__ import annotations

import asyncio
import contextlib
import fractions
import logging as log
import os
import time
import uuid
import warnings
from os.path import expanduser
from typing import Any

from ..helper import (
    Backend,
    import_dependency_safe,
    logcurr_vidgear_ver,
    logger_handler,
    retrieve_best_interpolation,
)

# import additional API(s)
from ..videogear import VideoGear

# import helper packages
from .helper import (
    create_blank_frame,
    generate_webdata,
    reducer,
)

# safe import critical Class modules
starlette = import_dependency_safe("starlette", error="silent")
if starlette is not None:
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.responses import JSONResponse, PlainTextResponse
    from starlette.routing import Mount, Route
    from starlette.staticfiles import StaticFiles
    from starlette.templating import Jinja2Templates

# define logger
logger = log.getLogger("WebGear_RTC")
if logger.hasHandlers():
    logger.handlers.clear()
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

# add global vars
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 1 / 30  # fallback 30fps, when source framerate is unknown
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)

aiortc = import_dependency_safe("aiortc", error="silent")
if aiortc is not None:
    from aiortc import (
        RTCConfiguration,
        RTCIceServer,
        RTCPeerConnection,
        RTCSessionDescription,
        VideoStreamTrack,
    )
    from aiortc.contrib.media import MediaRelay
    from aiortc.mediastreams import MediaStreamError
    from aiortc.rtcrtpsender import RTCRtpSender
    from av import VideoFrame  # aiortc dependency
    from av.frame import Frame as AVFrame  # imported for type annotation

    class RTC_VideoServer(VideoStreamTrack):
        """
        Default Internal Video-Server for WebGear_RTC,
        a inherit-class to aiortc's VideoStreamTrack API.
        """

        def __init__(
            self,
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
            logging: bool = False,
            time_delay: int = 0,
            # deprecated
            enablePiCamera: bool | None = None,
            **options: Any
        ):
            """
            This constructor method initializes the object state and attributes of the RTC_VideoServer class.

            Parameters:
                api (Backend): selects the capture backend. Accepted values are `Backend.CAMGEAR` _(default)_,
                    `Backend.PIGEAR`, and `Backend.FFGEAR`. Raises `TypeError` if an invalid value is given.
                stabilize (bool): enable access to Stabilizer Class for stabilizing frames.
                camera_num (int): [PiGear only] selects the camera module index. Must be `>= 0`.
                resolution (tuple): [PiGear only] sets `(width, height)` of the source. Default: `(640, 480)`.
                framerate (int/float): [PiGear only] sets the framerate of the source. Default: `25`.
                source (Any): [CamGear/FFGear] defines the source for the input stream (device index,
                    filepath, network URL, or image-sequence glob). Default: `None`.
                stream_mode (bool): [CamGear/FFGear] enables Stream-Mode for `yt_dlp`-backed streaming URLs.
                backend (int): [CamGear only] selects the OpenCV VideoCapture backend (e.g. `cv2.CAP_DSHOW`).
                source_demuxer (str): [FFGear only] specifies the FFmpeg demuxer for the source
                    (e.g. `"v4l2"`, `"dshow"`, `"avfoundation"`). Default: `None` (auto-detect).
                frame_format (str): [FFGear only] specifies the pixel layout for decoded frames
                    (any FFmpeg-supported pix_fmt, e.g. `"bgr24"`, `"gray"`, `"yuv420p"`). Default: `"bgr24"`.
                custom_ffmpeg (str): [FFGear only] path to a custom FFmpeg executable. Default: `""` (use PATH).
                colorspace (str): [CamGear/PiGear only] selects the colorspace of the input stream.
                logging (bool): enables/disables logging. Default: `False`.
                time_delay (int): [CamGear/PiGear only] time delay (in seconds) before reading frames.
                enablePiCamera (bool): **DEPRECATED** — use `api=Backend.PIGEAR` instead. Will be removed
                    in a future release.
                options (dict): additional tweak parameters forwarded to the selected backend gear
                    and/or WebGear_RTC and the Stabilizer class.
            """
            super().__init__()  # don't forget this!

            # handle deprecated `enablePiCamera`
            if enablePiCamera is not None:
                warnings.warn(
                    "`enablePiCamera` is deprecated in WebGear_RTC; use `api=Backend.PIGEAR` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                api = Backend.PIGEAR if enablePiCamera else Backend.CAMGEAR

            # initialize global params
            self.__logging = logging
            self.__enable_inf = False  # continue frames even when video ends.
            self.is_launched = False  # check if launched already
            self.is_running = False  # check if running
            self.__stream = None

            self.__frame_size_reduction = 20  # 20% reduction
            # retrieve interpolation for reduction
            self.__interpolation = retrieve_best_interpolation(
                ["INTER_LINEAR_EXACT", "INTER_LINEAR", "INTER_AREA"]
            )

            if options:
                if "frame_size_reduction" in options:
                    value = options["frame_size_reduction"]
                    if (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and value >= 0
                        and value < 90
                    ):
                        self.__frame_size_reduction = value
                    else:
                        logger.warning("Skipped invalid `frame_size_reduction` value!")
                    del options["frame_size_reduction"]  # clean
                if "enable_infinite_frames" in options:
                    value = options["enable_infinite_frames"]
                    if isinstance(value, bool):
                        self.__enable_inf = value
                    else:
                        logger.warning(
                            "Skipped invalid `enable_infinite_frames` value!"
                        )
                    del options["enable_infinite_frames"]  # clean
                if "custom_stream" in options:
                    value = options["custom_stream"]
                    if (hasattr(value, "read") and callable(value.read)) and (
                        hasattr(value, "stop") and callable(value.stop)
                    ):
                        self.__stream = value
                        self.__logging and logger.debug(
                            "Using custom stream for its Default Internal Video-Server."
                        )
                    else:
                        raise ValueError(
                            "[WebGear_RTC:ERROR] :: Invalid `custom_stream` value. Check VidGear docs!"
                        )
                    del options["custom_stream"]  # clean

            # define VideoGear stream if not already.
            if self.__stream is None:
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
                    **options
                )

            # log it
            self.__logging and logger.debug(
                "Setting params:: Size Reduction:{}%{}".format(
                    self.__frame_size_reduction,
                    " and emulating infinite frames" if self.__enable_inf else "",
                )
            )

            # pace frames at source framerate if known (VideoGear/custom stream `framerate`)
            fps = getattr(self.__stream, "framerate", 0)
            self.__ptime = (
                1 / fps
                if isinstance(fps, (int, float)) and 0 < fps <= 120
                else VIDEO_PTIME
            )

            # FFGear can output RGB pixel layouts, everything else is BGR
            self.__rgb = api == Backend.FFGEAR and frame_format in ("rgb24", "rgba")
            if api == Backend.FFGEAR and frame_format not in (
                "bgr24",
                "rgb24",
                "bgra",
                "rgba",
                "gray",
            ):
                logger.warning(
                    "`frame_format={}` isn't supported by WebGear_RTC, use `bgr24`, `rgb24`, `bgra`, `rgba` or `gray`!".format(
                        frame_format
                    )
                )

            # initialize blank & last frame
            self.blank_frame = None
            self.__last_frame = None

            # source is read only while peers are connected
            self.has_viewers = asyncio.Event()

        def launch(self) -> None:
            """
            Launches VideoGear stream
            """
            self.__logging and logger.debug("Launching Internal RTC Video-Server")
            self.is_launched = True
            self.is_running = True
            if hasattr(self.__stream, "start") and callable(self.__stream.start):
                self.__stream.start()

        async def next_timestamp(self) -> tuple[int, fractions.Fraction]:
            """
            VideoStreamTrack internal method for generating accurate timestamp.
            """
            if self.readyState != "live":
                raise MediaStreamError
            if hasattr(self, "_timestamp"):
                self._timestamp += int(self.__ptime * VIDEO_CLOCK_RATE)
                wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
                if wait < -0.5:
                    # resync clock after a pause (no viewers, slow source) instead of bursting frames
                    self._start = time.time() - self._timestamp / VIDEO_CLOCK_RATE
                else:
                    await asyncio.sleep(wait)
            else:
                self._start = time.time()
                self._timestamp = 0
            return self._timestamp, VIDEO_TIME_BASE

        async def recv(self) -> AVFrame:
            """
            A coroutine function that yields `av.frame.Frame`.
            """
            try:
                return await self.__recv()
            except MediaStreamError:
                raise
            except Exception as e:
                # MediaRelay only stops cleanly on `MediaStreamError`
                logger.exception(str(e))
                self.terminate()
                raise MediaStreamError from e

        async def __recv(self) -> AVFrame:
            # pause reading source until a peer is connected
            await self.has_viewers.wait()

            # get next time-stamp
            pts, time_base = await self.next_timestamp()

            # read video frame off event-loop, as `read()` can block
            stream = self.__stream
            if stream is None:
                raise MediaStreamError
            f_stream = await asyncio.to_thread(stream.read)

            # display blank if NoneType
            if f_stream is None:
                if self.__last_frame is None or not self.is_running:
                    raise MediaStreamError
                if self.blank_frame is None:
                    self.blank_frame = create_blank_frame(
                        frame=self.__last_frame,
                        text="No Input" if self.__enable_inf else "The End",
                        logging=self.__logging,
                    )
                f_stream = self.blank_frame[:]
                if not self.__enable_inf:
                    self.__logging and logger.debug("Video-Stream Ended.")
                    self.terminate()
            else:
                self.__last_frame = f_stream

            # reducer frames size if specified
            if self.__frame_size_reduction:
                f_stream = await reducer(
                    f_stream,
                    percentage=self.__frame_size_reduction,
                    interpolation=self.__interpolation,
                )

            # H.264 (and most HW decoders) need even dimensions, so crop odd row/column
            height, width = f_stream.shape[:2]
            f_stream = f_stream[: height - height % 2, : width - width % 2]

            # construct `av.frame.Frame` from `numpy.nd.array`
            # based on available channels in frames
            if f_stream.ndim == 3 and f_stream.shape[-1] == 4:
                f_format = "rgba" if self.__rgb else "bgra"
            elif f_stream.ndim == 2 or (f_stream.ndim == 3 and f_stream.shape[-1] == 1):
                # drop third dimension if defined, as only `ndim==2`
                # grayscale is supported by PyAV
                f_stream = f_stream[:, :, 0] if f_stream.ndim == 3 else f_stream
                f_format = "gray"
            elif f_stream.ndim == 3 and f_stream.shape[-1] == 3:
                f_format = "rgb24" if self.__rgb else "bgr24"
            else:
                raise ValueError(
                    "Input frame of shape: {}, Isn't supported!".format(f_stream.shape)
                )

            # convert to encoders' `yuv420p` once here, as relayed frame is shared by every peer's
            # encoder thread, and concurrent `reformat()` of same frame segfaults PyAV
            frame = VideoFrame.from_ndarray(f_stream, format=f_format).reformat(
                format="yuv420p"
            )
            frame.pts = pts
            frame.time_base = time_base

            # return `av.frame.Frame`
            return frame

        def terminate(self) -> None:
            """
            Gracefully terminates VideoGear stream
            """
            if self.__stream is not None:
                # terminate running flag
                self.is_running = False
                self.__logging and logger.debug("Terminating Internal RTC Video-Server")
                # terminate
                self.__stream.stop()
                self.__stream = None
            # release paused reader, so relay sees the end
            self.has_viewers.set()


class WebGear_RTC:
    """
    WebGear_RTC is similar to WeGear API in many aspects but utilizes WebRTC technology under the hood instead of Motion JPEG, which
    makes it suitable for building powerful video-streaming solutions for all modern browsers as well as native clients available on
    all major platforms.

    WebGear_RTC is implemented with the help of aiortc library which is built on top of asynchronous I/O framework for Web Real-Time
    Communication (WebRTC) and Object Real-Time Communication (ORTC) and supports many features like SDP generation/parsing, Interactive
    Connectivity Establishment with half-trickle and mDNS support, DTLS key and certificate generation, DTLS handshake, etc.

    WebGear_RTC can handle multiple consumers seamlessly and provides native support for ICE (Interactive Connectivity Establishment)
    protocol, STUN (Session Traversal Utilities for NAT), and TURN (Traversal Using Relays around NAT) servers that help us to easily
    establish direct media connection with the remote peers for uninterrupted data flow. It also allows us to define our custom Server
    as a source to transform frames easily before sending them across the network(see this doc example).

    WebGear_RTC API works in conjunction with Starlette ASGI application and can also flexibly interact with Starlette's ecosystem of
    shared middleware, mountable applications, Response classes, Routing tables, Static Files, Templating engine(with Jinja2), etc.

    Additionally, WebGear_RTC API also provides internal wrapper around VideoGear, which itself provides internal access to both
    CamGear and PiGear APIs.
    """

    def __init__(
        self,
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
        logging: bool = False,
        time_delay: int = 0,
        # deprecated
        enablePiCamera: bool | None = None,
        **options: Any
    ):
        """
        This constructor method initializes the object state and attributes of the WebGear_RTC class.

        Parameters:
            api (Backend): selects the capture backend. Accepted values are `Backend.CAMGEAR` _(default)_,
                `Backend.PIGEAR`, and `Backend.FFGEAR`. Raises `TypeError` if an invalid value is given.
            stabilize (bool): enable access to Stabilizer Class for stabilizing frames.
            camera_num (int): [PiGear only] selects the camera module index. Must be `>= 0`.
            resolution (tuple): [PiGear only] sets `(width, height)` of the source. Default: `(640, 480)`.
            framerate (int/float): [PiGear only] sets the framerate of the source. Default: `25`.
            source (Any): [CamGear/FFGear] defines the source for the input stream (device index,
                filepath, network URL, or image-sequence glob). Default: `None`.
            stream_mode (bool): [CamGear/FFGear] enables Stream-Mode for `yt_dlp`-backed streaming URLs.
            backend (int): [CamGear only] selects the OpenCV VideoCapture backend (e.g. `cv2.CAP_DSHOW`).
            source_demuxer (str): [FFGear only] specifies the FFmpeg demuxer for the source
                (e.g. `"v4l2"`, `"dshow"`, `"avfoundation"`). Default: `None` (auto-detect).
            frame_format (str): [FFGear only] specifies the pixel layout for decoded frames
                (any FFmpeg-supported pix_fmt, e.g. `"bgr24"`, `"gray"`, `"yuv420p"`). Default: `"bgr24"`.
            custom_ffmpeg (str): [FFGear only] path to a custom FFmpeg executable. Default: `""` (use PATH).
            colorspace (str): [CamGear/PiGear only] selects the colorspace of the input stream.
            logging (bool): enables/disables logging. Default: `False`.
            time_delay (int): [CamGear/PiGear only] time delay (in seconds) before reading frames.
            enablePiCamera (bool): **DEPRECATED** — use `api=Backend.PIGEAR` instead. Will be removed
                in a future release.
            options (dict): additional tweak parameters forwarded to the selected backend gear
                and/or WebGear_RTC and the Stabilizer class.
        """
        # enable logging if specified
        self.__logging = logging if isinstance(logging, bool) else False

        # print current version
        logcurr_vidgear_ver(logging=self.__logging)

        # handle deprecated `enablePiCamera`
        if enablePiCamera is not None:
            warnings.warn(
                "`enablePiCamera` is deprecated in WebGear_RTC; use `api=Backend.PIGEAR` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            api = Backend.PIGEAR if enablePiCamera else Backend.CAMGEAR

        # raise error(s) for critical Class imports
        import_dependency_safe("starlette" if starlette is None else "")
        import_dependency_safe("aiortc" if aiortc is None else "")

        # initialize global params
        custom_data_location = ""  # path to save data-files to custom location
        data_path = ""  # path to WebGear_RTC data-files
        overwrite_default = False
        self.__relay = MediaRelay()  # fans out video-server frames to every peer
        self.__codec = []  # preferred codec capabilities, empty = negotiate
        self.__ice_servers = []  # ICE servers for both, server and browser peers

        # reformat dictionary
        options = {str(k).strip(): v for k, v in options.items()}

        # assign values to global variables if specified and valid
        if options:
            if "custom_data_location" in options:
                value = options["custom_data_location"]
                if isinstance(value, str):
                    assert os.access(
                        value, os.W_OK
                    ), "[WebGear_RTC:ERROR] :: Permission Denied!, cannot write WebGear_RTC data-files to '{}' directory!".format(
                        value
                    )
                    assert os.path.isdir(
                        os.path.abspath(value)
                    ), "[WebGear_RTC:ERROR] :: `custom_data_location` value must be the path to a directory and not to a file!"
                    custom_data_location = os.path.abspath(value)
                else:
                    logger.warning("Skipped invalid `custom_data_location` value!")
                del options["custom_data_location"]  # clean

            if "overwrite_default_files" in options:
                value = options["overwrite_default_files"]
                if isinstance(value, bool):
                    overwrite_default = value
                else:
                    logger.warning("Skipped invalid `overwrite_default_files` value!")
                del options["overwrite_default_files"]  # clean

            if "enable_live_broadcast" in options:
                value = options["enable_live_broadcast"]
                if isinstance(value, bool):
                    if value:
                        options["enable_infinite_frames"] = (
                            True  # enforce infinite frames
                        )
                        self.__logging and logger.debug(
                            "Enabled live broadcasting for Peer connection(s)."
                        )
                else:
                    logger.warning("Skipped invalid `enable_live_broadcast` value!")
                del options["enable_live_broadcast"]  # clean

            if "codec" in options:
                value = options["codec"]
                capabilities = RTCRtpSender.getCapabilities("video").codecs
                if isinstance(value, str) and value.upper() in ("VP8", "H264"):
                    # keep `rtx` too, for retransmissions
                    self.__codec = [
                        c
                        for c in capabilities
                        if c.mimeType.lower()
                        in ("video/" + value.lower(), "video/rtx")
                    ]
                else:
                    logger.warning(
                        "Skipped invalid `codec` value, use `VP8` or `H264`!"
                    )
                del options["codec"]  # clean

            if "ice_servers" in options:
                value = options["ice_servers"]
                try:
                    servers = list(value)
                    # validates keys/values via aiortc's own `RTCIceServer`
                    [RTCIceServer(**x) for x in servers]
                    self.__ice_servers = servers
                except Exception:
                    logger.warning(
                        "Skipped invalid `ice_servers` value, use list of `RTCIceServer` dicts!"
                    )
                del options["ice_servers"]  # clean

        # check if custom certificates path is specified
        if custom_data_location:
            data_path = generate_webdata(
                custom_data_location,
                c_name="webgear_rtc",
                overwrite_default=overwrite_default,
                logging=logging,
            )
        else:
            # otherwise generate suitable path
            data_path = generate_webdata(
                os.path.join(expanduser("~"), ".vidgear"),
                c_name="webgear_rtc",
                overwrite_default=overwrite_default,
                logging=logging,
            )

        # log it
        self.__logging and logger.debug(
            "`{}` is the default location for saving WebGear_RTC data-files.".format(
                data_path
            )
        )

        # define Jinja2 templates handler
        self.__templates = Jinja2Templates(directory="{}/templates".format(data_path))

        # define custom exception handlers
        self.__exception_handlers = {404: self.__not_found, 500: self.__server_error}
        # define routing tables
        self.routes = [
            Route("/", endpoint=self.__homepage),
            Route("/offer", self.__offer, methods=["POST"]),
            Route("/close_connection", self.__close_connection, methods=["POST"]),
            Mount(
                "/static",
                app=StaticFiles(directory="{}/static".format(data_path)),
                name="static",
            ),
        ]

        # define middleware support
        self.middleware = []

        # Handle RTC video server
        if "custom_stream" in options or source is not None:
            # Handle video source
            self.__default_rtc_server = RTC_VideoServer(
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
                **options
            )
        else:
            raise ValueError(
                "[WebGear_RTC:ERROR] :: Source cannot be NoneType without Custom Stream(`custom_stream`) defined!"
            )

        # copying original routing tables for further validation
        self.__rt_org_copy = self.routes[:]
        # collects peer RTC connections by id
        self.__pcs = {}

    def __call__(self) -> Starlette:
        """
        Implements a custom Callable method for WebGear_RTC application.
        """
        # validate routing tables
        assert self.routes is not None, "Routing tables are NoneType!"
        if not isinstance(self.routes, list) or not all(
            x in self.routes for x in self.__rt_org_copy
        ):
            raise RuntimeError("[WebGear_RTC:ERROR] :: Routing tables are not valid!")

        # validate middlewares
        assert self.middleware is not None, "Middlewares are NoneType!"
        if self.middleware and (
            not isinstance(self.middleware, list)
            or not all(isinstance(x, Middleware) for x in self.middleware)
        ):
            raise RuntimeError("[WebGear_RTC:ERROR] :: Middlewares are not valid!")

        # return Starlette application
        self.__logging and logger.debug("Running Starlette application.")
        return Starlette(
            debug=(bool(self.__logging)),
            routes=self.routes,
            middleware=self.middleware,
            exception_handlers=self.__exception_handlers,
            lifespan=self.__lifespan,
        )

    async def __offer(self, request):
        """
        Generates JSON Response with a WebRTC Peer Connection of Video Server.
        """
        server = self.__default_rtc_server
        if server is None or (server.is_launched and not server.is_running):
            return JSONResponse({"error": "Video stream has ended."}, status_code=503)

        # get offer from params
        try:
            params = await request.json()
            offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
            if offer.type != "offer" or not isinstance(offer.sdp, str):
                raise ValueError("not an SDP offer")
        except Exception:
            return JSONResponse({"error": "Invalid SDP offer."}, status_code=400)

        # setup RTC peer connection - interface represents a WebRTC connection
        # between the local computer and a remote peer.
        pc = RTCPeerConnection(
            RTCConfiguration(iceServers=[RTCIceServer(**x) for x in self.__ice_servers])
        )
        pc_id = uuid.uuid4().hex

        # drop peer once connection fails or closes
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            self.__logging and logger.debug(
                "Peer `{}` connection state is {}".format(pc_id, pc.connectionState)
            )
            if pc.connectionState in ("failed", "closed"):
                await self.__close_peer(pc_id)

        # every peer gets its own latest-frame proxy of video server
        proxy = self.__relay.subscribe(server, buffered=False)
        try:
            # add track (and codec preferences) before offer, so offer's video m-line gets matched to it
            sender = pc.addTrack(proxy)
            video = next(t for t in pc.getTransceivers() if t.sender == sender)
            if self.__codec:
                video.setCodecPreferences(self.__codec)
            # Change the remote description associated with the connection.
            await pc.setRemoteDescription(offer)
            if video.mid is None:
                raise ValueError("offer has no video transceiver")
            # Create & set an SDP answer to an offer received from a remote peer
            await pc.setLocalDescription(await pc.createAnswer())
        except Exception as e:
            proxy.stop()
            await pc.close()
            self.__logging and logger.warning("Rejected SDP offer: {}".format(e))
            return JSONResponse({"error": "Invalid SDP offer."}, status_code=400)

        # initiate stream
        if not server.is_launched:
            self.__logging and logger.debug("Initiating Video Streaming.")
            server.launch()
        self.__pcs[pc_id] = pc
        server.has_viewers.set()
        self.__logging and logger.info("Created WebRTC Peer Connection.")

        # return Starlette json response
        return JSONResponse(
            {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "id": pc_id,
            }
        )

    async def __close_peer(self, pc_id: str) -> None:
        """
        Closes a peer connection and its relay proxy, and pauses video server when no peers are left.
        """
        pc = self.__pcs.pop(pc_id, None)
        if pc is None:
            return
        for sender in pc.getSenders():
            sender.track and sender.track.stop()
        await pc.close()
        if not self.__pcs and self.__default_rtc_server is not None:
            self.__default_rtc_server.has_viewers.clear()

    async def __homepage(self, request):
        """
        Return an HTML index page.
        """
        return self.__templates.TemplateResponse(
            request, "index.html", {"ice_servers": self.__ice_servers}
        )

    async def __not_found(self, request, exc):
        """
        Return an HTML 404 page.
        """
        return self.__templates.TemplateResponse(request, "404.html", status_code=404)

    async def __server_error(self, request, exc):
        """
        Return an HTML 500 page.
        """
        return self.__templates.TemplateResponse(request, "500.html", status_code=500)

    async def __close_connection(self, request):
        """
        Closes the peer connection with given `id` (as returned by `/offer`).
        """
        try:
            body = await request.json()
        except ValueError:
            body = None
        pc_id = body.get("id") if isinstance(body, dict) else None
        if isinstance(pc_id, str) and pc_id in self.__pcs:
            await self.__close_peer(pc_id)
            return PlainTextResponse("OK")
        return PlainTextResponse("DISABLED")

    @contextlib.asynccontextmanager
    async def __lifespan(self, context):
        try:
            yield
        finally:
            # close peer connections
            await asyncio.gather(*(self.__close_peer(x) for x in list(self.__pcs)))
            # close Video Server
            self.shutdown()

    def shutdown(self) -> None:
        """
        Gracefully shutdown video-server
        """
        if self.__default_rtc_server is not None:
            self.__logging and logger.debug("Closing Video Server.")
            self.__default_rtc_server.terminate()
            self.__default_rtc_server = None
