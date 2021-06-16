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
import sys
import time
import fractions
import asyncio
import logging as log
from collections import deque
from starlette.routing import Mount, Route
from starlette.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse, PlainTextResponse

from aiortc.rtcrtpsender import RTCRtpSender
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.contrib.media import MediaRelay
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame


from .helper import (
    reducer,
    logger_handler,
    generate_webdata,
    create_blank_frame,
)
from ..videogear import VideoGear

# define logger
logger = log.getLogger("WebGear_RTC")
if logger.hasHandlers():
    logger.handlers.clear()
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

# add global vars
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 1 / 30  # 30fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)


class RTC_VideoServer(VideoStreamTrack):
    """
    Default Internal Video-Server for WebGear_RTC,
    a inherit-class to aiortc's VideoStreamTrack API.
    """

    def __init__(
        self,
        enablePiCamera=False,
        stabilize=False,
        source=None,
        camera_num=0,
        stream_mode=False,
        backend=0,
        colorspace=None,
        resolution=(640, 480),
        framerate=25,
        logging=False,
        time_delay=0,
        **options
    ):
        """
        This constructor method initializes the object state and attributes of the RTC_VideoServer class.

        Parameters:
            enablePiCamera (bool): provide access to PiGear(if True) or CamGear(if False) APIs respectively.
            stabilize (bool): enable access to Stabilizer Class for stabilizing frames.
            camera_num (int): selects the camera module index which will be used as Rpi source.
            resolution (tuple): sets the resolution (i.e. `(width,height)`) of the Rpi source.
            framerate (int/float): sets the framerate of the Rpi source.
            source (based on input): defines the source for the input stream.
            stream_mode (bool): controls the exclusive YouTube Mode.
            backend (int): selects the backend for OpenCV's VideoCapture class.
            colorspace (str): selects the colorspace of the input stream.
            logging (bool): enables/disables logging.
            time_delay (int): time delay (in sec) before start reading the frames.
            options (dict): provides ability to alter Tweak Parameters of WebGear_RTC, CamGear, PiGear & Stabilizer.
        """

        super().__init__()  # don't forget this!

        # initialize global params
        self.__logging = logging
        self.__enable_inf = False  # continue frames even when video ends.
        self.__frame_size_reduction = 20  # 20% reduction
        self.is_launched = False  # check if launched already
        self.__is_running = False  # check if running

        if options:
            if "frame_size_reduction" in options:
                value = options["frame_size_reduction"]
                if isinstance(value, (int, float)) and value >= 0 and value <= 90:
                    self.__frame_size_reduction = value
                else:
                    logger.warning("Skipped invalid `frame_size_reduction` value!")
                del options["frame_size_reduction"]  # clean
            if "enable_infinite_frames" in options:
                value = options["enable_infinite_frames"]
                if isinstance(value, bool):
                    self.__enable_inf = value
                else:
                    logger.warning("Skipped invalid `enable_infinite_frames` value!")
                del options["enable_infinite_frames"]  # clean

        # define VideoGear stream.
        self.__stream = VideoGear(
            enablePiCamera=enablePiCamera,
            stabilize=stabilize,
            source=source,
            camera_num=camera_num,
            stream_mode=stream_mode,
            backend=backend,
            colorspace=colorspace,
            resolution=resolution,
            framerate=framerate,
            logging=logging,
            time_delay=time_delay,
            **options
        )

        # log it
        if self.__logging:
            logger.debug(
                "Setting params:: Size Reduction:{}%{}".format(
                    self.__frame_size_reduction,
                    " and emulating infinite frames" if self.__enable_inf else "",
                )
            )

        # initialize blank frame
        self.blank_frame = None

        # handles reset signal
        self.__reset_enabled = False

    def launch(self):
        """
        Launches VideoGear stream
        """
        if self.__logging:
            logger.debug("Launching Internal RTC Video-Server")
        self.is_launched = True
        self.__is_running = True
        self.__stream.start()

    async def next_timestamp(self):
        """
        VideoStreamTrack internal method for generating accurate timestamp.
        """
        # check if ready state not live
        if self.readyState != "live":
            # otherwise reset
            self.stop()
        if hasattr(self, "_timestamp") and not self.__reset_enabled:
            self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
            wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
            await asyncio.sleep(wait)
        else:
            if self.__logging:
                logger.debug(
                    "{} timestamps".format(
                        "Resetting" if self.__reset_enabled else "Setting"
                    )
                )
            self._start = time.time()
            self._timestamp = 0
            self.__reset_enabled = False
        return self._timestamp, VIDEO_TIME_BASE

    async def recv(self):
        """
        A coroutine function that yields `av.frame.Frame`.
        """
        # get next time-stamp
        pts, time_base = await self.next_timestamp()

        # read video frame
        f_stream = None
        if self.__stream is None:
            return None
        else:
            f_stream = self.__stream.read()

        # display blank if NoneType
        if f_stream is None:
            if self.blank_frame is None or not self.__is_running:
                return None
            else:
                f_stream = self.blank_frame[:]
            if not self.__enable_inf and not self.__reset_enabled:
                if self.__logging:
                    logger.debug("Video-Stream Ended.")
                self.terminate()
        else:
            # create blank
            if self.blank_frame is None:
                self.blank_frame = create_blank_frame(
                    frame=f_stream,
                    text="No Input" if self.__enable_inf else "The End",
                    logging=self.__logging,
                )

        # reducer frames size if specified
        if self.__frame_size_reduction:
            f_stream = await reducer(f_stream, percentage=self.__frame_size_reduction)

        # construct `av.frame.Frame` from `numpy.nd.array`
        frame = VideoFrame.from_ndarray(f_stream, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base

        # return `av.frame.Frame`
        return frame

    async def reset(self):
        """
        Resets timestamp clock
        """
        self.__reset_enabled = True
        self.__is_running = False

    def terminate(self):
        """
        Gracefully terminates VideoGear stream
        """
        if not (self.__stream is None):
            # terminate running flag
            self.__is_running = False
            self.is_launched = False
            if self.__logging:
                logger.debug("Terminating Internal RTC Video-Server")
            # terminate
            self.__stream.stop()
            self.__stream = None


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
    as a source to manipulate frames easily before sending them across the network(see this doc example).

    WebGear_RTC API works in conjunction with Starlette ASGI application and can also flexibly interact with Starlette's ecosystem of
    shared middleware, mountable applications, Response classes, Routing tables, Static Files, Templating engine(with Jinja2), etc.

    Additionally, WebGear_RTC API also provides internal wrapper around VideoGear, which itself provides internal access to both
    CamGear and PiGear APIs.
    """

    def __init__(
        self,
        enablePiCamera=False,
        stabilize=False,
        source=None,
        camera_num=0,
        stream_mode=False,
        backend=0,
        colorspace=None,
        resolution=(640, 480),
        framerate=25,
        logging=False,
        time_delay=0,
        **options
    ):

        """
        This constructor method initializes the object state and attributes of the WebGear_RTC class.

        Parameters:
            enablePiCamera (bool): provide access to PiGear(if True) or CamGear(if False) APIs respectively.
            stabilize (bool): enable access to Stabilizer Class for stabilizing frames.
            camera_num (int): selects the camera module index which will be used as Rpi source.
            resolution (tuple): sets the resolution (i.e. `(width,height)`) of the Rpi source.
            framerate (int/float): sets the framerate of the Rpi source.
            source (based on input): defines the source for the input stream.
            stream_mode (bool): controls the exclusive YouTube Mode.
            backend (int): selects the backend for OpenCV's VideoCapture class.
            colorspace (str): selects the colorspace of the input stream.
            logging (bool): enables/disables logging.
            time_delay (int): time delay (in sec) before start reading the frames.
            options (dict): provides ability to alter Tweak Parameters of WebGear_RTC, CamGear, PiGear & Stabilizer.
        """

        # initialize global params
        self.__logging = logging

        custom_data_location = ""  # path to save data-files to custom location
        data_path = ""  # path to WebGear_RTC data-files
        overwrite_default = False
        self.__relay = None  # act as broadcaster

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
                        self.__relay = MediaRelay()
                        options[
                            "enable_infinite_frames"
                        ] = True  # enforce infinite frames
                        logger.critical(
                            "Enabled live broadcasting with emulated infinite frames."
                        )
                    else:
                        None
                else:
                    logger.warning("Skipped invalid `enable_live_broadcast` value!")
                del options["enable_live_broadcast"]  # clean

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
            from os.path import expanduser

            data_path = generate_webdata(
                os.path.join(expanduser("~"), ".vidgear"),
                c_name="webgear_rtc",
                overwrite_default=overwrite_default,
                logging=logging,
            )

        # log it
        if self.__logging:
            logger.debug(
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
            Route("/offer", self.__offer, methods=["GET", "POST"]),
            Mount(
                "/static",
                app=StaticFiles(directory="{}/static".format(data_path)),
                name="static",
            ),
        ]

        # define middleware support
        self.middleware = []

        # Handle RTC video server
        if source is None:
            self.config = {"server": None}
            self.__default_rtc_server = None
            if self.__logging:
                logger.warning("Given source is of NoneType!")
        else:
            # Handle video source
            self.__default_rtc_server = RTC_VideoServer(
                enablePiCamera=enablePiCamera,
                stabilize=stabilize,
                source=source,
                camera_num=camera_num,
                stream_mode=stream_mode,
                backend=backend,
                colorspace=colorspace,
                resolution=resolution,
                framerate=framerate,
                logging=logging,
                time_delay=time_delay,
                **options
            )
            # define default frame generator in configuration
            self.config = {"server": self.__default_rtc_server}
            # add exclusive reset connection node
            self.routes.append(
                Route("/close_connection", self.__reset_connections, methods=["POST"])
            )
        # copying original routing tables for further validation
        self.__rt_org_copy = self.routes[:]
        # collects peer RTC connections
        self.__pcs = set()

    def __call__(self):
        """
        Implements a custom Callable method for WebGear_RTC application.
        """
        # validate routing tables
        assert not (self.routes is None), "Routing tables are NoneType!"
        if not isinstance(self.routes, list) or not all(
            x in self.routes for x in self.__rt_org_copy
        ):
            raise RuntimeError("[WebGear_RTC:ERROR] :: Routing tables are not valid!")

        # validate middlewares
        assert not (self.middleware is None), "Middlewares are NoneType!"
        if self.middleware and (
            not isinstance(self.middleware, list)
            or not all(isinstance(x, Middleware) for x in self.middleware)
        ):
            raise RuntimeError("[WebGear_RTC:ERROR] :: Middlewares are not valid!")

        # validate assigned RTC video-server in WebGear_RTC configuration
        if isinstance(self.config, dict) and "server" in self.config:
            # check if assigned RTC server class is inherit from `VideoStreamTrack` API.i
            if self.config["server"] is None or not issubclass(
                self.config["server"].__class__, VideoStreamTrack
            ):
                # otherwise raise error
                raise ValueError(
                    "[WebGear_RTC:ERROR] :: Invalid configuration. {}. Refer Docs for more information!".format(
                        "Video-Server not assigned"
                        if self.config["server"] is None
                        else "Assigned Video-Server class must be inherit from `aiortc.VideoStreamTrack` only"
                    )
                )
            # check if assigned server class has `terminate` function defined and callable
            if not (
                hasattr(self.config["server"], "terminate")
                and callable(self.config["server"].terminate)
            ):
                # otherwise raise error
                raise ValueError(
                    "[WebGear_RTC:ERROR] :: Invalid configuration. Assigned Video-Server Class must have `terminate` method defined. Refer Docs for more information!"
                )
        else:
            # raise error if validation fails
            raise RuntimeError(
                "[WebGear_RTC:ERROR] :: Assigned configuration is invalid!"
            )
        # return Starlette application
        if self.__logging:
            logger.debug("Running Starlette application.")
        return Starlette(
            debug=(True if self.__logging else False),
            routes=self.routes,
            middleware=self.middleware,
            exception_handlers=self.__exception_handlers,
            on_shutdown=[self.__on_shutdown],
        )

    async def __offer(self, request):
        """
        Generates JSON Response with a WebRTC Peer Connection of Video Server.
        """
        # get offer from params
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        # initiate stream
        if not (self.__default_rtc_server is None) and not (
            self.__default_rtc_server.is_launched
        ):
            if self.__logging:
                logger.debug("Initiating Video Streaming.")
            self.__default_rtc_server.launch()

        # setup RTC peer connection - interface represents a WebRTC connection
        # between the local computer and a remote peer.
        pc = RTCPeerConnection()
        self.__pcs.add(pc)
        if self.__logging:
            logger.info("Created WebRTC Peer Connection.")

        # track ICE connection state changes
        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.debug("ICE connection state is %s" % pc.iceConnectionState)
            if pc.iceConnectionState == "failed":
                logger.error("ICE connection state failed!")
                await pc.close()
                self.__pcs.discard(pc)

        # Change the remote description associated with the connection.
        await pc.setRemoteDescription(offer)
        # retrieve list of RTCRtpTransceiver objects that are currently attached to the connection
        for t in pc.getTransceivers():
            # Increments performance significantly, IDK why this works as H265 codec is not even supported :D
            capabilities = RTCRtpSender.getCapabilities("video")
            preferences = list(filter(lambda x: x.name == "H265", capabilities.codecs))
            t.setCodecPreferences(preferences)
            # add video server to peer track
            if t.kind == "video":
                pc.addTrack(
                    self.__relay.subscribe(self.config["server"])
                    if not (self.__relay is None)
                    else self.config["server"]
                )

        # Create an SDP answer to an offer received from a remote peer
        answer = await pc.createAnswer()

        # Change the local description for the answer
        await pc.setLocalDescription(answer)

        # return Starlette json response
        return JSONResponse(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        )

    async def __homepage(self, request):
        """
        Return an HTML index page.
        """
        return self.__templates.TemplateResponse("index.html", {"request": request})

    async def __not_found(self, request, exc):
        """
        Return an HTML 404 page.
        """
        return self.__templates.TemplateResponse(
            "404.html", {"request": request}, status_code=404
        )

    async def __server_error(self, request, exc):
        """
        Return an HTML 500 page.
        """
        return self.__templates.TemplateResponse(
            "500.html", {"request": request}, status_code=500
        )

    async def __reset_connections(self, request):
        """
        Resets all connections and recreates VideoServer timestamps
        """
        # check if `enable_infinite_frames` is enabled
        if self.__relay is None:
            logger.critical("Resetting Server")
            # collects peer RTC connections
            coros = [pc.close() for pc in self.__pcs]
            await asyncio.gather(*coros)
            self.__pcs.clear()
            await self.__default_rtc_server.reset()
            return PlainTextResponse("OK")
        else:
            return PlainTextResponse("DISABLED")

    async def __on_shutdown(self):
        """
        Implements a Callable to be run on application shutdown
        """
        # close Video Server
        self.shutdown()
        # collects peer RTC connections
        coros = [pc.close() for pc in self.__pcs]
        await asyncio.gather(*coros)
        self.__pcs.clear()

    def shutdown(self):
        """
        Gracefully shutdown video-server
        """
        if not (self.config["server"] is None):
            if self.__logging:
                logger.debug("Closing Video Server.")
            self.config["server"].terminate()
            self.config["server"] = None
        # terminate internal server aswell.
        self.__default_rtc_server = None
