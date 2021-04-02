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
import asyncio
import logging as log
from collections import deque
from starlette.routing import Mount, Route
from starlette.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from starlette.applications import Starlette
from starlette.responses import JSONResponse

from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame


from .helper import (
    reducer,
    logger_handler,
    generate_webdata,
    create_blank_frame,
)
from ..videogear import VideoGear

# define logger
logger = log.getLogger("WeGear_RTC")
if logger.hasHandlers():
    logger.handlers.clear()
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


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
            options (dict): provides ability to alter Tweak Parameters of WeGear_RTC, CamGear, PiGear & Stabilizer.
        """

        super().__init__()  # don't forget this!

        # initialize global params
        self.__logging = logging
        self.__enable_inf = False  # continue frames even when video ends.
        self.__frame_size_reduction = 10  # 20% reduction

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
                "Setting params:: Size Reduction:{}%".format(
                    self.__frame_size_reduction,
                )
            )

        # initialize blank frame
        self.blank_frame = None

    def launch(self):
        """
        Launches VideoGear stream
        """
        if self.__logging:
            logger.debug("Launching Internal RTC Video-Server")
        self.__stream.start()

    async def recv(self):
        """
        A coroutine function that yields `av.frame.Frame`.
        """
        # get next timestamp
        pts, time_base = await self.next_timestamp()

        # read video frame
        f_stream = None
        if self.__stream is None:
            return None
        else:
            f_stream = self.__stream.read()

        # display blank if NoneType
        if f_stream is None:
            if self.blank_frame is None:
                return None
            else:
                f_stream = self.blank_frame[:]
            if not self.__enable_inf:
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

        # contruct `av.frame.Frame` from `numpy.nd.array`
        frame = VideoFrame.from_ndarray(f_stream, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base

        # return `av.frame.Frame`
        return frame

    def terminate(self):
        """
        Gracefully terminates VideoGear stream
        """
        # log
        if not (self.__stream is None):
            if self.__logging:
                logger.debug("Terminating Internal RTC Video-Server")
            # terminate
            self.__stream.stop()
            self.__stream = None


class WebGear_RTC:

    """
    WebGear_RTC is similar to WeGear API in all aspects but utilizes WebRTC standard instead of Motion JPEG for streaming, that makes it possible to
    share data and perform teleconferencing peer-to-peer, without requiring that the user install plug-ins or any other third-party software
    and thereby is the most compatible API with all modern browsers.

    WebGear_RTC is primarily built upon `aiortc` - a library for Web Real-Time Communication (WebRTC) and Object Real-Time Communication (ORTC) in Python.
    aiortc allows us to exchange audio, video and data channels and interoperability is regularly tested against both Chrome and Firefox.

    Similar to WeGear API, WebGear_RTC API also additionally provides highly extensible and flexible asyncio wrapper around Starlette ASGI application, and
    provides easy access to its complete framework. It can flexibly interact with the Starlette's ecosystem of shared middleware and mountable applications,
    and its various Response classes, Routing tables, Static Files, Templating engine(with Jinja2), etc. It provides a special internal wrapper around
    VideoGear API, which itself provides internal access to both CamGear and PiGear APIs thereby granting it exclusive power for streaming frames incoming
    from any device/source, such as streaming Stabilization enabled Video in real-time.
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
        This constructor method initializes the object state and attributes of the WeGear_RTC class.

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
            options (dict): provides ability to alter Tweak Parameters of WeGear_RTC, CamGear, PiGear & Stabilizer.
        """

        # initialize global params
        self.__logging = logging

        custom_data_location = ""  # path to save data-files to custom location
        data_path = ""  # path to WeGear_RTC data-files
        overwrite_default = False

        # reformat dictionary
        options = {str(k).strip(): v for k, v in options.items()}

        # assign values to global variables if specified and valid
        if options:
            if "custom_data_location" in options:
                value = options["custom_data_location"]
                if isinstance(value, str):
                    assert os.access(
                        value, os.W_OK
                    ), "[WeGear_RTC:ERROR] :: Permission Denied!, cannot write WeGear_RTC data-files to '{}' directory!".format(
                        value
                    )
                    assert os.path.isdir(
                        os.path.abspath(value)
                    ), "[WeGear_RTC:ERROR] :: `custom_data_location` value must be the path to a directory and not to a file!"
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
                "`{}` is the default location for saving WeGear_RTC data-files.".format(
                    data_path
                )
            )

        # define Jinja2 templates handler
        self.__templates = Jinja2Templates(directory="{}/templates".format(data_path))

        # define custom exception handlers
        self.__exception_handlers = {404: self.__server_error, 500: self.__server_error}
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

        # copying original routing tables for further validation
        self.__rt_org_copy = self.routes[:]
        # keeps check if producer loop should be running
        self.__isrunning = True
        # collects peer RTC connections
        self.__pcs = set()

    def __call__(self):
        """
        Implements a custom Callable method for WeGear_RTC application.
        """
        # validate routing tables
        # validate routing tables
        assert not (self.routes is None), "Routing tables are NoneType!"
        if not isinstance(self.routes, list) or not all(
            x in self.routes for x in self.__rt_org_copy
        ):
            raise RuntimeError("[WeGear_RTC:ERROR] :: Routing tables are not valid!")

        # validate assigned RTC video-server in WeGear_RTC configuration
        if isinstance(self.config, dict) and "server" in self.config:
            # check if its  assigned server is inherit from `VideoStreamTrack` API.i
            if self.config["server"] is None or not issubclass(
                self.config["server"].__class__, VideoStreamTrack
            ):
                # otherwise raise error
                raise ValueError(
                    "[WeGear_RTC:ERROR] :: Invalid configuration. Assigned server must be inherit from aiortc's `VideoStreamTrack` Class only!"
                )
        else:
            # raise error if validation fails
            raise RuntimeError(
                "[WeGear_RTC:ERROR] :: Assigned configuration is invalid!"
            )

        # initiate stream
        if self.__logging:
            logger.debug("Initiating Video Streaming.")
        if not (self.__default_rtc_server is None):
            self.__default_rtc_server.launch()
        # return Starlette application
        if self.__logging:
            logger.debug("Running Starlette application.")
        return Starlette(
            debug=(True if self.__logging else False),
            routes=self.routes,
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
            if t.kind == "video":
                pc.addTrack(self.config["server"])

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

    async def __on_shutdown(self):
        # close Video Server
        if not (self.__default_rtc_server is None):
            if self.__logging:
                logger.debug("Closing Video Streaming.")
            self.__default_rtc_server.terminate()
        # collects peer RTC connections
        coros = [pc.close() for pc in self.__pcs]
        await asyncio.gather(*coros)
        self.__pcs.clear()