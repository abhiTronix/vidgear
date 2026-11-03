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
import subprocess
import warnings
from os.path import expanduser
from types import SimpleNamespace
from typing import Any

import numpy as np

from ..helper import (
    Backend,
    dict2Args,
    get_valid_ffmpeg_path,
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
    h264_access_units,
    reducer,
)

# safe import critical Class modules
starlette = import_dependency_safe("starlette", error="silent")
if starlette is not None:
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Mount, Route, WebSocketRoute
    from starlette.staticfiles import StaticFiles
    from starlette.templating import Jinja2Templates
simplejpeg = import_dependency_safe("simplejpeg", error="silent", min_version="1.6.1")

# define logger
logger = log.getLogger("WebGear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


class WebGear:
    """
    WebGear is a powerful ASGI Video-Broadcaster API ideal for transmitting Motion-JPEG-frames from a single source to multiple recipients via the browser.

    WebGear API works on Starlette's ASGI application and provides a highly extensible and flexible async wrapper around its complete framework. WebGear can
    flexibly interact with Starlette's ecosystem of shared middleware, mountable applications, Response classes, Routing tables, Static Files, Templating
    engine(with Jinja2), etc.

    WebGear API uses an intraframe-only compression scheme under the hood where the sequence of video-frames are first encoded as JPEG-DIB (JPEG with Device-Independent Bit compression)
    and then streamed over HTTP using Starlette's Multipart Streaming Response and a Uvicorn ASGI Server. This method imposes lower processing and memory requirements, but the quality
    is not the best, since JPEG compression is not very efficient for motion video. Therefore, WebGear by default uses a low-latency H.264 streaming mode
    (`streaming_mode="h264"`) where frames are encoded with FFmpeg and delivered over WebSockets to the browser's WebCodecs decoder, with automatic MJPEG fallback.

    In layman's terms, WebGear acts as a powerful Video Broadcaster that transmits live video-frames to any web-browser in the network. Additionally, WebGear API also provides internal
    wrapper around VideoGear, which itself provides internal access to both CamGear and PiGear APIs, thereby granting it exclusive power for transferring frames incoming from any source to the network.
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
        **options: dict
    ):
        """
        This constructor method initializes the object state and attributes of the WebGear class.

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
            custom_ffmpeg (str): [FFGear & H.264 streaming] path to a custom FFmpeg executable. Default: `""` (use PATH).
            colorspace (str): [CamGear/PiGear only] selects the colorspace of the input stream.
            logging (bool): enables/disables logging. Default: `False`.
            time_delay (int): [CamGear/PiGear only] time delay (in seconds) before reading frames.
            enablePiCamera (bool): **DEPRECATED** — use `api=Backend.PIGEAR` instead. Will be removed
                in a future release.
            options (dict): additional tweak parameters forwarded to the selected backend gear
                and/or WebGear and the Stabilizer class.
        """
        # enable logging if specified
        self.__logging = logging if isinstance(logging, bool) else False

        # print current version
        logcurr_vidgear_ver(logging=self.__logging)

        # handle deprecated `enablePiCamera`
        if enablePiCamera is not None:
            warnings.warn(
                "`enablePiCamera` is deprecated in WebGear; use `api=Backend.PIGEAR` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            api = Backend.PIGEAR if enablePiCamera else Backend.CAMGEAR

        # raise error(s) for critical Class imports
        import_dependency_safe("starlette" if starlette is None else "")
        import_dependency_safe(
            "simplejpeg" if simplejpeg is None else "", min_version="1.6.1"
        )

        # initialize global params
        self.__skip_generate_webdata = False  # generate webgear data by default
        # define frame-compression handler
        self.__jpeg_compression_quality = 90  # 90% quality
        self.__jpeg_compression_fastdct = True  # fastest DCT on by default
        self.__jpeg_compression_fastupsample = False  # fastupsample off by default
        self.__jpeg_compression_colorspace = "BGR"  # use BGR colorspace by default
        self.__frame_size_reduction = 25  # use 25% reduction
        # retrieve interpolation for reduction
        self.__interpolation = retrieve_best_interpolation(
            ["INTER_LINEAR_EXACT", "INTER_LINEAR", "INTER_AREA"]
        )

        custom_video_endpoint = ""  # custom video endpoint path
        custom_data_location = ""  # path to save data-files to custom location
        data_path = ""  # path to WebGear data-files
        overwrite_default = False
        self.__enable_inf = False  # continue frames even when video ends.
        self.__streaming_mode = ""  # low-latency "h264" by default, or "mjpeg"
        # default low-latency H.264 FFmpeg encoder parameters
        self.__h264_params = {
            "-c:v": "libx264",
            "-preset": "ultrafast",
            "-tune": "zerolatency",
            "-profile:v": "baseline",
            "-pix_fmt": "yuv420p",
        }

        # reformat dictionary
        options = {str(k).strip(): v for k, v in options.items()}

        # assign values to global variables if specified and valid
        if options:
            # check whether to disable Data-Files Auto-Generation WorkFlow
            if "skip_generate_webdata" in options:
                value = options["skip_generate_webdata"]
                # enable jpeg fastdct
                if isinstance(value, bool):
                    self.__skip_generate_webdata = value
                else:
                    logger.warning("Skipped invalid `skip_generate_webdata` value!")
                del options["skip_generate_webdata"]  # clean

            if "jpeg_compression_colorspace" in options:
                value = options["jpeg_compression_colorspace"]
                if isinstance(value, str) and value.strip().upper() in [
                    "RGB",
                    "BGR",
                    "RGBX",
                    "BGRX",
                    "XBGR",
                    "XRGB",
                    "GRAY",
                    "RGBA",
                    "BGRA",
                    "ABGR",
                    "ARGB",
                    "CMYK",
                ]:
                    # set encoding colorspace
                    self.__jpeg_compression_colorspace = value.strip().upper()
                else:
                    logger.warning(
                        "Skipped invalid `jpeg_compression_colorspace` value!"
                    )
                del options["jpeg_compression_colorspace"]  # clean

            if "jpeg_compression_quality" in options:
                value = options["jpeg_compression_quality"]
                # set valid jpeg quality
                if isinstance(value, (int, float)) and value >= 10 and value <= 100:
                    self.__jpeg_compression_quality = int(value)
                else:
                    logger.warning("Skipped invalid `jpeg_compression_quality` value!")
                del options["jpeg_compression_quality"]  # clean

            if "jpeg_compression_fastdct" in options:
                value = options["jpeg_compression_fastdct"]
                # enable jpeg fastdct
                if isinstance(value, bool):
                    self.__jpeg_compression_fastdct = value
                else:
                    logger.warning("Skipped invalid `jpeg_compression_fastdct` value!")
                del options["jpeg_compression_fastdct"]  # clean

            if "jpeg_compression_fastupsample" in options:
                value = options["jpeg_compression_fastupsample"]
                # enable jpeg  fastupsample
                if isinstance(value, bool):
                    self.__jpeg_compression_fastupsample = value
                else:
                    logger.warning(
                        "Skipped invalid `jpeg_compression_fastupsample` value!"
                    )
                del options["jpeg_compression_fastupsample"]  # clean

            if "frame_size_reduction" in options:
                value = options["frame_size_reduction"]
                if isinstance(value, (int, float)) and value >= 0 and value < 90:
                    self.__frame_size_reduction = value
                else:
                    logger.warning("Skipped invalid `frame_size_reduction` value!")
                del options["frame_size_reduction"]  # clean

            if "custom_video_endpoint" in options:
                value = options["custom_video_endpoint"]
                if value and isinstance(value, str) and value.strip().isalnum():
                    custom_video_endpoint = value.strip()
                    logging and logger.critical(
                        "Using custom video endpoint path: `/{}`".format(
                            custom_video_endpoint
                        )
                    )
                else:
                    logger.warning("Skipped invalid `custom_video_endpoint` value!")
                del options["custom_video_endpoint"]  # clean

            if "custom_data_location" in options:
                value = options["custom_data_location"]
                if value and isinstance(value, str):
                    assert os.access(
                        value, os.W_OK
                    ), "[WebGear:ERROR] :: Permission Denied!, cannot write WebGear data-files to '{}' directory!".format(
                        value
                    )
                    assert os.path.isdir(
                        os.path.abspath(value)
                    ), "[WebGear:ERROR] :: `custom_data_location` value must be the path to a directory and not to a file!"
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

            if "enable_infinite_frames" in options:
                value = options["enable_infinite_frames"]
                if isinstance(value, bool):
                    self.__enable_inf = value
                else:
                    logger.warning("Skipped invalid `enable_infinite_frames` value!")
                del options["enable_infinite_frames"]  # clean

            if "streaming_mode" in options:
                value = options["streaming_mode"]
                if isinstance(value, str) and value.strip().lower() in ["mjpeg", "h264"]:
                    self.__streaming_mode = value.strip().lower()
                else:
                    logger.warning("Skipped invalid `streaming_mode` value!")
                del options["streaming_mode"]  # clean

            if "h264_options" in options:
                value = options["h264_options"]
                if isinstance(value, dict):
                    # user-defined FFmpeg output parameters override defaults
                    self.__h264_params.update(
                        {str(k).strip(): v for k, v in value.items()}
                    )
                else:
                    logger.warning("Skipped invalid `h264_options` value!")
                del options["h264_options"]  # clean

        # validate low-latency H.264 streaming prerequisites
        self.__ffmpeg = ""
        explicit_h264 = self.__streaming_mode == "h264"
        self.__streaming_mode = self.__streaming_mode or "h264"
        if self.__streaming_mode == "h264":
            if source is None:
                # custom generators yield ready-made JPEG bytes, not frames
                (logger.warning if explicit_h264 else logger.debug)(
                    "H.264 streaming needs a valid `source`. Falling back to MJPEG streaming!"
                )
                self.__streaming_mode = "mjpeg"
            else:
                self.__ffmpeg = get_valid_ffmpeg_path(
                    custom_ffmpeg, os.name == "nt", logging=self.__logging
                )
                if not self.__ffmpeg:
                    logger.warning(
                        "Failed to find valid FFmpeg executable. Falling back to MJPEG streaming!"
                    )
                    self.__streaming_mode = "mjpeg"
        self.__video_endpoint = "/{}".format(
            custom_video_endpoint if custom_video_endpoint else "video"
        )

        # check if disable Data-Files Auto-Generation WorkFlow is disabled
        if not self.__skip_generate_webdata:
            # check if custom data path is specified
            if custom_data_location:
                data_path = generate_webdata(
                    custom_data_location,
                    c_name="webgear",
                    overwrite_default=overwrite_default,
                    logging=logging,
                )
            else:
                # otherwise generate suitable path
                data_path = generate_webdata(
                    os.path.join(expanduser("~"), ".vidgear"),
                    c_name="webgear",
                    overwrite_default=overwrite_default,
                    logging=logging,
                )

            # log it
            self.__logging and logger.debug(
                "`{}` is the default location for saving WebGear data-files.".format(
                    data_path
                )
            )
            # define Jinja2 templates handler
            self.__templates = Jinja2Templates(
                directory="{}/templates".format(data_path)
            )
            # define routing tables
            self.routes = [
                Route("/", endpoint=self.__homepage),
                Route(self.__video_endpoint, endpoint=self.__video),
                Mount(
                    "/static",
                    app=StaticFiles(directory="{}/static".format(data_path)),
                    name="static",
                ),
            ]
        else:
            # log it
            self.__logging and logger.critical(
                "WebGear Data-Files Auto-Generation WorkFlow has been manually disabled."
            )
            # define routing tables
            self.routes = [Route(self.__video_endpoint, endpoint=self.__video)]
            # log exceptions
            self.__logging and logger.warning(
                "Only `{}` route is available for this instance.".format(
                    self.__video_endpoint
                )
            )
        # add low-latency H.264 WebSocket route (MJPEG route stays as fallback)
        if self.__streaming_mode == "h264":
            self.routes.append(
                WebSocketRoute(
                    "/ws{}".format(self.__video_endpoint), endpoint=self.__ws_video
                )
            )
            self.__logging and logger.debug(
                "Enabling low-latency H.264 streaming at `/ws{}` with FFmpeg parameters: `{}`.".format(
                    self.__video_endpoint, self.__h264_params
                )
            )

        # define custom exception handlers
        self.__exception_handlers = {404: self.__not_found, 500: self.__server_error}
        # define middleware support
        self.middleware = []
        # Handle video source
        if source is None:
            self.config = {"generator": None}
            self.__stream = None
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
                **options
            )
            # define default frame generator in configuration
            self.config = {"generator": self.__producer}

        # log if specified
        if self.__logging:
            if source is None:
                logger.warning(
                    "Given source is of NoneType. Therefore, JPEG Frame-Compression is disabled!"
                )
            else:
                logger.debug(
                    "Enabling JPEG Frame-Compression with Colorspace:`{}`, Quality:`{}`%, Fastdct:`{}`, and Fastupsample:`{}`.".format(
                        self.__jpeg_compression_colorspace,
                        self.__jpeg_compression_quality,
                        "enabled" if self.__jpeg_compression_fastdct else "disabled",
                        (
                            "enabled"
                            if self.__jpeg_compression_fastupsample
                            else "disabled"
                        ),
                    )
                )

        # copying original routing tables for further validation
        self.__rt_org_copy = self.routes[:]
        # initialize blank frame
        self.blank_frame = None
        # keeps check if producer loop should be running
        self.__isrunning = True

    def __call__(self) -> Starlette:
        """
        Implements a custom Callable method for WebGear application.
        """
        # validate routing tables
        assert self.routes is not None, "Routing tables are NoneType!"
        if not isinstance(self.routes, list) or not all(
            x in self.routes for x in self.__rt_org_copy
        ):
            raise RuntimeError("[WebGear:ERROR] :: Routing tables are not valid!")

        # validate middlewares
        assert self.middleware is not None, "Middlewares are NoneType!"
        if self.middleware and (
            not isinstance(self.middleware, list)
            or not all(isinstance(x, Middleware) for x in self.middleware)
        ):
            raise RuntimeError("[WebGear:ERROR] :: Middlewares are not valid!")

        # validate assigned frame generator in WebGear configuration
        if isinstance(self.config, dict) and "generator" in self.config:
            # check if its  assigned value is a asynchronous generator
            if self.config["generator"] is None or not inspect.isasyncgen(
                self.config["generator"]()
            ):
                # otherwise raise error
                raise ValueError(
                    "[WebGear:ERROR] :: Invalid configuration. Assigned generator must be a asynchronous generator function/method only!"
                )
        else:
            # raise error if validation fails
            raise RuntimeError("[WebGear:ERROR] :: Assigned configuration is invalid!")

        # initiate stream
        self.__logging and logger.debug("Initiating Video Streaming.")
        if self.__stream is not None:
            self.__stream.start()
        # return Starlette application
        self.__logging and logger.debug("Running Starlette application.")
        return Starlette(
            debug=(bool(self.__logging)),
            routes=self.routes,
            middleware=self.middleware,
            exception_handlers=self.__exception_handlers,
            lifespan=self.__lifespan,
        )

    async def __frames(self):
        """
        WebGear's asynchronous raw frame generator, shared by MJPEG and H.264 streaming.
        """
        # loop over frames
        while self.__isrunning:
            # read frame
            frame = self.__stream.read()

            # display blank if NoneType
            if frame is None:
                frame = (
                    self.blank_frame
                    if self.blank_frame is None
                    else self.blank_frame[:]
                )
                if not self.__enable_inf:
                    self.__isrunning = False
            else:
                # create blank
                if self.blank_frame is None:
                    self.blank_frame = create_blank_frame(
                        frame=frame,
                        text="No Input" if self.__enable_inf else "The End",
                        logging=self.__logging,
                    )

            # reducer frames size if specified
            if self.__frame_size_reduction:
                frame = await reducer(
                    frame,
                    percentage=self.__frame_size_reduction,
                    interpolation=self.__interpolation,
                )
            yield frame

    async def __producer(self):
        """
        WebGear's default asynchronous frame producer/generator.
        """
        async for frame in self.__frames():
            # handle JPEG encoding
            if self.__jpeg_compression_colorspace == "GRAY":
                if frame.ndim == 2:
                    # patch for https://gitlab.com/jfolz/simplejpeg/-/issues/11
                    frame = np.expand_dims(frame, axis=2)
                encodedImage = simplejpeg.encode_jpeg(
                    frame,
                    quality=self.__jpeg_compression_quality,
                    colorspace=self.__jpeg_compression_colorspace,
                    fastdct=self.__jpeg_compression_fastdct,
                )
            else:
                encodedImage = simplejpeg.encode_jpeg(
                    frame,
                    quality=self.__jpeg_compression_quality,
                    colorspace=self.__jpeg_compression_colorspace,
                    colorsubsampling="422",
                    fastdct=self.__jpeg_compression_fastdct,
                )

            # yield frame in byte format
            yield (
                b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" + encodedImage + b"\r\n"
            )
            # sleep for sometime.
            await asyncio.sleep(0)

    async def __video(self, scope):
        """
        Returns a async video streaming response.
        """
        assert scope["type"] in ["http", "https"]
        return StreamingResponse(
            self.config["generator"](),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    async def __ws_video(self, websocket):
        """
        Streams low-latency H.264 Annex-B access units over WebSocket, encoded with FFmpeg.
        """
        await websocket.accept()
        frames = self.__frames()
        frame = await anext(frames, None)
        if frame is None:
            await websocket.close()
            return
        # yuv420p needs even dimensions
        height, width = frame.shape[:2]
        height, width = height - height % 2, width - width % 2
        fps = getattr(self.__stream, "framerate", 0) or 30
        pix_fmt = (
            "gray"
            if frame.ndim == 2 or self.__jpeg_compression_colorspace == "GRAY"
            else {
                "RGB": "rgb24",
                "RGBA": "rgba",
                "BGRA": "bgra",
                "ARGB": "argb",
                "ABGR": "abgr",
                "RGBX": "rgb0",
                "BGRX": "bgr0",
                "XRGB": "0rgb",
                "XBGR": "0bgr",
            }.get(self.__jpeg_compression_colorspace, "bgr24")
        )
        # ponytail: one encoder (+2 pipe threads) per client, share one with forced IDR on join if many viewers
        # blocking pipes on threads, since asyncio subprocesses need Proactor loop on Windows
        process = subprocess.Popen(
            [
                self.__ffmpeg,
                *["-hide_banner", "-loglevel", "error"],
                *["-f", "rawvideo", "-pix_fmt", pix_fmt],
                *["-s", "{}x{}".format(width, height), "-framerate", str(fps)],
                *["-i", "-"],
                *dict2Args({"-g": int(fps * 2), **self.__h264_params}),
                *["-flush_packets", "1", "-f", "h264", "-"],
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        def write(data):
            process.stdin.write(data)
            process.stdin.flush()

        async def feeder(frame):
            # pace raw frames at source framerate, EOF lets FFmpeg flush
            loop = asyncio.get_running_loop()
            deadline = loop.time()
            try:
                while frame is not None:
                    await asyncio.to_thread(write, frame[:height, :width].tobytes())
                    deadline += 1 / fps
                    await asyncio.sleep(max(0, deadline - loop.time()))
                    frame = await anext(frames, None)
            except OSError:
                pass  # FFmpeg gone (broken pipe)
            finally:
                with contextlib.suppress(OSError):
                    process.stdin.close()

        # `read1` returns whatever is available instead of waiting for full chunk
        reader = SimpleNamespace(
            read=lambda n: asyncio.to_thread(process.stdout.read1, n)
        )

        feeding = asyncio.create_task(feeder(frame))
        try:
            async for unit in h264_access_units(reader):
                await websocket.send_bytes(unit)
        except Exception as e:
            # client gone mid-stream
            self.__logging and logger.debug("H.264 client disconnected: {}".format(e))
        finally:
            # reap before any await (handler may be cancelled), killing also unblocks pipe threads
            process.poll() is None and process.kill()
            process.wait()
            process.stdout.close()
            feeding.cancel()
            await asyncio.gather(feeding, return_exceptions=True)
            await frames.aclose()
            with contextlib.suppress(Exception):
                await websocket.close()

    async def __homepage(self, request):
        """
        Returns an HTML index page.
        """
        return (
            self.__templates.TemplateResponse(
                request,
                "index.html",
                {
                    "streaming_mode": self.__streaming_mode,
                    "video_endpoint": self.__video_endpoint,
                },
            )
            if not self.__skip_generate_webdata
            else JSONResponse(
                {
                    "detail": "MESSAGE : WebGear Data-Files Auto-Generation WorkFlow is disabled!"
                },
                status_code=404,
            )
        )

    async def __not_found(self, request, exc):
        """
        Returns an HTML 404 page.
        """
        return (
            self.__templates.TemplateResponse(request, "404.html", status_code=404)
            if not self.__skip_generate_webdata
            else JSONResponse(
                {
                    "detail": "ERROR : {} :: MESSAGE : WebGear Data-Files Auto-Generation WorkFlow is disabled.".format(
                        exc.detail
                    )
                },
                status_code=404,
            )
        )

    async def __server_error(self, request, exc):
        """
        Returns an HTML 500 page.
        """
        return (
            self.__templates.TemplateResponse(request, "500.html", status_code=500)
            if not self.__skip_generate_webdata
            else JSONResponse(
                {
                    "detail": "ERROR : {} :: MESSAGE : WebGear Data-Files Auto-Generation WorkFlow is disabled.".format(
                        exc.detail if hasattr(exc, "detail") else repr(exc)
                    )
                },
                status_code=500,
            )
        )

    @contextlib.asynccontextmanager
    async def __lifespan(self, context):
        try:
            yield
        finally:
            # close Video Server
            self.shutdown()

    def shutdown(self) -> None:
        """
        Implements a Callable to be run on application shutdown
        """
        if self.__stream is not None:
            self.__logging and logger.debug("Closing Video Streaming.")
            # stops producer
            self.__isrunning = False
            # stops VideoGear stream
            self.__stream.stop()
            # prevent any re-iteration
            self.__stream = None
