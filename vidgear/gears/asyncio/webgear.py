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
from starlette.applications import Starlette
from starlette.responses import StreamingResponse
from starlette.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from starlette.routing import Mount
from starlette.routing import Route

from ..videogear import VideoGear
from .helper import reducer
from .helper import logger_handler
from .helper import generate_webdata
from collections import deque

import logging as log
import os, cv2, asyncio, sys


# define logger
logger = log.getLogger("WebGear")
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


class WebGear:

    """
    WebGear is a powerful ASGI Video-streamer API, that is built upon Starlette - a lightweight ASGI framework/toolkit, which is ideal 
    for building high-performance asyncio services.

    WebGear API provides a flexible but robust asyncio wrapper around Starlette ASGI application and can easily access its various components 
    independently. Thereby providing it the ability to interact with the Starlette's ecosystem of shared middleware and mountable applications 
    & seamless access to its various Response classes, Routing tables, Static Files, Templating engine(with Jinja2), etc.

    WebGear acts as robust Live Video Streaming Server that can stream live video frames to any web browser on a network in real-time. 
    It addition to this, WebGear provides a special internal wrapper around VideoGear API, which itself provides internal access to both CamGear and PiGear 
    APIs thereby granting it exclusive power for streaming frames incoming from any device/source. Also on the plus side, since WebGear has access 
    to all functions of VideoGear API, therefore it can stabilize video frames even while streaming live.


    WebGear specific parameters:

    - ** options:(dict): can be used in addition, to pass user-defined parameter to WebGear API in the form of python dictionary. Supported WebGear 
                        dictionary attributes are:

        - `custom_data_location` (str): Can be used to change/alter `default location` path to somewhere else. 

        - `overwrite_default_files` (bool): Can be used to force trigger the `Auto-generation process` to overwrite existing data-files. Remember only downloaded 
                                            files will be overwritten in this process, and any other file/folder will NOT be affected/overwritten.

        - `frame_size_reduction`: (int/float) This attribute controls the size reduction (in percentage) of the frame to be streamed on Server. 
                                                Its value can be between 0-90, and the recommended value is 40. 

        - JPEG Encoding Parameters: In WebGear, the input video frames are first encoded into Motion JPEG (M-JPEG or MJPEG) video compression format in which 
                                each video frame or interlaced field of a digital video sequence is compressed separately as a JPEG image, before sending onto 
                                a server. Therefore, WebGear API provides various attributes to have full control over JPEG encoding performance and quality, 
                                which are as follows:

            - `frame_jpeg_quality`(int) It controls the JPEG encoder quality and value varies from 0 to 100 (the higher is the better quality but 
                                            performance will be lower). Its default value is 95. 

            - `frame_jpeg_optimize`(bool) It enables various JPEG compression optimizations such as Chroma subsampling, Quantization table, etc. 
                                        Its default value is `False`. 

            - `frame_jpeg_progressive`(bool) It enables Progressive JPEG encoding instead of the Baseline. Progressive Mode. 
                                            Its default value is `False` means baseline mode. 


    VideoGear Specific parameters for WebGear:
    
        :param (boolean) enablePiCamera: set this flag to access PiGear or CamGear class respectively. 
                                        / This means the if enablePiCamera flag is `True`, PiGear class will be accessed 
                                        / and if `False`, the camGear Class will be accessed. Its default value is False.

        :param (boolean) stabilize: set this flag to enable access to VidGear's Stabilizer Class. This basically enables(if True) or disables(if False) 
                                        video stabilization in VidGear. Its default value is False.

        :param (dict) **options: can be used in addition, to pass parameter supported by VidGear's stabilizer class.
                                / Supported dict keys are: 
                                    - `SMOOTHING_RADIUS` (int) : to alter averaging window size. It handles the quality of stabilization at expense of latency and sudden panning. 
                                                            / Larger its value, less will be panning, more will be latency and vice-versa. It's default value is 30.
                                    - `BORDER_SIZE` (int) : to alter output border cropping. It's will crops the border to reduce the black borders from stabilization being too noticeable. 
                                                            / Larger its value, more will be cropping. It's default value is 0 (i.e. no cropping).          
                                    - `BORDER_TYPE` (string) : to change the border mode. Valid border types are 'black', 'reflect', 'reflect_101', 'replicate' and 'wrap'. It's default value is 'black'
    

    CamGear Specific supported parameters for WebGear:

        :param source : take the source value for CamGear Class. Its default value is 0. Valid Inputs are:
            - Index(integer): Valid index of the video device.
            - YouTube Url(string): Youtube URL as input.
            - Network_Stream_Address(string): Incoming Stream Valid Network address. 
            - GStreamer (string) videostream Support
        :param (boolean) y_tube: enables YouTube Mode in CamGear Class, i.e If enabled the class will interpret the given source string as YouTube URL. 
                                / Its default value is False.
        :param (int) backend: set the backend of the video stream (if specified). Its default value is 0.


    PiGear Specific supported parameters for WebGear:
        :param (integer) camera_num: selects the camera module index that will be used by API. 
                                /   Its default value is 0 and shouldn't be altered until unless 
                                /   if you using Raspberry Pi 3/3+ compute module in your project along with multiple camera modules. 
                                /   Furthermore, Its value can only be greater than zero, otherwise, it will throw ValueError for any negative value.
        :param (tuple) resolution: sets the resolution (width,height) in Picamera class. Its default value is (640,480).
        :param (integer) framerate: sets the framerate in Picamera class. Its default value is 25.


    Common parameters for WebGear: 
        :param (string) colorspace: set colorspace of the video stream. Its default value is None.
        :param (dict) **options: parameter supported by various API (whichever being accessed).
        :param (boolean) logging: set this flag to enable/disable error logging essential for debugging. Its default value is False.
        :param (integer) time_delay: sets time delay(in seconds) before start reading the frames. 
                            / This delay is essentially required for camera to warm-up. 
                            / Its default value is 0.
    """

    def __init__(
        self,
        enablePiCamera=False,
        stabilize=False,
        source=0,
        camera_num=0,
        y_tube=False,
        backend=0,
        colorspace=None,
        resolution=(640, 480),
        framerate=25,
        logging=False,
        time_delay=0,
        **options
    ):

        # initialize global params
        self.__jpeg_quality = 90  # 90% quality
        self.__jpeg_optimize = 0  # optimization off
        self.__jpeg_progressive = 0  # jpeg will be baseline instead
        self.__frame_size_reduction = 20  # 20% reduction
        self.__logging = logging

        custom_data_location = ""  # path to save data-files to custom location
        data_path = ""  # path to WebGear data-files
        overwrite_default = False

        # reformat dictionary
        options = {str(k).strip(): v for k, v in options.items()}

        # assign values to global variables if specified and valid
        if options:
            if "frame_size_reduction" in options:
                value = options["frame_size_reduction"]
                if isinstance(value, (int, float)) and value >= 0 and value <= 90:
                    self.__frame_size_reduction = value
                else:
                    logger.warning("Skipped invalid `frame_size_reduction` value!")
                del options["frame_size_reduction"]  # clean
            if "frame_jpeg_quality" in options:
                value = options["frame_jpeg_quality"]
                if isinstance(value, (int, float)) and value >= 10 and value <= 95:
                    self.__jpeg_quality = int(value)
                else:
                    logger.warning("Skipped invalid `frame_jpeg_quality` value!")
                del options["frame_jpeg_quality"]  # clean
            if "frame_jpeg_optimize" in options:
                value = options["frame_jpeg_optimize"]
                if isinstance(value, bool):
                    self.__jpeg_optimize = int(value)
                else:
                    logger.warning("Skipped invalid `frame_jpeg_optimize` value!")
                del options["frame_jpeg_optimize"]  # clean
            if "frame_jpeg_progressive" in options:
                value = options["frame_jpeg_progressive"]
                if isinstance(value, bool):
                    self.__jpeg_progressive = int(value)
                else:
                    logger.warning("Skipped invalid `frame_jpeg_progressive` value!")
                del options["frame_jpeg_progressive"]  # clean
            if "custom_data_location" in options:
                value = options["custom_data_location"]
                if isinstance(value, str):
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

        # define stream with necessary params
        self.stream = VideoGear(
            enablePiCamera=enablePiCamera,
            stabilize=stabilize,
            source=source,
            camera_num=camera_num,
            y_tube=y_tube,
            backend=backend,
            colorspace=colorspace,
            resolution=resolution,
            framerate=framerate,
            logging=logging,
            time_delay=time_delay,
            **options
        )

        loop = asyncio.get_event_loop()
        # check if custom certificates path is specified
        try:
            if custom_data_location:
                data_path = loop.run_until_complete(
                    generate_webdata(
                        custom_data_location,
                        overwrite_default=overwrite_default,
                        logging=logging,
                    )
                )
            else:
                # otherwise generate suitable path
                from os.path import expanduser

                data_path = loop.run_until_complete(
                    generate_webdata(
                        os.path.join(expanduser("~"), ".vidgear"),
                        overwrite_default=overwrite_default,
                        logging=logging,
                    )
                )
        except Exception as err:
            if isinstance(err, asyncio.CancelledError):
                logger.critical("WebGear Auto-generation terminated")
            else:
                logger.error(str(err))
            raise RuntimeError("Failed to generate webdata!")
        loop.stop()

        # log it
        if self.__logging:
            logger.debug(
                "`{}` is the default location for saving WebGear data-files.".format(
                    data_path
                )
            )
        if self.__logging:
            logger.debug(
                "Setting params:: Size Reduction:{}%, JPEG quality:{}%, JPEG optimizations:{}, JPEG progressive:{}".format(
                    self.__frame_size_reduction,
                    self.__jpeg_quality,
                    bool(self.__jpeg_optimize),
                    bool(self.__jpeg_progressive),
                )
            )

        # define Jinja2 templates handler
        self.__templates = Jinja2Templates(directory="{}/templates".format(data_path))

        # define custom exception handlers
        self.__exception_handlers = {404: self.__not_found, 500: self.__server_error}
        # define routing tables
        self.routes = [
            Route("/", endpoint=self.__homepage),
            Route("/video", endpoint=self.__video),
            Mount(
                "/static",
                app=StaticFiles(directory="{}/static".format(data_path)),
                name="static",
            ),
        ]
        # copying original routing tables for further validation
        self.__rt_org_copy = self.routes[:]
        # keeps check if producer loop should be running
        self.__isrunning = True

    def __call__(self):
        """
        Implements custom callable method
        """
        # validate routing tables
        assert not (self.routes is None), "Routing tables are NoneType!"
        if not isinstance(self.routes, list) or not all(
            x in self.routes for x in self.__rt_org_copy
        ):
            raise RuntimeError("Routing tables are not valid!")
        # initiate stream
        if self.__logging:
            logger.debug("Initiating Video Streaming.")
        self.stream.start()
        # return Starlette application
        if self.__logging:
            logger.debug("Running Starlette application.")
        return Starlette(
            debug=(True if self.__logging else False),
            routes=self.routes,
            exception_handlers=self.__exception_handlers,
            on_shutdown=[self.shutdown],
        )

    async def __producer(self):
        """
        Implements async frame producer.
        """
        # loop over frames
        while self.__isrunning:
            # read frame
            frame = self.stream.read()
            # break if NoneType
            if frame is None:
                break
            # reducer frames size if specified
            if self.__frame_size_reduction:
                frame = await reducer(frame, percentage=self.__frame_size_reduction)
            # handle JPEG encoding
            encodedImage = cv2.imencode(
                ".jpg",
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    self.__jpeg_quality,
                    cv2.IMWRITE_JPEG_PROGRESSIVE,
                    self.__jpeg_progressive,
                    cv2.IMWRITE_JPEG_OPTIMIZE,
                    self.__jpeg_optimize,
                ],
            )[1].tobytes()
            # yield frame in byte format
            yield (
                b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" + encodedImage + b"\r\n"
            )
            await asyncio.sleep(0.01)

    async def __video(self, scope):
        """
        Return a async video streaming response.
        """
        assert scope["type"] == "http"
        return StreamingResponse(
            self.__producer(), media_type="multipart/x-mixed-replace; boundary=frame"
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

    def shutdown(self):
        """
        Implements a callable to run on application shutdown
        """
        if not (self.stream is None):
            if self.__logging:
                logger.debug("Closing Video Streaming.")
            # stops producer
            self.__isrunning = False
            # stops VideoGear stream
            self.stream.stop()
            # prevent any re-iteration
            self.stream = None