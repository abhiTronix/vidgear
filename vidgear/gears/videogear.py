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
import logging as log
import warnings
from typing import Any, TypeVar

from numpy.typing import NDArray

# import additional API(s)
from .camgear import CamGear

# import helper packages
from .helper import Backend, logcurr_vidgear_ver, logger_handler

# define logger
logger = log.getLogger("VideoGear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

# Type variable `T` representing class `VideoGear`.
T = TypeVar("T", bound="VideoGear")


class VideoGear:
    """
    VideoGear API provides a special internal wrapper around VidGear's exclusive Video Stabilizer class.
    VideoGear also acts as a Common Video-Capture API that provides internal access for both CamGear and PiGear APIs and
    their parameters with an exclusive enablePiCamera boolean flag.

    VideoGear is ideal when you need to switch to different video sources without changing your code much. Also, it enables
    easy stabilization for various video-streams (real-time or not)
    with minimum effort and writing way fewer lines of code.
    """

    def __init__(
        self,
        # VideoGear parameters
        api: Backend = Backend.CAMGEAR,
        stabilize: bool = False,
        # PiGear parameters
        camera_num: int = 0,
        resolution: tuple[int, int] = (640, 480),
        framerate: int | float = 30,
        # CamGear/FFGear parameters
        source: Any = 0,
        stream_mode: bool = False,
        backend: int = 0,
        # FFGear parameters
        source_demuxer: str | None = None,
        frame_format: str = "bgr24",
        custom_ffmpeg: str = "",
        # common parameters
        time_delay: int = 0,
        colorspace: str | None = None,
        logging: bool = False,
        # deprecated
        enablePiCamera: bool | None = None,
        **options: dict
    ):
        """
        This constructor method initializes the object state and attributes of the VideoGear class.

        Parameters:
            api (Backend): selects the capture backend (`Backend.CAMGEAR`, `Backend.PIGEAR`, `Backend.FFGEAR`).
            stabilize (bool): enable access to Stabilizer Class for stabilizing frames.
            camera_num (int): [PiGear] selects the camera module index.
            resolution (tuple): [PiGear] sets `(width, height)` of the source.
            framerate (int/float): [PiGear] sets the framerate of the source.
            source (Any): [CamGear/FFGear] defines the source for the input stream.
            stream_mode (bool): [CamGear/FFGear] enables Stream-Mode for streaming URLs.
            backend (int): [CamGear] selects the backend for OpenCV's VideoCapture class.
            source_demuxer (str): [FFGear] specifies the demuxer for the source.
            frame_format (str): [FFGear] specifies the pixel layout (e.g. `bgr24`).
            custom_ffmpeg (str): [FFGear] specifies a custom FFmpeg executable path.
            colorspace (str): [CamGear/PiGear] selects the colorspace of the input stream.
            logging (bool): enables/disables logging.
            time_delay (int): [CamGear/PiGear] time delay (in sec) before reading frames.
            enablePiCamera (bool): **DEPRECATED** — use `api=Backend.PIGEAR` instead.
            options (dict): tweak parameters forwarded to the selected gear and Stabilizer.
        """
        # enable logging if specified
        self.__logging = logging if isinstance(logging, bool) else False

        # print current version
        logcurr_vidgear_ver(logging=self.__logging)

        # handle deprecated `enablePiCamera`
        if enablePiCamera is not None:
            warnings.warn(
                "`enablePiCamera` is deprecated; use `api=Backend.PIGEAR` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            api = Backend.PIGEAR if enablePiCamera else Backend.CAMGEAR

        # validate api selection
        if not isinstance(api, Backend):
            raise TypeError(
                "[VideoGear:ERROR] :: `api` must be a `Backend` enum member, got `{}`.".format(
                    type(api).__name__
                )
            )

        # initialize stabilizer
        self.__stabilization_mode = stabilize

        # reformat dictionary
        options = {str(k).strip(): v for k, v in options.items()}

        if self.__stabilization_mode:
            from .stabilizer import Stabilizer

            s_radius = options.pop("SMOOTHING_RADIUS", 25)
            if not isinstance(s_radius, int):
                s_radius = 25

            border_size = options.pop("BORDER_SIZE", 0)
            if not isinstance(border_size, int):
                border_size = 0

            border_type = options.pop("BORDER_TYPE", "black")
            if not isinstance(border_type, str):
                border_type = "black"

            crop_n_zoom = options.pop("CROP_N_ZOOM", False)
            if not isinstance(crop_n_zoom, bool):
                crop_n_zoom = False

            self.__stabilizer_obj = Stabilizer(
                smoothing_radius=s_radius,
                border_type=border_type,
                border_size=border_size,
                crop_n_zoom=crop_n_zoom,
                logging=logging,
            )
            self.__logging and logger.debug(
                "Enabling Stabilization Mode for the current video source!"
            )  # log info

        # dispatch table — register new gears here
        gear_builders = {
            Backend.CAMGEAR: lambda: CamGear(
                source=source,
                stream_mode=stream_mode,
                backend=backend,
                colorspace=colorspace,
                logging=logging,
                time_delay=time_delay,
                **options,
            ),
            Backend.PIGEAR: lambda: self.__build_pigear(
                camera_num=camera_num,
                resolution=resolution,
                framerate=framerate,
                colorspace=colorspace,
                logging=logging,
                time_delay=time_delay,
                options=options,
            ),
            Backend.FFGEAR: lambda: self.__build_ffgear(
                source=source,
                stream_mode=stream_mode,
                source_demuxer=source_demuxer,
                frame_format=frame_format,
                custom_ffmpeg=custom_ffmpeg,
                logging=logging,
                options=options,
            ),
        }

        self.__logging and logger.debug(
            "Selecting `{}` backend for VideoGear.".format(api.value)
        )
        self.stream = gear_builders[api]()

        # initialize framerate variable (FFGear has no `framerate` attr)
        self.framerate = getattr(self.stream, "framerate", 0.0)

    @staticmethod
    def __build_pigear(*, camera_num, resolution, framerate, colorspace, logging, time_delay, options):
        from .pigear import PiGear

        return PiGear(
            camera_num=camera_num,
            resolution=resolution,
            framerate=framerate,
            colorspace=colorspace,
            logging=logging,
            time_delay=time_delay,
            **options,
        )

    @staticmethod
    def __build_ffgear(*, source, stream_mode, source_demuxer, frame_format, custom_ffmpeg, logging, options):
        from .ffgear import FFGear

        return FFGear(
            source=source,
            stream_mode=stream_mode,
            source_demuxer=source_demuxer,
            frame_format=frame_format,
            custom_ffmpeg=custom_ffmpeg,
            logging=logging,
            **options,
        )

    def start(self) -> T:
        """
        Launches the internal *Threaded Frames Extractor* daemon of API in use.

        **Returns:** A reference to the selected class object.
        """
        self.stream.start()
        return self

    def read(self) -> NDArray:
        """
        Extracts frames synchronously from selected API's monitored deque, while maintaining a fixed-length frame
        buffer in the memory, and blocks the thread if the deque is full.

        **Returns:** A n-dimensional numpy array.
        """
        while self.__stabilization_mode:
            frame = self.stream.read()
            if frame is None:
                break
            frame_stab = self.__stabilizer_obj.stabilize(frame)
            if frame_stab is not None:
                return frame_stab
        return self.stream.read()

    def stop(self) -> None:
        """
        Safely terminates the thread, and release the respective multi-threaded resources.
        """
        self.stream.stop()
        # logged
        self.__logging and logger.debug("Terminating VideoGear.")
        # clean queue
        self.__stabilization_mode and self.__stabilizer_obj.clean()
