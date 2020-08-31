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
import logging as log

from .helper import logger_handler
from .camgear import CamGear

# define logger
logger = log.getLogger("VideoGear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


class VideoGear:

    """

    VideoGear provides a special internal wrapper around VidGear's exclusive **Video Stabilizer** class.

    VideoGear also act as a Common API, that provided an internal access to both CamGear and
    PiGear APIs and their parameters, with a special `enablePiCamera` boolean flag.

    VideoGear is basically ideal when you need to switch to different video sources without changing your code
    much. Also, it enables easy stabilization for various video-streams _(real-time or not)_  with minimum effort
    and using just fewer lines of code.

    """

    def __init__(
        self,
        # VideoGear parameters
        enablePiCamera=False,
        stabilize=False,
        # PiGear parameters
        camera_num=0,
        resolution=(640, 480),
        framerate=30,
        # CamGear parameters
        source=0,
        y_tube=False,
        backend=0,
        # common parameters
        time_delay=0,
        colorspace=None,
        logging=False,
        **options
    ):

        # initialize stabilizer
        self.__stablization_mode = stabilize

        # enable logging if specified
        self.__logging = False
        if logging:
            self.__logging = logging

        # reformat dictionary
        options = {str(k).strip(): v for k, v in options.items()}

        if self.__stablization_mode:
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
            if self.__logging:
                logger.debug(
                    "Enabling Stablization Mode for the current video source!"
                )  # log info

        if enablePiCamera:
            # only import the pigear module only if required
            from .pigear import PiGear

            # initialize the picamera stream by enabling PiGear API
            self.stream = PiGear(
                camera_num=camera_num,
                resolution=resolution,
                framerate=framerate,
                colorspace=colorspace,
                logging=logging,
                time_delay=time_delay,
                **options
            )
        else:
            # otherwise, we are using OpenCV so initialize the webcam
            # stream by activating CamGear API
            self.stream = CamGear(
                source=source,
                y_tube=y_tube,
                backend=backend,
                colorspace=colorspace,
                logging=logging,
                time_delay=time_delay,
                **options
            )

        # initialize framerate variable
        self.framerate = self.stream.framerate

    def start(self):
        """
        Launches the internal *Threaded Frames Extractor* daemon of API in use.

        **Returns:** A reference to the selected class object.
        """
        self.stream.start()
        return self

    def read(self):
        """
        Extracts frames synchronously from selected API's monitored deque, while maintaining a fixed-length frame
        buffer in the memory, and blocks the thread if the deque is full.

        **Returns:** A n-dimensional numpy array.
        """
        while self.__stablization_mode:
            frame = self.stream.read()
            if frame is None:
                break
            frame_stab = self.__stabilizer_obj.stabilize(frame)
            if not (frame_stab is None):
                return frame_stab
        return self.stream.read()

    def stop(self):
        """
        Safely terminates the thread, and release the respective VideoStream resources.
        """
        self.stream.stop()
        # logged
        if self.__logging:
            logger.debug("Terminating VideoGear.")
        # clean queue
        if self.__stablization_mode:
            self.__stabilizer_obj.clean()
