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

# import helper packages
from .helper import logger_handler, logcurr_vidgear_ver

# import additional API(s)
from .camgear import CamGear

# define logger
logger = log.getLogger("VideoGear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


class VideoGear:
    """
    VideoGear API provides a special internal wrapper around VidGear's exclusive Video Stabilizer class.
    VideoGear also acts as a Common Video-Capture API that provides internal access for both CamGear and PiGear APIs and their parameters with an exclusive enablePiCamera boolean flag.

    VideoGear is ideal when you need to switch to different video sources without changing your code much. Also, it enables easy stabilization for various video-streams (real-time or not)
    with minimum effort and writing way fewer lines of code.
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
        stream_mode=False,
        backend=0,
        # common parameters
        time_delay=0,
        colorspace=None,
        logging=False,
        **options
    ):

        """
        This constructor method initializes the object state and attributes of the VideoGear class.

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
            options (dict): provides ability to alter Tweak Parameters of CamGear, PiGear & Stabilizer.
        """
        # print current version
        logcurr_vidgear_ver(logging=logging)

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
            self.__logging and logger.debug(
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
                stream_mode=stream_mode,
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
        self.__logging and logger.debug("Terminating VideoGear.")
        # clean queue
        if self.__stablization_mode:
            self.__stabilizer_obj.clean()
