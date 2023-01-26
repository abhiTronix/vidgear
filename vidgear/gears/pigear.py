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
import cv2
import sys
import time
import logging as log
from threading import Thread

# import helper packages
from .helper import (
    capPropId,
    logger_handler,
    import_dependency_safe,
    logcurr_vidgear_ver,
)

# safe import critical Class modules
picamera = import_dependency_safe("picamera", error="silent")
if not (picamera is None):
    from picamera import PiCamera
    from picamera.array import PiRGBArray

# define logger
logger = log.getLogger("PiGear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


class PiGear:
    """
    PiGear is similar to CamGear API but exclusively made to support various Raspberry Pi Camera Modules (such as OmniVision OV5647 Camera Module and Sony IMX219 Camera Module).
    PiGear provides a flexible multi-threaded framework around complete picamera python library, and provide us the ability to exploit almost all of its parameters like brightness,
    saturation, sensor_mode, iso, exposure, etc. effortlessly. Furthermore, PiGear also supports multiple camera modules, such as in the case of Raspberry-Pi Compute Module IO boards.

    Best of all, PiGear contains Threaded Internal Timer - that silently keeps active track of any frozen-threads/hardware-failures and exit safely, if any does occur. That means that
    if you're running PiGear API in your script and someone accidentally pulls the Camera-Module cable out, instead of going into possible kernel panic, API will exit safely to save resources.

    !!! warning "Make sure to enable [Raspberry Pi hardware-specific settings](https://picamera.readthedocs.io/en/release-1.13/quickstart.html) prior using this API, otherwise nothing will work."
    """

    def __init__(
        self,
        camera_num=0,
        resolution=(640, 480),
        framerate=30,
        colorspace=None,
        logging=False,
        time_delay=0,
        **options
    ):
        """
        This constructor method initializes the object state and attributes of the PiGear class.

        Parameters:
            camera_num (int): selects the camera module index which will be used as source.
            resolution (tuple): sets the resolution (i.e. `(width,height)`) of the source..
            framerate (int/float): sets the framerate of the source.
            colorspace (str): selects the colorspace of the input stream.
            logging (bool): enables/disables logging.
            time_delay (int): time delay (in sec) before start reading the frames.
            options (dict): provides ability to alter Source Tweak Parameters.
        """
        # print current version
        logcurr_vidgear_ver(logging=logging)

        # raise error(s) for critical Class imports
        import_dependency_safe(
            "picamera" if picamera is None else "",
        )

        # enable logging if specified
        self.__logging = False
        if logging:
            self.__logging = logging

        assert (
            isinstance(framerate, (int, float)) and framerate > 5.0
        ), "[PiGear:ERROR] :: Input framerate value `{}` is a Invalid! Kindly read docs.".format(
            framerate
        )
        assert (
            isinstance(resolution, (tuple, list)) and len(resolution) == 2
        ), "[PiGear:ERROR] :: Input resolution value `{}` is a Invalid! Kindly read docs.".format(
            resolution
        )
        if not (isinstance(camera_num, int) and camera_num >= 0):
            camera_num = 0
            logger.warning(
                "Input camera_num value `{}` is invalid, Defaulting to index 0!"
            )

        # initialize the picamera stream at given index
        self.__camera = PiCamera(camera_num=camera_num)
        self.__camera.resolution = tuple(resolution)
        self.__camera.framerate = framerate
        self.__logging and logger.debug(
            "Activating Pi camera at index: {} with resolution: {} & framerate: {}".format(
                camera_num, resolution, framerate
            )
        )

        # initialize framerate variable
        self.framerate = framerate

        # initializing colorspace variable
        self.color_space = None

        # reformat dict
        options = {str(k).strip(): v for k, v in options.items()}

        # define timeout variable default value(handles hardware failures)
        self.__failure_timeout = options.pop("HWFAILURE_TIMEOUT", 2.0)
        if isinstance(self.__failure_timeout, (int, float)):
            if not (10.0 > self.__failure_timeout > 1.0):
                raise ValueError(
                    "[PiGear:ERROR] :: `HWFAILURE_TIMEOUT` value can only be between 1.0 ~ 10.0"
                )
            self.__logging and logger.debug(
                "Setting HW Failure Timeout: {} seconds".format(self.__failure_timeout)
            )
        else:
            # reset improper values
            self.__failure_timeout = 2.0

        try:
            # apply attributes to source if specified
            for key, value in options.items():
                self.__logging and logger.debug(
                    "Setting Parameter: {} = '{}'".format(key, value)
                )
                setattr(self.__camera, key, value)
        except Exception as e:
            # Catch if any error occurred
            logger.exception(str(e))

        # separately handle colorspace value to int conversion
        if not (colorspace is None):
            self.color_space = capPropId(colorspace.strip())
            if self.__logging and not (self.color_space is None):
                logger.debug(
                    "Enabling `{}` colorspace for this video stream!".format(
                        colorspace.strip()
                    )
                )

        # enable rgb capture array thread and capture stream
        self.__rawCapture = PiRGBArray(self.__camera, size=resolution)
        self.stream = self.__camera.capture_continuous(
            self.__rawCapture, format="bgr", use_video_port=True
        )

        # frame variable initialization
        self.frame = None
        try:
            stream = next(self.stream)
            self.frame = stream.array
            self.__rawCapture.seek(0)
            self.__rawCapture.truncate()
            # render colorspace if defined
            if not (self.frame is None) and not (self.color_space is None):
                self.frame = cv2.cvtColor(self.frame, self.color_space)
        except Exception as e:
            logger.exception(str(e))
            raise RuntimeError("[PiGear:ERROR] :: Camera Module failed to initialize!")

        # applying time delay to warm-up picamera only if specified
        if time_delay and isinstance(time_delay, (int, float)):
            time.sleep(time_delay)

        # thread initialization
        self.__thread = None

        # timer thread initialization(Keeps check on frozen thread)
        self.__timer = None
        self.__t_elasped = 0.0  # records time taken by thread

        # catching thread exceptions
        self.__exceptions = None

        # initialize termination flag
        self.__terminate = False

    def start(self):
        """
        Launches the internal *Threaded Frames Extractor* daemon

        **Returns:** A reference to the CamGear class object.
        """

        # Start frame producer thread
        self.__thread = Thread(target=self.__update, name="PiGear", args=())
        self.__thread.daemon = True
        self.__thread.start()

        # Start internal timer thread
        self.__timer = Thread(target=self.__timeit, name="PiTimer", args=())
        self.__timer.daemon = True
        self.__timer.start()

        return self

    def __timeit(self):
        """
        Threaded Internal Timer that keep checks on thread execution timing
        """

        # assign current time
        self.__t_elasped = time.time()

        # loop until termainated
        while not (self.__terminate):
            # check for frozen thread
            if time.time() - self.__t_elasped > self.__failure_timeout:
                # log failure
                self.__logging and logger.critical("Camera Module Disconnected!")
                # prepare for clean exit
                self.__exceptions = True
                self.__terminate = True  # self-terminate

    def __update(self):
        """
        A **Threaded Frames Extractor**, that keep iterating frames from PiCamera API to a internal monitored deque,
        until the thread is terminated, or frames runs out.
        """
        # keep looping infinitely until the thread is terminated
        while not (self.__terminate):

            try:
                # Try to iterate next frame from generator
                stream = next(self.stream)
            except Exception:
                # catch and save any exceptions
                self.__exceptions = sys.exc_info()
                break  # exit

            # __update timer
            self.__t_elasped = time.time()

            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            frame = stream.array
            self.__rawCapture.seek(0)
            self.__rawCapture.truncate()

            # apply colorspace if specified
            if not (self.color_space is None):
                # apply colorspace to frames
                color_frame = None
                try:
                    if isinstance(self.color_space, int):
                        color_frame = cv2.cvtColor(frame, self.color_space)
                    else:
                        self.__logging and logger.warning(
                            "Global color_space parameter value `{}` is not a valid!".format(
                                self.color_space
                            )
                        )
                        self.color_space = None
                except Exception as e:
                    # Catch if any error occurred
                    self.color_space = None
                    if self.__logging:
                        logger.exception(str(e))
                        logger.warning("Input colorspace is not a valid colorspace!")

                if not (color_frame is None):
                    self.frame = color_frame
                else:
                    self.frame = frame
            else:
                self.frame = frame

        # terminate processes
        if not (self.__terminate):
            self.__terminate = True

        # release picamera resources
        self.__rawCapture.close()
        self.__camera.close()

    def read(self):
        """
        Extracts frames synchronously from monitored deque, while maintaining a fixed-length frame buffer in the memory,
        and blocks the thread if the deque is full.

        **Returns:** A n-dimensional numpy array.
        """
        # check if there are any thread exceptions
        if not (self.__exceptions is None):
            if isinstance(self.__exceptions, bool):
                # clear frame
                self.frame = None
                # notify user about hardware failure
                raise SystemError(
                    "[PiGear:ERROR] :: Hardware failure occurred, Kindly reconnect Camera Module and restart your Pi!"
                )
            else:
                # clear frame
                self.frame = None
                # re-raise error for debugging
                error_msg = (
                    "[PiGear:ERROR] :: Camera Module API failure occured: {}".format(
                        self.__exceptions[1]
                    )
                )
                raise RuntimeError(error_msg).with_traceback(self.__exceptions[2])

        # return the frame
        return self.frame

    def stop(self):
        """
        Safely terminates the thread, and release the VideoStream resources.
        """
        self.__logging and logger.debug("Terminating PiGear Processes.")

        # make sure that the threads should be terminated
        self.__terminate = True

        # stop timer thread
        if not (self.__timer is None):
            self.__timer.join()
            self.__timer = None

        # handle camera thread
        if not (self.__thread is None):
            # check if hardware failure occured
            if not (self.__exceptions is None) and isinstance(self.__exceptions, bool):
                # force release picamera resources
                self.__rawCapture.close()
                self.__camera.close()
            # properly handle thread exit
            self.__thread.join()  # wait if still process is still processing some information
            # remove any threads
            self.__thread = None
