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
from threading import Thread
from pkg_resources import parse_version
from .helper import capPropId
from .helper import logger_handler
from mss import mss
from mss.exception import ScreenShotError

import numpy as np
import cv2, time
import logging as log


# define logger
logger = log.getLogger("ScreenGear")
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


class ScreenGear:

    """
	With ScreenGear, we can easily define an area on the computer screen or an open window to record the live screen frames in 
	real-time at the expense of insignificant latency. To achieve this, ScreenGear provides a high-level multi-threaded wrapper 
	around mss python library API and also supports the flexible direct parameter manipulation. Furthermore, ScreenGear relies on 
	Threaded Queue mode for ultra-fast live frame handling and which is enabled by default.

	Threaded Queue Mode => Sequentially adds and releases frames to/from deque and handles overflow of this queue. It utilizes 
	Deques that support thread-safe, memory efficient appends and pops from either side of the deque with approximately the 
	same O(1) performance in either direction.  


	:param monitor(int): sets the Positions/Location of monitor where to grab frame from. More information can be found in the docs. 
						/ It default value is 1 (means current monitor will be used).

	:param **options(dict): can be used to pass parameters to ScreenGear Class. 
							/This attribute provides the flexibility to manipulate mss input parameters 
							/directly like the dimensions of the region of the given monitor from where the 
							frames to be grabbed. Checkout VidGear docs for usage details.

	:param (string) colorspace: set the colorspace of the video stream. Its default value is None.

	:param (boolean) logging: set this flag to enable/disable error logging essential for debugging. Its default value is False.

	"""

    def __init__(self, monitor=1, colorspace=None, logging=False, **options):

        # enable logging if specified
        self.__logging = False
        if logging:
            self.__logging = logging

        # create mss object
        self.__mss_object = mss()
        # create monitor instance for the user-defined monitor
        monitor_instance = None
        if monitor >= 0:
            try:
                monitor_instance = self.__mss_object.monitors[monitor]
            except Exception as e:
                logger.exception(str(e))
                monitor_instance = None
        else:
            raise ValueError(
                "[ScreenGear:ERROR] :: `monitor` value cannot be negative, Read Docs!"
            )

        # Initialize Queue
        self.__queue = None

        # import deque
        from collections import deque

        # define deque and assign it to global var
        self.__queue = deque(maxlen=96)  # max len 96 to check overflow
        # log it
        if logging:
            logger.debug("Enabling Threaded Queue Mode by default for ScreenGear!")

        # intiate screen dimension handler
        screen_dims = {}
        # initializing colorspace variable
        self.color_space = None

        # reformat proper mss dict and assign to screen dimension handler
        screen_dims = {
            k.strip(): v
            for k, v in options.items()
            if k.strip() in ["top", "left", "width", "height"]
        }
        # separately handle colorspace value to int conversion
        if not (colorspace is None):
            self.color_space = capPropId(colorspace.strip())
            if logging and not (self.color_space is None):
                logger.debug(
                    "Enabling `{}` colorspace for this video stream!".format(
                        colorspace.strip()
                    )
                )

        # intialize mss capture instance
        self.__mss_capture_instance = None
        try:
            # check whether user-defined dimensions are provided
            if screen_dims and len(screen_dims) == 4:
                if logging:
                    logger.debug("Setting capture dimensions: {}!".format(screen_dims))
                self.__mss_capture_instance = (
                    screen_dims  # create instance from dimensions
                )
            elif not (monitor_instance is None):
                self.__mss_capture_instance = (
                    monitor_instance  # otherwise create instance from monitor
                )
            else:
                raise RuntimeError("[ScreenGear:ERROR] :: API Failure occurred!")
            # extract global frame from instance
            self.frame = np.asanyarray(
                self.__mss_object.grab(self.__mss_capture_instance)
            )
            # intitialize and append to queue
            self.__queue.append(self.frame)
        except Exception as e:
            if isinstance(e, ScreenShotError):
                # otherwise catch and log errors
                if logging:
                    logger.exception(self.__mss_object.get_error_details())
                raise ValueError(
                    "[ScreenGear:ERROR] :: ScreenShotError caught, Wrong dimensions passed to python-mss, Kindly Refer Docs!"
                )
            else:
                raise SystemError(
                    "[ScreenGear:ERROR] :: Unable to initiate any MSS instance on this system, Are you running headless?"
                )

        # thread initialization
        self.__thread = None
        # initialize termination flag
        self.__terminate = False

    def start(self):
        """
		start the thread to read frames from the video stream
		"""
        self.__thread = Thread(target=self.__update, name="ScreenGear", args=())
        self.__thread.daemon = True
        self.__thread.start()
        return self

    def __update(self):
        """
		Update frames from stream
		"""
        # intialize frame variable
        frame = None
        # keep looping infinitely until the thread is terminated
        while not(self.__terminate):

            # check queue buffer for overflow
            if len(self.__queue) >= 96:
                # stop iterating if overflowing occurs
                time.sleep(0.000001)
                continue

            try:
                frame = np.asanyarray(
                    self.__mss_object.grab(self.__mss_capture_instance)
                )
                assert not(frame is None or np.shape(frame) == ()), "[ScreenGear:ERROR] :: Failed to retreive any valid frames!"
            except Exception as e:
                if isinstance(e, ScreenShotError):
                    raise RuntimeError(self.__mss_object.get_error_details())
                else:
                    logger.exception(str(e))
                self.__terminate = True
                continue

            if not (self.color_space is None):
                # apply colorspace to frames
                color_frame = None
                try:
                    if isinstance(self.color_space, int):
                        color_frame = cv2.cvtColor(frame, self.color_space)
                    else:
                        if self.__logging:
                            logger.warning(
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
            # append to queue
            self.__queue.append(self.frame)
        # finally release mss resources
        self.__mss_object.close()

    def read(self):
        """
		return the frame
		"""
        # check whether or not termination flag is enabled
        while not self.__terminate:
            # check if queue is empty
            if len(self.__queue) > 0:
                return self.__queue.popleft()
            else:
                continue
        # otherwise return NoneType
        return None

    def stop(self):
        """
		Terminates the Read process
		"""
        if self.__logging:
            logger.debug("Terminating ScreenGear Processes.")
        # indicate that the thread should be terminated
        self.__terminate = True
        # terminate Threaded queue mode seperately
        if not (self.__queue is None):
            self.__queue.clear()
        # wait until stream resources are released (producer thread might be still grabbing frame)
        if self.__thread is not None:
            self.__thread.join()
            # properly handle thread exit
