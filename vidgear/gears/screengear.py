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
import time
import queue
import numpy as np
import logging as log
from threading import Thread, Event
from collections import deque, OrderedDict

# import helper packages
from .helper import import_dependency_safe, capPropId, logger_handler

# safe import critical Class modules
mss = import_dependency_safe("from mss import mss", error="silent")
if not (mss is None):
    from mss.exception import ScreenShotError
pysct = import_dependency_safe("pyscreenshot", error="silent")
if not (pysct is None):
    from pyscreenshot.err import FailedBackendError

# define logger
logger = log.getLogger("ScreenGear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


class ScreenGear:
    """
    ScreenGear is designed exclusively for ultra-fast Screencasting, which means it can grab frames from your monitor in real-time, either by defining an area on the computer screen or full-screen,
    at the expense of inconsiderable latency. ScreenGear also seamlessly support frame capturing from multiple monitors as well as supports multiple backends.

    ScreenGear API implements a multi-threaded wrapper around pyscreenshot & python-mss python library, and also flexibly supports its internal parameter.
    """

    def __init__(
        self, monitor=None, backend="", colorspace=None, logging=False, **options
    ):
        """
        This constructor method initializes the object state and attributes of the ScreenGear class.

        Parameters:
            monitor (int): enables `mss` backend and sets the index of the monitor screen.
            backend (str): enables `pyscreenshot` and select suitable backend for extracting frames.
            colorspace (str): selects the colorspace of the input stream.
            logging (bool): enables/disables logging.
            options (dict): provides the flexibility to manually set the dimensions of capture screen area.
        """
        # raise error(s) for critical Class imports
        import_dependency_safe("mss.mss" if mss is None else "")
        import_dependency_safe("pyscreenshot" if pysct is None else "")

        # enable logging if specified:
        self.__logging = logging if isinstance(logging, bool) else False

        # create monitor instance for the user-defined monitor
        self.__monitor_instance = None
        self.__backend = ""
        if monitor is None:
            self.__capture_object = pysct
            self.__backend = backend.lower().strip()
        else:
            self.__capture_object = mss()
            if backend.strip():
                logger.warning(
                    "Backends are disabled for Monitor Indexing(monitor>=0)!"
                )
            try:
                self.__monitor_instance = self.__capture_object.monitors[monitor]
            except Exception as e:
                logger.exception(str(e))
                self.__monitor_instance = None

        # assigns special parameter to global variable and clear
        # Thread Timeout
        self.__thread_timeout = options.pop("THREAD_TIMEOUT", None)
        if self.__thread_timeout and isinstance(self.__thread_timeout, (int, float)):
            # set values
            self.__thread_timeout = int(self.__thread_timeout)
        else:
            # defaults to 5mins timeout
            self.__thread_timeout = None

        # define deque and assign it to global var
        self.__queue = queue.Queue(maxsize=96)  # max len 96 to check overflow
        # log it
        if logging:
            logger.debug("Enabling Threaded Queue Mode by default for ScreenGear!")
            if self.__thread_timeout:
                logger.debug(
                    "Setting Video-Thread Timeout to {}s.".format(self.__thread_timeout)
                )

        # intiate screen dimension handler
        screen_dims = {}
        # reformat proper mss dict and assign to screen dimension handler
        screen_dims = {
            k.strip(): v
            for k, v in options.items()
            if k.strip() in ["top", "left", "width", "height"]
        }
        # check whether user-defined dimensions are provided
        if screen_dims and len(screen_dims) == 4:
            key_order = ("top", "left", "width", "height")
            screen_dims = OrderedDict((k, screen_dims[k]) for k in key_order)
            if logging:
                logger.debug("Setting Capture-Area dimensions: {}!".format(screen_dims))
        else:
            screen_dims.clear()

        # separately handle colorspace value to int conversion
        if colorspace:
            self.color_space = capPropId(colorspace.strip())
            if logging and not (self.color_space is None):
                logger.debug(
                    "Enabling `{}` colorspace for this video stream!".format(
                        colorspace.strip()
                    )
                )
        else:
            self.color_space = None

        # intialize mss capture instance
        self.__mss_capture_instance = ""
        try:
            if self.__monitor_instance is None:
                if screen_dims:
                    self.__mss_capture_instance = tuple(screen_dims.values())
                # extract global frame from instance
                self.frame = np.asanyarray(
                    self.__capture_object.grab(
                        bbox=self.__mss_capture_instance,
                        childprocess=False,
                        backend=self.__backend,
                    )
                )
            else:
                if screen_dims:
                    self.__mss_capture_instance = {
                        "top": self.__monitor_instance["top"] + screen_dims["top"],
                        "left": self.__monitor_instance["left"] + screen_dims["left"],
                        "width": screen_dims["width"],
                        "height": screen_dims["height"],
                        "mon": monitor,
                    }
                else:
                    self.__mss_capture_instance = (
                        self.__monitor_instance  # otherwise create instance from monitor
                    )
                # extract global frame from instance
                self.frame = np.asanyarray(
                    self.__capture_object.grab(self.__mss_capture_instance)
                )
            # initialize and append to queue
            self.__queue.put(self.frame)
        except Exception as e:
            if isinstance(e, ScreenShotError):
                # otherwise catch and log errors
                if logging:
                    logger.exception(self.__capture_object.get_error_details())
                raise ValueError(
                    "[ScreenGear:ERROR] :: ScreenShotError caught, Wrong dimensions passed to python-mss, Kindly Refer Docs!"
                )
            elif isinstance(e, KeyError):
                raise ValueError(
                    "[ScreenGear:ERROR] :: ScreenShotError caught, Invalid backend: `{}`, Kindly Refer Docs!".format(
                        backend
                    )
                )
            else:
                raise SystemError(
                    "[ScreenGear:ERROR] :: Unable to grab any instance on this system, Are you running headless?"
                )

        # thread initialization
        self.__thread = None
        # initialize termination flag
        self.__terminate = Event()

    def start(self):
        """
        Launches the internal *Threaded Frames Extractor* daemon

        **Returns:** A reference to the ScreenGear class object.
        """
        self.__thread = Thread(target=self.__update, name="ScreenGear", args=())
        self.__thread.daemon = True
        self.__thread.start()
        return self

    def __update(self):
        """
        A **Threaded Frames Extractor**, that keep iterating frames from `mss` API to a internal monitored deque,
        until the thread is terminated, or frames runs out.
        """
        # intialize frame variable
        frame = None
        # keep looping infinitely until the thread is terminated
        while True:

            # if the thread indicator variable is set, stop the thread
            if self.__terminate.is_set():
                break

            try:
                if self.__monitor_instance:
                    frame = np.asanyarray(
                        self.__capture_object.grab(self.__mss_capture_instance)
                    )
                else:
                    frame = np.asanyarray(
                        self.__capture_object.grab(
                            bbox=self.__mss_capture_instance,
                            childprocess=False,
                            backend=self.__backend,
                        )
                    )
                    if not self.__backend or self.__backend == "pil":
                        frame = frame[:, :, ::-1]
                assert not (
                    frame is None or np.shape(frame) == ()
                ), "[ScreenGear:ERROR] :: Failed to retreive any valid frames!"
            except Exception as e:
                if isinstance(e, ScreenShotError):
                    raise RuntimeError(self.__capture_object.get_error_details())
                else:
                    logger.exception(str(e))
                self.__terminate.set()
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
            self.__queue.put(self.frame)

        # finally release mss resources
        if self.__monitor_instance:
            self.__capture_object.close()

    def read(self):
        """
        Extracts frames synchronously from monitored deque, while maintaining a fixed-length frame buffer in the memory,
        and blocks the thread if the deque is full.

        **Returns:** A n-dimensional numpy array.
        """
        # check whether or not termination flag is enabled
        while not self.__terminate.is_set():
            return self.__queue.get(timeout=self.__thread_timeout)
        # otherwise return NoneType
        return None

    def stop(self):
        """
        Safely terminates the thread, and release the resources.
        """
        if self.__logging:
            logger.debug("Terminating ScreenGear Processes.")

        # indicate that the thread should be terminate
        self.__terminate.set()

        # wait until stream resources are released (producer thread might be still grabbing frame)
        if self.__thread is not None:
            if not (self.__queue is None):
                while not self.__queue.empty():
                    try:
                        self.__queue.get_nowait()
                    except queue.Empty:
                        continue
                    self.__queue.task_done()
            self.__thread.join()
