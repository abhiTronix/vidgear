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
import os
import time
import logging as log
from threading import Thread
from typing import Tuple, Union, TypeVar
from numpy.typing import NDArray

# import helper packages
from .helper import (
    capPropId,
    logger_handler,
    import_dependency_safe,
    logcurr_vidgear_ver,
)

# safe import critical Class modules
### LEGACY picamera API ###
picamera = import_dependency_safe("picamera", error="silent")
if not (picamera is None):
    from picamera import PiCamera
    from picamera.array import PiRGBArray

### NEW picamera2 API ###
picamera2 = import_dependency_safe("picamera2", error="silent")
if not (picamera2 is None):
    from picamera2 import Picamera2
    from libcamera import Transform

# define logger
logger = log.getLogger("PiGear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


PIGear = TypeVar("PIGear", bound="PiGear")


class PiGear:
    """
    PiGear implements a seamless and robust wrapper around the [picamera2](https://github.com/raspberrypi/picamera2) python library, simplifying integration with minimal code changes and ensuring a
    smooth transition for developers already familiar with the Picamera2 API. PiGear leverages the `libcamera` API under the hood with multi-threading, providing high-performance :fire:, enhanced
    control and functionality for Raspberry Pi camera modules.

    PiGear handles common configuration parameters and non-standard settings for various camera types, simplifying the integration process. PiGear currently supports picamera2 API parameters such as
    `sensor`, `controls`, `transform`, and `stride`, with internal type and sanity checks for robust performance.

    While primarily focused on Raspberry Pi camera modules, PiGear also provides basic functionality for USB webcams only with Picamera2 API, along with the ability to accurately differentiate between
    USB and Raspberry Pi cameras using metadata.

    ???+ info "Backward compatibility with `picamera` library"
        PiGear seamlessly switches to the legacy [picamera](https://picamera.readthedocs.io/en/release-1.13/index.html) library if the `picamera2` library is unavailable, ensuring seamless backward
        compatibility. For this, PiGear also provides a flexible multi-threaded framework around complete `picamera` API, allowing developers to effortlessly exploit a wide range of parameters, such
        as `brightness`, `saturation`, `sensor_mode`, `iso`, `exposure`, and more.

    Furthermore, PiGear supports the use of multiple camera modules, including those found on Raspberry Pi Compute Module IO boards and USB cameras _(only with Picamera2 API)_.

    ??? new "Threaded Internal Timer :material-camera-timer:"
        PiGear ensures proper resource release during the termination of the API, preventing potential issues or resource leaks. PiGear API internally implements a
        ==Threaded Internal Timer== that silently keeps active track of any frozen-threads or hardware-failures and exits safely if any do occur. This means that if
        you're running the PiGear API in your script and someone accidentally pulls the Camera-Module cable out, instead of going into a possible kernel panic,
        the API will exit safely to save resources.

    !!! failure "Make sure to [enable Raspberry Pi hardware-specific settings](https://picamera.readthedocs.io/en/release-1.13/quickstart.html) prior using this API, otherwise nothing will work."
    """

    def __init__(
        self,
        camera_num: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        framerate: Union[int, float] = 30,
        colorspace: str = None,
        logging: bool = False,
        time_delay: int = 0,
        **options: dict
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
        # enable logging if specified
        self.__logging = logging if isinstance(logging, bool) else False

        # print current version
        logcurr_vidgear_ver(logging=self.__logging)

        # raise error(s) for critical Class imports
        global picamera2
        if picamera2:
            # log if picamera2
            self.__logging and logger.info("picamera2 API is currently being accessed.")
        elif picamera:
            # switch to picamera otherwise
            logger.critical(
                "picamera2 library not installed on this system. Defaulting to legacy picamera API."
            )
        else:
            # raise error if none
            import_dependency_safe("picamera")

        assert (
            isinstance(framerate, (int, float)) and framerate > 0.0
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

        # reformat dict
        options = {str(k).strip(): v for k, v in options.items()}

        # check if legacy picamera backend is enforced
        enforce_legacy_picamera = options.pop("enforce_legacy_picamera", False)
        if isinstance(enforce_legacy_picamera, bool) and enforce_legacy_picamera:
            # check if picamera library is available.
            if picamera:
                logger.critical(
                    "Enforcing legacy picamera API for this run. picamera2 API access will be disabled!"
                )
                # disable picamera2
                picamera2 = None
            else:
                # raise error otherwise
                logger.error(
                    "`picamera` is unavailable or unsupported on this system, `enforce_legacy_picamera` will be discarded!"
                )
                import_dependency_safe("picamera")

        if picamera2:
            # handle logging
            not (self.__logging) and not os.getenv(
                "LIBCAMERA_LOG_LEVELS", False
            ) and logger.info(
                "Kindly set `LIBCAMERA_LOG_LEVELS=2` environment variable to disable common libcamera API messages."
            )
            # collect metadata
            cameras_metadata = Picamera2.global_camera_info()
            # initialize the picamera stream at given index
            self.__camera = Picamera2(camera_num=camera_num)
            # extract metadata for current camera
            camera_metadata = [x for x in cameras_metadata if x["Num"] == camera_num][0]
            # check connected camera is USB or I2C
            self.__camera_is_usb = True if "usb" in camera_metadata["Id"] else False
            # handle framerate control
            if not self.__camera_is_usb:
                self.__camera.set_controls({"FrameRate": framerate})
            else:
                logger.warning(
                    "USB camera detected. Setting input framerate is NOT supported with Picamera2 API!"
                )
            # log
            self.__logging and logger.debug(
                "Activating Picamera2 API for `{}` camera at index: {} with resolution: {} & framerate: {}".format(
                    camera_metadata["Model"],
                    camera_num,
                    resolution if not self.__camera_is_usb else "default",
                    framerate,
                )
            )
        else:
            # initialize the picamera stream at given index
            self.__camera = PiCamera(camera_num=camera_num)
            self.__camera.resolution = tuple(resolution)
            self.__camera.framerate = framerate
            self.__logging and logger.debug(
                "Activating Picamera API at index: {} with resolution: {} & framerate: {}".format(
                    camera_num, resolution, framerate
                )
            )

        # initialize framerate (Read-only) variable
        self.framerate = framerate

        # initializing colorspace variable
        self.color_space = None

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
            if picamera2:
                # define common supported picamera2 config parameters
                valid_config_options = [
                    "auto_align_output_size",  # internal
                    "enable_verbose_logs",  # internal
                    "format",
                    "sensor",
                ]

                # define non-USB supported picamera2 config parameters
                non_usb_options = [
                    "controls",  # not-supported on USB
                    "transform",  # not-working on USB
                    "buffer_count",  # not-supported on USB
                    "queue",  # not-supported on USB
                ]  # Less are supported (will be changed in future)

                # filter parameter supported with non-USB cameras only
                if self.__camera_is_usb:
                    unsupported_config_keys = set(list(options.keys())).intersection(
                        set(non_usb_options)
                    )
                    unsupported_config_keys and logger.warning(
                        "Setting parameters: `{}` for USB camera is NOT supported with Picamera2 API!".format(
                            "`, `".join(unsupported_config_keys)
                        )
                    )
                else:
                    valid_config_options += non_usb_options

                # log all invalid keys
                invalid_config_keys = set(list(options.keys())) - set(
                    valid_config_options
                )
                invalid_config_keys and logger.warning(
                    "Discarding invalid options NOT supported by Picamera2 API for current Camera Sensor: `{}`".format(
                        "`, `".join(invalid_config_keys)
                    )
                )
                # delete all unsupported options
                options = {
                    x: y for x, y in options.items() if x in valid_config_options
                }

                # setting size, already defined
                options.update({"size": tuple(resolution)})

                # set 24-bit, BGR format by default
                if not "format" in options:
                    # auto defaults for USB cameras
                    not self.__camera_is_usb and options.update({"format": "RGB888"})
                elif self.__camera_is_usb:
                    # check the supported formats, if USB camera
                    avail_formats = [
                        mode["format"] for mode in self.__camera.sensor_modes
                    ]
                    # handle unsupported formats
                    if not options["format"] in avail_formats:
                        logger.warning(
                            "Discarding `format={}`. `{}` are the only available formats for USB camera in use!".format(
                                options["format"], "`, `".join(avail_formats)
                            )
                        )
                        del options["format"]
                    else:
                        # `colorspace` parameter must define with  `format` optional parameter
                        # unless format is MPEG (tested)
                        (
                            not (colorspace is None) or options["format"] == "MPEG"
                        ) and logger.warning(
                            "Custom Output frames `format={}` detected. It is advised to define `colorspace` parameter or handle this format manually in your code!".format(
                                options["format"]
                            )
                        )
                else:
                    # `colorspace` parameter must define with  `format` optional parameter
                    # unless format is either BGR or BGRA
                    (
                        not (colorspace is None)
                        or options["format"]
                        in [
                            "RGB888",
                            "XRGB8888",
                        ]
                    ) and logger.warning(
                        "Custom Output frames `format={}` detected. It is advised to define `colorspace` parameter or handle this format manually in your code!".format(
                            options["format"]
                        )
                    )

                # enable verbose logging mode (handled by Picamera2 API)
                verbose = options.pop("enable_verbose_logs", False)
                if self.__logging and isinstance(verbose, bool) and verbose:
                    self.__camera.set_logging(Picamera2.DEBUG)
                else:
                    # setup logging
                    self.__camera.set_logging(Picamera2.WARNING)

                # handle transformations, if specified
                transform = options.pop("transform", Transform())
                if not isinstance(transform, Transform):
                    logger.warning("`transform` value is of invalid type, Discarding!")
                    transform = Transform()

                # handle sensor configurations, if specified
                sensor = options.pop("sensor", {})
                if isinstance(sensor, dict):
                    # extract all valid sensor keys
                    valid_sensor = ["output_size", "bit_depth"]
                    # log all invalid keys
                    invalid_sensor_keys = set(list(sensor)) - set(valid_sensor)
                    invalid_sensor_keys and logger.warning(
                        "Discarding sensor properties NOT supported by current Camera Sensor: `{}`. Only supported are: (`{}`)".format(
                            "`, `".join(invalid_sensor_keys),
                            "`, `".join(valid_sensor),
                        )
                    )
                    # delete all unsupported control keys
                    sensor = {x: y for x, y in sensor.items() if x in valid_sensor}
                    # remove size if output size is defined
                    if "output_size" in sensor:
                        del options["size"]
                        logger.critical(
                            "Overriding output frame size with `output_size={}!".format(
                                sensor["output_size"]
                            )
                        )
                else:
                    logger.warning("`sensor` value is of invalid type, Discarding!")
                    sensor = {}

                # handle controls, if specified
                controls = options.pop("controls", {})
                if isinstance(controls, dict):
                    # extract all valid control keys
                    valid_controls = self.__camera.camera_controls
                    # remove any fps controls, assigned already
                    valid_controls.pop("FrameDuration", None)
                    valid_controls.pop("FrameDurationLimits", None)
                    # log all invalid keys
                    invalid_control_keys = set(list(controls.keys())) - set(
                        list(valid_controls.keys())
                    )
                    invalid_control_keys and logger.warning(
                        "Discarding control properties NOT supported by current Camera Sensor: `{}`. Only supported are: (`{}`)".format(
                            "`, `".join(invalid_control_keys),
                            "`, `".join(list(valid_controls.keys())),
                        )
                    )
                    # delete all unsupported control keys
                    controls = {
                        x: y for x, y in controls.items() if x in valid_controls.keys()
                    }
                else:
                    logger.warning("`controls` value is of invalid type, Discarding!")
                    controls = {}

                # handle buffer_count, if specified
                buffer_count = options.pop("buffer_count", 4)
                if (
                    not isinstance(buffer_count, int) or buffer_count < 1
                ):  # must be greater than 1
                    logger.warning(
                        "`buffer_count` value is of invalid type, Discarding!"
                    )
                    # `create_preview_configuration` requests 4 sets of buffers
                    buffer_count = 4

                # handle queue, if specified
                queue = options.pop("queue", True)
                if not isinstance(queue, bool):
                    logger.warning("`queue` value is of invalid type, Discarding!")
                    queue = True

                # check if auto-align camera configuration is specified
                auto_align_output_size = options.pop("auto_align_output_size", False)

                # create default configuration for camera
                config = self.__camera.create_preview_configuration(
                    main=options,
                    transform=transform,
                    sensor=sensor,
                    controls=controls,
                    buffer_count=buffer_count,
                    queue=queue,
                )

                # auto-align camera configuration, if specified
                if isinstance(auto_align_output_size, bool) and auto_align_output_size:
                    self.__logging and logger.debug(
                        "Re-aligning Output frames to optimal size supported by current Camera Sensor."
                    )
                    self.__camera.align_configuration(config)

                # configure camera
                self.__camera.configure(config)
                self.__logging and logger.debug(
                    "Setting Picamera2 API Parameters: `{}`, controls: `{}`, sensor: `{}`, buffer_count: `{}`, and queue: `{}`.".format(
                        self.__camera.camera_configuration()["main"],
                        controls,
                        sensor,
                        buffer_count,
                        queue,
                    )
                )
            else:
                # apply attributes to source if specified
                for key, value in options.items():
                    self.__logging and logger.debug(
                        "Setting {} API Parameter for Picamera: `{}`".format(key, value)
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
        if not picamera2:
            self.__rawCapture = PiRGBArray(self.__camera, size=resolution)
            self.stream = self.__camera.capture_continuous(
                self.__rawCapture, format="bgr", use_video_port=True
            )

        # initialize frame variable
        # with captured frame
        try:
            if picamera2:
                # start camera thread
                self.__camera.start()
                # capture frame array
                self.frame = self.__camera.capture_array("main")
                # assign camera as stream for setting
                # parameters after starting the camera
                self.stream = self.__camera
            else:
                # capture frame array from stream
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
        self.__t_elapsed = 0.0  # records time taken by thread

        # catching thread exceptions
        self.__exceptions = None

        # initialize termination flag
        self.__terminate = False

    def start(self) -> PIGear:
        """
        Launches the internal *Threaded Frames Extractor* daemon

        **Returns:** A reference to the PiGear class object.
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
        self.__t_elapsed = time.time()

        # loop until terminated
        while not (self.__terminate):
            # check for frozen thread
            if time.time() - self.__t_elapsed > self.__failure_timeout:
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
            if not picamera2:
                try:
                    # Try to iterate next frame from generator
                    stream = next(self.stream)
                except Exception:
                    # catch and save any exceptions
                    self.__exceptions = sys.exc_info()
                    break  # exit

            # __update timer
            self.__t_elapsed = time.time()

            # grab the frame from the stream
            if picamera2:
                frame = self.__camera.capture_array("main")
            else:
                frame = stream.array
                # clear the stream in preparation
                # for the next frame
                self.__rawCapture.seek(0)
                self.__rawCapture.truncate()

            # apply colorspace if specified
            if not (self.color_space is None):
                # apply colorspace to frames
                color_frame = None
                try:
                    color_frame = cv2.cvtColor(frame, self.color_space)
                except Exception as e:
                    # Catch if any error occurred
                    color_frame = None
                    self.color_space = None
                    self.__logging and logger.exception(str(e))
                    logger.warning("Assigned colorspace value is invalid. Discarding!")
                self.frame = color_frame if not (color_frame is None) else frame
            else:
                self.frame = frame

        # terminate processes
        if not (self.__terminate):
            self.__terminate = True

        # release resources
        if picamera2:
            self.__camera.stop()
        else:
            self.__rawCapture.close()
            self.__camera.close()

    def read(self) -> NDArray:
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
                    "[PiGear:ERROR] :: Camera Module API failure occurred: {}".format(
                        self.__exceptions[1]
                    )
                )
                raise RuntimeError(error_msg).with_traceback(self.__exceptions[2])
        # return the frame
        return self.frame

    def stop(self) -> None:
        """
        Safely terminates the thread, and release the multi-threaded resources.
        """
        # log termination
        self.__logging and logger.debug("Terminating PiGear Processes.")

        # make sure that the threads should be terminated
        self.__terminate = True

        # stop timer thread
        if not (self.__timer is None):
            self.__timer.join()
            self.__timer = None

        # handle camera thread
        if not (self.__thread is None):
            # check if hardware failure occurred
            if not (self.__exceptions is None) and isinstance(self.__exceptions, bool):
                if picamera2:
                    # release picamera2 resources
                    self.__camera.stop()
                else:
                    # force release picamera resources
                    self.__rawCapture.close()
                    self.__camera.close()
            # properly handle thread exit
            # wait if still process is still
            # processing some information
            self.__thread.join()
            # remove any threads
            self.__thread = None
