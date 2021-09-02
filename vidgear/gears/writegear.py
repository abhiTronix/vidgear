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
import os
import cv2
import sys
import time
import logging as log
import subprocess as sp

# import helper packages
from .helper import (
    capPropId,
    dict2Args,
    is_valid_url,
    logger_handler,
    check_WriteAccess,
    get_valid_ffmpeg_path,
    get_supported_vencoders,
)

# define logger
logger = log.getLogger("WriteGear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


class WriteGear:

    """
    WriteGear handles various powerful Video-Writer Tools that provide us the freedom to do almost anything imaginable with multimedia data.

    WriteGear API provides a complete, flexible, and robust wrapper around FFmpeg, a leading multimedia framework. WriteGear can process real-time frames into a lossless
    compressed video-file with any suitable specification (such asbitrate, codec, framerate, resolution, subtitles, etc.). It is powerful enough to perform complex tasks such as
    Live-Streaming (such as for Twitch) and Multiplexing Video-Audio with real-time frames in way fewer lines of code.

    Best of all, WriteGear grants users the complete freedom to play with any FFmpeg parameter with its exclusive Custom Commands function without relying on any
    third-party API.

    In addition to this, WriteGear also provides flexible access to OpenCV's VideoWriter API tools for video-frames encoding without compression.

    ??? tip "Modes of Operation"

        WriteGear primarily operates in following modes:

        * **Compression Mode**: In this mode, WriteGear utilizes powerful **FFmpeg** inbuilt encoders to encode lossless multimedia files.
                                This mode provides us the ability to exploit almost any parameter available within FFmpeg, effortlessly and flexibly,
                                and while doing that it robustly handles all errors/warnings quietly.

        * **Non-Compression Mode**: In this mode, WriteGear utilizes basic **OpenCV's inbuilt VideoWriter API** tools. This mode also supports all
                                    parameters manipulation available within VideoWriter API, but it lacks the ability to manipulate encoding parameters
                                    and other important features like video compression, audio encoding, etc.

    """

    def __init__(
        self,
        output_filename="",
        compression_mode=True,
        custom_ffmpeg="",
        logging=False,
        **output_params
    ):

        """
        This constructor method initializes the object state and attributes of the WriteGear class.

        Parameters:
            output_filename (str): sets the valid filename/path/URL for the video output.
            compression_mode (bool): selects the WriteGear's Primary Mode of Operation.
            custom_ffmpeg (str): assigns the location of custom path/directory for custom FFmpeg executables.
            logging (bool): enables/disables logging.
            output_params (dict): provides the flexibility to control supported internal parameters and FFmpeg properities.
        """

        # assign parameter values to class variables
        self.__compression = compression_mode
        self.__os_windows = (
            True if os.name == "nt" else False
        )  # checks if machine in-use is running windows os or not

        # enable logging if specified
        self.__logging = False
        if logging:
            self.__logging = logging

        # initialize various important class variables
        self.__output_parameters = {}
        self.__inputheight = None
        self.__inputwidth = None
        self.__inputchannels = None
        self.__process = None  # handle process to be frames written
        self.__cmd = ""  # handle FFmpeg Pipe command
        self.__ffmpeg = ""  # handle valid FFmpeg binaries location
        self.__initiate = (
            True  # initiate one time process for valid process initialization
        )
        self.__out_file = None  # handles output filename

        # handles output file name (if not given)
        if not output_filename:
            raise ValueError(
                "[WriteGear:ERROR] :: Kindly provide a valid `output_filename` value. Refer Docs for more information."
            )
        else:
            # validate this class has the access rights to specified directory or not
            abs_path = os.path.abspath(output_filename)

            if check_WriteAccess(
                os.path.dirname(abs_path), is_windows=self.__os_windows
            ):
                if os.path.isdir(abs_path):  # check if given path is directory
                    abs_path = os.path.join(
                        abs_path,
                        "VidGear-{}.mp4".format(time.strftime("%Y%m%d-%H%M%S")),
                    )  # auto-assign valid name and adds it to path

                # assign output file absolute path to class variable
                self.__out_file = abs_path
            else:
                # log warning if
                logger.warning(
                    "The given path:`{}` does not have write access permission. Skipped!".format(
                        output_filename
                    )
                )

        # cleans and reformat output parameters
        self.__output_parameters = {
            str(k).strip(): str(v).strip()
            if not isinstance(v, (list, tuple, int, float))
            else v
            for k, v in output_params.items()
        }

        # handles FFmpeg binaries validity tests
        if self.__compression:

            if self.__logging:
                logger.debug(
                    "Compression Mode is enabled therefore checking for valid FFmpeg executables."
                )
                logger.debug("Output Parameters: {}".format(self.__output_parameters))

            # handles where to save the downloaded FFmpeg Static Binaries on Windows(if specified)
            __ffmpeg_download_path = self.__output_parameters.pop(
                "-ffmpeg_download_path", ""
            )
            if not isinstance(__ffmpeg_download_path, (str)):
                # reset improper values
                __ffmpeg_download_path = ""

            # handle user defined output dimensions(must be a tuple or list)
            self.__output_dimensions = self.__output_parameters.pop(
                "-output_dimensions", None
            )
            if not isinstance(self.__output_dimensions, (list, tuple)):
                # reset improper values
                self.__output_dimensions = None

            # handle user defined framerate
            self.__inputframerate = self.__output_parameters.pop(
                "-input_framerate", 0.0
            )
            if not isinstance(self.__inputframerate, (float, int)):
                # reset improper values
                self.__inputframerate = 0.0
            else:
                # must be float
                self.__inputframerate = float(self.__inputframerate)

            # handle special-case force-termination in compression mode
            self.__force_termination = self.__output_parameters.pop(
                "-disable_force_termination", False
            )
            if not isinstance(self.__force_termination, bool):
                # handle improper values
                self.__force_termination = (
                    True if ("-i" in self.__output_parameters) else False
                )
            else:
                self.__force_termination = (
                    self.__force_termination
                    if ("-i" in self.__output_parameters)
                    else False
                )

            # validate the FFmpeg path/binaries and returns valid FFmpeg file executable location (also downloads static binaries on windows)
            self.__ffmpeg = get_valid_ffmpeg_path(
                custom_ffmpeg,
                self.__os_windows,
                ffmpeg_download_path=__ffmpeg_download_path,
                logging=self.__logging,
            )

            # check if valid path returned
            if self.__ffmpeg:
                if self.__logging:
                    logger.debug(
                        "Found valid FFmpeg executables: `{}`.".format(self.__ffmpeg)
                    )
            else:
                # otherwise disable Compression Mode
                logger.warning(
                    "Disabling Compression Mode since no valid FFmpeg executables found on this machine!"
                )
                if self.__logging and not self.__os_windows:
                    logger.debug(
                        "Kindly install working FFmpeg or provide a valid custom FFmpeg binary path. See docs for more info."
                    )
                self.__compression = False  # compression mode disabled

        # display confirmation if logging is enabled/disabled
        if self.__compression and self.__ffmpeg:
            # check whether url is valid instead
            if self.__out_file is None:
                if is_valid_url(
                    self.__ffmpeg, url=output_filename, logging=self.__logging
                ):
                    if self.__logging:
                        logger.debug(
                            "URL:`{}` is successfully configured for streaming.".format(
                                output_filename
                            )
                        )
                    self.__out_file = output_filename
                else:
                    raise ValueError(
                        "[WriteGear:ERROR] :: output_filename value:`{}` is not valid/supported in Compression Mode!".format(
                            output_filename
                        )
                    )
            if self.__logging:
                logger.debug("Compression Mode is configured properly!")
        else:
            if self.__out_file is None:
                raise ValueError(
                    "[WriteGear:ERROR] :: output_filename value:`{}` is not vaild in Non-Compression Mode!".format(
                        output_filename
                    )
                )
            if self.__logging:
                logger.debug(
                    "Compression Mode is disabled, Activating OpenCV built-in Writer!"
                )

    def write(self, frame, rgb_mode=False):

        """
        Pipelines `ndarray` frames to respective API _(FFmpeg in Compression Mode & OpenCV VideoWriter API in Non-Compression Mode)_.

        Parameters:
            frame (ndarray): a valid numpy frame
            rgb_mode (boolean): enable this flag to activate RGB mode _(i.e. specifies that incoming frames are of RGB format(instead of default BGR)_.

        """
        if frame is None:  # None-Type frames will be skipped
            return

        # get height, width and number of channels of current frame
        height, width = frame.shape[:2]
        channels = frame.shape[-1] if frame.ndim == 3 else 1

        # assign values to class variables on first run
        if self.__initiate:
            self.__inputheight = height
            self.__inputwidth = width
            self.__inputchannels = channels
            if self.__logging:
                logger.debug(
                    "InputFrame => Height:{} Width:{} Channels:{}".format(
                        self.__inputheight, self.__inputwidth, self.__inputchannels
                    )
                )

        # validate size of frame
        if height != self.__inputheight or width != self.__inputwidth:
            raise ValueError("[WriteGear:ERROR] :: All frames must have same size!")
        # validate number of channels
        if channels != self.__inputchannels:
            raise ValueError(
                "[WriteGear:ERROR] :: All frames must have same number of channels!"
            )

        if self.__compression:
            # checks if compression mode is enabled

            # initiate FFmpeg process on first run
            if self.__initiate:
                # start pre-processing and initiate process
                self.__Preprocess(channels, rgb=rgb_mode)
                # Check status of the process
                assert self.__process is not None

            # write the frame
            try:
                self.__process.stdin.write(frame.tostring())
            except (OSError, IOError):
                # log something is wrong!
                logger.error(
                    "BrokenPipeError caught, Wrong values passed to FFmpeg Pipe, Kindly Refer Docs!"
                )
                raise ValueError  # for testing purpose only
        else:
            # otherwise initiate OpenCV's VideoWriter Class
            if self.__initiate:
                # start VideoWriter Class process
                self.__startCV_Process()
                # Check status of the process
                assert self.__process is not None
                if self.__logging:
                    # log OpenCV warning
                    logger.info(
                        "RGBA and 16-bit grayscale video frames are not supported by OpenCV yet, switch to `compression_mode` to use them!"
                    )
            # write the frame
            self.__process.write(frame)

    def __Preprocess(self, channels, rgb=False):
        """
        Internal method that pre-processes FFmpeg Parameters before beginning pipelining.

        Parameters:
            channels (int): Number of channels
            rgb_mode (boolean): activates RGB mode _(if enabled)_.
        """
        # turn off initiate flag
        self.__initiate = False
        # initialize input parameters
        input_parameters = {}

        # handle dimensions
        dimensions = ""
        if self.__output_dimensions is None:  # check if dimensions are given
            dimensions += "{}x{}".format(
                self.__inputwidth, self.__inputheight
            )  # auto derive from frame
        else:
            dimensions += "{}x{}".format(
                self.__output_dimensions[0], self.__output_dimensions[1]
            )  # apply if defined
        input_parameters["-s"] = str(dimensions)

        # handles pix_fmt based on channels(HACK)
        if channels == 1:
            input_parameters["-pix_fmt"] = "gray"
        elif channels == 2:
            input_parameters["-pix_fmt"] = "ya8"
        elif channels == 3:
            input_parameters["-pix_fmt"] = "rgb24" if rgb else "bgr24"
        elif channels == 4:
            input_parameters["-pix_fmt"] = "rgba" if rgb else "bgra"
        else:
            raise ValueError(
                "[WriteGear:ERROR] :: Frames with channels outside range 1-to-4 are not supported!"
            )

        if self.__inputframerate > 5:
            # set input framerate - minimum threshold is 5.0
            if self.__logging:
                logger.debug(
                    "Setting Input framerate: {}".format(self.__inputframerate)
                )
            input_parameters["-framerate"] = str(self.__inputframerate)

        # initiate FFmpeg process
        self.__startFFmpeg_Process(
            input_params=input_parameters, output_params=self.__output_parameters
        )

    def __startFFmpeg_Process(self, input_params, output_params):

        """
        An Internal method that launches FFmpeg subprocess, that pipelines frames to
        stdin, in Compression Mode.

        Parameters:
            input_params (dict): Input FFmpeg parameters
            output_params (dict): Output FFmpeg parameters
        """
        # convert input parameters to list
        input_parameters = dict2Args(input_params)

        # dynamically pre-assign a default video-encoder (if not assigned by user).
        supported_vcodecs = get_supported_vencoders(self.__ffmpeg)
        default_vcodec = [
            vcodec
            for vcodec in ["libx264", "libx265", "libxvid", "mpeg4"]
            if vcodec in supported_vcodecs
        ][0] or "unknown"
        if "-c:v" in output_params:
            output_params["-vcodec"] = output_params.pop("-c:v", default_vcodec)
        if not "-vcodec" in output_params:
            output_params["-vcodec"] = default_vcodec
        if (
            default_vcodec != "unknown"
            and not output_params["-vcodec"] in supported_vcodecs
        ):
            logger.critical(
                "Provided FFmpeg does not support `{}` video-encoder. Switching to default supported `{}` encoder!".format(
                    output_params["-vcodec"], default_vcodec
                )
            )
            output_params["-vcodec"] = default_vcodec

        # assign optimizations
        if output_params["-vcodec"] in supported_vcodecs:
            if output_params["-vcodec"] in ["libx265", "libx264"]:
                if not "-crf" in output_params:
                    output_params["-crf"] = "18"
                if not "-preset" in output_params:
                    output_params["-preset"] = "fast"
            if output_params["-vcodec"] in ["libxvid", "mpeg4"]:
                if not "-qscale:v" in output_params:
                    output_params["-qscale:v"] = "3"
        else:
            raise RuntimeError(
                "[WriteGear:ERROR] :: Provided FFmpeg does not support any suitable/usable video-encoders for compression."
                " Kindly disable compression mode or switch to another FFmpeg(if available)."
            )

        # convert output parameters to list
        output_parameters = dict2Args(output_params)
        # format command
        cmd = (
            [self.__ffmpeg, "-y"]
            + ["-f", "rawvideo", "-vcodec", "rawvideo"]
            + input_parameters
            + ["-i", "-"]
            + output_parameters
            + [self.__out_file]
        )
        # assign value to class variable
        self.__cmd += " ".join(cmd)
        # Launch the FFmpeg process
        if self.__logging:
            logger.debug("Executing FFmpeg command: `{}`".format(self.__cmd))
            # In debugging mode
            self.__process = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=None)
        else:
            # In silent mode
            self.__process = sp.Popen(
                cmd, stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.STDOUT
            )

    def execute_ffmpeg_cmd(self, cmd=None):
        """

        Executes user-defined FFmpeg Terminal command, formatted as a python list(in Compression Mode only).

        Parameters:
            cmd (list): inputs list data-type command.

        """
        # check if valid command
        if cmd is None or not (cmd):
            logger.warning("Input FFmpeg command is empty, Nothing to execute!")
            return
        else:
            if not (isinstance(cmd, list)):
                raise ValueError(
                    "[WriteGear:ERROR] :: Invalid input FFmpeg command datatype! Kindly read docs."
                )

        # check if Compression Mode is enabled
        if not (self.__compression):
            raise RuntimeError(
                "[WriteGear:ERROR] :: Compression Mode is disabled, Kindly enable it to access this function!"
            )

        # add configured FFmpeg path
        cmd = [self.__ffmpeg] + cmd

        try:
            # write to pipeline
            if self.__logging:
                logger.debug("Executing FFmpeg command: `{}`".format(" ".join(cmd)))
                # In debugging mode
                sp.run(cmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=None)
            else:
                sp.run(cmd, stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.STDOUT)
        except (OSError, IOError):
            # log something is wrong!
            logger.error(
                "BrokenPipeError caught, Wrong command passed to FFmpeg Pipe, Kindly Refer Docs!"
            )
            raise ValueError  # for testing purpose only

    def __startCV_Process(self):
        """
        An Internal method that launches OpenCV VideoWriter process for given settings, in Non-Compression Mode.
        """
        # turn off initiate flag
        self.__initiate = False

        # initialize essential parameter variables
        FPS = 0
        BACKEND = ""
        FOURCC = 0
        COLOR = True

        # pre-assign default encoder parameters (if not assigned by user).
        if "-fourcc" not in self.__output_parameters:
            FOURCC = cv2.VideoWriter_fourcc(*"MJPG")
        if "-fps" not in self.__output_parameters:
            FPS = 25

        # auto assign dimensions
        HEIGHT = self.__inputheight
        WIDTH = self.__inputwidth

        # assign parameter dict values to variables
        try:
            for key, value in self.__output_parameters.items():
                if key == "-fourcc":
                    FOURCC = cv2.VideoWriter_fourcc(*(value.upper()))
                elif key == "-fps":
                    FPS = int(value)
                elif key == "-backend":
                    BACKEND = capPropId(value.upper())
                elif key == "-color":
                    COLOR = bool(value)
                else:
                    pass

        except Exception as e:
            # log if something is wrong
            if self.__logging:
                logger.exception(str(e))
            raise ValueError(
                "[WriteGear:ERROR] :: Wrong Values passed to OpenCV Writer, Kindly Refer Docs!"
            )

        if self.__logging:
            # log values for debugging
            logger.debug(
                "FILE_PATH: {}, FOURCC = {}, FPS = {}, WIDTH = {}, HEIGHT = {}, BACKEND = {}".format(
                    self.__out_file, FOURCC, FPS, WIDTH, HEIGHT, BACKEND
                )
            )
        # start different process for with/without Backend.
        if BACKEND:
            self.__process = cv2.VideoWriter(
                self.__out_file,
                apiPreference=BACKEND,
                fourcc=FOURCC,
                fps=FPS,
                frameSize=(WIDTH, HEIGHT),
                isColor=COLOR,
            )
        else:
            self.__process = cv2.VideoWriter(
                self.__out_file,
                fourcc=FOURCC,
                fps=FPS,
                frameSize=(WIDTH, HEIGHT),
                isColor=COLOR,
            )

    def close(self):
        """
        Safely terminates various WriteGear process.
        """
        if self.__logging:
            logger.debug("Terminating WriteGear Processes.")

        if self.__compression:
            # if Compression Mode is enabled
            if self.__process is None or not (self.__process.poll() is None):
                return  # no process was initiated at first place
            if self.__process.stdin:
                self.__process.stdin.close()  # close `stdin` output
            if self.__force_termination:
                self.__process.terminate()
            self.__process.wait()  # wait if still process is still processing some information
            self.__process = None
        else:
            # if Compression Mode is disabled
            if self.__process is None:
                return  # no process was initiated at first place
            self.__process.release()  # close it
