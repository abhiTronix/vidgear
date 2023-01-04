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
import time
import platform
import pathlib
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
    get_supported_pixfmts,
    get_supported_vencoders,
    check_gstreamer_support,
    logcurr_vidgear_ver,
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
        output="",
        compression_mode=True,
        custom_ffmpeg="",
        logging=False,
        **output_params
    ):

        """
        This constructor method initializes the object state and attributes of the WriteGear class.

        Parameters:
            output (str): sets the valid filename/path/URL for encoding.
            compression_mode (bool): selects the WriteGear's Primary Mode of Operation.
            custom_ffmpeg (str): assigns the location of custom path/directory for custom FFmpeg executables.
            logging (bool): enables/disables logging.
            output_params (dict): provides the flexibility to control supported internal parameters and FFmpeg properities.
        """
        # print current version
        logcurr_vidgear_ver(logging=logging)

        # check if user not using depreciated `output_filename` parameter
        assert (
            not "output_filename" in output_params
        ), "[WriteGear:ERROR] :: The `output_filename` parameter has been renamed to `output`. Refer Docs for more info."

        # assign parameter values to class variables
        # enables compression if enabled
        self.__compression = (
            compression_mode if isinstance(compression_mode, bool) else False
        )
        # specifies if machine in-use is running Windows OS or not
        self.__os_windows = True if os.name == "nt" else False
        # enable logging if specified
        self.__logging = logging if isinstance(logging, bool) else False
        # initialize various important class variables
        self.__output_parameters = {}  # handles output parameters
        self.__inputheight = None  # handles input frames height
        self.__inputwidth = None  # handles input frames width
        self.__inputchannels = None  # handles input frames channels
        self.__inputdtype = None  # handles input frames dtype
        self.__process = None  # handles Encoding class/process
        self.__ffmpeg = ""  # handles valid FFmpeg binaries location
        self.__initiate_process = (
            True  # handles initiate one-time process for generating pipeline
        )
        self.__out_file = None  # handles output
        gstpipeline_mode = False  # handles GStreamer Pipeline Mode

        # handles output
        if not output:
            # raise error otherwise
            raise ValueError(
                "[WriteGear:ERROR] :: Kindly provide a valid `output` value. Refer Docs for more info."
            )
        else:
            # validate output is a system file/directory
            # and Whether WriteGear has the write rights
            # to specified file/directory or not
            abs_path = os.path.abspath(output)
            if check_WriteAccess(
                os.path.dirname(abs_path),
                is_windows=self.__os_windows,
                logging=self.__logging,
            ):
                # check if given path is directory
                if os.path.isdir(abs_path):
                    # then, auto-assign valid name and adds it to path
                    abs_path = os.path.join(
                        abs_path,
                        "VidGear-{}.mp4".format(time.strftime("%Y%m%d-%H%M%S")),
                    )
                # assign output file absolute
                # path to class variable if valid
                self.__out_file = abs_path
            else:
                # log note otherwise
                logger.info(
                    "`{}` isn't a valid system path or directory. Skipped!".format(
                        output
                    )
                )

        # cleans and reformat output parameters
        self.__output_parameters = {
            str(k).strip(): str(v).strip()
            if not isinstance(v, (list, tuple, int, float))
            else v
            for k, v in output_params.items()
        }
        # log it if specified
        self.__logging and logger.debug(
            "Output Parameters: `{}`".format(self.__output_parameters)
        )

        # handles FFmpeg binaries validity
        # in Compression mode
        if self.__compression:
            # log it if specified
            self.__logging and logger.debug(
                "Compression Mode is enabled therefore checking for valid FFmpeg executable."
            )

            # handles where to save the downloaded FFmpeg Static Binaries
            # on Windows(if specified)
            __ffmpeg_download_path = self.__output_parameters.pop(
                "-ffmpeg_download_path", ""
            )
            # check if value is valid
            if not isinstance(__ffmpeg_download_path, (str)):
                # reset improper values
                __ffmpeg_download_path = ""

            # handle user-defined output resolution (must be a tuple or list)
            # in Compression Mode only.
            self.__output_dimensions = self.__output_parameters.pop(
                "-output_dimensions", None
            )
            # check if value is valid
            if not isinstance(self.__output_dimensions, (list, tuple)):
                # reset improper values
                self.__output_dimensions = None

            # handle user defined input framerate of encoding pipeline
            # in Compression Mode only.
            self.__inputframerate = self.__output_parameters.pop(
                "-input_framerate", 0.0
            )
            # check if value is valid
            if not isinstance(self.__inputframerate, (float, int)):
                # reset improper values
                self.__inputframerate = 0.0
            else:
                # must be float
                self.__inputframerate = float(self.__inputframerate)

            # handle user-defined input frames pixel-format in Compression Mode only.
            self.__inputpixfmt = self.__output_parameters.pop("-input_pixfmt", None)
            # check if value is valid
            if not isinstance(self.__inputpixfmt, str):
                # reset improper values
                self.__inputpixfmt = None
            else:
                # must be exact
                self.__inputpixfmt = self.__inputpixfmt.strip()

            # handle user-defined FFmpeg command pre-headers(must be a list)
            # in Compression Mode only.
            self.__ffmpeg_preheaders = self.__output_parameters.pop("-ffpreheaders", [])
            # check if value is valid
            if not isinstance(self.__ffmpeg_preheaders, list):
                # reset improper values
                self.__ffmpeg_preheaders = []

            # handle the special-case of forced-termination (only for Compression mode)
            disable_force_termination = self.__output_parameters.pop(
                "-disable_force_termination",
                False if ("-i" in self.__output_parameters) else True,
            )
            # check if value is valid
            if isinstance(disable_force_termination, bool):
                self.__forced_termination = not (disable_force_termination)
            else:
                # handle improper values
                self.__forced_termination = (
                    True if ("-i" in self.__output_parameters) else False
                )

            # validate the FFmpeg path/binaries and returns valid executable FFmpeg
            # location/path (also auto-downloads static binaries on Windows OS)
            self.__ffmpeg = get_valid_ffmpeg_path(
                custom_ffmpeg,
                self.__os_windows,
                ffmpeg_download_path=__ffmpeg_download_path,
                logging=self.__logging,
            )
            # check if valid executable FFmpeg location/path
            if self.__ffmpeg:
                # log it if found
                self.__logging and logger.debug(
                    "Found valid FFmpeg executable: `{}`.".format(self.__ffmpeg)
                )
            else:
                # otherwise disable Compression Mode
                # and switch to Non-compression mode
                logger.warning(
                    "Disabling Compression Mode since no valid FFmpeg executable found on this machine!"
                )
                if self.__logging and not self.__os_windows:
                    logger.debug(
                        "Kindly install a working FFmpeg module or provide a valid custom FFmpeg binary path. See docs for more info."
                    )
                # compression mode disabled
                self.__compression = False
        else:
            # handle GStreamer Pipeline Mode (only for Non-compression mode)
            if "-gst_pipeline_mode" in self.__output_parameters:
                # check if value is valid
                if isinstance(self.__output_parameters["-gst_pipeline_mode"], bool):
                    gstpipeline_mode = self.__output_parameters[
                        "-gst_pipeline_mode"
                    ] and check_gstreamer_support(logging=logging)
                    self.__logging and logger.debug(
                        "GStreamer Pipeline Mode successfully activated!"
                    )
                else:
                    # reset improper values
                    gstpipeline_mode = False
                    # log it
                    self.__logging and logger.warning(
                        "GStreamer Pipeline Mode failed to activate!"
                    )

        # handle output differently in Compression/Non-compression Modes
        if self.__compression and self.__ffmpeg:
            # check if output falls in exclusive cases
            if self.__out_file is None:
                if (
                    platform.system() == "Linux"
                    and pathlib.Path(output).is_char_device()
                ):
                    # check whether output is a Linux video device path (such as `/dev/video0`)
                    self.__logging and logger.debug(
                        "Path:`{}` is a valid Linux Video Device path.".format(output)
                    )
                    self.__out_file = output
                elif is_valid_url(self.__ffmpeg, url=output, logging=self.__logging):
                    # check whether output is a valid URL instead
                    self.__logging and logger.debug(
                        "URL:`{}` is valid and successfully configured for streaming.".format(
                            output
                        )
                    )
                    self.__out_file = output
                else:
                    # raise error otherwise
                    raise ValueError(
                        "[WriteGear:ERROR] :: output value:`{}` is not supported in Compression Mode.".format(
                            output
                        )
                    )
            # log if forced termination is enabled
            self.__forced_termination and logger.debug(
                "Forced termination is enabled for this FFmpeg process."
            )
            # log Compression is enabled
            self.__logging and logger.debug(
                "Compression Mode with FFmpeg backend is configured properly."
            )
        else:
            # raise error if not valid input
            if self.__out_file is None and not gstpipeline_mode:
                raise ValueError(
                    "[WriteGear:ERROR] :: output value:`{}` is not supported in Non-Compression Mode.".format(
                        output
                    )
                )

            # check if GStreamer Pipeline Mode is enabled
            if gstpipeline_mode:
                # enforce GStreamer backend
                self.__output_parameters["-backend"] = "CAP_GSTREAMER"
                # enforce original output value
                self.__out_file = output

            # log it
            self.__logging and logger.debug(
                "Non-Compression Mode is successfully configured in GStreamer Pipeline Mode."
            )

            # log if Compression is disabled
            logger.critical(
                "Compression Mode is disabled, Activating OpenCV built-in Writer!"
            )

    def write(self, frame, rgb_mode=False):

        """
        Pipelines `ndarray` frames to respective API _(**FFmpeg** in Compression Mode & **OpenCV's VideoWriter API** in Non-Compression Mode)_.

        Parameters:
            frame (ndarray): a valid numpy frame
            rgb_mode (boolean): enable this flag to activate RGB mode _(i.e. specifies that incoming frames are of RGB format(instead of default BGR)_.

        """
        if frame is None:  # None-Type frames will be skipped
            return

        # get height, width, number of channels, and dtype of current frame
        height, width = frame.shape[:2]
        channels = frame.shape[-1] if frame.ndim == 3 else 1
        dtype = frame.dtype

        # assign values to class variables on first run
        if self.__initiate_process:
            self.__inputheight = height
            self.__inputwidth = width
            self.__inputchannels = channels
            self.__inputdtype = dtype
            self.__logging and logger.debug(
                "InputFrame => Height:{} Width:{} Channels:{} Datatype:{}".format(
                    self.__inputheight,
                    self.__inputwidth,
                    self.__inputchannels,
                    self.__inputdtype,
                )
            )

        # validate frame size
        if height != self.__inputheight or width != self.__inputwidth:
            raise ValueError(
                "[WriteGear:ERROR] :: All video-frames must have same size!"
            )
        # validate number of channels in frame
        if channels != self.__inputchannels:
            raise ValueError(
                "[WriteGear:ERROR] :: All video-frames must have same number of channels!"
            )
        # validate frame datatype
        if dtype != self.__inputdtype:
            raise ValueError(
                "[WriteGear:ERROR] :: All video-frames must have same datatype!"
            )

        # checks if compression mode is enabled
        if self.__compression:
            # initiate FFmpeg process on first run
            if self.__initiate_process:
                # start pre-processing of FFmpeg parameters, and initiate process
                self.__PreprocessFFParams(channels, dtype=dtype, rgb=rgb_mode)
                # Check status of the process
                assert self.__process is not None
            try:
                # try writing the frame bytes to the subprocess pipeline
                self.__process.stdin.write(frame.tobytes())
            except (OSError, IOError):
                # log if something is wrong!
                logger.error(
                    "BrokenPipeError caught, Wrong values passed to FFmpeg Pipe. Kindly Refer Docs!"
                )
                raise ValueError  # for testing purpose only
        else:
            # otherwise initiate OpenCV's VideoWriter Class process
            if self.__initiate_process:
                # start VideoWriter Class process
                self.__start_CVProcess()
                # Check status of the process
                assert self.__process is not None
                # log one-time OpenCV warning
                self.__logging and logger.info(
                    "RGBA and 16-bit grayscale video frames are not supported by OpenCV yet. Kindly switch on `compression_mode` to use them!"
                )
            # write frame directly to
            # VideoWriter Class process
            self.__process.write(frame)

    def __PreprocessFFParams(self, channels, dtype=None, rgb=False):
        """
        Internal method that pre-processes FFmpeg Parameters before beginning to pipeline frames.

        Parameters:
            channels (int): Number of channels in input frame.
            dtype (str): Datatype of input frame.
            rgb_mode (boolean): Whether to activate `RGB mode`?
        """
        # turn off initiate flag
        self.__initiate_process = False
        # initialize input parameters
        input_parameters = {}

        # handle output frames dimensions
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

        # handles user-defined and auto-assigned input pixel-formats
        if not (
            self.__inputpixfmt is None
        ) and self.__inputpixfmt in get_supported_pixfmts(self.__ffmpeg):
            # assign directly if valid
            input_parameters["-pix_fmt"] = self.__inputpixfmt
        else:
            # handles pix_fmt based on channels and dtype(HACK)
            if dtype.kind == "u" and dtype.itemsize == 2:
                # handle pix_fmt for frames with higher than 8-bit depth
                pix_fmt = None
                if channels == 1:
                    pix_fmt = "gray16"
                elif channels == 2:
                    pix_fmt = "ya16"
                elif channels == 3:
                    pix_fmt = "rgb48" if rgb else "bgr48"
                elif channels == 4:
                    pix_fmt = "rgba64" if rgb else "bgra64"
                else:
                    # raise error otherwise
                    raise ValueError(
                        "[WriteGear:ERROR] :: Frames with channels outside range 1-to-4 are not supported!"
                    )
                # Add endianness suffix (w.r.t byte-order)
                input_parameters["-pix_fmt"] = pix_fmt + (
                    "be" if dtype.byteorder == ">" else "le"
                )
            else:
                # handle pix_fmt for frames with exactly 8-bit depth(`uint8`)
                if channels == 1:
                    input_parameters["-pix_fmt"] = "gray"
                elif channels == 2:
                    input_parameters["-pix_fmt"] = "ya8"
                elif channels == 3:
                    input_parameters["-pix_fmt"] = "rgb24" if rgb else "bgr24"
                elif channels == 4:
                    input_parameters["-pix_fmt"] = "rgba" if rgb else "bgra"
                else:
                    # raise error otherwise
                    raise ValueError(
                        "[WriteGear:ERROR] :: Frames with channels outside range 1-to-4 are not supported!"
                    )

        # handles user-defined output video framerate
        if self.__inputframerate > 0.0:
            # assign input framerate if valid
            self.__logging and logger.debug(
                "Setting Input framerate: {}".format(self.__inputframerate)
            )
            input_parameters["-framerate"] = str(self.__inputframerate)

        # initiate FFmpeg process
        self.__start_FFProcess(
            input_params=input_parameters, output_params=self.__output_parameters
        )

    def __start_FFProcess(self, input_params, output_params):

        """
        An Internal method that launches FFmpeg subprocess pipeline in Compression Mode
        for pipelining frames to `stdin`.

        Parameters:
            input_params (dict): Input FFmpeg parameters
            output_params (dict): Output FFmpeg parameters
        """
        # convert input parameters to argument list
        input_parameters = dict2Args(input_params)

        # handle output video encoder.
        # get list of supported video-encoders
        supported_vcodecs = get_supported_vencoders(self.__ffmpeg)
        # dynamically select default encoder
        default_vcodec = [
            vcodec
            for vcodec in ["libx264", "libx265", "libxvid", "mpeg4"]
            if vcodec in supported_vcodecs
        ][0] or "unknown"
        # extract any user-defined encoder
        if "-c:v" in output_params:
            # assign it to the pipeline
            output_params["-vcodec"] = output_params.pop("-c:v", default_vcodec)
        if not "-vcodec" in output_params:
            # auto-assign default video-encoder (if not assigned by user).
            output_params["-vcodec"] = default_vcodec
        if (
            default_vcodec != "unknown"
            and not output_params["-vcodec"] in supported_vcodecs
        ):
            # reset to default if not supported
            logger.critical(
                "Provided FFmpeg does not support `{}` video-encoder. Switching to default supported `{}` encoder!".format(
                    output_params["-vcodec"], default_vcodec
                )
            )
            output_params["-vcodec"] = default_vcodec

        # assign optimizations based on selected video encoder(if any)
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
            # raise error otherwise
            raise RuntimeError(
                "[WriteGear:ERROR] :: Provided FFmpeg does not support any suitable/usable video-encoders for compression."
                " Kindly disable compression mode or switch to another FFmpeg binaries(if available)."
            )

        # convert output parameters to argument list
        output_parameters = dict2Args(output_params)

        # format FFmpeg command
        cmd = (
            [self.__ffmpeg, "-y"]
            + self.__ffmpeg_preheaders
            + ["-f", "rawvideo", "-vcodec", "rawvideo"]
            + input_parameters
            + ["-i", "-"]
            + output_parameters
            + [self.__out_file]
        )
        # Launch the process with FFmpeg command
        if self.__logging:
            # log command in logging mode
            logger.debug("Executing FFmpeg command: `{}`".format(" ".join(cmd)))
            # In logging mode
            self.__process = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=None)
        else:
            # In silent mode
            self.__process = sp.Popen(
                cmd, stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.STDOUT
            )

    def __enter__(self):
        """
        Handles entry with the `with` statement. See [PEP343 -- The 'with' statement'](https://peps.python.org/pep-0343/).

        **Returns:** Returns a reference to the WriteGear Class
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Handles exit with the `with` statement. See [PEP343 -- The 'with' statement'](https://peps.python.org/pep-0343/).
        """
        self.close()

    def execute_ffmpeg_cmd(self, command=None):
        """

        Executes user-defined FFmpeg Terminal command, formatted as a python list(in Compression Mode only).

        Parameters:
            command (list): inputs list data-type command.

        """
        # check if valid command
        if command is None or not (command):
            logger.warning("Input command is empty, Nothing to execute!")
            return
        else:
            if not (isinstance(command, list)):
                raise ValueError(
                    "[WriteGear:ERROR] :: Invalid input command datatype! Kindly read docs."
                )

        # check if Compression Mode is enabled
        if not (self.__compression):
            # raise error otherwise
            raise RuntimeError(
                "[WriteGear:ERROR] :: Compression Mode is disabled, Kindly enable it to access this function."
            )

        # add configured FFmpeg path
        cmd = [self.__ffmpeg] + command

        try:
            # write frames to pipeline
            if self.__logging:
                # log command in logging mode
                logger.debug("Executing FFmpeg command: `{}`".format(" ".join(cmd)))
                # In logging mode
                sp.run(cmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=None)
            else:
                # In silent mode
                sp.run(cmd, stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.STDOUT)
        except (OSError, IOError):
            # raise error and log if something is wrong.
            logger.error(
                "BrokenPipeError caught, Wrong command passed to FFmpeg Pipe, Kindly Refer Docs!"
            )
            raise ValueError  # for testing purpose only

    def __start_CVProcess(self):
        """
        An Internal method that launches OpenCV VideoWriter process in Non-Compression
        Mode with given settings.
        """
        # turn off initiate flag
        self.__initiate_process = False

        # initialize essential variables
        FPS = 0
        BACKEND = ""
        FOURCC = 0
        COLOR = True

        # pre-assign default parameters (if not assigned by user).
        if "-fourcc" not in self.__output_parameters:
            FOURCC = cv2.VideoWriter_fourcc(*"MJPG")
        if "-fps" not in self.__output_parameters:
            FPS = 25

        # auto-assign frame dimensions
        HEIGHT = self.__inputheight
        WIDTH = self.__inputwidth

        # assign dict parameter values to variables
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
            # log and raise error if something is wrong
            self.__logging and logger.exception(str(e))
            raise ValueError(
                "[WriteGear:ERROR] :: Wrong Values passed to OpenCV Writer, Kindly Refer Docs!"
            )

        # log values for debugging
        self.__logging and logger.debug(
            "FILE_PATH: {}, FOURCC = {}, FPS = {}, WIDTH = {}, HEIGHT = {}, BACKEND = {}".format(
                self.__out_file, FOURCC, FPS, WIDTH, HEIGHT, BACKEND
            )
        )
        # start different OpenCV VideoCapture processes
        # for with and without Backend.
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
        # check if OpenCV VideoCapture is opened successfully
        assert (
            self.__process.isOpened()
        ), "[WriteGear:ERROR] :: Failed to intialize OpenCV Writer!"

    def close(self):
        """
        Safely terminates various WriteGear process.
        """
        # log termination
        if self.__logging:
            logger.debug("Terminating WriteGear Processes.")
        # handle termination separately
        if self.__compression:
            # when Compression Mode is enabled
            if self.__process is None or not (self.__process.poll() is None):
                # return if no process initiated
                # at first place
                return
            # close `stdin` output
            self.__process.stdin and self.__process.stdin.close()
            # close `stdout` output
            self.__process.stdout and self.__process.stdout.close()
            # forced termination if specified.
            self.__forced_termination and self.__process.terminate()
            # wait if process is still processing
            self.__process.wait()
        else:
            # when Compression Mode is disabled
            if self.__process is None:
                # return if no process initiated
                # at first place
                return
            # close it
            self.__process.release()
        # discard process
        self.__process = None
