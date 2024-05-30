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
import time
import math
import signal
import difflib
import logging as log
import subprocess as sp
from tqdm import tqdm
from fractions import Fraction
from collections import OrderedDict

# import helper packages
from .helper import (
    deprecated,
    dict2Args,
    delete_ext_safe,
    extract_time,
    is_valid_url,
    logger_handler,
    validate_audio,
    validate_video,
    check_WriteAccess,
    get_video_bitrate,
    get_valid_ffmpeg_path,
    logcurr_vidgear_ver,
)

# define logger
logger = log.getLogger("StreamGear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


class StreamGear:
    """
    StreamGear automates transcoding workflow for generating Ultra-Low Latency, High-Quality, Dynamic & Adaptive Streaming Formats (such as MPEG-DASH and HLS) in just few lines of python code.
    StreamGear provides a standalone, highly extensible, and flexible wrapper around FFmpeg multimedia framework for generating chunked-encoded media segments of the content.

    SteamGear easily transcodes source videos/audio files & real-time video-frames and breaks them into a sequence of multiple smaller chunks/segments of suitable length. These segments make it
    possible to stream videos at different quality levels _(different bitrate or spatial resolutions)_ and can be switched in the middle of a video from one quality level to another - if bandwidth
    permits - on a per-segment basis. A user can serve these segments on a web server that makes it easier to download them through HTTP standard-compliant GET requests.

    SteamGear also creates a Manifest/Playlist file (such as MPD in-case of DASH and M3U8 in-case of HLS) besides segments that describe these segment information
    (timing, URL, media characteristics like video resolution and bit rates) and is provided to the client before the streaming session.

    SteamGear currently supports MPEG-DASH (Dynamic Adaptive Streaming over HTTP, ISO/IEC 23009-1) and Apple HLS (HTTP live streaming).
    """

    def __init__(
        self, output="", format="dash", custom_ffmpeg="", logging=False, **stream_params
    ):
        """
        This constructor method initializes the object state and attributes of the StreamGear class.

        Parameters:
            output (str): sets the valid filename/path for generating the StreamGear assets.
            format (str): select the adaptive HTTP streaming format(DASH and HLS).
            custom_ffmpeg (str): assigns the location of custom path/directory for custom FFmpeg executables.
            logging (bool): enables/disables logging.
            stream_params (dict): provides the flexibility to control supported internal parameters and FFmpeg properties.
        """
        # enable logging if specified
        self.__logging = logging if isinstance(logging, bool) else False

        # print current version
        logcurr_vidgear_ver(logging=self.__logging)

        # checks if machine in-use is running windows os or not
        self.__os_windows = True if os.name == "nt" else False

        # initialize various class variables
        # handles user-defined parameters
        self.__params = {}
        # handle input video/frame resolution and channels
        self.__inputheight = None
        self.__inputwidth = None
        self.__inputchannels = None
        self.__sourceframerate = None
        # handle process to be frames written
        self.__process = None
        # handle valid FFmpeg assets location
        self.__ffmpeg = ""
        # handle one time process for valid process initialization
        self.__initiate_stream = True

        # cleans and reformat user-defined parameters
        self.__params = {
            str(k).strip(): (v.strip() if isinstance(v, str) else v)
            for k, v in stream_params.items()
        }

        # handle where to save the downloaded FFmpeg Static assets on Windows(if specified)
        __ffmpeg_download_path = self.__params.pop("-ffmpeg_download_path", "")
        if not isinstance(__ffmpeg_download_path, (str)):
            # reset improper values
            __ffmpeg_download_path = ""

        # validate the FFmpeg assets and return location (also downloads static assets on windows)
        self.__ffmpeg = get_valid_ffmpeg_path(
            str(custom_ffmpeg),
            self.__os_windows,
            ffmpeg_download_path=__ffmpeg_download_path,
            logging=self.__logging,
        )

        # check if valid FFmpeg path returned
        if self.__ffmpeg:
            self.__logging and logger.debug(
                "Found valid FFmpeg executables: `{}`.".format(self.__ffmpeg)
            )
        else:
            # else raise error
            raise RuntimeError(
                "[StreamGear:ERROR] :: Failed to find FFmpeg assets on this system. Kindly compile/install FFmpeg or provide a valid custom FFmpeg binary path!"
            )

        # handle Audio-Input
        audio = self.__params.pop("-audio", False)
        if audio and isinstance(audio, str):
            if os.path.isfile(audio):
                self.__audio = os.path.abspath(audio)
            elif is_valid_url(self.__ffmpeg, url=audio, logging=self.__logging):
                self.__audio = audio
            else:
                self.__audio = False
        elif audio and isinstance(audio, list):
            self.__audio = audio
        else:
            self.__audio = False
        # log external audio source
        self.__audio and self.__logging and logger.debug(
            "External audio source `{}` detected.".format(self.__audio)
        )

        # handle Video-Source input
        source = self.__params.pop("-video_source", False)
        # Check if input is valid string
        if source and isinstance(source, str) and len(source) > 1:
            # Differentiate input
            if os.path.isfile(source):
                self.__video_source = os.path.abspath(source)
            elif is_valid_url(self.__ffmpeg, url=source, logging=self.__logging):
                self.__video_source = source
            else:
                # discard the value otherwise
                self.__video_source = False

            # Validate input
            if self.__video_source:
                validation_results = validate_video(
                    self.__ffmpeg, video_path=self.__video_source
                )
                assert not (
                    validation_results is None
                ), "[StreamGear:ERROR] :: Given `{}` video_source is Invalid, Check Again!".format(
                    self.__video_source
                )
                self.__aspect_source = validation_results["resolution"]
                self.__fps_source = validation_results["framerate"]
                # log it
                self.__logging and logger.debug(
                    "Given video_source is valid and has {}x{} resolution, and a framerate of {} fps.".format(
                        self.__aspect_source[0],
                        self.__aspect_source[1],
                        self.__fps_source,
                    )
                )
            else:
                # log warning
                logger.warning("Discarded invalid `-video_source` value provided.")
        else:
            if source:
                # log warning if source provided
                logger.warning("Invalid `-video_source` value provided.")
            else:
                # log normally
                logger.info("No `-video_source` value provided.")
            # discard the value otherwise
            self.__video_source = False

        # handle user-defined framerate
        self.__inputframerate = self.__params.pop("-input_framerate", 0.0)
        if isinstance(self.__inputframerate, (float, int)):
            # must be float
            self.__inputframerate = float(self.__inputframerate)
        else:
            # reset improper values
            self.__inputframerate = 0.0

        # handle old assets
        clear_assets = self.__params.pop("-clear_prev_assets", False)
        if isinstance(clear_assets, bool):
            self.__clear_assets = clear_assets
            # log if clearing assets is enabled
            clear_assets and logger.debug(
                "Previous StreamGear API assets will be deleted in this run."
            )
        else:
            # reset improper values
            self.__clear_assets = False

        # handle whether to livestream?
        livestreaming = self.__params.pop("-livestream", False)
        if isinstance(livestreaming, bool):
            self.__livestreaming = livestreaming
            # log if live streaming is enabled
            livestreaming and logger.info(
                "Live-Streaming Mode is enabled for this run."
            )
        else:
            # reset improper values
            self.__livestreaming = False

        # handle the special-case of forced-termination
        enable_force_termination = self.__params.pop("-enable_force_termination", False)
        # check if value is valid
        if isinstance(enable_force_termination, bool):
            self.__forced_termination = enable_force_termination
            # log if forced termination is enabled
            self.__forced_termination and logger.info(
                "Forced termination is enabled for this run."
            )
        else:
            # handle improper values
            self.__forced_termination = False

        # handle streaming format
        supported_formats = ["dash", "hls"]  # TODO will be extended in future
        if format and isinstance(format, str):
            _format = format.strip().lower()
            if _format in supported_formats:
                self.__format = _format
                logger.info(
                    "StreamGear will generate asset files for {} streaming format.".format(
                        self.__format.upper()
                    )
                )
            elif difflib.get_close_matches(_format, supported_formats):
                raise ValueError(
                    "[StreamGear:ERROR] :: Incorrect `format` parameter value! Did you mean `{}`?".format(
                        difflib.get_close_matches(_format, supported_formats)[0]
                    )
                )
            else:
                raise ValueError(
                    "[StreamGear:ERROR] :: The `format` parameter value `{}` not valid/supported!".format(
                        format
                    )
                )
        else:
            raise ValueError(
                "[StreamGear:ERROR] :: The `format` parameter value is Missing or Invalid!"
            )

        # handles output asset filenames
        if output:
            # validate this class has the access rights to specified directory or not
            abs_path = os.path.abspath(output)
            # check if given output is a valid system path
            if check_WriteAccess(
                os.path.dirname(abs_path),
                is_windows=self.__os_windows,
                logging=self.__logging,
            ):
                # get all assets extensions
                valid_extension = "mpd" if self.__format == "dash" else "m3u8"
                assets_exts = [
                    ("chunk-stream", ".m4s"),  # filename prefix, extension
                    ("chunk-stream", ".ts"),  # filename prefix, extension
                    ".{}".format(valid_extension),
                ]
                # add source file extension too
                self.__video_source and assets_exts.append(
                    (
                        "chunk-stream",
                        os.path.splitext(self.__video_source)[1],
                    )  # filename prefix, extension
                )
                # handle output
                # check if path is a directory
                if os.path.isdir(abs_path):
                    # clear previous assets if specified
                    self.__clear_assets and delete_ext_safe(
                        abs_path, assets_exts, logging=self.__logging
                    )
                    # auto-assign valid name and adds it to path
                    abs_path = os.path.join(
                        abs_path,
                        "{}-{}.{}".format(
                            self.__format,
                            time.strftime("%Y%m%d-%H%M%S"),
                            valid_extension,
                        ),
                    )
                # or check if path is a file
                elif os.path.isfile(abs_path) and self.__clear_assets:
                    # clear previous assets if specified
                    delete_ext_safe(
                        os.path.dirname(abs_path),
                        assets_exts,
                        logging=self.__logging,
                    )
                # check if path has valid file extension
                assert abs_path.endswith(
                    valid_extension
                ), "Given `{}` path has invalid file-extension w.r.t selected format: `{}`!".format(
                    output, self.__format.upper()
                )
                self.__logging and logger.debug(
                    "Output Path:`{}` is successfully configured for generating streaming assets.".format(
                        abs_path
                    )
                )
                # workaround patch for Windows only,
                # others platforms will not be affected
                self.__out_file = abs_path.replace("\\", "/")
            # check if given output is a valid URL
            elif is_valid_url(self.__ffmpeg, url=output, logging=self.__logging):
                self.__logging and logger.debug(
                    "URL:`{}` is valid and successfully configured for generating streaming assets.".format(
                        output
                    )
                )
                self.__out_file = output
            # raise ValueError otherwise
            else:
                raise ValueError(
                    "[StreamGear:ERROR] :: The output parameter value:`{}` is not valid/supported!".format(
                        output
                    )
                )
        else:
            # raise ValueError otherwise
            raise ValueError(
                "[StreamGear:ERROR] :: Kindly provide a valid `output` parameter value. Refer Docs for more information."
            )

        # log Mode of operation
        self.__video_source and logger.info(
            "StreamGear has been successfully configured for {} Mode.".format(
                "Single-Source" if self.__video_source else "Real-time Frames"
            )
        )

    @deprecated(
        parameter="rgb_mode",
        message="The `rgb_mode` parameter is deprecated and will be removed in a future version. Only BGR format frames will be supported going forward.",
    )
    def stream(self, frame, rgb_mode=False):
        """
        Pipes `ndarray` frames to FFmpeg Pipeline for transcoding them into chunked-encoded media segments of
        streaming formats such as MPEG-DASH and HLS.

        !!! warning "[DEPRECATION NOTICE]: The `rgb_mode` parameter is deprecated and will be removed in a future version."

        Parameters:
            frame (ndarray): a valid numpy frame
            rgb_mode (boolean): enable this flag to activate RGB mode _(i.e. specifies that incoming frames are of RGB format instead of default BGR)_.
        """
        # check if function is called in correct context
        if self.__video_source:
            raise RuntimeError(
                "[StreamGear:ERROR] :: The `stream()` method cannot be used when streaming from a `-video_source` input file. Kindly refer vidgear docs!"
            )
        # None-Type frames will be skipped
        if frame is None:
            return
        # extract height, width and number of channels of frame
        height, width = frame.shape[:2]
        channels = frame.shape[-1] if frame.ndim == 3 else 1
        # assign values to class variables on first run
        if self.__initiate_stream:
            self.__inputheight = height
            self.__inputwidth = width
            self.__inputchannels = channels
            self.__sourceframerate = (
                25.0 if not (self.__inputframerate) else self.__inputframerate
            )
            self.__logging and logger.debug(
                "InputFrame => Height:{} Width:{} Channels:{}".format(
                    self.__inputheight, self.__inputwidth, self.__inputchannels
                )
            )
        # validate size of frame
        if height != self.__inputheight or width != self.__inputwidth:
            raise ValueError("[StreamGear:ERROR] :: All frames must have same size!")
        # validate number of channels
        if channels != self.__inputchannels:
            raise ValueError(
                "[StreamGear:ERROR] :: All frames must have same number of channels!"
            )
        # initiate FFmpeg process on first run
        if self.__initiate_stream:
            # launch pre-processing
            self.__PreProcess(channels=channels, rgb=rgb_mode)
            # Check status of the process
            assert self.__process is not None

        # write the frame to pipeline
        try:
            self.__process.stdin.write(frame.tobytes())
        except (OSError, IOError):
            # log something is wrong!
            logger.error(
                "BrokenPipeError caught, Wrong values passed to FFmpeg Pipe, Kindly Refer Docs!"
            )
            raise ValueError  # for testing purpose only

    def transcode_source(self):
        """
        Transcodes an entire video file _(with or without audio)_ into chunked-encoded media segments of
        streaming formats such as MPEG-DASH and HLS.
        """
        # check if function is called in correct context
        if not (self.__video_source):
            raise RuntimeError(
                "[StreamGear:ERROR] :: The `transcode_source()` method cannot be used without a valid `-video_source` input. Kindly refer vidgear docs!"
            )
        # assign height, width and framerate
        self.__inputheight = int(self.__aspect_source[1])
        self.__inputwidth = int(self.__aspect_source[0])
        self.__sourceframerate = float(self.__fps_source)
        # launch pre-processing
        self.__PreProcess()

    def __PreProcess(self, channels=0, rgb=False):
        """
        Internal method that pre-processes default FFmpeg parameters before starting pipelining.

        Parameters:
            channels (int): Number of channels
            rgb (boolean): activates RGB mode _(if enabled)_.
        """
        # turn off initiate flag
        self.__initiate_stream = False
        # initialize I/O parameters
        input_parameters = OrderedDict()
        output_parameters = OrderedDict()
        # pre-assign default codec parameters (if not assigned by user).
        default_codec = "libx264rgb" if rgb else "libx264"
        output_parameters["-vcodec"] = self.__params.pop("-vcodec", default_codec)

        # enforce compatibility
        if output_parameters["-vcodec"] != "copy":
            # NOTE: these parameters only supported when stream copy not defined
            output_parameters["-vf"] = self.__params.pop("-vf", "format=yuv420p")
            # Non-essential `-aspect` parameter is removed from the default pipeline.
        else:
            # log warnings for these parameters
            self.__params.pop("-vf", False) and logger.warning(
                "Filtering and stream copy cannot be used together. Discarding specified `-vf` parameter!"
            )
            self.__params.pop("-aspect", False) and logger.warning(
                "Overriding aspect ratio with stream copy may produce invalid files. Discarding specified `-aspect` parameter!"
            )

        # enable optimizations w.r.t selected codec
        ### OPTIMIZATION-1 ###
        if output_parameters["-vcodec"] in [
            "libx264",
            "libx264rgb",
            "libx265",
            "libvpx-vp9",
        ]:
            output_parameters["-crf"] = self.__params.pop("-crf", "20")
        ### OPTIMIZATION-2 ###
        if output_parameters["-vcodec"] == "libx264":
            if not (self.__video_source):
                output_parameters["-profile:v"] = self.__params.pop(
                    "-profile:v", "high"
                )
        ### OPTIMIZATION-3 ###
        if output_parameters["-vcodec"] in ["libx264", "libx264rgb"]:
            output_parameters["-tune"] = self.__params.pop("-tune", "zerolatency")
            output_parameters["-preset"] = self.__params.pop("-preset", "veryfast")
        ### OPTIMIZATION-4 ###
        if output_parameters["-vcodec"] == "libx265":
            output_parameters["-x265-params"] = self.__params.pop(
                "-x265-params", "lossless=1"
            )

        # enable audio (if present)
        if self.__audio:
            # validate audio source
            bitrate = validate_audio(self.__ffmpeg, source=self.__audio)
            if bitrate:
                logger.info(
                    "Detected External Audio Source is valid, and will be used for generating streams."
                )
                # assign audio source
                output_parameters[
                    "{}".format(
                        "-core_asource" if isinstance(self.__audio, list) else "-i"
                    )
                ] = self.__audio
                # assign audio codec
                output_parameters["-acodec"] = self.__params.pop(
                    "-acodec", "aac" if isinstance(self.__audio, list) else "copy"
                )
                output_parameters["a_bitrate"] = bitrate  # temporary handler
                output_parameters["-core_audio"] = (
                    ["-map", "1:a:0"] if self.__format == "dash" else []
                )
            else:
                logger.warning(
                    "Audio source `{}` is not valid, Skipped!".format(self.__audio)
                )
        # validate input video's audio source if available
        elif self.__video_source:
            bitrate = validate_audio(self.__ffmpeg, source=self.__video_source)
            if bitrate:
                logger.info("Input Video's audio source will be used for this run.")
                # assign audio codec
                output_parameters["-acodec"] = (
                    "aac" if self.__format == "hls" else "copy"
                )
                output_parameters["a_bitrate"] = bitrate  # temporary handler
            else:
                logger.info(
                    "No valid audio source available in the input video. Disabling audio while generating streams."
                )
        else:
            logger.info(
                "No valid audio source provided. Disabling audio while generating streams."
            )
        # enable audio optimizations based on audio codec
        if "-acodec" in output_parameters and output_parameters["-acodec"] == "aac":
            output_parameters["-movflags"] = "+faststart"

        # set input framerate
        if self.__sourceframerate > 0.0 and not (self.__video_source):
            # set input framerate
            self.__logging and logger.debug(
                "Setting Input framerate: {}".format(self.__sourceframerate)
            )
            input_parameters["-framerate"] = str(self.__sourceframerate)

        # handle input resolution and pixel format
        if not (self.__video_source):
            dimensions = "{}x{}".format(self.__inputwidth, self.__inputheight)
            input_parameters["-video_size"] = str(dimensions)
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
                    "[StreamGear:ERROR] :: Frames with channels outside range 1-to-4 are not supported!"
                )
        # process assigned format parameters
        process_params = self.__handle_streams(
            input_params=input_parameters, output_params=output_parameters
        )
        # check if processing completed successfully
        assert not (
            process_params is None
        ), "[StreamGear:ERROR] :: `{}` stream cannot be initiated properly!".format(
            self.__format.upper()
        )
        # Finally start FFmpef pipline and process everything
        self.__Build_n_Execute(process_params[0], process_params[1])

    def __handle_streams(self, input_params, output_params):
        """
        An internal function that parses various streams and its parameters.

        Parameters:
            input_params (dict): Input FFmpeg parameters
            output_params (dict): Output FFmpeg parameters
        """
        # handle bit-per-pixels
        bpp = self.__params.pop("-bpp", 0.1000)
        if isinstance(bpp, float) and bpp >= 0.001:
            bpp = float(bpp)
        else:
            # reset to default if invalid
            bpp = 0.1000
        # log it
        bpp and self.__logging and logger.debug(
            "Setting bit-per-pixels: {} for this stream.".format(bpp)
        )

        # handle gop
        gop = self.__params.pop("-gop", 2 * int(self.__sourceframerate))
        if isinstance(gop, (int, float)) and gop >= 0:
            gop = int(gop)
        else:
            # reset to some recommended value
            gop = 2 * int(self.__sourceframerate)
        # log it
        gop and self.__logging and logger.debug(
            "Setting GOP: {} for this stream.".format(gop)
        )

        # define default stream and its mapping
        if self.__format == "hls":
            output_params["-corev0"] = ["-map", "0:v"]
            if "-acodec" in output_params:
                output_params["-corea0"] = [
                    "-map",
                    "{}:a".format(1 if "-core_audio" in output_params else 0),
                ]
        else:
            output_params["-map"] = 0

        # assign default output resolution
        if "-s:v:0" in self.__params:
            # prevent duplicates
            del self.__params["-s:v:0"]
        output_params["-s:v:0"] = "{}x{}".format(self.__inputwidth, self.__inputheight)
        # assign default output video-bitrate
        if "-b:v:0" in self.__params:
            # prevent duplicates
            del self.__params["-b:v:0"]
        output_params["-b:v:0"] = (
            str(
                get_video_bitrate(
                    int(self.__inputwidth),
                    int(self.__inputheight),
                    self.__sourceframerate,
                    bpp,
                )
            )
            + "k"
        )

        # assign default output audio-bitrate
        if "-b:a:0" in self.__params:
            # prevent duplicates
            del self.__params["-b:a:0"]
        # extract and assign audio-bitrate from temporary handler
        a_bitrate = output_params.pop("a_bitrate", False)
        if "-acodec" in output_params and a_bitrate:
            output_params["-b:a:0"] = a_bitrate

        # handle user-defined streams
        streams = self.__params.pop("-streams", {})
        output_params = self.__evaluate_streams(streams, output_params, bpp)

        # define additional streams optimization parameters
        if output_params["-vcodec"] in ["libx264", "libx264rgb"]:
            if not "-bf" in self.__params:
                output_params["-bf"] = 1
            if not "-sc_threshold" in self.__params:
                output_params["-sc_threshold"] = 0
            if not "-keyint_min" in self.__params:
                output_params["-keyint_min"] = gop
        if (
            output_params["-vcodec"] in ["libx264", "libx264rgb", "libvpx-vp9"]
            and not "-g" in self.__params
        ):
            output_params["-g"] = gop
        if output_params["-vcodec"] == "libx265":
            output_params["-core_x265"] = [
                "-x265-params",
                "keyint={}:min-keyint={}".format(gop, gop),
            ]

        # process given dash/hls stream and return it
        if self.__format == "dash":
            processed_params = self.__generate_dash_stream(
                input_params=input_params,
                output_params=output_params,
            )
        else:
            processed_params = self.__generate_hls_stream(
                input_params=input_params,
                output_params=output_params,
            )
        return processed_params

    def __evaluate_streams(self, streams, output_params, bpp):
        """
        Internal function that Extracts, Evaluates & Validates user-defined streams

        Parameters:
            streams (dict): Individual streams formatted as list of dict.
            output_params (dict): Output FFmpeg parameters
        """
        # temporary streams count variable
        output_params["stream_count"] = 1  # default is 1

        # check if streams are empty
        if not streams:
            logger.info("No additional `-streams` are provided.")
            return output_params

        # check if streams are valid
        if isinstance(streams, list) and all(isinstance(x, dict) for x in streams):
            # keep track of streams
            stream_count = 1
            # calculate source aspect-ratio
            source_aspect_ratio = self.__inputwidth / self.__inputheight
            # log the process
            self.__logging and logger.debug(
                "Processing {} streams.".format(len(streams))
            )
            # iterate over given streams
            for idx, stream in enumerate(streams):
                # log stream processing
                self.__logging and logger.debug("Processing #{} stream ::".format(idx))
                # make copy
                stream_copy = stream.copy()
                # handle intermediate stream data as dictionary
                intermediate_dict = {}
                # define and map stream to intermediate dict
                if self.__format == "hls":
                    intermediate_dict["-corev{}".format(stream_count)] = ["-map", "0:v"]
                    if "-acodec" in output_params:
                        intermediate_dict["-corea{}".format(stream_count)] = [
                            "-map",
                            "{}:a".format(1 if "-core_audio" in output_params else 0),
                        ]
                else:
                    intermediate_dict["-core{}".format(stream_count)] = ["-map", "0"]

                # extract resolution & individual dimension of stream
                resolution = stream.pop("-resolution", "")
                dimensions = (
                    resolution.lower().split("x")
                    if (resolution and isinstance(resolution, str))
                    else []
                )
                # validate resolution
                if (
                    len(dimensions) == 2
                    and dimensions[0].isnumeric()
                    and dimensions[1].isnumeric()
                ):
                    # verify resolution is w.r.t source aspect-ratio
                    expected_width = math.floor(
                        int(dimensions[1]) * source_aspect_ratio
                    )
                    if int(dimensions[0]) != expected_width:
                        logger.warning(
                            "The provided stream resolution '{}' does not align with the source aspect ratio. Output stream may appear distorted!".format(
                                resolution
                            )
                        )
                    # assign stream resolution to intermediate dict
                    intermediate_dict["-s:v:{}".format(stream_count)] = resolution
                else:
                    # otherwise log error and skip stream
                    logger.error(
                        "Missing `-resolution` value. Invalid stream `{}` Skipped!".format(
                            stream_copy
                        )
                    )
                    continue

                # verify given stream video-bitrate
                video_bitrate = stream.pop("-video_bitrate", "")
                if (
                    video_bitrate
                    and isinstance(video_bitrate, str)
                    and video_bitrate.endswith(("k", "M"))
                ):
                    # assign it
                    intermediate_dict["-b:v:{}".format(stream_count)] = video_bitrate
                else:
                    # otherwise calculate video-bitrate
                    fps = stream.pop("-framerate", 0.0)
                    if dimensions and isinstance(fps, (float, int)) and fps > 0:
                        intermediate_dict["-b:v:{}".format(stream_count)] = (
                            "{}k".format(
                                get_video_bitrate(
                                    int(dimensions[0]), int(dimensions[1]), fps, bpp
                                )
                            )
                        )
                    else:
                        # If everything fails, log and skip the stream!
                        logger.error(
                            "Unable to determine Video-Bitrate for the stream `{}`. Skipped!".format(
                                stream_copy
                            )
                        )
                        continue
                # verify given stream audio-bitrate
                audio_bitrate = stream.pop("-audio_bitrate", "")
                if "-acodec" in output_params:
                    if audio_bitrate and audio_bitrate.endswith(("k", "M")):
                        intermediate_dict["-b:a:{}".format(stream_count)] = (
                            audio_bitrate
                        )
                    else:
                        # otherwise calculate audio-bitrate
                        if dimensions:
                            aspect_width = int(dimensions[0])
                            intermediate_dict["-b:a:{}".format(stream_count)] = (
                                "{}k".format(128 if (aspect_width > 800) else 96)
                            )
                # update output parameters
                output_params.update(intermediate_dict)
                # clear intermediate dict
                intermediate_dict.clear()
                # clear stream copy
                stream_copy.clear()
                # increment to next stream
                stream_count += 1
                # log stream processing
                self.__logging and logger.debug(
                    "Processed #{} stream successfully.".format(idx)
                )
            # store stream count
            output_params["stream_count"] = stream_count
            # log streams processing
            self.__logging and logger.debug("All streams processed successfully!")
        else:
            # skip and log
            logger.warning("Invalid type `-streams` skipped!")

        return output_params

    def __generate_hls_stream(self, input_params, output_params):
        """
        An internal function that parses user-defined parameters and generates
        suitable FFmpeg Terminal Command for transcoding input into HLS Stream.

        Parameters:
            input_params (dict): Input FFmpeg parameters
            output_params (dict): Output FFmpeg parameters
        """
        # validate `hls_segment_type`
        default_hls_segment_type = self.__params.pop("-hls_segment_type", "mpegts")
        if isinstance(
            default_hls_segment_type, str
        ) and default_hls_segment_type.strip() in ["fmp4", "mpegts"]:
            output_params["-hls_segment_type"] = default_hls_segment_type.strip()
        else:
            # otherwise reset to default
            logger.warning("Invalid `-hls_segment_type` value skipped!")
            output_params["-hls_segment_type"] = "mpegts"
        # gather required parameters
        if self.__livestreaming:
            # `hls_list_size` must be greater than or equal to 0
            default_hls_list_size = self.__params.pop("-hls_list_size", 6)
            if isinstance(default_hls_list_size, int) and default_hls_list_size >= 0:
                output_params["-hls_list_size"] = default_hls_list_size
            else:
                # otherwise reset to default
                logger.warning("Invalid `-hls_list_size` value skipped!")
                output_params["-hls_list_size"] = 6
            # `hls_init_time` must be greater than or equal to 0
            default_hls_init_time = self.__params.pop("-hls_init_time", 4)
            if isinstance(default_hls_init_time, int) and default_hls_init_time >= 0:
                output_params["-hls_init_time"] = default_hls_init_time
            else:
                # otherwise reset to default
                logger.warning("Invalid `-hls_init_time` value skipped!")
                output_params["-hls_init_time"] = 4
            # `hls_time` must be greater than or equal to 0
            default_hls_time = self.__params.pop("-hls_time", 4)
            if isinstance(default_hls_time, int) and default_hls_time >= 0:
                output_params["-hls_time"] = default_hls_time
            else:
                # otherwise reset to default
                logger.warning("Invalid `-hls_time` value skipped!")
                output_params["-hls_time"] = 6
            # `hls_flags` must be string
            default_hls_flags = self.__params.pop(
                "-hls_flags", "delete_segments+discont_start+split_by_time"
            )
            if isinstance(default_hls_flags, str):
                output_params["-hls_flags"] = default_hls_flags
            else:
                # otherwise reset to default
                logger.warning("Invalid `-hls_flags` value skipped!")
                output_params["-hls_flags"] = (
                    "delete_segments+discont_start+split_by_time"
                )
            # clean everything at exit?
            remove_at_exit = self.__params.pop("-remove_at_exit", 0)
            if isinstance(remove_at_exit, int) and remove_at_exit in [
                0,
                1,
            ]:
                output_params["-remove_at_exit"] = remove_at_exit
            else:
                # otherwise reset to default
                logger.warning("Invalid `-remove_at_exit` value skipped!")
                output_params["-remove_at_exit"] = 0
        else:
            # enforce "contain all the segments"
            output_params["-hls_list_size"] = 0
            output_params["-hls_playlist_type"] = "vod"

        # handle base URL for absolute paths
        hls_base_url = self.__params.pop("-hls_base_url", "")
        if isinstance(hls_base_url, str):
            output_params["-hls_base_url"] = hls_base_url
        else:
            # otherwise reset to default
            logger.warning("Invalid `-hls_base_url` value skipped!")
            output_params["-hls_base_url"] = ""

        # Hardcoded HLS parameters (Refer FFmpeg docs for more info.)
        output_params["-allowed_extensions"] = "ALL"
        # Handling <hls_segment_filename>
        # Here filename will be based on `stream_count` dict parameter that
        # would be used to check whether stream is multi-variant(>1) or single(0-1)
        segment_template = (
            "{}-stream%v-%03d.{}"
            if output_params["stream_count"] > 1
            else "{}-stream-%03d.{}"
        )
        output_params["-hls_segment_filename"] = segment_template.format(
            os.path.join(os.path.dirname(self.__out_file), "chunk"),
            "m4s" if output_params["-hls_segment_type"] == "fmp4" else "ts",
        )
        # Hardcoded HLS parameters (Refer FFmpeg docs for more info.)
        output_params["-hls_allow_cache"] = 0
        # enable hls formatting
        output_params["-f"] = "hls"
        # return HLS params
        return (input_params, output_params)

    def __generate_dash_stream(self, input_params, output_params):
        """
        An internal function that parses user-defined parameters and generates
        suitable FFmpeg Terminal Command for transcoding input into MPEG-dash Stream.

        Parameters:
            input_params (dict): Input FFmpeg parameters
            output_params (dict): Output FFmpeg parameters
        """

        # Check if live-streaming or not?
        if self.__livestreaming:
            # `extra_window_size` must be greater than or equal to 0
            window_size = self.__params.pop("-window_size", 5)
            if isinstance(window_size, int) and window_size >= 0:
                output_params["-window_size"] = window_size
            else:
                # otherwise reset to default
                logger.warning("Invalid `-window_size` value skipped!")
                output_params["-window_size"] = 5
            # `extra_window_size` must be greater than or equal to 0
            extra_window_size = self.__params.pop("-extra_window_size", 5)
            if isinstance(extra_window_size, int) and extra_window_size >= 0:
                output_params["-extra_window_size"] = window_size
            else:
                # otherwise reset to default
                logger.warning("Invalid `-extra_window_size` value skipped!")
                output_params["-extra_window_size"] = 5
            # clean everything at exit?
            remove_at_exit = self.__params.pop("-remove_at_exit", 0)
            if isinstance(remove_at_exit, int) and remove_at_exit in [
                0,
                1,
            ]:
                output_params["-remove_at_exit"] = remove_at_exit
            else:
                # otherwise reset to default
                logger.warning("Invalid `-remove_at_exit` value skipped!")
                output_params["-remove_at_exit"] = 0
            # `seg_duration` must be greater than or equal to 0
            seg_duration = self.__params.pop("-seg_duration", 20)
            if isinstance(seg_duration, int) and seg_duration >= 0:
                output_params["-seg_duration"] = seg_duration
            else:
                # otherwise reset to default
                logger.warning("Invalid `-seg_duration` value skipped!")
                output_params["-seg_duration"] = 20
            # Disable (0) the use of a SegmentTimeline inside a SegmentTemplate.
            output_params["-use_timeline"] = 0
        else:
            # `seg_duration` must be greater than or equal to 0
            output_params["-seg_duration"] = self.__params.pop("-seg_duration", 5)
            if isinstance(seg_duration, int) and seg_duration >= 0:
                output_params["-seg_duration"] = seg_duration
            else:
                # otherwise reset to default
                logger.warning("Invalid `-seg_duration` value skipped!")
                output_params["-seg_duration"] = 5
            # Enable (1) the use of a SegmentTimeline inside a SegmentTemplate.
            output_params["-use_timeline"] = 1

        # Finally, some hardcoded DASH parameters (Refer FFmpeg docs for more info.)
        output_params["-use_template"] = 1
        output_params["-adaptation_sets"] = "id=0,streams=v {}".format(
            "id=1,streams=a" if ("-acodec" in output_params) else ""
        )
        # enable dash formatting
        output_params["-f"] = "dash"
        # return DASH params
        return (input_params, output_params)

    def __Build_n_Execute(self, input_params, output_params):
        """
        An Internal function that launches FFmpeg subprocess and pipelines commands.

        Parameters:
            input_params (dict): Input FFmpeg parameters
            output_params (dict): Output FFmpeg parameters
        """
        # handle audio source if present
        "-core_asource" in output_params and output_params.move_to_end(
            "-core_asource", last=False
        )
        # handle `-i` parameter
        "-i" in output_params and output_params.move_to_end("-i", last=False)
        # copy streams count
        stream_count = output_params.pop("stream_count", 1)

        # convert input parameters to list
        input_commands = dict2Args(input_params)
        # convert output parameters to list
        output_commands = dict2Args(output_params)
        # convert any additional parameters to list
        stream_commands = dict2Args(self.__params)

        # create exclusive HLS params
        hls_commands = []
        # handle HLS multi-variant streams
        if self.__format == "hls" and stream_count > 1:
            stream_map = ""
            for count in range(0, stream_count):
                stream_map += "v:{}{} ".format(
                    count, ",a:{}".format(count) if "-acodec" in output_params else ","
                )
            hls_commands += [
                "-master_pl_name",
                os.path.basename(self.__out_file),
                "-var_stream_map",
                stream_map.strip(),
                os.path.join(os.path.dirname(self.__out_file), "stream_%v.m3u8"),
            ]

        # log it if enabled
        self.__logging and logger.debug(
            "User-Defined Output parameters: `{}`".format(
                " ".join(output_commands) if output_commands else None
            )
        )
        self.__logging and logger.debug(
            "Additional parameters: `{}`".format(
                " ".join(stream_commands) if stream_commands else None
            )
        )
        # build FFmpeg command from parameters
        ffmpeg_cmd = None
        # ensuring less cluttering if silent mode
        hide_banner = [] if self.__logging else ["-hide_banner"]
        # format commands
        if self.__video_source:
            ffmpeg_cmd = (
                [self.__ffmpeg, "-y"]
                + (["-re"] if self.__livestreaming else [])  # pseudo live-streaming
                + hide_banner
                + ["-i", self.__video_source]
                + input_commands
                + output_commands
                + stream_commands
            )
        else:
            ffmpeg_cmd = (
                [self.__ffmpeg, "-y"]
                + hide_banner
                + ["-f", "rawvideo", "-vcodec", "rawvideo"]
                + input_commands
                + ["-i", "-"]
                + output_commands
                + stream_commands
            )
        # format outputs
        ffmpeg_cmd.extend([self.__out_file] if not (hls_commands) else hls_commands)
        # Launch the FFmpeg pipeline with built command
        logger.critical("Transcoding streaming chunks. Please wait...")  # log it
        self.__process = sp.Popen(
            ffmpeg_cmd,
            stdin=sp.PIPE,
            stdout=(
                sp.DEVNULL
                if (not self.__video_source and not self.__logging)
                else sp.PIPE
            ),
            stderr=None if self.__logging else sp.STDOUT,
        )
        # post handle progress bar and runtime errors in case of video_source
        if self.__video_source:
            return_code = 0
            pbar = None
            sec_prev = 0
            if self.__logging:
                self.__process.communicate()
                return_code = self.__process.returncode
            else:
                # iterate until stdout runs out
                while True:
                    # read and process data
                    data = self.__process.stdout.readline()
                    if data:
                        data = data.decode("utf-8")
                        # extract duration and time-left
                        if pbar is None and "Duration:" in data:
                            # extract time in seconds
                            sec_duration = extract_time(data)
                            # initiate progress bar
                            pbar = tqdm(
                                total=sec_duration,
                                desc="Processing Frames",
                                unit="frame",
                            )
                        elif "time=" in data:
                            # extract time in seconds
                            sec_current = extract_time(data)
                            # update progress bar
                            if sec_current:
                                pbar.update(sec_current - sec_prev)
                                sec_prev = sec_current
                    else:
                        # poll if no data
                        if self.__process.poll() is not None:
                            break
                return_code = self.__process.poll()
            # close progress bar
            not (pbar is None) and pbar.close()
            # handle return_code
            if return_code != 0:
                # log and raise error if return_code is `1`
                logger.error(
                    "StreamGear failed to initiate stream for this video source!"
                )
                raise sp.CalledProcessError(return_code, ffmpeg_cmd)
            else:
                # log if successful
                logger.critical(
                    "Transcoding Ended. {} Streaming assets are successfully generated at specified path.".format(
                        self.__format.upper()
                    )
                )

    def __enter__(self):
        """
        Handles entry with the `with` statement. See [PEP343 -- The 'with' statement'](https://peps.python.org/pep-0343/).

        **Returns:** Returns a reference to the StreamGear Class
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Handles exit with the `with` statement. See [PEP343 -- The 'with' statement'](https://peps.python.org/pep-0343/).
        """
        self.close()

    @deprecated(
        message="The `terminate()` method will be removed in the next release. Kindly use `close()` method instead."
    )
    def terminate(self):
        """
        !!! warning "[DEPRECATION NOTICE]: This method is now deprecated and will be removed in a future release."

        This function ensures backward compatibility for the `terminate()` method to maintain the API on existing systems.
        It achieves this by calling the new `close()` method to terminate various
        StreamGear processes.
        """

        self.close()

    def close(self):
        """
        Safely terminates various StreamGear process.
        """
        # log termination
        self.__logging and logger.debug("Terminating StreamGear Processes.")
        # return if no process was initiated at first place
        if self.__process is None or not (self.__process.poll() is None):
            return
        # close `stdin` output
        self.__process.stdin and self.__process.stdin.close()
        # close `stdout` output
        self.__process.stdout and self.__process.stdout.close()
        # forced termination if specified.
        if self.__forced_termination:
            self.__process.terminate()
        else:
            # send CTRL_BREAK_EVENT signal
            self.__process.send_signal(signal.CTRL_BREAK_EVENT)
        # wait if process is still processing
        self.__process.wait()
        # discard process
        self.__process = None
