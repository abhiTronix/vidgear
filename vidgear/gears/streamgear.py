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
import platform
import pathlib
import difflib
import logging as log
import subprocess as sp
from tqdm import tqdm
from fractions import Fraction
from collections import OrderedDict

# import helper packages
from .helper import (
    capPropId,
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
    possible to stream videos at different quality levels (different bitrates or spatial resolutions) and can be switched in the middle of a video from one quality level to another – if bandwidth
    permits – on a per-segment basis. A user can serve these segments on a web server that makes it easier to download them through HTTP standard-compliant GET requests.

    SteamGear also creates a Manifest/Playlist file (such as MPD in-case of DASH and M3U8 in-case of HLS) besides segments that describe these segment information (timing, URL, media characteristics like video resolution and bit rates)
     and is provided to the client before the streaming session.

    SteamGear currently supports MPEG-DASH (Dynamic Adaptive Streaming over HTTP, ISO/IEC 23009-1) and Apple HLS (HTTP live streaming).
    """

    def __init__(
        self, output="", format="dash", custom_ffmpeg="", logging=False, **stream_params
    ):
        """
        This constructor method initializes the object state and attributes of the StreamGear class.

        Parameters:
            output (str): sets the valid filename/path for storing the StreamGear assets.
            format (str): select the adaptive HTTP streaming format(DASH and HLS).
            custom_ffmpeg (str): assigns the location of custom path/directory for custom FFmpeg executables.
            logging (bool): enables/disables logging.
            stream_params (dict): provides the flexibility to control supported internal parameters and FFmpeg properities.
        """
        # print current version
        logcurr_vidgear_ver(logging=logging)

        # checks if machine in-use is running windows os or not
        self.__os_windows = True if os.name == "nt" else False
        # enable logging if specified
        self.__logging = logging if (logging and isinstance(logging, bool)) else False

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
            str(k).strip(): str(v).strip()
            if not isinstance(v, (dict, list, int, float))
            else v
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
        audio = self.__params.pop("-audio", "")
        if audio and isinstance(audio, str):
            if os.path.isfile(audio):
                self.__audio = os.path.abspath(audio)
            elif is_valid_url(self.__ffmpeg, url=audio, logging=self.__logging):
                self.__audio = audio
            else:
                self.__audio = ""
        elif audio and isinstance(audio, list):
            self.__audio = audio
        else:
            self.__audio = ""

        if self.__audio and self.__logging:
            logger.debug("External audio source detected!")

        # handle Video-Source input
        source = self.__params.pop("-video_source", "")
        # Check if input is valid string
        if source and isinstance(source, str) and len(source) > 1:
            # Differentiate input
            if os.path.isfile(source):
                self.__video_source = os.path.abspath(source)
            elif is_valid_url(self.__ffmpeg, url=source, logging=self.__logging):
                self.__video_source = source
            else:
                # discard the value otherwise
                self.__video_source = ""
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
                logger.warning("No valid video_source provided.")
        else:
            # discard the value otherwise
            self.__video_source = ""

        # handle user-defined framerate
        self.__inputframerate = self.__params.pop("-input_framerate", 0.0)
        if isinstance(self.__inputframerate, (float, int)):
            # must be float
            self.__inputframerate = float(self.__inputframerate)
        else:
            # reset improper values
            self.__inputframerate = 0.0

        # handle old assests
        self.__clear_assets = self.__params.pop("-clear_prev_assets", False)
        if not isinstance(self.__clear_assets, bool):
            # reset improper values
            self.__clear_assets = False

        # handle whether to livestream?
        self.__livestreaming = self.__params.pop("-livestream", False)
        if not isinstance(self.__livestreaming, bool):
            # reset improper values
            self.__livestreaming = False

        # handle Streaming formats
        supported_formats = ["dash", "hls"]  # will be extended in future
        # Validate
        if not (format is None) and format and isinstance(format, str):
            _format = format.strip().lower()
            if _format in supported_formats:
                self.__format = _format
                logger.info(
                    "StreamGear will generate files for {} HTTP streaming format.".format(
                        self.__format.upper()
                    )
                )
            elif difflib.get_close_matches(_format, supported_formats):
                raise ValueError(
                    "[StreamGear:ERROR] :: Incorrect format! Did you mean `{}`?".format(
                        difflib.get_close_matches(_format, supported_formats)[0]
                    )
                )
            else:
                raise ValueError(
                    "[StreamGear:ERROR] :: format value `{}` not valid/supported!".format(
                        format
                    )
                )
        else:
            raise ValueError(
                "[StreamGear:ERROR] :: format value is Missing/Incorrect. Check vidgear docs!"
            )

        # handles output name
        if not output:
            raise ValueError(
                "[StreamGear:ERROR] :: Kindly provide a valid `output` value. Refer Docs for more information."
            )
        else:
            # validate this class has the access rights to specified directory or not
            abs_path = os.path.abspath(output)

            if check_WriteAccess(
                os.path.dirname(abs_path),
                is_windows=self.__os_windows,
                logging=self.__logging,
            ):
                # check if given path is directory
                valid_extension = "mpd" if self.__format == "dash" else "m3u8"
                # get all assets extensions
                assets_exts = [
                    ("chunk-stream", ".m4s"),  # filename prefix, extension
                    ("chunk-stream", ".ts"),  # filename prefix, extension
                    ".{}".format(valid_extension),
                ]
                # add source file extension too
                if self.__video_source:
                    assets_exts.append(
                        (
                            "chunk-stream",
                            os.path.splitext(self.__video_source)[1],
                        )  # filename prefix, extension
                    )
                if os.path.isdir(abs_path):
                    if self.__clear_assets:
                        delete_ext_safe(abs_path, assets_exts, logging=self.__logging)
                    abs_path = os.path.join(
                        abs_path,
                        "{}-{}.{}".format(
                            self.__format,
                            time.strftime("%Y%m%d-%H%M%S"),
                            valid_extension,
                        ),
                    )  # auto-assign valid name and adds it to path
                elif self.__clear_assets and os.path.isfile(abs_path):
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
                    "Path:`{}` is sucessfully configured for streaming.".format(
                        abs_path
                    )
                )
                # assign it
                self.__out_file = abs_path.replace(
                    "\\", "/"
                )  # workaround for Windows platform only, others will not be affected
            elif platform.system() == "Linux" and pathlib.Path(output).is_char_device():
                # check if linux video device path (such as `/dev/video0`)
                self.__logging and logger.debug(
                    "Path:`{}` is a valid Linux Video Device path.".format(output)
                )
                self.__out_file = output
            # check if given output is a valid URL
            elif is_valid_url(self.__ffmpeg, url=output, logging=self.__logging):
                self.__logging and logger.debug(
                    "URL:`{}` is valid and sucessfully configured for streaming.".format(
                        output
                    )
                )
                self.__out_file = output
            else:
                raise ValueError(
                    "[StreamGear:ERROR] :: Output value:`{}` is not valid/supported!".format(
                        output
                    )
                )
        # log Mode of operation
        logger.info(
            "StreamGear has been successfully configured for {} Mode.".format(
                "Single-Source" if self.__video_source else "Real-time Frames"
            )
        )

    def stream(self, frame, rgb_mode=False):
        """
        Pipelines `ndarray` frames to FFmpeg Pipeline for transcoding into multi-bitrate streamable assets.

        Parameters:
            frame (ndarray): a valid numpy frame
            rgb_mode (boolean): enable this flag to activate RGB mode _(i.e. specifies that incoming frames are of RGB format instead of default BGR)_.

        """
        # check if function is called in correct context
        if self.__video_source:
            raise RuntimeError(
                "[StreamGear:ERROR] :: `stream()` function cannot be used when streaming from a `-video_source` input file. Kindly refer vidgear docs!"
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
        Transcodes entire Video Source _(with audio)_ into multi-bitrate streamable assets
        """
        # check if function is called in correct context
        if not (self.__video_source):
            raise RuntimeError(
                "[StreamGear:ERROR] :: `transcode_source()` function cannot be used without a valid `-video_source` input. Kindly refer vidgear docs!"
            )
        # assign height, width and framerate
        self.__inputheight = int(self.__aspect_source[1])
        self.__inputwidth = int(self.__aspect_source[0])
        self.__sourceframerate = float(self.__fps_source)
        # launch pre-processing
        self.__PreProcess()

    def __PreProcess(self, channels=0, rgb=False):
        """
        Internal method that pre-processes default FFmpeg parameters before beginning pipelining.

        Parameters:
            channels (int): Number of channels
            rgb_mode (boolean): activates RGB mode _(if enabled)_.
        """
        # turn off initiate flag
        self.__initiate_stream = False
        # initialize parameters
        input_parameters = OrderedDict()
        output_parameters = OrderedDict()
        # pre-assign default codec parameters (if not assigned by user).
        default_codec = "libx264rgb" if rgb else "libx264"
        output_parameters["-vcodec"] = self.__params.pop("-vcodec", default_codec)
        # enable optimizations and enforce compatibility
        output_parameters["-vf"] = self.__params.pop("-vf", "format=yuv420p")
        aspect_ratio = Fraction(
            self.__inputwidth / self.__inputheight
        ).limit_denominator(10)
        output_parameters["-aspect"] = ":".join(str(aspect_ratio).split("/"))
        # w.r.t selected codec
        if output_parameters["-vcodec"] in [
            "libx264",
            "libx264rgb",
            "libx265",
            "libvpx-vp9",
        ]:
            output_parameters["-crf"] = self.__params.pop("-crf", "20")
        if output_parameters["-vcodec"] in ["libx264", "libx264rgb"]:
            if not (self.__video_source):
                output_parameters["-profile:v"] = self.__params.pop(
                    "-profile:v", "high"
                )
            output_parameters["-tune"] = self.__params.pop("-tune", "zerolatency")
            output_parameters["-preset"] = self.__params.pop("-preset", "veryfast")
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
                    "Detected External Audio Source is valid, and will be used for streams."
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
        elif self.__video_source:
            # validate audio source
            bitrate = validate_audio(self.__ffmpeg, source=self.__video_source)
            if bitrate:
                logger.info("Source Audio will be used for streams.")
                # assign audio codec
                output_parameters["-acodec"] = (
                    "aac" if self.__format == "hls" else "copy"
                )
                output_parameters["a_bitrate"] = bitrate  # temporary handler
            else:
                logger.warning(
                    "No valid audio_source available. Disabling audio for streams!"
                )
        else:
            logger.warning(
                "No valid audio_source provided. Disabling audio for streams!"
            )
        # enable audio optimizations based on audio codec
        if "-acodec" in output_parameters and output_parameters["-acodec"] == "aac":
            output_parameters["-movflags"] = "+faststart"

        # set input framerate
        if self.__sourceframerate > 0 and not (self.__video_source):
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
        ), "[StreamGear:ERROR] :: {} stream cannot be initiated!".format(
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
        if isinstance(bpp, (float, int)) and bpp > 0.0:
            bpp = float(bpp) if (bpp > 0.001) else 0.1000
        else:
            # reset to defaut if invalid
            bpp = 0.1000
        # log it
        self.__logging and logger.debug(
            "Setting bit-per-pixels: {} for this stream.".format(bpp)
        )

        # handle gop
        gop = self.__params.pop("-gop", 0)
        if isinstance(gop, (int, float)) and gop > 0:
            gop = int(gop)
        else:
            # reset to some recommended value
            gop = 2 * int(self.__sourceframerate)
        # log it
        self.__logging and logger.debug("Setting GOP: {} for this stream.".format(gop))

        # define and map default stream
        if self.__format != "hls":
            output_params["-map"] = 0
        else:
            output_params["-corev0"] = ["-map", "0:v"]
            if "-acodec" in output_params:
                output_params["-corea0"] = [
                    "-map",
                    "{}:a".format(1 if "-core_audio" in output_params else 0),
                ]
        # assign resolution
        if "-s:v:0" in self.__params:
            # prevent duplicates
            del self.__params["-s:v:0"]
        output_params["-s:v:0"] = "{}x{}".format(self.__inputwidth, self.__inputheight)
        # assign video-bitrate
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
        # assign audio-bitrate
        if "-b:a:0" in self.__params:
            # prevent duplicates
            del self.__params["-b:a:0"]
        # extract audio-bitrate from temporary handler
        a_bitrate = output_params.pop("a_bitrate", "")
        if "-acodec" in output_params and a_bitrate:
            output_params["-b:a:0"] = a_bitrate

        # handle user-defined streams
        streams = self.__params.pop("-streams", {})
        output_params = self.__evaluate_streams(streams, output_params, bpp)

        # define additional stream optimization parameters
        if output_params["-vcodec"] in ["libx264", "libx264rgb"]:
            if not "-bf" in self.__params:
                output_params["-bf"] = 1
            if not "-sc_threshold" in self.__params:
                output_params["-sc_threshold"] = 0
            if not "-keyint_min" in self.__params:
                output_params["-keyint_min"] = gop
        if output_params["-vcodec"] in ["libx264", "libx264rgb", "libvpx-vp9"]:
            if not "-g" in self.__params:
                output_params["-g"] = gop
        if output_params["-vcodec"] == "libx265":
            output_params["-core_x265"] = [
                "-x265-params",
                "keyint={}:min-keyint={}".format(gop, gop),
            ]

        # process given dash/hls stream
        processed_params = None
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
            streams (dict): Indivisual streams formatted as list of dict.
            output_params (dict): Output FFmpeg parameters
        """
        # temporary streams count variable
        output_params["stream_count"] = 1  # default is 1

        # check if streams are empty
        if not streams:
            logger.warning("No `-streams` are provided!")
            return output_params

        # check if streams are valid
        if isinstance(streams, list) and all(isinstance(x, dict) for x in streams):
            stream_count = 1  # keep track of streams
            # calculate source aspect-ratio
            source_aspect_ratio = self.__inputwidth / self.__inputheight
            # log the process
            self.__logging and logger.debug(
                "Processing {} streams.".format(len(streams))
            )
            # iterate over given streams
            for stream in streams:
                stream_copy = stream.copy()  # make copy
                intermediate_dict = {}  # handles intermediate stream data as dictionary

                # define and map stream to intermediate dict
                if self.__format != "hls":
                    intermediate_dict["-core{}".format(stream_count)] = ["-map", "0"]
                else:
                    intermediate_dict["-corev{}".format(stream_count)] = ["-map", "0:v"]
                    if "-acodec" in output_params:
                        intermediate_dict["-corea{}".format(stream_count)] = [
                            "-map",
                            "{}:a".format(1 if "-core_audio" in output_params else 0),
                        ]

                # extract resolution & indivisual dimension of stream
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
                            "Given stream resolution `{}` is not in accordance with the Source Aspect-Ratio. Stream Output may appear Distorted!".format(
                                resolution
                            )
                        )
                    # assign stream resolution to intermediate dict
                    intermediate_dict["-s:v:{}".format(stream_count)] = resolution
                else:
                    # otherwise log error and skip stream
                    logger.error(
                        "Missing `-resolution` value, Stream `{}` Skipped!".format(
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
                        intermediate_dict[
                            "-b:v:{}".format(stream_count)
                        ] = "{}k".format(
                            get_video_bitrate(
                                int(dimensions[0]), int(dimensions[1]), fps, bpp
                            )
                        )
                    else:
                        # If everything fails, log and skip the stream!
                        logger.error(
                            "Unable to determine Video-Bitrate for the stream `{}`, Skipped!".format(
                                stream_copy
                            )
                        )
                        continue
                # verify given stream audio-bitrate
                audio_bitrate = stream.pop("-audio_bitrate", "")
                if "-acodec" in output_params:
                    if audio_bitrate and audio_bitrate.endswith(("k", "M")):
                        intermediate_dict[
                            "-b:a:{}".format(stream_count)
                        ] = audio_bitrate
                    else:
                        # otherwise calculate audio-bitrate
                        if dimensions:
                            aspect_width = int(dimensions[0])
                            intermediate_dict[
                                "-b:a:{}".format(stream_count)
                            ] = "{}k".format(128 if (aspect_width > 800) else 96)
                # update output parameters
                output_params.update(intermediate_dict)
                # clear intermediate dict
                intermediate_dict.clear()
                # clear stream copy
                stream_copy.clear()
                # increment to next stream
                stream_count += 1
            output_params["stream_count"] = stream_count
            self.__logging and logger.debug("All streams processed successfully!")
        else:
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
        # Check if live-streaming or not?

        # validate `hls_segment_type`
        default_hls_segment_type = self.__params.pop("-hls_segment_type", "mpegts")
        if isinstance(
            default_hls_segment_type, str
        ) and default_hls_segment_type.strip() in ["fmp4", "mpegts"]:
            output_params["-hls_segment_type"] = default_hls_segment_type.strip()
        else:
            output_params["-hls_segment_type"] = "mpegts"

        # gather required parameters
        if self.__livestreaming:
            # `hls_list_size` must be greater than 0
            default_hls_list_size = self.__params.pop("-hls_list_size", 6)
            if isinstance(default_hls_list_size, int) and default_hls_list_size > 0:
                output_params["-hls_list_size"] = default_hls_list_size
            else:
                # otherwise reset to  default
                output_params["-hls_list_size"] = 6
            # default behaviour
            output_params["-hls_init_time"] = self.__params.pop("-hls_init_time", 4)
            output_params["-hls_time"] = self.__params.pop("-hls_time", 6)
            output_params["-hls_flags"] = self.__params.pop(
                "-hls_flags", "delete_segments+discont_start+split_by_time"
            )
            # clean everything at exit?
            output_params["-remove_at_exit"] = self.__params.pop("-remove_at_exit", 0)
        else:
            # enforce "contain all the segments"
            output_params["-hls_list_size"] = 0
            output_params["-hls_playlist_type"] = "vod"

        # handle base URL for absolute paths
        output_params["-hls_base_url"] = self.__params.pop("-hls_base_url", "")

        # Finally, some hardcoded HLS parameters (Refer FFmpeg docs for more info.)
        output_params["-allowed_extensions"] = "ALL"
        # Handling <hls_segment_filename>
        # Here filenname will be based on `stream_count` dict parameter that
        # would be used to check whether stream is multivariant(>1) or single(0-1)
        segment_template = (
            "{}-stream%v-%03d.{}"
            if output_params["stream_count"] > 1
            else "{}-stream-%03d.{}"
        )
        output_params["-hls_segment_filename"] = segment_template.format(
            os.path.join(os.path.dirname(self.__out_file), "chunk"),
            "m4s" if output_params["-hls_segment_type"] == "fmp4" else "ts",
        )
        output_params["-hls_allow_cache"] = 0
        # enable hls formatting
        output_params["-f"] = "hls"
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
            output_params["-window_size"] = self.__params.pop("-window_size", 5)
            output_params["-extra_window_size"] = self.__params.pop(
                "-extra_window_size", 5
            )
            # clean everything at exit?
            output_params["-remove_at_exit"] = self.__params.pop("-remove_at_exit", 0)
            # default behaviour
            output_params["-seg_duration"] = self.__params.pop("-seg_duration", 20)
            # Disable (0) the use of a SegmentTimline inside a SegmentTemplate.
            output_params["-use_timeline"] = 0
        else:
            # default behaviour
            output_params["-seg_duration"] = self.__params.pop("-seg_duration", 5)
            # Enable (1) the use of a SegmentTimline inside a SegmentTemplate.
            output_params["-use_timeline"] = 1

        # Finally, some hardcoded DASH parameters (Refer FFmpeg docs for more info.)
        output_params["-use_template"] = 1
        output_params["-adaptation_sets"] = "id=0,streams=v {}".format(
            "id=1,streams=a" if ("-acodec" in output_params) else ""
        )
        # enable dash formatting
        output_params["-f"] = "dash"
        return (input_params, output_params)

    def __Build_n_Execute(self, input_params, output_params):

        """
        An Internal function that launches FFmpeg subprocess and pipelines commands.

        Parameters:
            input_params (dict): Input FFmpeg parameters
            output_params (dict): Output FFmpeg parameters
        """
        # handle audio source if present
        if "-core_asource" in output_params:
            output_params.move_to_end("-core_asource", last=False)

        # finally handle `-i`
        if "-i" in output_params:
            output_params.move_to_end("-i", last=False)

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
        if self.__logging:
            logger.debug(
                "User-Defined Output parameters: `{}`".format(
                    " ".join(output_commands) if output_commands else None
                )
            )
            logger.debug(
                "Additional parameters: `{}`".format(
                    " ".join(stream_commands) if stream_commands else None
                )
            )
        # build FFmpeg command from parameters
        ffmpeg_cmd = None
        hide_banner = (
            [] if self.__logging else ["-hide_banner"]
        )  # ensuring less cluterring if specified
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
            stdout=sp.DEVNULL
            if (not self.__video_source and not self.__logging)
            else sp.PIPE,
            stderr=None if self.__logging else sp.STDOUT,
        )
        # post handle progress bar and runtime errors in case of video_source
        if self.__video_source:
            return_code = 0
            pbar = None
            sec_prev = 0
            if not self.__logging:
                # iterate until stdout runs out
                while True:
                    # read and process data
                    data = self.__process.stdout.readline()
                    if data:
                        data = data.decode("utf-8")
                        # extract duration and time-left
                        if pbar is None:
                            if "Duration:" in data:
                                sec_duration = extract_time(data)
                                # initate progress bar
                                pbar = tqdm(
                                    total=sec_duration,
                                    desc="Processing Frames",
                                    unit="frame",
                                )
                        else:
                            if "time=" in data:
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
            else:
                self.__process.communicate()
                return_code = self.__process.returncode
            # close progress bar
            if pbar:
                pbar.close()
            # handle return_code
            if return_code:
                # log and raise error if return_code is `1`
                logger.error(
                    "StreamGear failed to initiate stream for this video source!"
                )
                error = sp.CalledProcessError(return_code, ffmpeg_cmd)
                raise error
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
        self.terminate()

    def terminate(self):
        """
        Safely terminates StreamGear.
        """
        # return if no process was initiated at first place
        if self.__process is None or not (self.__process.poll() is None):
            return
        # close `stdin` output
        if self.__process.stdin:
            self.__process.stdin.close()
        # force terminate if external audio source
        if isinstance(self.__audio, list):
            self.__process.terminate()
        # wait if still process is still processing some information
        self.__process.wait()
        self.__process = None
        # log it
        logger.critical(
            "Transcoding Ended. {} Streaming assets are successfully generated at specified path.".format(
                self.__format.upper()
            )
        )
