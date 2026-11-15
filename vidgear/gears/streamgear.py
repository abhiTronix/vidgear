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
import difflib
import logging as log
import os
import subprocess as sp
import time
from collections import OrderedDict, deque
from threading import Event, Thread

from numpy.typing import NDArray
from tqdm import tqdm

# import helper packages
from .helper import (
    _pump_stderr,
    _stop_process,
    check_WriteAccess,
    delete_ext_safe,
    delete_file_safe,
    deprecated,
    dict2Args,
    get_valid_ffmpeg_path,
    get_video_bitrate,
    has_muxer_option,
    is_number,
    is_valid_url,
    logcurr_vidgear_ver,
    logger_handler,
    run_ffmpeg,
    to_float,
    validate_audio,
    validate_video,
)

# define logger
logger = log.getLogger("StreamGear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

# muxer option that exists only in FFmpeg builds StreamGear works with (dash: >= 4.1)
_REQUIRED_MUXER_OPTION = {"dash": "-seg_duration", "hls": "-var_stream_map"}


def _is_bitrate(value) -> bool:
    """
    Checks value is a bitrate string such as `"850k"` or `"2.5M"`.
    """
    return (
        isinstance(value, str)
        and value.endswith(("k", "M"))
        and to_float(value[:-1]) > 0
    )


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
        self,
        output: str = "",
        format: str = "dash",
        custom_ffmpeg: str = "",
        logging: bool = False,
        **stream_params: dict
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
        self.__os_windows = os.name == "nt"

        # initialize various class variables
        # handles user-defined parameters
        self.__params = {}
        # handle input video/frame resolution and channels
        self.__inputheight = None
        self.__inputwidth = None
        self.__inputchannels = None
        self.__sourceframerate = None
        # handle source display aspect ratio (as `width / height`)
        self.__source_dar = None
        # handle process to be frames written
        self.__process = None
        # handle valid FFmpeg assets location
        self.__ffmpeg = ""
        # handle one time process for valid process initialization
        self.__initiate_stream = True
        # handle closed state
        self.__closed = False
        # handle cancellation of single-source transcoding
        self.__cancel = Event()
        # handle captured FFmpeg stderr tail and its reader thread
        self.__stderr = None
        self.__stderr_reader = None
        # FFmpeg exit code (set on completion/close)
        self.returncode = None

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

        # handle streaming format
        supported_formats = ["dash", "hls"]  # TODO will be extended in future
        if format and isinstance(format, str):
            _format = format.strip().lower()
            if _format in supported_formats:
                self.__format = _format
                self.__logging and logger.info(
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

        # check FFmpeg has the required muxer (capabilities, not version numbers)
        if not has_muxer_option(
            self.__ffmpeg, self.__format, _REQUIRED_MUXER_OPTION[self.__format]
        ):
            raise RuntimeError(
                "[StreamGear:ERROR] :: FFmpeg at `{}` lacks the `{}` muxer or its `{}` option. Kindly use FFmpeg 6.0 or newer!".format(
                    self.__ffmpeg,
                    self.__format,
                    _REQUIRED_MUXER_OPTION[self.__format],
                )
            )

        # handle Audio-Input
        audio = self.__params.pop("-audio", False)
        if isinstance(audio, str) and audio:
            if os.path.isfile(audio):
                self.__audio = os.path.abspath(audio)
            elif "://" in audio and is_valid_url(
                self.__ffmpeg, url=audio, logging=self.__logging
            ):
                self.__audio = audio
            else:
                raise ValueError(
                    "[StreamGear:ERROR] :: The `-audio` value `{}` is neither an existing file nor a supported URL!".format(
                        audio
                    )
                )
        elif isinstance(audio, list) and audio:
            self.__audio = audio
        elif audio:
            raise ValueError(
                "[StreamGear:ERROR] :: The `-audio` value must be a file path, URL or list of FFmpeg input arguments!"
            )
        else:
            self.__audio = False
        # validate external audio source
        if self.__audio:
            self.__audio_bitrate = validate_audio(self.__ffmpeg, source=self.__audio)
            if not self.__audio_bitrate:
                raise ValueError(
                    "[StreamGear:ERROR] :: The `-audio` source `{}` has no readable audio stream!".format(
                        self.__audio
                    )
                )
            self.__logging and logger.debug(
                "External audio source `{}` detected.".format(self.__audio)
            )

        # handle Video-Source input
        source = self.__params.pop("-video_source", False)
        self.__video_source = False
        if source:
            # Differentiate input
            if isinstance(source, str) and os.path.isfile(source):
                self.__video_source = os.path.abspath(source)
            elif (
                isinstance(source, str)
                and "://" in source
                and is_valid_url(self.__ffmpeg, url=source, logging=self.__logging)
            ):
                self.__video_source = source
            else:
                raise ValueError(
                    "[StreamGear:ERROR] :: The `-video_source` value `{}` is neither an existing file nor a supported URL!".format(
                        source
                    )
                )
            # Validate input
            meta = validate_video(self.__ffmpeg, video_path=self.__video_source)
            if meta is None:
                raise ValueError(
                    "[StreamGear:ERROR] :: The `-video_source` value `{}` has no readable video stream!".format(
                        self.__video_source
                    )
                )
            width, height = (int(x) for x in meta["resolution"])
            dar = meta["dar"]
            # FFmpeg auto-rotates, so portrait sources come out with swapped dimensions
            if abs(meta["rotation"]) % 180 == 90:
                width, height = height, width
                dar = dar and dar[::-1]
            self.__aspect_source = (width, height)
            self.__source_dar = dar[0] / dar[1] if dar else None
            self.__fps_source = float(meta["framerate"])
            self.__source_audio_codec = meta["audio_codec"]
            # warn about HDR sources, which aren't tone-mapped by default
            meta["hdr"] and logger.warning(
                "The `-video_source` is HDR (PQ/HLG), but will be encoded as 8-bit SDR without tone-mapping and may look washed out. Kindly add a tone-mapping `-vf` filter (e.g. `zscale` + `tonemap`)!"
            )
            # log it
            self.__logging and logger.debug(
                "Given video_source is valid and has {}x{} resolution, and a framerate of {} fps.".format(
                    width, height, self.__fps_source
                )
            )
        else:
            # log normally
            self.__logging and logger.info("No `-video_source` value provided.")

        # handle user-defined framerate
        self.__inputframerate = max(
            to_float(self.__params.pop("-input_framerate", 0.0)), 0.0
        )

        # handle old assets
        clear_assets = self.__params.pop("-clear_prev_assets", False)
        self.__clear_assets = clear_assets if isinstance(clear_assets, bool) else False
        # log if clearing assets is enabled
        self.__clear_assets and self.__logging and logger.info(
            "The `-clear_prev_assets` parameter is enabled successfully. All previous StreamGear API assets for `{}` format will be removed for this run.".format(
                self.__format.upper()
            )
        )

        # handle whether to livestream?
        livestreaming = self.__params.pop("-livestream", False)
        self.__livestreaming = livestreaming is True
        # log if live streaming is enabled
        self.__livestreaming and self.__logging and logger.info(
            "Live-Streaming is successfully enabled for this run."
        )

        # handle the special-case of forced-termination
        enable_force_termination = self.__params.pop("-enable_force_termination", False)
        # check if value is valid
        self.__forced_termination = enable_force_termination is True
        # log if forced termination is enabled
        self.__forced_termination and logger.warning(
            "Forced termination is enabled for this run. This may result in corrupted output in certain scenarios!"
        )

        # FFmpeg's stderr tail is now always captured
        self.__params.pop("-capture_stderr", None)

        # handle user-defined progress callback (Single-Source Mode)
        self.__progress_callback = self.__params.pop("-progress_callback", None)
        if not callable(self.__progress_callback):
            # reset improper values
            self.__progress_callback = None

        # handles output asset filenames
        if not output or not isinstance(output, str):
            # raise ValueError otherwise
            raise ValueError(
                "[StreamGear:ERROR] :: Kindly provide a valid `output` parameter value. Refer Docs for more information."
            )
        valid_extension = ".mpd" if self.__format == "dash" else ".m3u8"
        if "://" in output:
            # check if given output is a valid URL
            if not is_valid_url(self.__ffmpeg, url=output, logging=self.__logging):
                raise ValueError(
                    "[StreamGear:ERROR] :: The output parameter value:`{}` is not valid/supported!".format(
                        output
                    )
                )
            self.__logging and logger.debug(
                "URL:`{}` is valid and successfully configured for generating streaming assets.".format(
                    output
                )
            )
            self.__out_file = output
        else:
            abs_path = os.path.abspath(output)
            # auto-assign valid name if path is a directory
            if os.path.isdir(abs_path):
                abs_path = os.path.join(
                    abs_path,
                    "{}-{}{}".format(
                        self.__format,
                        time.strftime("%Y%m%d-%H%M%S"),
                        valid_extension,
                    ),
                )
            # validate this class has the access rights to specified directory or not
            if not check_WriteAccess(
                os.path.dirname(abs_path),
                is_windows=self.__os_windows,
                logging=self.__logging,
            ):
                raise ValueError(
                    "[StreamGear:ERROR] :: The parent directory of output `{}` must exist and be writable!".format(
                        output
                    )
                )
            # check if path has valid file extension
            if os.path.splitext(abs_path)[1].lower() != valid_extension:
                raise ValueError(
                    "[StreamGear:ERROR] :: Given `{}` path has invalid file-extension w.r.t selected format: `{}`!".format(
                        output, self.__format.upper()
                    )
                )
            # clear previous assets if specified
            if self.__clear_assets:
                delete_file_safe(abs_path)
                delete_ext_safe(
                    os.path.dirname(abs_path),
                    [
                        ("chunk-stream", ""),  # filename prefix, extension
                        ("init-stream", ""),  # DASH init segments
                        ("stream_", ".m3u8"),  # HLS variant playlists
                        ("init", ".mp4"),  # HLS fMP4 init segments
                    ],
                    logging=self.__logging,
                )
            self.__logging and logger.debug(
                "Output Path:`{}` is successfully configured for generating streaming assets.".format(
                    abs_path
                )
            )
            # workaround patch for Windows only,
            # others platforms will not be affected
            self.__out_file = abs_path.replace("\\", "/")

        # keep user parameters, so every run starts from them
        self.__user_params = self.__params

        # log Mode of operation
        self.__logging and logger.info(
            "StreamGear has been successfully configured for {} Mode.".format(
                "Single-Source" if self.__video_source else "Real-time Frames"
            )
        )

    @deprecated(
        parameter="rgb_mode",
        message="The `rgb_mode` parameter is deprecated and will be removed in a future version. Only BGR format frames will be supported going forward.",
    )
    def stream(self, frame: NDArray, rgb_mode: bool = False) -> None:
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
        if self.__closed:
            raise RuntimeError("[StreamGear:ERROR] :: Can't stream, StreamGear is closed!")
        # None-Type frames will be skipped
        if frame is None:
            return
        # anything else gets written as garbage
        if frame.dtype != "uint8":
            raise ValueError(
                "[StreamGear:ERROR] :: Frames must be `uint8`, not `{}`. Convert them first, e.g. `(frame * 255).astype(np.uint8)`.".format(
                    frame.dtype
                )
            )
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

        # write the frame to pipeline
        try:
            self.__process.stdin.write(
                frame.data if frame.flags.c_contiguous else frame.tobytes()
            )
        except OSError as e:
            # let the reader drain FFmpeg's last words
            self.__stderr_reader.join(1.0)
            raise ValueError(
                "[StreamGear:ERROR] :: FFmpeg pipe broke (returncode: {}), likely wrong `stream_params` or output. FFmpeg's last words:\n{}".format(
                    self.__process.poll(), "\n".join(self.stderr_tail)
                )
            ) from e

    def transcode_source(self) -> None:
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
        self.__inputwidth, self.__inputheight = self.__aspect_source
        self.__sourceframerate = self.__fps_source
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
        # start from user parameters on every run
        self.__params = dict(self.__user_params)
        # initialize I/O parameters
        input_parameters = OrderedDict()
        output_parameters = OrderedDict()
        # pre-assign default codec parameters (if not assigned by user).
        # NOTE: `libx264rgb` outputs High 4:4:4, which browsers can't play, so RGB only changes the input
        default_codec = "libx264"
        output_vcodec = self.__params.pop("-vcodec", default_codec)
        # enforce default encoder if stream copy specified
        # in Real-time Frames Mode
        output_parameters["-vcodec"] = (
            default_codec
            if output_vcodec == "copy"
            and (not (self.__video_source) or "-streams" in self.__params)
            else output_vcodec
        )
        # enforce compatibility with stream copy
        if output_parameters["-vcodec"] != "copy":
            # NOTE: these parameters only supported when stream copy not defined
            output_parameters["-vf"] = self.__params.pop("-vf", "format=yuv420p")
            # Non-essential `-aspect` parameter is removed from the default pipeline.
        else:
            # log warnings if stream copy specified in Real-time Frames Mode
            not (self.__video_source) and logger.error(
                "Stream copy is not compatible with Real-time Frames Mode as it require re-encoding of incoming frames. Discarding the `-vcodec copy` parameter!"
            )
            ("-streams" in self.__params) and logger.error(
                "Stream copying is incompatible with Custom Streams as it require re-encoding for each additional stream. Discarding the `-vcodec copy` parameter!"
            )
            # log warnings for these parameters
            self.__params.pop("-vf", False) and logger.warning(
                "Filtering and stream copy cannot be used together. Discarding specified `-vf` parameter!"
            )
            self.__params.pop("-aspect", False) and logger.warning(
                "Overriding aspect ratio with stream copy may produce invalid files. Discarding specified `-aspect` parameter!"
            )

        # enable optimizations w.r.t selected codec
        # NOTE: no default `-crf`, since it overrides the bitrate of every stream.
        # A user-defined `-crf` passes through, capped by each stream's `-maxrate`.
        if output_parameters["-vcodec"] == "libx264":
            # profile for Real-time Frames Mode
            not (self.__video_source) and output_parameters.update(
                {"-profile:v": self.__params.pop("-profile:v", "high")}
            )
            # zero latency only matters when streaming live
            if not (self.__video_source) or self.__livestreaming:
                output_parameters["-tune"] = self.__params.pop("-tune", "zerolatency")
            output_parameters["-preset"] = self.__params.pop("-preset", "veryfast")

        # enable audio (if present)
        if self.__audio:
            # assign audio source
            output_parameters[
                "{}".format("-core_asource" if isinstance(self.__audio, list) else "-i")
            ] = self.__audio
            # assign audio codec
            output_parameters["-acodec"] = self.__params.pop("-acodec", "aac")
            output_parameters["a_bitrate"] = self.__audio_bitrate  # temporary handler
            # stop at the end of video, padding shorter audio with silence
            # (copied audio can't be padded, and `-shortest` would then cut the video)
            if output_parameters["-acodec"] != "copy":
                "-af" in self.__params or output_parameters.update({"-af": "apad"})
                output_parameters["-shortest"] = None
        # validate input video's audio source if available
        elif self.__video_source and self.__source_audio_codec:
            bitrate = validate_audio(self.__ffmpeg, source=self.__video_source)
            if bitrate:
                self.__logging and logger.info(
                    "Input video's audio source will be used for this run."
                )
                # copy only AAC, which every DASH/HLS client plays, else encode it
                output_parameters["-acodec"] = self.__params.pop(
                    "-acodec",
                    (
                        "copy"
                        if self.__source_audio_codec == "aac"
                        and "-streams" not in self.__params
                        else "aac"
                    ),
                )
                output_parameters["a_bitrate"] = bitrate  # temporary handler
        # discard audio bitrate with stream copy, else clamp it to sane AAC values
        a_bitrate = output_parameters.pop("a_bitrate", "")
        if output_parameters.get("-acodec", "copy") != "copy" and a_bitrate:
            kbps = to_float(a_bitrate[:-1]) * {"k": 1, "m": 1e3, "g": 1e6}.get(
                a_bitrate[-1], 0
            )
            # raw PCM/Opus rates (e.g. `1536k`) aren't sensible AAC targets
            output_parameters["a_bitrate"] = (
                a_bitrate if 0 < kbps <= 320 else "128k"
            )
        "-acodec" not in output_parameters and self.__logging and logger.info(
            "No valid audio source available. Disabling audio while generating streams."
        )

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
        # Finally start FFmpeg pipeline and process everything
        self.__Build_n_Execute(process_params[0], process_params[1])

    def __stream_maps(self, output_params):
        """
        Internal method that returns the `-map` arguments of one output stream: the first real video
        stream _(no cover art)_, and the first audio stream _(if audio is enabled)_.
        """
        maps = ["-map", "0:V:0"]
        if "-acodec" in output_params:
            maps += ["-map", "{}:a:0".format(1 if self.__audio else 0)]
        return maps

    def __handle_streams(self, input_params, output_params):
        """
        An internal function that parses various streams and its parameters.

        Parameters:
            input_params (dict): Input FFmpeg parameters
            output_params (dict): Output FFmpeg parameters
        """
        # handle bit-per-pixels
        bpp = self.__params.pop("-bpp", 0.1000)
        if not (is_number(bpp) and bpp >= 0.001):
            # reset to default if invalid
            bpp = 0.1000
        # log it
        self.__logging and logger.debug(
            "Setting bit-per-pixels: {} for this stream.".format(bpp)
        )

        # handle gop (2 seconds by default)
        gop = self.__params.pop("-gop", None)
        gop = (
            int(gop)
            if is_number(gop) and gop >= 1
            else max(round(2 * self.__sourceframerate), 1)
        )
        # log it
        self.__logging and logger.debug("Setting GOP: {} for this stream.".format(gop))

        # define default stream and its mapping
        output_params["-core0"] = self.__stream_maps(output_params)

        # assign default output resolution and video-bitrate (user values first)
        resolution = self.__params.pop("-s:v:0", "")
        video_bitrate = self.__params.pop("-b:v:0", "")
        if output_params["-vcodec"] != "copy":
            dimensions = (
                resolution.lower().split("x") if isinstance(resolution, str) else []
            )
            if not (len(dimensions) == 2 and all(x.isnumeric() for x in dimensions)):
                dimensions = [self.__inputwidth, self.__inputheight]
            output_params["-s:v:0"] = "{}x{}".format(*dimensions)
            output_params["-b:v:0"] = (
                video_bitrate
                if _is_bitrate(video_bitrate)
                else "{}k".format(
                    get_video_bitrate(
                        int(dimensions[0]),
                        int(dimensions[1]),
                        self.__sourceframerate,
                        bpp,
                    )
                )
            )

        # assign default output audio-bitrate (user values first)
        audio_bitrate = self.__params.pop("-b:a:0", "")
        a_bitrate = output_params.pop("a_bitrate", "")
        if "-acodec" in output_params and (a_bitrate or _is_bitrate(audio_bitrate)):
            output_params["-b:a:0"] = (
                audio_bitrate if _is_bitrate(audio_bitrate) else a_bitrate
            )

        # handle user-defined streams
        streams = self.__params.pop("-streams", {})
        output_params = self.__evaluate_streams(streams, output_params, bpp)

        # cap every stream at its bitrate, so manifests declare what clients get
        if not ({"-maxrate", "-bufsize"} & set(self.__params)):
            for key in [k for k in output_params if k.startswith("-b:v:")]:
                bitrate = output_params[key]
                output_params[key.replace("-b:", "-maxrate:")] = bitrate
                output_params[key.replace("-b:", "-bufsize:")] = "{:g}{}".format(
                    to_float(bitrate[:-1]) * 2, bitrate[-1]
                )

        # define additional streams optimization parameters
        if output_params["-vcodec"] == "libx264":
            if "-bf" not in self.__params:
                output_params["-bf"] = 1
            if "-sc_threshold" not in self.__params:
                output_params["-sc_threshold"] = 0
            if "-keyint_min" not in self.__params:
                output_params["-keyint_min"] = gop
        if (
            output_params["-vcodec"] in ["libx264", "libvpx-vp9"]
            and "-g" not in self.__params
        ):
            output_params["-g"] = gop
        if output_params["-vcodec"] == "libx265":
            # merge user's `-x265-params` with the GOP, since only the last one would win
            x265_params = self.__params.pop("-x265-params", "")
            output_params["-x265-params"] = ":".join(
                x
                for x in [
                    x265_params if isinstance(x265_params, str) else "",
                    "keyint={0}:min-keyint={0}".format(gop),
                ]
                if x
            )
        # streams with own framerate need time-based keyframes to keep segments aligned
        if any(k.startswith("-r:v:") for k in output_params):
            output_params["-force_key_frames"] = "expr:gte(t,n_forced*{:g})".format(
                gop / self.__sourceframerate
            )

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
            self.__logging and logger.info("No additional `-streams` are provided.")
            return output_params

        # check if streams are valid
        if isinstance(streams, list) and all(isinstance(x, dict) for x in streams):
            # keep track of streams
            stream_count = 1
            # calculate source aspect-ratio (display aspect, for anamorphic sources)
            source_aspect_ratio = self.__source_dar or (
                self.__inputwidth / self.__inputheight
            )
            # log the process
            self.__logging and logger.debug(
                "Processing {} streams.".format(len(streams))
            )
            # iterate over given streams
            for idx, stream in enumerate(streams):
                # log stream processing
                self.__logging and logger.debug("Processing Stream: #{}".format(idx))
                # work on a copy, so user's dict stays intact
                stream = dict(stream)
                # handle intermediate stream data as dictionary
                intermediate_dict = {}
                # define and map stream to intermediate dict
                intermediate_dict["-core{}".format(stream_count)] = self.__stream_maps(
                    output_params
                )

                # extract resolution & individual dimension of stream
                resolution = stream.pop("-resolution", "")
                dimensions = (
                    resolution.lower().split("x")
                    if (resolution and isinstance(resolution, str))
                    else []
                )
                # validate resolution
                if not (len(dimensions) == 2 and all(x.isnumeric() for x in dimensions)):
                    # otherwise log error and skip stream
                    logger.error(
                        "Missing `-resolution` value. Invalid stream `{}` Skipped!".format(
                            stream
                        )
                    )
                    continue
                # encoders need even dimensions
                width, height = (int(x) + int(x) % 2 for x in dimensions)
                (width, height) != tuple(int(x) for x in dimensions) and logger.warning(
                    "The stream resolution '{}' is rounded up to even '{}x{}'.".format(
                        resolution, width, height
                    )
                )
                # verify resolution is w.r.t source aspect-ratio
                if abs(width - height * source_aspect_ratio) >= 2:
                    logger.warning(
                        "The provided stream resolution '{}' does not align with the source aspect ratio. Output stream may appear distorted!".format(
                            resolution
                        )
                    )
                # assign stream resolution to intermediate dict
                intermediate_dict["-s:v:{}".format(stream_count)] = "{}x{}".format(
                    width, height
                )

                # verify given stream framerate (can't exceed source framerate)
                fps = to_float(stream.pop("-framerate", 0.0))
                if fps > self.__sourceframerate:
                    logger.warning(
                        "The stream framerate `{}` exceeds source framerate `{}`, Ignored!".format(
                            fps, self.__sourceframerate
                        )
                    )
                    fps = self.__sourceframerate
                elif fps > 0 and fps < self.__sourceframerate:
                    intermediate_dict["-r:v:{}".format(stream_count)] = "{:g}".format(
                        fps
                    )

                # verify given stream video-bitrate
                video_bitrate = stream.pop("-video_bitrate", "")
                if _is_bitrate(video_bitrate):
                    # assign it
                    intermediate_dict["-b:v:{}".format(stream_count)] = video_bitrate
                elif fps > 0:
                    # otherwise calculate video-bitrate
                    intermediate_dict["-b:v:{}".format(stream_count)] = "{}k".format(
                        get_video_bitrate(width, height, fps, bpp)
                    )
                else:
                    # If everything fails, log and skip the stream!
                    logger.error(
                        "Unable to determine Video-Bitrate for the stream `{}`. Skipped!".format(
                            resolution
                        )
                    )
                    continue
                # verify given stream audio-bitrate
                audio_bitrate = stream.pop("-audio_bitrate", "")
                if "-acodec" in output_params:
                    intermediate_dict["-b:a:{}".format(stream_count)] = (
                        audio_bitrate
                        if _is_bitrate(audio_bitrate)
                        else "{}k".format(128 if (width > 800) else 96)
                    )
                # update output parameters
                output_params.update(intermediate_dict)
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

    def __pop_duration(self, key, default):
        """
        Internal method that pops a positive duration _(seconds, fractional allowed)_ parameter.
        """
        value = self.__params.pop(key, default)
        if is_number(value) and value > 0:
            return value
        # otherwise reset to default
        logger.warning("Invalid `{}` value skipped!".format(key))
        return default

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
        # `remove_at_exit` is a DASH-only option
        self.__params.pop("-remove_at_exit", None) is not None and logger.warning(
            "The `-remove_at_exit` parameter is only supported by DASH, Skipped!"
        )
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
            # `hls_init_time` & `hls_time` must be greater than 0
            output_params["-hls_init_time"] = self.__pop_duration("-hls_init_time", 4)
            output_params["-hls_time"] = self.__pop_duration("-hls_time", 4)
            # `hls_flags` must be string
            default_hls_flags = self.__params.pop(
                "-hls_flags", "delete_segments+independent_segments+temp_file"
            )
            if isinstance(default_hls_flags, str):
                output_params["-hls_flags"] = default_hls_flags
            else:
                # otherwise reset to default
                logger.warning("Invalid `-hls_flags` value skipped!")
                output_params["-hls_flags"] = (
                    "delete_segments+independent_segments+temp_file"
                )
        else:
            # enforce "contain all the segments"
            output_params["-hls_list_size"] = 0
            output_params["-hls_playlist_type"] = "vod"

        # handle base URL for absolute paths
        hls_base_url = self.__params.pop("-hls_base_url", "")
        if hls_base_url and isinstance(hls_base_url, str):
            output_params["-hls_base_url"] = hls_base_url
        elif hls_base_url:
            logger.warning("Invalid `-hls_base_url` value skipped!")

        # Handling <hls_segment_filename>
        # Here filename will be based on `stream_count` dict parameter that
        # would be used to check whether stream is multi-variant(>1) or single(0-1)
        segment_template = (
            "{}-stream%v-%03d.{}"
            if output_params["stream_count"] > 1
            else "{}-stream-%03d.{}"
        )
        # NOTE: `/` joins, since segment paths may be URLs
        output_params["-hls_segment_filename"] = segment_template.format(
            os.path.dirname(self.__out_file) + "/chunk",
            "m4s" if output_params["-hls_segment_type"] == "fmp4" else "ts",
        )
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
            # `window_size` must be greater than or equal to 0
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
                output_params["-extra_window_size"] = extra_window_size
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
            # `seg_duration` must be greater than 0
            output_params["-seg_duration"] = self.__pop_duration("-seg_duration", 4)
            # Disable (0) the use of a SegmentTimeline inside a SegmentTemplate.
            output_params["-use_timeline"] = 0
        else:
            # `seg_duration` must be greater than 0
            output_params["-seg_duration"] = self.__pop_duration("-seg_duration", 5)
            # Enable (1) the use of a SegmentTimeline inside a SegmentTemplate.
            output_params["-use_timeline"] = 1

        # Finally, some hardcoded DASH parameters (Refer FFmpeg docs for more info.)
        output_params["-use_template"] = 1
        output_params["-adaptation_sets"] = "id=0,streams=v {}".format(
            "id=1,streams=a" if ("-acodec" in output_params) else ""
        ).strip()
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
            for count in range(stream_count):
                stream_map += "v:{}{} ".format(
                    count, ",a:{}".format(count) if "-acodec" in output_params else ","
                )
            hls_commands += [
                "-master_pl_name",
                os.path.basename(self.__out_file),
                "-var_stream_map",
                stream_map.strip(),
                os.path.dirname(self.__out_file) + "/stream_%v.m3u8",
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
        # ensuring less cluttering if silent mode
        hide_banner = [] if self.__logging else ["-hide_banner"]
        # format commands
        if self.__video_source:
            ffmpeg_cmd = (
                [self.__ffmpeg, "-y"]
                # read files at native speed when live-streaming them
                + (
                    ["-re"]
                    if self.__livestreaming and os.path.isfile(self.__video_source)
                    else []
                )
                + hide_banner
                + ["-i", self.__video_source]
                + input_commands
                + output_commands
                + stream_commands
            )
        else:
            ffmpeg_cmd = (
                [self.__ffmpeg, "-y", *hide_banner, "-f", "rawvideo", "-vcodec", "rawvideo", *input_commands, "-i", "-", *output_commands, *stream_commands]
            )
        # format outputs
        ffmpeg_cmd.extend(hls_commands if hls_commands else [self.__out_file])
        # Launch the FFmpeg pipeline with built command
        self.__logging and logger.debug(
            "Executing FFmpeg command: `{}`".format(" ".join(ffmpeg_cmd))
        )
        self.__logging and logger.critical(
            "Transcoding streaming chunks. Please wait..."
        )
        if self.__video_source:
            # run single-source transcoding to completion
            pbar = None

            def progress(done, total):
                nonlocal pbar
                # drive progress bar in silent mode
                if not self.__logging and total:
                    if pbar is None:
                        pbar = tqdm(total=total, desc="Processing", unit="s")
                    # FFmpeg may report a little past the source duration
                    pbar.update(min(done, total) - pbar.n)
                if self.__progress_callback:
                    try:
                        self.__progress_callback(done, total)
                    except Exception:
                        # keep transcoding (and draining stderr), drop the callback
                        logger.exception(
                            "The `-progress_callback` raised, and is disabled for this run!"
                        )
                        self.__progress_callback = None

            result = run_ffmpeg(
                ffmpeg_cmd, progress=progress, cancel=self.__cancel, echo=self.__logging
            )
            # close progress bar
            pbar is not None and pbar.close()
            self.returncode = result.returncode
            if self.__cancel.is_set():
                # stopped by `close()`
                logger.warning("Transcoding cancelled by user!")
            elif result.returncode != 0:
                # log and raise error if return_code is non-zero
                logger.error(
                    "StreamGear failed to initiate stream for this video source!"
                )
                raise sp.CalledProcessError(
                    result.returncode, ffmpeg_cmd, stderr=result.stderr
                )
            else:
                # log if successful
                self.__logging and logger.critical(
                    "Transcoding Ended. {} Streaming assets are successfully generated at specified path.".format(
                        self.__format.upper()
                    )
                )
        else:
            # pipe frames, capture stderr tail (mirrored to console in logging mode)
            self.__stderr = deque(maxlen=50)
            self.__process = sp.Popen(
                ffmpeg_cmd, stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.PIPE
            )
            self.__stderr_reader = Thread(
                target=_pump_stderr,
                args=(self.__process.stderr, self.__stderr, self.__logging),
                daemon=True,
            )
            self.__stderr_reader.start()

    @property
    def stderr_tail(self) -> list[str]:
        """
        Last lines of FFmpeg's stderr _(Real-time Frames Mode only, else empty)_.
        """
        return list(self.__stderr or ())

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
    def terminate(self) -> None:
        """
        !!! warning "[DEPRECATION NOTICE]: This method is now deprecated and will be removed in a future release."

        This function ensures backward compatibility for the `terminate()` method to maintain the API on existing systems.
        It achieves this by calling the new `close()` method to terminate various
        StreamGear processes.
        """

        self.close()

    def close(self, timeout: float | None = None) -> None:
        """
        Safely terminates various StreamGear process. Can be called from another thread to stop a running `transcode_source()`.

        Parameters:
            timeout (float): seconds to wait for FFmpeg to finish, before stopping it forcefully _(Real-time Frames Mode only)_.
        """
        # log termination
        self.__logging and logger.debug("Terminating StreamGear Processes.")
        # no frames after close
        self.__closed = True
        # stop any running single-source transcoding
        self.__cancel.set()

        # return if no process was initiated at first place
        if self.__process is None:
            return
        # close `stdin` (EOF lets FFmpeg finish encoding)
        try:
            self.__process.stdin.close()
        except OSError:
            pass  # FFmpeg already died, `stream()` has reported it
        # forced termination if specified.
        self.__forced_termination and self.__process.poll() is None and (
            self.__process.terminate()
        )
        # wait if process is still processing
        try:
            self.__process.wait(timeout)
        except sp.TimeoutExpired:
            logger.warning(
                "FFmpeg didn't exit within {}s. Stopping it forcefully!".format(timeout)
            )
            _stop_process(self.__process)
        self.__stderr_reader.join(1.0)
        self.returncode = self.__process.returncode
        # discard process
        self.__process = None
