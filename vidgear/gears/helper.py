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

# Contains all the support functions/modules required by Vidgear packages

# import the necessary packages
import os
import re
import sys
import cv2
import types
import errno
import stat
import shutil
import importlib
import requests
import numpy as np
import logging as log
import platform
import socket
from tqdm import tqdm
from contextlib import closing
from pathlib import Path
from colorlog import ColoredFormatter
from pkg_resources import parse_version
from requests.adapters import HTTPAdapter, Retry
from ..version import __version__


def logger_handler():
    """
    ## logger_handler

    Returns the logger handler

    **Returns:** A logger handler
    """
    # logging formatter
    formatter = ColoredFormatter(
        "{green}{asctime}{reset} :: {bold_purple}{name:^13}{reset} :: {log_color}{levelname:^8}{reset} :: {bold_white}{message}",
        datefmt="%H:%M:%S",
        reset=True,
        log_colors={
            "INFO": "bold_cyan",
            "DEBUG": "bold_yellow",
            "WARNING": "bold_red,fg_thin_yellow",
            "ERROR": "bold_red",
            "CRITICAL": "bold_red,bg_white",
        },
        style="{",
    )
    # check if VIDGEAR_LOGFILE defined
    file_mode = os.environ.get("VIDGEAR_LOGFILE", False)
    # define handler
    handler = log.StreamHandler()
    if file_mode and isinstance(file_mode, str):
        file_path = os.path.abspath(file_mode)
        if (os.name == "nt" or os.access in os.supports_effective_ids) and os.access(
            os.path.dirname(file_path), os.W_OK
        ):
            file_path = (
                os.path.join(file_path, "vidgear.log")
                if os.path.isdir(file_path)
                else file_path
            )
            handler = log.FileHandler(file_path, mode="a")
            formatter = log.Formatter(
                "{asctime} :: {name} :: {levelname} :: {message}",
                datefmt="%H:%M:%S",
                style="{",
            )

    handler.setFormatter(formatter)
    return handler


# global var to check
# if version is logged
ver_is_logged = False

# define logger
logger = log.getLogger("Helper")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


def logcurr_vidgear_ver(logging=False):
    """
    ## logcurr_vidgear_ver

    A auxiliary function to log current vidgear version for debugging.

    Parameters:
        logging (bool): enables logging for its operations
    """
    # making changes to global var
    global ver_is_logged
    # log current vidgear version
    logging and not (ver_is_logged) and logger.info(
        "Running VidGear Version: {}".format(str(__version__))
    )
    # disable logging same thing again
    if logging and not (ver_is_logged):
        ver_is_logged = True


def get_module_version(module=None):
    """
    ## get_module_version

    Retrieves version of specified module

    Parameters:
        name (ModuleType): module of datatype `ModuleType`.

    **Returns:** version of specified module as string
    """
    # check if module type is valid
    assert not (module is None) and isinstance(
        module, types.ModuleType
    ), "[Vidgear:ERROR] :: Invalid module!"

    # get version from attribute
    version = getattr(module, "__version__", None)
    # retry if failed
    if version is None:
        # some modules uses a capitalized attribute name
        version = getattr(module, "__VERSION__", None)
    # raise if still failed
    if version is None:
        raise ImportError(
            "[Vidgear:ERROR] ::  Can't determine version for module: `{}`!".format(
                module.__name__
            )
        )
    return str(version)


def import_dependency_safe(
    name,
    error="raise",
    pkg_name=None,
    min_version=None,
    custom_message=None,
):
    """
    ## import_dependency_safe

    Imports specified dependency safely. By default(`error = raise`), if a dependency is missing,
    an ImportError with a meaningful message will be raised. Otherwise if `error = log` a warning
    will be logged and on `error = silent` everything will be quit. But If a dependency is present,
    but older than specified, an error is raised if specified.

    Parameters:
        name (string): name of dependency to be imported.
        error (string): raise or Log or silence ImportError. Possible values are `"raise"`, `"log"` and `silent`. Default is `"raise"`.
        pkg_name (string): (Optional) package name of dependency(if different `pip` name). Otherwise `name` will be used.
        min_version (string): (Optional) required minimum version of the dependency to be imported.
        custom_message (string): (Optional) custom Import error message to be raised or logged.

    **Returns:** The imported module, when found and the version is correct(if specified). Otherwise `None`.
    """
    # check specified parameters
    sub_class = ""
    if not name or not isinstance(name, str):
        return None
    else:
        # extract name in case of relative import
        name = name.strip()
        if name.startswith("from"):
            name = name.split(" ")
            name, sub_class = (name[1].strip(), name[-1].strip())

    assert error in [
        "raise",
        "log",
        "silent",
    ], "[Vidgear:ERROR] :: Invalid value at `error` parameter."

    # specify package name of dependency(if defined). Otherwise use name
    install_name = pkg_name if not (pkg_name is None) else name

    # create message
    msg = (
        custom_message
        if not (custom_message is None)
        else "Failed to find required dependency '{}'. Install it with  `pip install {}` command.".format(
            name, install_name
        )
    )
    # try importing dependency
    try:
        module = importlib.import_module(name)
        if sub_class:
            module = getattr(module, sub_class)
    except Exception:
        # handle errors.
        if error == "raise":
            raise ImportError(msg) from None
        elif error == "log":
            logger.error(msg)
            return None
        else:
            return None

    # check if minimum required version
    if not (min_version) is None:
        # Handle submodules
        parent_module = name.split(".")[0]
        if parent_module != name:
            # grab parent module
            module_to_get = sys.modules[parent_module]
        else:
            module_to_get = module
        # extract version
        version = get_module_version(module_to_get)
        # verify
        if parse_version(version) < parse_version(min_version):
            # create message
            msg = """Unsupported version '{}' found. Vidgear requires '{}' dependency installed with version '{}' or greater. 
            Update it with  `pip install -U {}` command.""".format(
                parent_module, min_version, version, install_name
            )
            # handle errors.
            if error == "silent":
                return None
            else:
                # raise
                raise ImportError(msg)

    return module


# set default timer for download requests
DEFAULT_TIMEOUT = 3


class TimeoutHTTPAdapter(HTTPAdapter):
    """
    A custom Transport Adapter with default timeouts
    """

    def __init__(self, *args, **kwargs):
        self.timeout = DEFAULT_TIMEOUT
        if "timeout" in kwargs:
            self.timeout = kwargs["timeout"]
            del kwargs["timeout"]
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        timeout = kwargs.get("timeout")
        if timeout is None:
            kwargs["timeout"] = self.timeout
        return super().send(request, **kwargs)


def check_CV_version():
    """
    ## check_CV_version

    **Returns:** OpenCV's version first bit
    """
    if parse_version(cv2.__version__) >= parse_version("4"):
        return 4
    else:
        return 3


def check_open_port(address, port=22):
    """
    ## check_open_port

    Checks whether specified port open at given IP address.

    Parameters:
        address (string): given IP address.
        port (int): check if port is open at given address.

    **Returns:** A boolean value, confirming whether given port is open at given IP address.
    """
    if not address:
        return False
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        if sock.connect_ex((address, port)) == 0:
            return True
        else:
            return False


def check_WriteAccess(path, is_windows=False, logging=False):
    """
    ## check_WriteAccess

    Checks whether given path directory has Write-Access.

    Parameters:
        path (string): absolute path of directory
        is_windows (boolean): is running on Windows OS?
        logging (bool): enables logging for its operations

    **Returns:** A boolean value, confirming whether Write-Access available, or not?.
    """
    # check if path exists
    dirpath = Path(path)
    try:
        if not (dirpath.exists() and dirpath.is_dir()):
            logger.warning(
                "Specified directory `{}` doesn't exists or valid.".format(path)
            )
            return False
        else:
            path = dirpath.resolve()
    except:
        return False
    # check filepath on *nix systems
    if not is_windows:
        uid = os.geteuid()
        gid = os.getegid()
        s = os.stat(path)
        mode = s[stat.ST_MODE]
        return (
            ((s[stat.ST_UID] == uid) and (mode & stat.S_IWUSR))
            or ((s[stat.ST_GID] == gid) and (mode & stat.S_IWGRP))
            or (mode & stat.S_IWOTH)
        )
    # otherwise, check filepath on windows
    else:
        write_accessible = False
        temp_fname = os.path.join(path, "temp.tmp")
        try:
            fd = os.open(temp_fname, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
            os.close(fd)
            write_accessible = True
        except Exception as e:
            if isinstance(e, PermissionError):
                logger.error(
                    "You don't have adequate access rights to use `{}` directory!".format(
                        path
                    )
                )
            logging and logger.exception(str(e))
        finally:
            delete_file_safe(temp_fname)
        return write_accessible


def check_gstreamer_support(logging=False):
    """
    ## check_gstreamer_support

    Checks whether OpenCV is compiled with Gstreamer(`>=1.0.0`) support.

    Parameters:
        logging (bool): enables logging for its operations

    **Returns:** A Boolean value
    """
    raw = cv2.getBuildInformation()
    gst = [
        x.strip()
        for x in raw.split("\n")
        if x and re.search(r"GStreamer[,-:]+\s*(?:YES|NO)", x)
    ]
    if gst and "YES" in gst[0]:
        version = re.search(r"(\d+\.)?(\d+\.)?(\*|\d+)", gst[0])
        logging and logger.debug("Found GStreamer version:{}".format(version[0]))
        return version[0] >= "1.0.0"
    else:
        logger.warning("GStreamer not found!")
        return False


def get_supported_resolution(value, logging=False):
    """
    ## get_supported_resolution

    Parameters:
        value (string): value to be validated
        logging (bool): enables logging for its operations

    **Returns:** Valid stream resolution
    """
    # default to best
    stream_resolution = "best"
    supported_stream_qualities = [
        "144p",
        "240p",
        "360p",
        "480p",
        "720p",
        "1080p",
        "1440p",
        "2160p",
        "4320p",
        "worst",
        "best",
    ]
    if isinstance(value, str):
        if value.strip().lower() in supported_stream_qualities:
            stream_resolution = value.strip().lower()
            logging and logger.debug(
                "Selecting `{}` resolution for streams.".format(stream_resolution)
            )
        else:
            logger.warning(
                "Specified stream-resolution `{}` is not supported. Reverting to `best`!".format(
                    value
                )
            )
    else:
        logger.warning(
            "Specified stream-resolution `{}` is Invalid. Reverting to `best`!".format(
                value
            )
        )
    return stream_resolution


def dimensions_to_resolutions(value):
    """
    ## dimensions_to_resolutions

    Parameters:
        value (list): list of dimensions (e.g. `640x360`)

    **Returns:** list of resolutions (e.g. `360p`)
    """
    supported_resolutions = {
        "256x144": "144p",
        "426x240": "240p",
        "640x360": "360p",
        "854x480": "480p",
        "1280x720": "720p",
        "1920x1080": "1080p",
        "2560x1440": "1440p",
        "3840x2160": "2160p",
        "7680x4320": "4320p",
    }
    return (
        list(map(supported_resolutions.get, value, value))
        if isinstance(value, list)
        else []
    )


def get_supported_vencoders(path):
    """
    ## get_supported_vencoders

    Find and returns FFmpeg's supported video encoders

    Parameters:
        path (string): absolute path of FFmpeg binaries

    **Returns:** List of supported encoders.
    """
    encoders = check_output([path, "-hide_banner", "-encoders"])
    splitted = encoders.split(b"\n")
    # extract video encoders
    supported_vencoders = [
        x.decode("utf-8").strip()
        for x in splitted[2 : len(splitted) - 1]
        if x.decode("utf-8").strip().startswith("V")
    ]
    # compile regex
    finder = re.compile(r"[A-Z]*[\.]+[A-Z]*\s[a-z0-9_-]*")
    # find all outputs
    outputs = finder.findall("\n".join(supported_vencoders))
    # return output findings
    return [[s for s in o.split(" ")][-1] for o in outputs]


def get_supported_demuxers(path):
    """
    ## get_supported_demuxers

    Find and returns FFmpeg's supported demuxers

    Parameters:
        path (string): absolute path of FFmpeg binaries

    **Returns:** List of supported demuxers.
    """
    demuxers = check_output([path, "-hide_banner", "-demuxers"])
    splitted = [x.decode("utf-8").strip() for x in demuxers.split(b"\n")]
    supported_demuxers = splitted[splitted.index("--") + 1 : len(splitted) - 1]
    # compile regex
    finder = re.compile(r"\s\s[a-z0-9_,-]+\s+")
    # find all outputs
    outputs = finder.findall("\n".join(supported_demuxers))
    # return output findings
    return [o.strip() for o in outputs]


def get_supported_pixfmts(path):
    """
    ## get_supported_pixfmts

    Find and returns all FFmpeg's supported pixel formats.

    Parameters:
        path (string): absolute path of FFmpeg binaries

    **Returns:** List of supported pixel formats.
    """
    pxfmts = check_output([path, "-hide_banner", "-pix_fmts"])
    splitted = pxfmts.split(b"\n")
    srtindex = [i for i, s in enumerate(splitted) if b"-----" in s]
    # extract video encoders
    supported_pxfmts = [
        x.decode("utf-8").strip()
        for x in splitted[srtindex[0] + 1 :]
        if x.decode("utf-8").strip()
    ]
    # compile regex
    finder = re.compile(r"([A-Z]*[\.]+[A-Z]*\s[a-z0-9_-]*)(\s+[0-4])(\s+[0-9]+)")
    # find all outputs
    outputs = finder.findall("\n".join(supported_pxfmts))
    # return output findings
    return [[s for s in o[0].split(" ")][-1] for o in outputs if len(o) == 3]


def is_valid_url(path, url=None, logging=False):
    """
    ## is_valid_url

    Checks URL validity by testing its scheme against
    FFmpeg's supported protocols

    Parameters:
        path (string): absolute path of FFmpeg binaries
        url (string): URL to be validated
        logging (bool): enables logging for its operations

    **Returns:** A boolean value, confirming whether tests passed, or not?.
    """
    if url is None or not (url):
        logger.warning("URL is empty!")
        return False
    # extract URL scheme
    extracted_scheme_url = url.split("://", 1)[0]
    # extract all FFmpeg supported protocols
    protocols = check_output([path, "-hide_banner", "-protocols"])
    splitted = [x.decode("utf-8").strip() for x in protocols.split(b"\n")]
    supported_protocols = splitted[splitted.index("Output:") + 1 : len(splitted) - 1]
    # rtsp is a demuxer somehow
    supported_protocols += ["rtsp"] if "rtsp" in get_supported_demuxers(path) else []
    # Test and return result whether scheme is supported
    if extracted_scheme_url and extracted_scheme_url in supported_protocols:
        logging and logger.debug(
            "URL scheme `{}` is supported by FFmpeg.".format(extracted_scheme_url)
        )
        return True
    else:
        logger.warning(
            "URL scheme `{}` isn't supported by FFmpeg!".format(extracted_scheme_url)
        )
        return False


def validate_video(path, video_path=None, logging=False):
    """
    ## validate_video

    Validates video by retrieving resolution/size and framerate from file.

    Parameters:
        path (string): absolute path of FFmpeg binaries
        video_path (string): absolute path to Video.

    **Returns:** A dictionary of retieved Video resolution _(as tuple(width, height))_ and framerate _(as float)_.
    """
    if video_path is None or not (video_path):
        logger.warning("Video path is empty!")
        return None

    # extract metadata
    metadata = check_output(
        [path, "-hide_banner", "-i", video_path], force_retrieve_stderr=True
    )
    # clean and search
    stripped_data = [x.decode("utf-8").strip() for x in metadata.split(b"\n")]
    logging and logger.debug(stripped_data)
    result = {}
    for data in stripped_data:
        output_a = re.findall(r"([1-9]\d+)x([1-9]\d+)", data)
        output_b = re.findall(r"\d+(?:\.\d+)?\sfps", data)
        if len(result) == 2:
            break
        if output_b and not "framerate" in result:
            result["framerate"] = re.findall(r"[\d\.\d]+", output_b[0])[0]
        if output_a and not "resolution" in result:
            result["resolution"] = output_a[-1]

    # return values
    return result if (len(result) == 2) else None


def create_blank_frame(frame=None, text="", logging=False):
    """
    ## create_blank_frame

    Create blank frames of given frame size with text

    Parameters:
        frame (numpy.ndarray): inputs numpy array(frame).
        text (str): Text to be written on frame.
    **Returns:**  A reduced numpy ndarray array.
    """
    # check if frame is valid
    if frame is None or not (isinstance(frame, np.ndarray)):
        raise ValueError("[Helper:ERROR] :: Input frame is invalid!")
    # grab the frame size
    (height, width) = frame.shape[:2]
    # create blank frame
    blank_frame = np.zeros(frame.shape, frame.dtype)
    # setup text
    if text and isinstance(text, str):
        logging and logger.debug("Adding text: {}".format(text))
        # setup font
        font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        # get boundary of this text
        fontScale = min(height, width) / (25 / 0.25)
        textsize = cv2.getTextSize(text, font, fontScale, 5)[0]
        # get coords based on boundary
        textX = (width - textsize[0]) // 2
        textY = (height + textsize[1]) // 2
        # put text
        cv2.putText(
            blank_frame, text, (textX, textY), font, fontScale, (125, 125, 125), 6
        )

    # return frame
    return blank_frame


def extract_time(value):
    """
    ## extract_time

    Extract time from give string value.

    Parameters:
        value (string): string value.

    **Returns:** Time _(in seconds)_ as integer.
    """
    if not (value):
        logger.warning("Value is empty!")
        return 0
    else:
        stripped_data = value.strip()
        t_duration = re.findall(
            r"(?:[01]\d|2[0123]):(?:[012345]\d):(?:[012345]\d)", stripped_data
        )
        return (
            sum(
                int(x) * 60**i
                for i, x in enumerate(reversed(t_duration[0].split(":")))
            )
            if t_duration
            else 0
        )


def validate_audio(path, source=None):
    """
    ## validate_audio

    Validates audio by retrieving audio-bitrate from file.

    Parameters:
        path (string): absolute path of FFmpeg binaries
        source (string/list): source to be validated.

    **Returns:** A string value, confirming whether audio is present, or not?.
    """
    if source is None or not (source):
        logger.warning("Audio input source is empty!")
        return ""

    # create ffmpeg command
    cmd = [path, "-hide_banner"] + (
        source if isinstance(source, list) else ["-i", source]
    )
    # extract metadata
    metadata = check_output(cmd, force_retrieve_stderr=True)
    # extract bitrate
    audio_bitrate_meta = [
        line.strip()
        for line in metadata.decode("utf-8").split("\n")
        if "Audio:" in line
    ]
    audio_bitrate = (
        re.findall(r"([0-9]+)\s(kb|mb|gb)\/s", audio_bitrate_meta[0])[-1]
        if audio_bitrate_meta
        else ""
    )
    # extract samplerate
    audio_samplerate_metadata = [
        line.strip()
        for line in metadata.decode("utf-8").split("\n")
        if all(x in line for x in ["Audio:", "Hz"])
    ]
    audio_samplerate = (
        re.findall(r"[0-9]+\sHz", audio_samplerate_metadata[0])[0]
        if audio_samplerate_metadata
        else ""
    )
    # format into actual readable bitrate value
    if audio_bitrate:
        # return bitrate directly
        return "{}{}".format(int(audio_bitrate[0].strip()), audio_bitrate[1].strip()[0])
    elif audio_samplerate:
        # convert samplerate to bitrate first
        sample_rate_value = int(audio_samplerate.split(" ")[0])
        channels_value = 1 if "mono" in audio_samplerate_metadata[0] else 2
        bit_depth_value = re.findall(
            r"(u|s|f)([0-9]+)(le|be)", audio_samplerate_metadata[0]
        )[0][1]
        return (
            (
                str(
                    get_audio_bitrate(
                        sample_rate_value, channels_value, int(bit_depth_value)
                    )
                )
                + "k"
            )
            if bit_depth_value
            else ""
        )
    else:
        return ""


def get_audio_bitrate(samplerate, channels, bit_depth):
    """
    ## get_audio_bitrate

    Calculate optimum bitrate from audio samplerate, channels, bit-depth values

    Parameters:
        samplerate (int): audio samplerate value
        channels (int): number of channels
        bit_depth (float): audio bit depth value

    **Returns:** Audio bitrate _(in Kbps)_ as integer.
    """
    return round((samplerate * channels * bit_depth) / 1000)


def get_video_bitrate(width, height, fps, bpp):
    """
    ## get_video_bitrate

    Calculate optimum Bitrate from resolution, framerate, bits-per-pixels values

    Parameters:
        width (int): video-width
        height (int): video-height
        fps (float): video-framerate
        bpp (float): bit-per-pixels value

    **Returns:** Video bitrate _(in Kbps)_ as integer.
    """
    return round((width * height * bpp * fps) / 1000)


def delete_file_safe(file_path):
    """
    ## delete_ext_safe

    Safely deletes files at given path.

    Parameters:
        file_path (string): path to the file
    """
    try:
        dfile = Path(file_path)
        if sys.version_info >= (3, 8, 0):
            dfile.unlink(missing_ok=True)
        else:
            dfile.exists() and dfile.unlink()
    except Exception as e:
        logger.exception(str(e))


def mkdir_safe(dir_path, logging=False):
    """
    ## mkdir_safe

    Safely creates directory at given path.

    Parameters:
        dir_path (string): path to the directory
        logging (bool): enables logging for its operations

    """
    try:
        os.makedirs(dir_path)
        logging and logger.debug("Created directory at `{}`".format(dir_path))
    except (OSError, IOError) as e:
        if e.errno != errno.EACCES and e.errno != errno.EEXIST:
            raise


def delete_ext_safe(dir_path, extensions=[], logging=False):
    """
    ## delete_ext_safe

    Safely deletes files with given extensions at given path.

    Parameters:
        dir_path (string): path to the directory
        extensions (list): list of extensions to be deleted
        logging (bool): enables logging for its operations

    """
    if not extensions or not os.path.exists(dir_path):
        logger.warning("Invalid input provided for deleting!")
        return

    logger.critical("Clearing Assets at `{}`!".format(dir_path))

    for ext in extensions:
        if len(ext) == 2:
            files_ext = [
                os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if f.startswith(ext[0]) and f.endswith(ext[1])
            ]
        else:
            files_ext = [
                os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if f.endswith(ext)
            ]
        for file in files_ext:
            delete_file_safe(file)
            logging and logger.debug("Deleted file: `{}`".format(file))


def capPropId(property, logging=True):
    """
    ## capPropId

    Retrieves the OpenCV property's Integer(Actual) value from string.

    Parameters:
        property (string): inputs OpenCV property as string.
        logging (bool): enables logging for its operations

    **Returns:** Resultant integer value.
    """
    integer_value = 0
    try:
        integer_value = getattr(cv2, property)
    except Exception as e:
        if logging:
            logger.exception(str(e))
            logger.critical("`{}` is not a valid OpenCV property!".format(property))
        return None
    return integer_value


def retrieve_best_interpolation(interpolations):
    """
    ## retrieve_best_interpolation
    Retrieves best interpolation for resizing

    Parameters:
        interpolations (list): list of interpolations as string.
    **Returns:**  Resultant integer value of found interpolation.
    """
    if isinstance(interpolations, list):
        for intp in interpolations:
            interpolation = capPropId(intp, logging=False)
            if not (interpolation is None):
                return interpolation
    return None


def reducer(frame=None, percentage=0, interpolation=cv2.INTER_LANCZOS4):
    """
    ## reducer

    Reduces frame size by given percentage

    Parameters:
        frame (numpy.ndarray): inputs numpy array(frame).
        percentage (int/float): inputs size-reduction percentage.
        interpolation (int): Change resize interpolation.

    **Returns:**  A reduced numpy ndarray array.
    """
    # check if frame is valid
    if frame is None:
        raise ValueError("[Helper:ERROR] :: Input frame cannot be NoneType!")

    # check if valid reduction percentage is given
    if not (percentage > 0 and percentage < 90):
        raise ValueError(
            "[Helper:ERROR] :: Given frame-size reduction percentage is invalid, Kindly refer docs."
        )

    if not (isinstance(interpolation, int)):
        raise ValueError(
            "[Helper:ERROR] :: Given interpolation is invalid, Kindly refer docs."
        )

    # grab the frame size
    (height, width) = frame.shape[:2]

    # calculate the ratio of the width from percentage
    reduction = ((100 - percentage) / 100) * width
    ratio = reduction / float(width)
    # construct the dimensions
    dimensions = (int(reduction), int(height * ratio))

    # return the resized frame
    return cv2.resize(frame, dimensions, interpolation=interpolation)


def dict2Args(param_dict):
    """
    ## dict2Args

    Converts dictionary attributes to list(args)

    Parameters:
        param_dict (dict): Parameters dictionary

    **Returns:** Arguments list
    """
    args = []
    for key in param_dict.keys():
        if key in ["-clones"] or key.startswith("-core"):
            if isinstance(param_dict[key], list):
                args.extend(param_dict[key])
            else:
                logger.warning(
                    "{} with invalid datatype:`{}`, Skipped!".format(
                        "Core parameter" if key.startswith("-core") else "Clone",
                        param_dict[key],
                    )
                )
        else:
            args.append(key)
            args.append(str(param_dict[key]))
    return args


def get_valid_ffmpeg_path(
    custom_ffmpeg="", is_windows=False, ffmpeg_download_path="", logging=False
):
    """
    ## get_valid_ffmpeg_path

    Validate the given FFmpeg path/binaries, and returns a valid FFmpeg executable path.

    Parameters:
        custom_ffmpeg (string): path to custom FFmpeg executables
        is_windows (boolean): is running on Windows OS?
        ffmpeg_download_path (string): FFmpeg static binaries download location _(Windows only)_
        logging (bool): enables logging for its operations

    **Returns:** A valid FFmpeg executable path string.
    """
    final_path = ""
    if is_windows:
        # checks if current os is windows
        if custom_ffmpeg:
            # if custom FFmpeg path is given assign to local variable
            final_path += custom_ffmpeg
        else:
            # otherwise auto-download them
            try:
                if not (ffmpeg_download_path):
                    # otherwise save to Temp Directory
                    import tempfile

                    ffmpeg_download_path = tempfile.gettempdir()

                logging and logger.debug(
                    "FFmpeg Windows Download Path: {}".format(ffmpeg_download_path)
                )

                # download Binaries
                os_bit = (
                    ("win64" if platform.machine().endswith("64") else "win32")
                    if is_windows
                    else ""
                )
                _path = download_ffmpeg_binaries(
                    path=ffmpeg_download_path, os_windows=is_windows, os_bit=os_bit
                )
                # assign to local variable
                final_path += _path

            except Exception as e:
                # log if any error occurred
                logger.exception(str(e))
                logger.error(
                    "Error in downloading FFmpeg binaries, Check your network and Try again!"
                )
                return False

        if os.path.isfile(final_path):
            # check if valid FFmpeg file exist
            pass
        elif os.path.isfile(os.path.join(final_path, "ffmpeg.exe")):
            # check if FFmpeg directory exists, if does, then check for valid file
            final_path = os.path.join(final_path, "ffmpeg.exe")
        else:
            # else return False
            logging and logger.debug(
                "No valid FFmpeg executables found at Custom FFmpeg path!"
            )
            return False
    else:
        # otherwise perform test for Unix
        if custom_ffmpeg:
            # if custom FFmpeg path is given assign to local variable
            if os.path.isfile(custom_ffmpeg):
                # check if valid FFmpeg file exist
                final_path += custom_ffmpeg
            elif os.path.isfile(os.path.join(custom_ffmpeg, "ffmpeg")):
                # check if FFmpeg directory exists, if does, then check for valid file
                final_path = os.path.join(custom_ffmpeg, "ffmpeg")
            else:
                # else return False
                logging and logger.debug(
                    "No valid FFmpeg executables found at Custom FFmpeg path!"
                )
                return False
        else:
            # otherwise assign ffmpeg binaries from system
            final_path += "ffmpeg"

    logging and logger.debug("Final FFmpeg Path: {}".format(final_path))

    # Final Auto-Validation for FFmeg Binaries. returns final path if test is passed
    return final_path if validate_ffmpeg(final_path, logging=logging) else False


def download_ffmpeg_binaries(path, os_windows=False, os_bit=""):
    """
    ## download_ffmpeg_binaries

    Generates FFmpeg Static Binaries for windows(if not available)

    Parameters:
        path (string): path for downloading custom FFmpeg executables
        os_windows (boolean): is running on Windows OS?
        os_bit (string): 32-bit or 64-bit OS?

    **Returns:** A valid FFmpeg executable path string.
    """
    final_path = ""
    if os_windows and os_bit:
        # initialize with available FFmpeg Static Binaries GitHub Server
        file_url = "https://github.com/abhiTronix/FFmpeg-Builds/releases/latest/download/ffmpeg-static-{}-gpl.zip".format(
            os_bit
        )

        file_name = os.path.join(
            os.path.abspath(path), "ffmpeg-static-{}-gpl.zip".format(os_bit)
        )
        file_path = os.path.join(
            os.path.abspath(path),
            "ffmpeg-static-{}-gpl/bin/ffmpeg.exe".format(os_bit),
        )
        base_path, _ = os.path.split(file_name)  # extract file base path
        # check if file already exists
        if os.path.isfile(file_path):
            final_path += file_path  # skip download if does
        else:
            # import libs
            import zipfile

            # check if given path has write access
            assert os.access(path, os.W_OK), (
                "[Helper:ERROR] :: Permission Denied, Cannot write binaries to directory = "
                + path
            )
            # remove leftovers if exists
            os.path.isfile(file_name) and delete_file_safe(file_name)
            # download and write file to the given path
            with open(file_name, "wb") as f:
                logger.debug(
                    "No Custom FFmpeg path provided. Auto-Installing FFmpeg static binaries from GitHub Mirror now. Please wait..."
                )
                # create session
                with requests.Session() as http:
                    # setup retry strategy
                    retries = Retry(
                        total=3,
                        backoff_factor=1,
                        status_forcelist=[429, 500, 502, 503, 504],
                    )
                    # Mount it for https usage
                    adapter = TimeoutHTTPAdapter(timeout=2.0, max_retries=retries)
                    http.mount("https://", adapter)
                    response = http.get(file_url, stream=True)
                    response.raise_for_status()
                    total_length = (
                        response.headers.get("content-length")
                        if "content-length" in response.headers
                        else len(response.content)
                    )
                    assert not (
                        total_length is None
                    ), "[Helper:ERROR] :: Failed to retrieve files, check your Internet connectivity!"
                    bar = tqdm(total=int(total_length), unit="B", unit_scale=True)
                    for data in response.iter_content(chunk_size=4096):
                        f.write(data)
                        len(data) > 0 and bar.update(len(data))
                    bar.close()
            logger.debug("Extracting executables.")
            with zipfile.ZipFile(file_name, "r") as zip_ref:
                zip_fname, _ = os.path.split(zip_ref.infolist()[0].filename)
                zip_ref.extractall(base_path)
            # perform cleaning
            delete_file_safe(file_name)
            logger.debug("FFmpeg binaries for Windows configured successfully!")
            final_path += file_path
    # return final path
    return final_path


def validate_ffmpeg(path, logging=False):
    """
    ## validate_ffmpeg

    Validate FFmeg Binaries. returns `True` if tests are passed.

    Parameters:
        path (string): absolute path of FFmpeg binaries
        logging (bool): enables logging for its operations

    **Returns:** A boolean value, confirming whether tests passed, or not?.
    """
    try:
        # get the FFmpeg version
        version = check_output([path, "-version"])
        firstline = version.split(b"\n")[0]
        version = firstline.split(b" ")[2].strip()
        if logging:  # log if test are passed
            logger.debug("FFmpeg validity Test Passed!")
            logger.debug(
                "Found valid FFmpeg Version: `{}` installed on this system".format(
                    version
                )
            )
    except Exception as e:
        # log if test are failed
        if logging:
            logger.exception(str(e))
            logger.warning("FFmpeg validity Test Failed!")
        return False
    return True


def check_output(*args, **kwargs):
    """
    ## check_output

    Returns stdin output from subprocess module
    """
    # import libs
    import subprocess as sp

    # workaround for python bug: https://bugs.python.org/issue37380
    if platform.system() == "Windows":
        # see comment https://bugs.python.org/msg370334
        sp._cleanup = lambda: None

    # handle additional params
    retrieve_stderr = kwargs.pop("force_retrieve_stderr", False)

    # execute command in subprocess
    process = sp.Popen(
        stdout=sp.PIPE,
        stderr=sp.DEVNULL if not (retrieve_stderr) else sp.PIPE,
        *args,
        **kwargs,
    )
    output, stderr = process.communicate()
    retcode = process.poll()

    # handle return code
    if retcode and not (retrieve_stderr):
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = args[0]
        error = sp.CalledProcessError(retcode, cmd)
        error.output = output
        raise error

    return output if not (retrieve_stderr) else stderr


def generate_auth_certificates(path, overwrite=False, logging=False):
    """
    ## generate_auth_certificates

    Auto-Generates, and Auto-validates CURVE ZMQ key-pairs for NetGear API's Secure Mode.

    Parameters:
        path (string): path for generating CURVE key-pairs
        overwrite (boolean): overwrite existing key-pairs or not?
        logging (bool): enables logging for its operations

    **Returns:** A valid CURVE key-pairs path as string.
    """
    # import necessary lib
    import zmq.auth

    # check if path corresponds to vidgear only
    if os.path.basename(path) != ".vidgear":
        path = os.path.join(path, ".vidgear")

    # generate keys dir
    keys_dir = os.path.join(path, "keys")
    mkdir_safe(keys_dir, logging=logging)

    # generate separate public and private key dirs
    public_keys_dir = os.path.join(keys_dir, "public_keys")
    secret_keys_dir = os.path.join(keys_dir, "private_keys")

    # check if overwriting is allowed
    if overwrite:
        # delete previous certificates
        for dirs in [public_keys_dir, secret_keys_dir]:
            if os.path.exists(dirs):
                shutil.rmtree(dirs)
            mkdir_safe(dirs, logging=logging)

        # generate new keys
        server_public_file, server_secret_file = zmq.auth.create_certificates(
            keys_dir, "server"
        )
        client_public_file, client_secret_file = zmq.auth.create_certificates(
            keys_dir, "client"
        )

        # move keys to their appropriate directory respectively
        for key_file in os.listdir(keys_dir):
            if key_file.endswith(".key"):
                shutil.move(os.path.join(keys_dir, key_file), public_keys_dir)
            elif key_file.endswith(".key_secret"):
                shutil.move(os.path.join(keys_dir, key_file), secret_keys_dir)
            else:
                # clean redundant keys if present
                redundant_key = os.path.join(keys_dir, key_file)
                if os.path.isfile(redundant_key):
                    delete_file_safe(redundant_key)
    else:
        # otherwise validate available keys
        status_public_keys = validate_auth_keys(public_keys_dir, ".key")
        status_private_keys = validate_auth_keys(secret_keys_dir, ".key_secret")

        # check if all valid keys are found
        if status_private_keys and status_public_keys:
            return (keys_dir, secret_keys_dir, public_keys_dir)

        # check if valid public keys are found
        if not (status_public_keys):
            mkdir_safe(public_keys_dir, logging=logging)

        # check if valid private keys are found
        if not (status_private_keys):
            mkdir_safe(secret_keys_dir, logging=logging)

        # generate new keys
        server_public_file, server_secret_file = zmq.auth.create_certificates(
            keys_dir, "server"
        )
        client_public_file, client_secret_file = zmq.auth.create_certificates(
            keys_dir, "client"
        )

        # move keys to their appropriate directory respectively
        for key_file in os.listdir(keys_dir):
            if key_file.endswith(".key") and not (status_public_keys):
                shutil.move(
                    os.path.join(keys_dir, key_file), os.path.join(public_keys_dir, ".")
                )
            elif key_file.endswith(".key_secret") and not (status_private_keys):
                shutil.move(
                    os.path.join(keys_dir, key_file), os.path.join(secret_keys_dir, ".")
                )
            else:
                # clean redundant keys if present
                redundant_key = os.path.join(keys_dir, key_file)
                if os.path.isfile(redundant_key):
                    delete_file_safe(redundant_key)

    # validate newly generated keys
    status_public_keys = validate_auth_keys(public_keys_dir, ".key")
    status_private_keys = validate_auth_keys(secret_keys_dir, ".key_secret")

    # raise error is validation test fails
    if not (status_private_keys) or not (status_public_keys):
        raise RuntimeError(
            "[Helper:ERROR] :: Unable to generate valid ZMQ authentication certificates at `{}`!".format(
                keys_dir
            )
        )

    # finally return valid key paths
    return (keys_dir, secret_keys_dir, public_keys_dir)


def validate_auth_keys(path, extension):
    """
    ## validate_auth_keys

    Validates, and also maintains generated ZMQ CURVE Key-pairs.

    Parameters:
        path (string): path of generated CURVE key-pairs
        extension (string): type of key-pair to be validated

    **Returns:** A boolean value, confirming whether tests passed, or not?.
    """
    # check for valid path
    if not (os.path.exists(path)):
        return False

    # check if directory empty
    if not (os.listdir(path)):
        return False

    keys_buffer = []  # stores auth-keys

    # loop over auth-keys
    for key_file in os.listdir(path):
        key = os.path.splitext(key_file)
        # check if valid key is generated
        if key and (key[0] in ["server", "client"]) and (key[1] == extension):
            keys_buffer.append(key_file)  # store it

    # remove invalid keys if found
    len(keys_buffer) == 1 and delete_file_safe(os.path.join(path, keys_buffer[0]))

    # return results
    return True if (len(keys_buffer) == 2) else False
