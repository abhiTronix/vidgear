"""
===============================================
vidgear library source-code is deployed under the Apache 2.0 License:

Copyright (c) 2019-2020 Abhishek Thakur(@abhiTronix) <abhi.una12@gmail.com>

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
import errno
import numpy as np
import logging as log
import platform
import requests
from tqdm import tqdm
from colorlog import ColoredFormatter
from pkg_resources import parse_version

try:
    # import OpenCV Binaries
    import cv2

    # check whether OpenCV Binaries are 3.x+
    if parse_version(cv2.__version__) < parse_version("3"):
        raise ImportError(
            "[Vidgear:ERROR] :: Installed OpenCV API version(< 3.0) is not supported!"
        )
except ImportError:
    raise ImportError(
        "[Vidgear:ERROR] :: Failed to detect correct OpenCV executables, install it with `pip3 install opencv-python` command."
    )


def logger_handler():
    """
    ### logger_handler

    Returns a color formatted logger handler

    **Returns:** A logger handler
    """
    # logging formatter
    formatter = ColoredFormatter(
        "%(bold_blue)s%(name)s%(reset)s :: %(log_color)s%(levelname)s%(reset)s :: %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "INFO": "bold_green",
            "DEBUG": "bold_yellow",
            "WARNING": "bold_purple",
            "ERROR": "bold_red",
            "CRITICAL": "bold_red,bg_white",
        },
    )
    # define handler
    handler = log.StreamHandler()
    handler.setFormatter(formatter)
    return handler


# define logger
logger = log.getLogger("Helper")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


def check_CV_version():
    """
    ### check_CV_version

    **Returns:** OpenCV's version first bit
    """
    if parse_version(cv2.__version__) >= parse_version("4"):
        return 4
    else:
        return 3


def is_valid_url(path, url=None, logging=False):
    """
    ### is_valid_url

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
    splitted = protocols.split(b"\n")
    supported_protocols = [
        x.decode("utf-8").strip() for x in splitted[2 : len(splitted) - 1]
    ]
    # Test and return result whether scheme is supported
    if extracted_scheme_url and extracted_scheme_url in supported_protocols:
        if logging:
            logger.debug(
                "URL scheme `{}` is supported by FFmpeg.".format(extracted_scheme_url)
            )
        return True
    else:
        logger.warning(
            "URL scheme `{}` is not supported by FFmpeg!".format(extracted_scheme_url)
        )
        return False


def validate_video(path, video_path=None):
    """
    ### validate_video

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
    result = {}
    for data in stripped_data:
        output_a = re.findall(r"(\d+)x(\d+)", data)
        output_b = re.findall(r"\d+(?:\.\d+)?\sfps", data)
        if len(result) == 2:
            break
        if output_b and not "framerate" in result:
            result["framerate"] = re.findall(r"[\d\.\d]+", output_b[0])[0]
        if output_a and not "resolution" in result:
            result["resolution"] = output_a[-1]

    # return values
    return result if (len(result) == 2) else None


def extract_time(value):
    """
    ### extract_time

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
                int(x) * 60 ** i
                for i, x in enumerate(reversed(t_duration[0].split(":")))
            )
            if t_duration
            else 0
        )


def validate_audio(path, file_path=None):
    """
    ### validate_audio

    Validates audio by retrieving audio-bitrate from file.

    Parameters:
        path (string): absolute path of FFmpeg binaries
        file_path (string): absolute path to file to be validated.

    **Returns:** A string value, confirming whether audio is present, or not?.
    """
    if file_path is None or not (file_path):
        logger.warning("File path is empty!")
        return ""

    # extract audio sample-rate from metadata
    metadata = check_output(
        [path, "-hide_banner", "-i", file_path], force_retrieve_stderr=True
    )
    audio_bitrate = re.findall(r"fltp,\s[0-9]+\s\w\w[/]s", metadata.decode("utf-8"))
    if audio_bitrate:
        filtered = audio_bitrate[0].split(" ")[1:3]
        final_bitrate = "{}{}".format(
            int(filtered[0].strip()),
            "k" if (filtered[1].strip().startswith("k")) else "M",
        )
        return final_bitrate
    else:
        return ""


def get_video_bitrate(width, height, fps, bpp):
    """
    ### get_video_bitrate

    Calculate optimum Bitrate from resolution, framerate, bits-per-pixels values

    Parameters:
        width (int): video-width
        height (int): video-height
        fps (float): video-framerate
        bpp (float): bit-per-pixels value

    **Returns:** Video bitrate _(in Kbps)_ as integer.
    """
    return round((width * height * bpp * fps) / 1000)


def mkdir_safe(dir_path, logging=False):
    """
    ### mkdir_safe

    Safely creates directory at given path.

    Parameters:
        dir_path (string): path to the directory
        logging (bool): enables logging for its operations

    """
    try:
        os.makedirs(dir_path)
        if logging:
            logger.debug("Created directory at `{}`".format(dir_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        if logging:
            logger.debug("Directory already exists at `{}`".format(dir_path))


def delete_safe(dir_path, extensions=[], logging=False):
    """
    ### delete_safe

    Safely deletes files with given extensions at given path.

    Parameters:
        dir_path (string): path to the directory
        extensions (list): list of extensions to be deleted
        logging (bool): enables logging for its operations

    """
    if not extensions or not os.path.exists(dir_path):
        logger.warning("Invalid input provided for deleting!")
        return

    if logging:
        logger.debug("Clearing Assets at `{}`!".format(dir_path))

    for ext in extensions:
        files_ext = [
            os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(ext)
        ]
        for file in files_ext:
            os.remove(file)
            if logging:
                logger.debug("Deleted file: `{}`".format(file))


def capPropId(property):
    """
    ### capPropId

    Retrieves the OpenCV property's Integer(Actual) value from string.

    Parameters:
        property (string): inputs OpenCV property as string.

    **Returns:** Resultant integer value.
    """
    integer_value = 0
    try:
        integer_value = getattr(cv2, property)
    except Exception as e:
        logger.exception(str(e))
        logger.critical("`{}` is not a valid OpenCV property!".format(property))
        return None
    return integer_value


def youtube_url_validator(url):
    """
    ### youtube_url_validator

    Validates & extracts Youtube video ID from URL.

    Parameters:
        url (string): inputs URL.

    **Returns:**  A valid Youtube video string ID.
    """
    youtube_regex = (
        r"(?:http:|https:)*?\/\/(?:www\.|)(?:youtube\.com|m\.youtube\.com|youtu\.|youtube-nocookie\.com).*"
        "(?:v=|v%3D|v\/|(?:a|p)\/(?:a|u)\/\d.*\/|watch\?|vi(?:=|\/)|\/embed\/|oembed\?|be\/|e\/)([^&?%#\/\n]*)"
    )
    matched = re.search(youtube_regex, url)
    # check for None-type
    if not (matched is None):
        return matched.groups()[0]
    else:
        return ""


def reducer(frame=None, percentage=0):
    """
    ### reducer

    Reduces frame size by given percentage

    Parameters:
        frame (numpy.ndarray): inputs numpy array(frame).
        percentage (int/float): inputs size-reduction percentage.

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

    # grab the frame size
    (height, width) = frame.shape[:2]

    # calculate the ratio of the width from percentage
    reduction = ((100 - percentage) / 100) * width
    ratio = reduction / float(width)
    # construct the dimensions
    dimensions = (int(reduction), int(height * ratio))

    # return the resized frame
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_LANCZOS4)


def dict2Args(param_dict):
    """
    ### dict2Args

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
    ### get_valid_ffmpeg_path

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

                if logging:
                    logger.debug(
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
                if logging:
                    logger.exception(str(e))
                    logger.debug(
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
            if logging:
                logger.debug("No valid FFmpeg executables found at Custom FFmpeg path!")
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
                if logging:
                    logger.debug(
                        "No valid FFmpeg executables found at Custom FFmpeg path!"
                    )
                return False
        else:
            # otherwise assign ffmpeg binaries from system
            final_path += "ffmpeg"

    if logging:
        logger.debug("Final FFmpeg Path: {}".format(final_path))

    # Final Auto-Validation for FFmeg Binaries. returns final path if test is passed
    return final_path if validate_ffmpeg(final_path, logging=logging) else False


def download_ffmpeg_binaries(path, os_windows=False, os_bit=""):
    """
    ### download_ffmpeg_binaries

    Generates FFmpeg Static Binaries for windows(if not available)

    Parameters:
        path (string): path for downloading custom FFmpeg executables
        os_windows (boolean): is running on Windows OS?
        os_bit (string): 32-bit or 64-bit OS?

    **Returns:** A valid FFmpeg executable path string.
    """
    final_path = ""
    if os_windows and os_bit:
        # initialize variables
        file_url = "https://ffmpeg.zeranoe.com/builds/{}/static/ffmpeg-latest-{}-static.zip".format(
            os_bit, os_bit
        )
        file_name = os.path.join(
            os.path.abspath(path), "ffmpeg-latest-{}-static.zip".format(os_bit)
        )
        file_path = os.path.join(
            os.path.abspath(path),
            "ffmpeg-latest-{}-static/bin/ffmpeg.exe".format(os_bit),
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
            if os.path.isfile(file_name):
                os.remove(file_name)
            # download and write file to the given path
            with open(file_name, "wb") as f:
                logger.debug(
                    "No Custom FFmpeg path provided. Auto-Installing FFmpeg static binaries now. Please wait..."
                )
                try:
                    response = requests.get(file_url, stream=True, timeout=2)
                    response.raise_for_status()
                except Exception as e:
                    logger.exception(str(e))
                    logger.warning("Downloading Failed. Trying GitHub mirror now!")
                    file_url = "https://raw.githubusercontent.com/abhiTronix/ffmpeg-static-builds/master/windows/ffmpeg-latest-{}-static.zip".format(
                        os_bit, os_bit
                    )
                    response = requests.get(file_url, stream=True, timeout=2)
                    response.raise_for_status()
                total_length = response.headers.get("content-length")
                assert not (
                    total_length is None
                ), "[Helper:ERROR] :: Failed to retrieve files, check your Internet connectivity!"
                bar = tqdm(total=int(total_length), unit="B", unit_scale=True)
                for data in response.iter_content(chunk_size=4096):
                    f.write(data)
                    if len(data) > 0:
                        bar.update(len(data))
                bar.close()
            logger.debug("Extracting executables.")
            with zipfile.ZipFile(file_name, "r") as zip_ref:
                zip_ref.extractall(base_path)
            # perform cleaning
            os.remove(file_name)
            logger.debug("FFmpeg binaries for Windows configured successfully!")
            final_path += file_path
    # return final path
    return final_path


def validate_ffmpeg(path, logging=False):
    """
    ### validate_ffmpeg

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
        if logging:
            # log if test are passed
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
    ### check_output

    Returns stdin output from subprocess module
    """
    # import libs
    import subprocess as sp

    # handle additional params
    retrieve_stderr = kwargs.pop("force_retrieve_stderr", False)

    # execute command in subprocess
    process = sp.Popen(
        stdout=sp.PIPE,
        stderr=sp.DEVNULL if not (retrieve_stderr) else sp.PIPE,
        *args,
        **kwargs
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
    ### generate_auth_certificates

    Auto-Generates, and Auto-validates CURVE ZMQ key-pairs for NetGear API's Secure Mode.

    Parameters:
        path (string): path for generating CURVE key-pairs
        overwrite (boolean): overwrite existing key-pairs or not?
        logging (bool): enables logging for its operations

    **Returns:** A valid CURVE key-pairs path as string.
    """
    # import necessary libs
    import shutil
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
                    os.remove(redundant_key)
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
                    os.remove(redundant_key)

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
    ### validate_auth_keys

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
    if len(keys_buffer) == 1:
        os.remove(os.path.join(path, keys_buffer[0]))

    # return results
    return True if (len(keys_buffer) == 2) else False
