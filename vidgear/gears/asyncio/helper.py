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

# Contains all the support functions/modules required by Vidgear Asyncio packages

# import the necessary packages

import os
import cv2
import sys
import errno
import numpy as np
import aiohttp
import asyncio
import logging as log
import platform
import requests
from tqdm import tqdm
from colorlog import ColoredFormatter
from pkg_resources import parse_version


def logger_handler():
    """
    ### logger_handler

    Returns a color formatted logger handler

    **Returns:** A asyncio package logger handler
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
logger = log.getLogger("Helper Asyncio")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


def mkdir_safe(dir, logging=False):
    """
    ### mkdir_safe

    Safely creates directory at given path.

    Parameters:
        logging (bool): enables logging for its operations

    """
    try:
        os.makedirs(dir)
        if logging:
            logger.debug("Created directory at `{}`".format(dir))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        if logging:
            logger.debug("Directory already exists at `{}`".format(dir))


async def reducer(frame=None, percentage=0):
    """
    ### reducer

    Asynchronous method that reduces frame size by given percentage.

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


def generate_webdata(path, overwrite_default=False, logging=False):
    """
    ### generate_webdata

    Auto-Generates, and Auto-validates default data for WebGear API.

    Parameters:
        path (string): path for generating data
        overwrite_default (boolean): overwrite existing data or not?
        logging (bool): enables logging for its operations

    **Returns:** A valid data path as string.
    """
    # check if path corresponds to vidgear only
    if os.path.basename(path) != ".vidgear":
        path = os.path.join(path, ".vidgear")

    # self-generate dirs
    template_dir = os.path.join(path, "templates")  # generates HTML templates dir
    static_dir = os.path.join(path, "static")  # generates static dir
    # generate js & css static and favicon img subdirs
    js_static_dir = os.path.join(static_dir, "js")
    css_static_dir = os.path.join(static_dir, "css")
    favicon_dir = os.path.join(static_dir, "img")

    mkdir_safe(static_dir, logging=logging)
    mkdir_safe(template_dir, logging=logging)
    mkdir_safe(js_static_dir, logging=logging)
    mkdir_safe(css_static_dir, logging=logging)
    mkdir_safe(favicon_dir, logging=logging)

    if len(logger.handlers) > 1:
        logger.handlers.clear()
        logger.addHandler(logger_handler())
        logger.setLevel(log.DEBUG)

    # check if overwriting is enabled
    if overwrite_default:
        logger.critical(
            "Overwriting existing WebGear data-files with default data-files from the server!"
        )
        download_webdata(
            template_dir,
            files=["index.html", "404.html", "500.html", "base.html"],
            logging=logging,
        )
        download_webdata(
            css_static_dir, files=["bootstrap.min.css", "cover.css"], logging=logging
        )
        download_webdata(
            js_static_dir,
            files=["bootstrap.min.js", "jquery-3.4.1.slim.min.js", "popper.min.js"],
            logging=logging,
        )
        download_webdata(favicon_dir, files=["favicon-32x32.png"], logging=logging)
    else:
        # validate important data-files
        if validate_webdata(template_dir, ["index.html", "404.html", "500.html"]):
            if logging:
                logger.debug("Found valid WebGear data-files successfully.")
        else:
            # otherwise download default files
            logger.critical(
                "Failed to detect critical WebGear data-files: index.html, 404.html & 500.html!"
            )
            logger.warning("Re-downloading default data-files from the server.")
            download_webdata(
                template_dir,
                files=["index.html", "404.html", "500.html", "base.html"],
                logging=logging,
            )
            download_webdata(
                css_static_dir,
                files=["bootstrap.min.css", "cover.css"],
                logging=logging,
            )
            download_webdata(
                js_static_dir,
                files=["bootstrap.min.js", "jquery-3.4.1.slim.min.js", "popper.min.js"],
                logging=logging,
            )
            download_webdata(favicon_dir, files=["favicon-32x32.png"], logging=logging)
    return path


def download_webdata(path, files=[], logging=False):
    """
    ### download_webdata

    Downloads given list of files for WebGear API(if not available) from GitHub Server,
    and also Validates them.

    Parameters:
        path (string): path for downloading data
        files (list): list of files to be downloaded
        logging (bool): enables logging for its operations

    **Returns:** A valid path as string.
    """
    basename = os.path.basename(path)
    if logging:
        logger.debug("Downloading {} data-files at `{}`".format(basename, path))
    for file in files:
        # get filename
        file_name = os.path.join(path, file)
        # get URL
        if basename == "templates":
            file_url = "https://raw.githubusercontent.com/abhiTronix/webgear_data/master/{}/{}".format(
                basename, file
            )
        else:
            file_url = "https://raw.githubusercontent.com/abhiTronix/webgear_data/master/static/{}/{}".format(
                basename, file
            )
        # download and write file to the given path
        if logging:
            logger.debug("Downloading {} data-file: {}.".format(basename, file))

        response = requests.get(file_url, stream=True, timeout=2)
        response.raise_for_status()
        total_length = response.headers.get("content-length")
        assert not (
            total_length is None
        ), "[Helper:ERROR] :: Failed to retrieve files, check your Internet connectivity!"
        bar = tqdm(total=int(total_length), unit="B", unit_scale=True)
        with open(file_name, "wb") as f:
            for data in response.iter_content(chunk_size=256):
                f.write(data)
                if len(data) > 0:
                    bar.update(len(data))
        bar.close()
    if logging:
        logger.debug("Verifying downloaded data:")
    if validate_webdata(path, files=files, logging=logging):
        if logging:
            logger.info("Successful!")
        return path
    else:
        raise RuntimeError(
            "[Helper:ERROR] :: Failed to download required {} data-files at: {}, Check your Internet connectivity!".format(
                basename, path
            )
        )


def validate_webdata(path, files=[], logging=False):
    """
    ### validate_auth_keys

    Validates, and also maintains downloaded list of files.

    Parameters:
        path (string): path of downloaded files
        files (list): list of files to be validated
        logging (bool): enables logging for its operations

    **Returns:** A  boolean value, confirming whether tests passed, or not?.
    """
    # check if valid path or directory empty
    if not (os.path.exists(path)) or not (os.listdir(path)):
        return False

    files_buffer = []
    # loop over files
    for file in os.listdir(path):
        if file in files:
            files_buffer.append(file)  # store them

    # return results
    if len(files_buffer) < len(files):
        if logging:
            logger.warning(
                "`{}` file(s) missing from data-files!".format(
                    " ,".join(list(set(files_buffer) ^ set(files)))
                )
            )
        return False
    else:
        return True
