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

# Contains all the support functions/modules required by Vidgear Asyncio packages

# import the necessary packages
import os
import cv2
import requests
import numpy as np
import logging as log
from tqdm import tqdm
from colorlog import ColoredFormatter
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# import helper packages
from ..helper import logger_handler, mkdir_safe

# define logger
logger = log.getLogger("Helper_Async")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


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
        if logging:
            logger.debug("Adding text: {}".format(text))
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


async def reducer(frame=None, percentage=0, interpolation=cv2.INTER_LANCZOS4):
    """
    ## reducer

    Asynchronous method that reduces frame size by given percentage.

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


def generate_webdata(path, c_name="webgear", overwrite_default=False, logging=False):
    """
    ## generate_webdata

    Auto-Generates, and Auto-validates default data for WebGear and WebGear_RTC APIs.

    Parameters:
        path (string): path for generating data
        c_name (string): class name that is generating files
        overwrite_default (boolean): overwrite existing data or not?
        logging (bool): enables logging for its operations

    **Returns:** A valid data path as string.
    """
    # check if path corresponds to vidgear only
    if os.path.basename(path) != ".vidgear":
        path = os.path.join(path, ".vidgear")

    # generate parent directory
    path = os.path.join(path, c_name)
    mkdir_safe(path, logging=logging)

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

    # check if overwriting is enabled
    if overwrite_default or not validate_webdata(
        template_dir, ["index.html", "404.html", "500.html"]
    ):
        logger.critical(
            "Overwriting existing {} data-files with default data-files from the server!".format(
                c_name.capitalize()
            )
            if overwrite_default
            else "Failed to detect critical {} data-files: index.html, 404.html & 500.html!".format(
                c_name.capitalize()
            )
        )
        # download default files
        logging and logger.info(
            "Downloading default data-files from the Gitlab Server: {}".format(
                "https://gitlab.com/abhiTronix/vidgear-vitals"
            )
        )
        download_webdata(
            template_dir,
            c_name=c_name,
            files=["index.html", "404.html", "500.html", "base.html"],
            logging=logging,
        )
        download_webdata(
            css_static_dir, c_name=c_name, files=["custom.css"], logging=logging
        )
        download_webdata(
            js_static_dir,
            c_name=c_name,
            files=["custom.js"],
            logging=logging,
        )
        download_webdata(
            favicon_dir, c_name=c_name, files=["favicon-32x32.png"], logging=logging
        )
    else:
        # validate important data-files
        if logging:
            logger.debug("Found valid WebGear data-files successfully.")

    return path


def download_webdata(path, c_name="webgear", files=[], logging=False):
    """
    ## download_webdata

    Downloads given list of files for WebGear and WebGear_RTC APIs(if not available) from GitHub/Gitlab Servers,
    and also Validates them.

    Parameters:
        path (string): path for downloading data
        c_name (string): class name that is generating files
        files (list): list of files to be downloaded
        logging (bool): enables logging for its operations

    **Returns:** A valid path as string.
    """
    basename = os.path.basename(path)
    if logging:
        logger.debug("Downloading {} data-files at `{}`".format(basename, path))

    # list all registered urls
    reg_urls = [
        "https://gitlab.com/abhiTronix/vidgear-vitals/-/raw/main",
        "https://raw.githubusercontent.com/abhiTronix/vidgear-vitals/main",
    ]

    # create session
    with requests.Session() as http:
        for url in reg_urls:
            try:
                for file in files:
                    # get filename
                    file_name = os.path.join(path, file)
                    # get URL
                    file_url = "{}/{}{}/{}/{}".format(
                        url,
                        c_name,
                        "/static" if basename != "templates" else "",
                        basename,
                        file,
                    )
                    # download and write file to the given path
                    logging and logger.debug(
                        "Downloading {} data-file: {}.".format(basename, file)
                    )

                    with open(file_name, "wb") as f:
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
                        for data in response.iter_content(chunk_size=256):
                            f.write(data)
                            if len(data) > 0:
                                bar.update(len(data))
                        bar.close()
            except AssertionError as e:
                # raise if connection error
                raise e
            except Exception as e:
                # log error
                logger.exception(str(e))
                # log event if necessary
                url != reg_urls[1] and logger.error(
                    "Download failed for Gitlab Server! Retrying from GitHub Server: {}".format(
                        url, "https://github.com/abhiTronix/vidgear-vitals"
                    )
                )
            else:
                # break otherwise
                break

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
    ## validate_auth_keys

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
