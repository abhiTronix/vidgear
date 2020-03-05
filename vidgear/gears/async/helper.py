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

# Contains all the support functions/modules required by Vidgear

# import the necessary packages
import os, sys, requests, platform, errno
import numpy as np
from pkg_resources import parse_version
from colorlog import ColoredFormatter
import progressbar
import logging as log
import asyncio
import aiohttp

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
	returns logger handler
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
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


def mkdir_safe(dir, logging=False):
    """
	Simply creates directory safely
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


def generate_auth_certificates(path, overwrite=False):

    """ 
	auto-Generates and auto-validates CURVE ZMQ keys/certificates for Netgear 
	"""

    # import necessary libs
    import shutil
    import zmq.auth

    # check if path corresponds to vidgear only
    if os.path.basename(path) != ".vidgear":
        path = os.path.join(path, ".vidgear")

    # generate keys dir
    keys_dir = os.path.join(path, "keys")
    mkdir_safe(keys_dir)

    # generate separate public and private key dirs
    public_keys_dir = os.path.join(keys_dir, "public_keys")
    secret_keys_dir = os.path.join(keys_dir, "private_keys")

    # check if overwriting is allowed
    if overwrite:
        # delete previous certificates
        for d in [public_keys_dir, secret_keys_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.mkdir(d)

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
            mkdir_safe(public_keys_dir)

        # check if valid private keys are found
        if not (status_private_keys):
            mkdir_safe(secret_keys_dir)

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
	validates and maintains ZMQ Auth Keys/Certificates
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


async def reducer(frame=None, percentage=0):

    """
	Reduces frame size by given percentage
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


async def generate_webdata(path, overwrite_default=False, logging=False):
    """ 
	handles WebGear API data-files validation and generation 
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

    mkdir_safe(static_dir)
    mkdir_safe(template_dir)
    mkdir_safe(js_static_dir)
    mkdir_safe(css_static_dir)
    mkdir_safe(favicon_dir)

    if len(logger.handlers) > 1:
        logger.handlers.clear()
        logger.addHandler(logger_handler())
        logger.setLevel(log.DEBUG)

    # check if overwriting is enabled
    if overwrite_default:
        logger.critical(
            "Overwriting existing WebGear data-files with default data-files from the server!"
        )
        await download_webdata(
            template_dir,
            files=["index.html", "404.html", "500.html", "base.html"],
            logging=logging,
        )
        await download_webdata(
            css_static_dir, files=["bootstrap.min.css", "cover.css"], logging=logging
        )
        await download_webdata(
            js_static_dir,
            files=["bootstrap.min.js", "jquery-3.4.1.slim.min.js", "popper.min.js"],
            logging=logging,
        )
        await download_webdata(
            favicon_dir, files=["favicon-32x32.png"], logging=logging
        )
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
            await download_webdata(
                template_dir,
                files=["index.html", "404.html", "500.html", "base.html"],
                logging=logging,
            )
            await download_webdata(
                css_static_dir,
                files=["bootstrap.min.css", "cover.css"],
                logging=logging,
            )
            await download_webdata(
                js_static_dir,
                files=["bootstrap.min.js", "jquery-3.4.1.slim.min.js", "popper.min.js"],
                logging=logging,
            )
            await download_webdata(
                favicon_dir, files=["favicon-32x32.png"], logging=logging
            )
    return path


async def download_webdata(path, files=[], logging=False):
    """
	Downloads default WebGear data-files from the server
	"""
    basename = os.path.basename(path)
    if logging:
        logger.debug("Downloading {} data-files at `{}`".format(basename, path))

    # collect data
    targets = []
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
        targets.append((file_url, file_name))

    # download data asynchronously
    async with aiohttp.ClientSession(raise_for_status=True, read_timeout=4) as session:
        tasks = [
            download(session, file_path, url, logging=logging)
            for (url, file_path) in targets
        ]
        await asyncio.gather(*tasks)

    # verify downloaded data
    if logging:
        logger.debug("Verifying downloaded data at `{}`:".format(path))
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


async def download(session, target, url, logging=False):
    """
	Downloads data asynchronously
	"""
    async with session.get(url) as response:
        # download and write file to the given path
        if logging:
            logger.debug("Downloading  data-file: {}.".format(target))
        total_length = int(response.headers.get("content-length", 0)) or None
        assert not (
            total_length is None and total_length > 0
        ), "[Helper:ERROR] :: Failed to retrieve files, check your Internet connectivity!"
        bar = progressbar.ProgressBar(max_value=total_length)
        with open(target, mode="wb") as f:
            async for chunk in response.content.iter_chunked(512):
                f.write(chunk)
                if len(chunk) > 0:
                    bar.update(len(chunk))
        bar.finish()


def validate_webdata(path, files=[], logging=False):
    """
	validates WebGear API data-files
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