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
import json
import sys
import platform
import setuptools
import urllib.request

from distutils.version import LooseVersion
from distutils.util import convert_path
from setuptools import setup


def test_opencv():
    """
    This function is workaround to
    test if correct OpenCV Library version has already been installed
    on the machine or not. Returns True if previously not installed.
    """
    try:
        # import OpenCV Binaries
        import cv2

        # check whether OpenCV Binaries are 3.x+
        if LooseVersion(cv2.__version__) < LooseVersion("3"):
            raise ImportError(
                "Incompatible (< 3.0) OpenCV version-{} Installation found on this machine!".format(
                    LooseVersion(cv2.__version__)
                )
            )
    except ImportError:
        return True
    return False


def latest_version(package_name):
    """
    Get latest package version from pypi (Hack)
    """
    url = "https://pypi.python.org/pypi/%s/json" % (package_name,)
    versions = []
    try:
        response = urllib.request.urlopen(urllib.request.Request(url), timeout=1)
        data = json.load(response)
        versions = list(data["releases"].keys())
        versions.sort(key=LooseVersion)
        return ">={}".format(versions[-1])
    except Exception as e:
        if versions and isinstance(e, TypeError):
            return ">={}".format(versions[-1])
    return ""


pkg_version = {}
ver_path = convert_path("vidgear/version.py")
with open(ver_path) as ver_file:
    exec(ver_file.read(), pkg_version)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    long_description = long_description.replace(  # patch for images
        "docs/overrides/assets", "https://abhitronix.github.io/vidgear/latest/assets"
    )
    long_description = long_description.replace(
        "(#", "(https://github.com/abhiTronix/vidgear#"
    )
    # patch for unicodes
    long_description = long_description.replace("➶", ">>")
    long_description = long_description.replace("©", "(c)")

setup(
    name="vidgear",
    packages=["vidgear", "vidgear.gears", "vidgear.gears.asyncio"],
    version=pkg_version["__version__"],
    description="High-performance cross-platform Video Processing Python framework powerpacked with unique trailblazing features.",
    license="Apache License 2.0",
    author="Abhishek Thakur",
    install_requires=[
        "pafy{}".format(latest_version("pafy")),
        "yt_dlp{}".format(latest_version("yt_dlp")),  # pafy backend
        "mss{}".format(latest_version("mss")),
        "cython",  # helper for numpy install
        "numpy",
        "streamlink",
        "requests",
        "pyzmq{}".format(latest_version("pyzmq")),
        "simplejpeg{}".format(latest_version("simplejpeg")),
        "colorlog",
        "tqdm",
        "Pillow",
        "pyscreenshot{}".format(latest_version("pyscreenshot")),
    ]
    + (["opencv-python"] if test_opencv() else [])
    + (["picamera"] if ("arm" in platform.uname()[4][:3]) else []),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email="abhi.una12@gmail.com",
    url="https://abhitronix.github.io/vidgear",
    extras_require={
        "asyncio": [
            "starlette{}".format(latest_version("starlette")),
            "jinja2",
            "uvicorn{}".format(latest_version("uvicorn")),
            "msgpack{}".format(latest_version("msgpack")),
            "msgpack_numpy{}".format(latest_version("msgpack_numpy")),
            "aiortc{}".format(latest_version("aiortc")),
        ]
        + (
            (
                ["uvloop{}".format(latest_version("uvloop"))]
                if sys.version_info[:2] >= (3, 7)  # dropped support for 3.6.x legacies
                else ["uvloop==0.14.0"]
            )
            if (platform.system() != "Windows")
            else []
        )
    },
    keywords=[
        "OpenCV",
        "multithreading",
        "FFmpeg",
        "picamera",
        "starlette",
        "mss",
        "pyzmq",
        "aiortc",
        "uvicorn",
        "uvloop",
        "pafy",
        "yt-dlp",
        "asyncio",
        "dash",
        "streamlink",
        "Video Processing",
        "Video Stablization",
        "Computer Vision",
        "Video Streaming",
        "raspberrypi",
        "YouTube",
        "Twitch",
        "WebRTC",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Operating System :: POSIX",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.6",
    scripts=[],
    project_urls={
        "Bug Reports": "https://github.com/abhiTronix/vidgear/issues",
        "Funding": "https://ko-fi.com/W7W8WTYO",
        "Source": "https://github.com/abhiTronix/vidgear",
        "Documentation": "https://abhitronix.github.io/vidgear",
        "Changelog": "https://abhitronix.github.io/vidgear/latest/changelog/",
    },
)
