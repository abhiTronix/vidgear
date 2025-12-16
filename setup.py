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
import platform
import urllib.request
from packaging.version import parse
from setuptools import setup
import re


def latest_version(package_name):
    """
    Get latest package version from pypi (Hack)
    """
    url = "https://pypi.python.org/pypi/%s/json" % (package_name,)
    versions = []
    try:
        response = urllib.request.urlopen(
            urllib.request.Request(url),
            timeout=1,
        )
        data = json.load(response)
        versions = list(data["releases"].keys())
        versions.sort(key=parse)
        return ">={}".format(versions[-1])
    except Exception as e:
        if versions:
            return ">={}".format(versions[-1])
        else:
            print(str(e))
    return ""


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    long_description = long_description.replace(
        "(#", "(https://github.com/abhiTronix/vidgear#"
    )
    # Add patch to remove sponsor block
    long_description = re.sub(
        r"<!-- SPONSORS START -->.*?<!-- SPONSORS END -->",
        "",
        long_description,
        flags=re.DOTALL,
    )
    # patch for unicodes
    long_description = long_description.replace("➶", ">>")
    long_description = long_description.replace("©", "(c)")

setup(
    name="vidgear",
    packages=["vidgear", "vidgear.gears", "vidgear.gears.asyncio"],
    package_data={"vidgear": ["py.typed"]},
    version="0.3.4",
    description="High-performance cross-platform Video Processing Python framework powerpacked with unique trailblazing features.",
    license="Apache License 2.0",
    author="Abhishek Thakur",
    install_requires=[
        "cython",  # helper for numpy install
        "numpy",
        "requests",
        "colorlog",
        "tqdm",
        "packaging",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email="abhi.una12@gmail.com",
    url="https://abhitronix.github.io/vidgear",
    extras_require={
        # API specific deps
        "core": [
            "yt-dlp[default]{}".format(latest_version("yt-dlp")),
            "pyzmq{}".format(latest_version("pyzmq")),
            "Pillow",
            "simplejpeg{}".format(latest_version("simplejpeg")),
            "mss{}".format(latest_version("mss")),
            "pyscreenshot{}".format(latest_version("pyscreenshot")),
        ]
        + (
            ["dxcam{}".format(latest_version("dxcam"))]
            if (platform.system() == "Windows")  # windows is only supported
            else []
        ),
        # API specific + Asyncio deps
        "asyncio": [
            "yt-dlp[default]{}".format(latest_version("yt-dlp")),
            "pyzmq{}".format(latest_version("pyzmq")),
            "simplejpeg{}".format(latest_version("simplejpeg")),
            "mss{}".format(latest_version("mss")),
            "Pillow",
            "pyscreenshot{}".format(latest_version("pyscreenshot")),
            "starlette{}".format(latest_version("starlette")),
            "jinja2",
            "msgpack{}".format(latest_version("msgpack")),
            "msgpack_numpy{}".format(latest_version("msgpack_numpy")),
            "aiortc{}".format(latest_version("aiortc")),
            "uvicorn{}".format(latest_version("uvicorn")),
        ]
        + (
            ["dxcam{}".format(latest_version("dxcam"))]
            if (platform.system() == "Windows")  # windows is only supported
            else []
        )
        + (
            ["uvloop{}".format(latest_version("uvloop"))]
            if (platform.system() != "Windows")  # windows not supported
            else []
        ),
    },
    keywords=[
        "OpenCV",
        "multithreading",
        "FFmpeg",
        "picamera",
        "starlette",
        "mss",
        "pyzmq",
        "dxcam",
        "aiortc",
        "uvicorn",
        "uvloop",
        "yt-dlp",
        "asyncio",
        "dash",
        "hls",
        "Video Processing",
        "Video Stabilization",
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
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.10",
    scripts=[],
    project_urls={
        "Bug Reports": "https://github.com/abhiTronix/vidgear/issues",
        "Funding": "https://ko-fi.com/W7W8WTYO",
        "Source": "https://github.com/abhiTronix/vidgear",
        "Documentation": "https://abhitronix.github.io/vidgear",
        "Changelog": "https://abhitronix.github.io/vidgear/latest/changelog/",
    },
)
