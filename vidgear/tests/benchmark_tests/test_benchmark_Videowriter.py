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

import os, platform
import pytest
import tempfile
from vidgear.gears import WriteGear
from vidgear.gears import VideoGear
from vidgear.gears.helper import logger_handler
from .fps import FPS
import logging as log

logger = log.getLogger("Test_benchmark_videowriter")
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


def return_testvideo_path():
    """
	returns Test Video path
	"""
    path = "{}/Downloads/Test_videos/BigBuckBunny_4sec.mp4".format(
        tempfile.gettempdir()
    )
    return os.path.abspath(path)


def return_static_ffmpeg():
    """
	returns system specific FFmpeg static path
	"""
    path = ""
    if platform.system() == "Windows":
        path += os.path.join(
            tempfile.gettempdir(), "Downloads/FFmpeg_static/ffmpeg/bin/ffmpeg.exe"
        )
    elif platform.system() == "Darwin":
        path += os.path.join(
            tempfile.gettempdir(), "Downloads/FFmpeg_static/ffmpeg/bin/ffmpeg"
        )
    else:
        path += os.path.join(
            tempfile.gettempdir(), "Downloads/FFmpeg_static/ffmpeg/ffmpeg"
        )
    return os.path.abspath(path)


def WriteGear_non_compression_mode():
    """
	Function to Benchmark WriteGear's Non-Compression Mode(OpenCV)
	"""
    stream = VideoGear(source=return_testvideo_path()).start()
    writer = WriteGear(output_filename="Output_vnc.mp4", compression_mode=False)
    fps_CV = FPS().start()
    while True:
        frame = stream.read()
        if frame is None:
            break
        writer.write(frame)
        fps_CV.update()
    stream.stop()
    writer.close()
    logger.debug("OpenCV Writer")
    logger.info("approx. FPS: {:.2f}".format(fps_CV.average_fps()))
    os.remove(os.path.abspath("Output_vnc.mp4"))


def WriteGear_compression_mode():
    """
	Function to Benchmark WriteGear's Compression Mode(FFmpeg)
	"""
    stream = VideoGear(source=return_testvideo_path()).start()
    writer = WriteGear(
        output_filename="Output_vc.mp4", custom_ffmpeg=return_static_ffmpeg()
    )
    fps_Vid = FPS().start()
    while True:
        frame = stream.read()
        if frame is None:
            break
        writer.write(frame)
        fps_Vid.update()
    stream.stop()
    writer.close()
    logger.debug("FFmpeg Writer")
    logger.info("approx. FPS: {:.2f}".format(fps_Vid.average_fps()))
    os.remove(os.path.abspath("Output_vc.mp4"))


def test_benchmark_videowriter():
    """
	Benchmarking WriteGear's optimized Compression Mode(FFmpeg) against Non-Compression Mode(OpenCV)
	"""
    if platform.system() != "Darwin":
        try:
            WriteGear_non_compression_mode()
            WriteGear_compression_mode()
        except Exception as e:
            pytest.fail(str(e))
    else:
        logger.debug("Skipping this test for macOS!")
