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
import cv2
import time
import numpy as np
import pytest
import logging as log
import platform
import tempfile

from vidgear.gears import CamGear
from vidgear.gears.helper import logger_handler

# define test logger
logger = log.getLogger("Test_camgear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

# define machine os
_windows = True if os.name == "nt" else False


def return_youtubevideo_params(url):
    """
    returns Youtube Video parameters(FPS, dimensions) directly using Youtube-dl
    """
    import yt_dlp

    ydl = yt_dlp.YoutubeDL(
        {
            "outtmpl": "%(id)s%(ext)s",
            "noplaylist": True,
            "quiet": True,
            "format": "bestvideo",
        }
    )
    with ydl:
        result = ydl.extract_info(
            url, download=False
        )  # We just want to extract the info
    return (int(result["width"]), int(result["height"]), float(result["fps"]))


def return_testvideo_path():
    """
    returns Test Video path
    """
    path = "{}/Downloads/Test_videos/BigBuckBunny_4sec.mp4".format(
        tempfile.gettempdir()
    )
    return os.path.abspath(path)


def return_total_frame_count():
    """
    simply counts the total frames in a given video
    """
    stream = cv2.VideoCapture(return_testvideo_path())
    num_cv = 0
    while True:
        (grabbed, frame) = stream.read()
        if not grabbed:
            logger.debug("Total frames: {}".format(num_cv))
            break
        num_cv += 1
    stream.release()
    return num_cv


test_data = [
    (
        return_testvideo_path(),
        {
            "THREAD_TIMEOUT": 300,
            "CAP_PROP_FRAME_WIDTH ": 320,
            "CAP_PROP_FRAME_HEIGHT": 240,
        },
    ),
    (
        return_testvideo_path(),
        {"THREAD_TIMEOUT": "wrong", "im_wrong": True, "THREADED_QUEUE_MODE": False},
    ),
    ("im_not_a_source.mp4", {"THREADED_QUEUE_MODE": "invalid"}),
]


@pytest.mark.parametrize("source, options", test_data)
def test_threaded_queue_mode(source, options):
    """
    Test for the Thread Queue Mode in CamGear API
    """
    try:
        if platform.system() == "Linux":
            stream_camgear = CamGear(
                source=source, backend=cv2.CAP_FFMPEG, logging=True, **options
            ).start()
        else:
            stream_camgear = CamGear(source=source, logging=True, **options).start()
        camgear_frames_num = 0
        while True:
            frame = stream_camgear.read()
            if frame is None:
                logger.debug("VidGear Total frames: {}".format(camgear_frames_num))
                break

            time.sleep(0.2)  # dummy computational task

            camgear_frames_num += 1
        stream_camgear.stop()
        actual_frame_num = return_total_frame_count()
        if "THREADED_QUEUE_MODE" in options and not options["THREADED_QUEUE_MODE"]:
            # emulate frame skipping
            assert camgear_frames_num < actual_frame_num
        else:
            assert camgear_frames_num == actual_frame_num
    except Exception as e:
        if isinstance(e, RuntimeError) and source == "im_not_a_source.mp4":
            pass
        else:
            pytest.fail(str(e))


@pytest.mark.parametrize(
    "url, quality, parameters",
    [
        ("https://youtu.be/uCy5OuSQnyA", "73p", "invalid"),
        ("https://www.dailymotion.com/video/x7xsoud", "73p", "invalid"),
        ("https://youtu.be/uCy5OuSQnyA", "720p", "invalid"),
        ("https://youtu.be/NMre6IAAAiU", "invalid", {"nocheckcertificate": True}),
        (
            "https://www.dailymotion.com/video/x7xsoud",
            "invalid",
            {"hls-live-edge": 3.0},
        ),
        ("im_not_a_url", "", {}),
    ],
)
def test_stream_mode(url, quality, parameters):
    """
    Testing Stream Mode Playback capabilities of CamGear
    """
    try:
        height = 0
        width = 0
        fps = 0
        options = {"STREAM_RESOLUTION": quality, "STREAM_PARAMS": parameters}
        # get params
        stream = CamGear(
            source=url, stream_mode=True, logging=True, **options
        ).start()  # YouTube Video URL as input
        while True:
            frame = stream.read()
            if frame is None:
                break
            if height == 0 or width == 0:
                fps = stream.framerate
                height, width = frame.shape[:2]
                break
        stream.stop()
        logger.debug("WIDTH: {} HEIGHT: {} FPS: {}".format(width, height, fps))
    except Exception as e:
        # if isinstance(e, (RuntimeError, ValueError, cv2.error)) and (
        #    url == "im_not_a_url" or platform.system() in ["Windows", "Darwin"]
        # ):
        pytest.xfail(str(e))
        # else:
        #    pytest.fail(str(e))


def test_network_playback():
    """
    Testing Direct Network Video Playback capabilities of VidGear(with rtsp streaming)
    """
    Publictest_rstp_urls = [
        "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov",
        "rtsp://freja.hiof.no:1935/rtplive/definst/hessdalen03.stream",
        "rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa",
        "rtmp://semerkandglb.mediatriple.net:1935/semerkandliveedge/semerkand2",
    ]

    index = 0

    while index < len(Publictest_rstp_urls):
        try:
            output_stream = CamGear(
                source=Publictest_rstp_urls[index], logging=True
            ).start()
            i = 0
            Output_data = []
            while i < 10:
                frame = output_stream.read()
                if frame is None:
                    break
                Output_data.append(frame)
                i += 1
            output_stream.stop()
            logger.debug("Output data shape:", np.array(Output_data).shape)
            if Output_data[-1].shape[:2] > (50, 50):
                break
        except Exception as e:
            if isinstance(e, RuntimeError):
                logger.debug(
                    "`{}` URL is not working".format(Publictest_rstp_urls[index])
                )
                index += 1
                continue
            else:
                pytest.fail(str(e))

    if index == len(Publictest_rstp_urls):
        pytest.fail("Test failed to play any URL!")
