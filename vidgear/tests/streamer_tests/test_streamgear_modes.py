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
# import libraries
import logging as log
import os
import platform
import subprocess
import tempfile
import cv2
import pytest

from mpegdash.parser import MPEGDASHParser
from vidgear.gears import CamGear, StreamGear
from vidgear.gears.helper import logger_handler

# define test logger
logger = log.getLogger("Test_Streamgear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


def return_testvideo_path(fmt="av"):
    """
    returns Test video path
    """
    supported_fmts = {
        "av": "BigBuckBunny_4sec.mp4",
        "vo": "BigBuckBunny_4sec_VO.mp4",
        "ao": "BigBuckBunny_4sec_AO.mp4",
    }
    req_fmt = fmt if (fmt in supported_fmts) else "av"
    path = "{}/Downloads/Test_videos/{}".format(
        tempfile.gettempdir(), supported_fmts[req_fmt]
    )
    return os.path.abspath(path)


def check_valid_mpd(file="", exp_reps=1):
    """
    checks if given file is a valid MPD(MPEG-DASH Manifest file)
    """
    if not file or not os.path.isfile(file):
        return False
    try:
        mpd = MPEGDASHParser.parse(file)
        all_reprs = []
        for period in mpd.periods:
            for adapt_set in period.adaptation_sets:
                for rep in adapt_set.representations:
                    all_reprs.append(rep)
    except Exception as e:
        logger.error(str(e))
        return False
    return all_reprs if (len(all_reprs) >= exp_reps) else False


def extract_meta_mpd(file):
    """
    Extracts metadata from a valid MPD(MPEG-DASH Manifest file)
    """
    reprs = check_valid_mpd(file)
    if reprs:
        metas = []
        for rep in reprs:
            meta = {}
            meta["mime_type"] = rep.mime_type
            if meta["mime_type"].startswith("audio"):
                meta["audioSamplingRate"] = rep.audio_sampling_rate
            else:
                meta["width"] = rep.width
                meta["height"] = rep.height
                meta["framerate"] = rep.frame_rate
            metas.append(meta)
        return metas
    else:
        return []


def return_mpd_path():
    """
    returns MPD assets temp path
    """
    return os.path.join(tempfile.gettempdir(), "temp_mpd")


def string_to_float(value):
    """
    Converts fraction to float
    """
    extracted = value.strip().split("/")
    cleaned = [float(x.strip()) for x in extracted]
    return cleaned[0] / cleaned[1]


def extract_resolutions(source, streams):
    """
    Extracts resolution value from dictionaries
    """
    if not (source) or not (streams):
        return []
    results = []
    assert os.path.isfile(source), "Not a valid source"
    s_cv = cv2.VideoCapture(source)
    results.append({"width": s_cv.get(cv2.CAP_PROP_FRAME_WIDTH), "height": s_cv.get(cv2.CAP_PROP_FRAME_HEIGHT)})
    for stream in streams:
        if "-resolution" in stream:
            try:
                res = stream["-resolution"].split("x")
                assert len(res) == 2
                width, height = res[0].strip(), res[1].strip()
                assert width.isnumeric() and height.isnumeric()
                results.append({"width": width, "height": height})
            except Exception as e:
                logger.error(str(e))
                continue
        else:
            continue
    return results


def test_ss_stream():
    """
    Testing Single-Source Mode
    """
    mpd_file_path = os.path.join(return_mpd_path(), "dash_test.mpd")
    try:
        stream_params = {
            "-video_source": return_testvideo_path(),
            "-clear_prev_assets": True,
        }
        streamer = StreamGear(output=mpd_file_path, logging=True, **stream_params)
        streamer.transcode_source()
        streamer.terminate()
        assert check_valid_mpd(mpd_file_path)
    except Exception as e:
        pytest.fail(str(e))


@pytest.mark.parametrize("conversion", [None, "COLOR_BGR2GRAY", "COLOR_BGR2BGRA"])
def test_rtf_stream(conversion):
    """
    Testing Real-Time Frames Mode
    """
    mpd_file_path = return_mpd_path()
    try:
        # Open stream
        stream = CamGear(source=return_testvideo_path(), colorspace=conversion).start()
        stream_params = {
            "-clear_prev_assets": True,
            "-input_framerate": "invalid",
        }
        streamer = StreamGear(output=mpd_file_path, **stream_params)
        while True:
            frame = stream.read()
            # check if frame is None
            if frame is None:
                break
            if conversion == "COLOR_BGR2RGBA":
                streamer.stream(frame, rgb_mode=True)
            else:
                streamer.stream(frame)
        stream.stop()
        streamer.terminate()
        mpd_file = [
            os.path.join(mpd_file_path, f)
            for f in os.listdir(mpd_file_path)
            if f.endswith(".mpd")
        ]
        assert len(mpd_file) == 1, "Failed to create MPD file!"
        assert check_valid_mpd(mpd_file[0])
    except Exception as e:
        pytest.fail(str(e))


def test_input_framerate_rtf():
    """
    Testing "-input_framerate" parameter provided by StreamGear
    """
    try:
        mpd_file_path = os.path.join(return_mpd_path(), "dash_test.mpd")
        stream = cv2.VideoCapture(return_testvideo_path())  # Open stream
        test_video_framerate = stream.get(cv2.CAP_PROP_FPS)
        stream_params = {
            "-clear_prev_assets": True,
            "-input_framerate": test_video_framerate,
        }
        streamer = StreamGear(output=mpd_file_path, logging=True, **stream_params)
        while True:
            (grabbed, frame) = stream.read()
            if not grabbed:
                break
            streamer.stream(frame)
        stream.release()
        streamer.terminate()
        meta_data = extract_meta_mpd(mpd_file_path)
        assert meta_data, "Test Failed!"
        meta_vids = [x for x in meta_data if x["mime_type"].startswith("video")]
        assert round(string_to_float(meta_vids[0]["framerate"])) == round(
            test_video_framerate
        ), "Test Failed!"
    except Exception as e:
        pytest.fail(str(e))


@pytest.mark.parametrize(
    "stream_params",
    [
        {"-clear_prev_assets": True, "-bpp": 0.2000, "-gop": 125, "-vcodec": "libx265"},
        {
            "-clear_prev_assets": True,
            "-bpp": "unknown",
            "-gop": "unknown",
            "-s:v:0": "unknown",
            "-b:v:0": "unknown",
            "-b:a:0": "unknown",
        },
    ],
)
def test_params(stream_params):
    """
    Testing "-input_framerate" parameter provided by StreamGear
    """
    try:
        mpd_file_path = os.path.join(return_mpd_path(), "dash_test.mpd")
        stream = cv2.VideoCapture(return_testvideo_path())  # Open stream
        streamer = StreamGear(output=mpd_file_path, logging=True, **stream_params)
        while True:
            (grabbed, frame) = stream.read()
            if not grabbed:
                break
            streamer.stream(frame)
        stream.release()
        streamer.terminate()
        assert check_valid_mpd(mpd_file_path)
    except Exception as e:
        pytest.fail(str(e))


@pytest.mark.parametrize(
    "stream_params",
    [
        {
            "-clear_prev_assets": True,
            "-video_source": return_testvideo_path(fmt="vo"),
            "-audio": "https://raw.githubusercontent.com/abhiTronix/Imbakup/master/Images/invalid.aac",
        },
        {
            "-clear_prev_assets": True,
            "-video_source": return_testvideo_path(fmt="vo"),
            "-audio": return_testvideo_path(fmt="ao"),
        },
        {
            "-clear_prev_assets": True,
            "-video_source": return_testvideo_path(fmt="vo"),
            "-audio": "https://raw.githubusercontent.com/abhiTronix/Imbakup/master/Images/big_buck_bunny_720p_1mb_ao.aac",
        },
    ],
)
def test_audio(stream_params):
    """
    Testing Single-Source Mode
    """
    mpd_file_path = os.path.join(return_mpd_path(), "dash_test.mpd")
    try:
        streamer = StreamGear(output=mpd_file_path, **stream_params)
        streamer.transcode_source()
        streamer.terminate()
        assert check_valid_mpd(mpd_file_path)
    except Exception as e:
        pytest.fail(str(e))


@pytest.mark.parametrize(
    "stream_params",
    [
        {
            "-clear_prev_assets": True,
            "-video_source": return_testvideo_path(fmt="vo"),
            "-streams": [
                {
                    "-video_bitrate": "unknown",
                },  # Invalid Stream 1
                {
                    "-resolution": "unxun",
                },  # Invalid Stream 2
                {
                    "-resolution": "640x480",
                    "-video_bitrate": "unknown",
                },  # Invalid Stream 3
                {
                    "-resolution": "640x480",
                    "-framerate": "unknown",
                },  # Invalid Stream 4
                {
                    "-resolution": "320x240",
                    "-framerate": 20.0,
                },  # Stream: 320x240 at 20fps framerate
            ],
        },
        {
            "-clear_prev_assets": True,
            "-video_source": return_testvideo_path(fmt="vo"),
            "-audio": return_testvideo_path(fmt="ao"),
            "-streams": [
                {
                    "-resolution": "640x480",
                    "-video_bitrate": "850k",
                    "-audio_bitrate": "128k",
                },  # Stream1: 640x480 at 850kbps bitrate
                {
                    "-resolution": "320x240",
                    "-framerate": 20.0,
                },  # Stream2: 320x240 at 20fps framerate
            ],
        },
        {
            "-clear_prev_assets": True,
            "-video_source": return_testvideo_path(),
            "-streams": [
                {
                    "-resolution": "960x540",
                    "-video_bitrate": "1350k",
                },  # Stream1: 960x540 at 1350kbps bitrate
            ],
        },
    ],
)
def test_multistreams(stream_params):
    """
    Testing Support for additional Secondary Streams of variable bitrates or spatial resolutions.
    """
    mpd_file_path = os.path.join(return_mpd_path(), "dash_test.mpd")
    results = extract_resolutions(
        stream_params["-video_source"], stream_params["-streams"]
    )
    try:
        streamer = StreamGear(output=mpd_file_path, **stream_params)
        streamer.transcode_source()
        streamer.terminate()
        metadata = extract_meta_mpd(mpd_file_path)
        assert metadata and (len(metadata) == len(results)), "Test Failed!"
        for x, y in zip(results, metadata):
            assert int(x["width"]) == int(y["width"]), "Width check failed!"
            assert int(x["height"]) == int(y["height"]), "Height check failed!"
    except Exception as e:
        pytest.fail(str(e))
