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
import queue
import pytest
import m3u8
import logging as log
import platform
import tempfile
from mpegdash.parser import MPEGDASHParser

from vidgear.gears import CamGear, StreamGear
from vidgear.gears.helper import logger_handler, validate_video

# define test logger
logger = log.getLogger("Test_Streamgear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

# define machine os
_windows = True if os.name == "nt" else False


def return_testvideo_path(fmt="av"):
    """
    returns Test video path
    """
    supported_fmts = {
        "av": "BigBuckBunny_4sec.mp4",
        "vo": "BigBuckBunny_4sec_VO.mp4",
        "ao": "BigBuckBunny_4sec_AO.aac",
    }
    req_fmt = fmt if (fmt in supported_fmts) else "av"
    path = "{}/Downloads/Test_videos/{}".format(
        tempfile.gettempdir(), supported_fmts[req_fmt]
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


def check_valid_mpd(file="", exp_reps=1):
    """
    checks if given file is a valid MPD(MPEG-DASH Manifest file)
    """
    if not file or not os.path.isfile(file):
        return False
    all_reprs = []
    all_adapts = []
    try:
        mpd = MPEGDASHParser.parse(file)
        for period in mpd.periods:
            for adapt_set in period.adaptation_sets:
                all_adapts.append(adapt_set)
                for rep in adapt_set.representations:
                    all_reprs.append(rep)
    except Exception as e:
        logger.error(str(e))
        return False
    return (all_adapts, all_reprs) if (len(all_reprs) >= exp_reps) else False


def extract_meta_video(file):
    """
    Extracts metadata from a valid video file
    """
    logger.debug("Extracting Metadata from {}".format(file))
    meta = validate_video(return_static_ffmpeg(), file, logging=True)
    return meta


def check_valid_m3u8(file=""):
    """
    checks if given file is a valid M3U8 file
    """
    if not file or not os.path.isfile(file):
        logger.error("No file provided")
        return False
    metas = []
    try:
        data = open(file).read()
        playlist = m3u8.loads(data)
        if playlist.is_variant:
            for pl in playlist.playlists:
                meta = {}
                meta["resolution"] = pl.stream_info.resolution
                meta["framerate"] = pl.stream_info.frame_rate
                metas.append(meta)
        else:
            for seg in playlist.segments:
                metas.append(extract_meta_video(seg))
    except Exception as e:
        logger.error(str(e))
        return False
    return metas


def extract_meta_mpd(file):
    """
    Extracts metadata from a valid MPD(MPEG-DASH Manifest file)
    """
    adapts, reprs = check_valid_mpd(file)
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
                meta["framerate"] = (
                    rep.frame_rate
                    if not (rep.frame_rate is None)
                    else adapts[0].frame_rate
                )
            logger.debug("Found Meta: {}".format(meta))
            metas.append(meta)
        logger.debug("MetaData: {}".format(metas))
        return metas
    else:
        return []


def return_assets_path(hls=False):
    """
    returns assets temp path
    """
    return os.path.join(tempfile.gettempdir(), "temp_m3u8" if hls else "temp_mpd")


def string_to_float(value):
    """
    Converts fraction to float
    """
    if value is None:
        logger.error("Input value is None!")
        return 0.0
    extracted = value.strip().split("/")
    cleaned = [float(x.strip()) for x in extracted]
    return cleaned[0] / cleaned[1]


def extract_resolutions(source, streams):
    """
    Extracts resolution value from dictionaries
    """
    if not (source) or not (streams):
        return {}
    results = {}
    assert os.path.isfile(source), "Not a valid source"
    results["source"] = extract_meta_video(source)
    num = 0
    for stream in streams:
        if "-resolution" in stream:
            try:
                res = stream["-resolution"].split("x")
                assert len(res) == 2
                width, height = (res[0].strip(), res[1].strip())
                assert width.isnumeric() and height.isnumeric()
                results["streams{}".format(num)] = {"resolution": (width, height)}
                num += 1
            except Exception as e:
                logger.error(str(e))
                continue
        else:
            continue
    return results


@pytest.mark.parametrize("format", ["dash", "hls"])
def test_ss_stream(format):
    """
    Testing Single-Source Mode
    """
    assets_file_path = os.path.join(
        return_assets_path(False if format == "dash" else True),
        "format_test{}".format(".mpd" if format == "dash" else ".m3u8"),
    )
    try:
        stream_params = {
            "-video_source": return_testvideo_path(),
            "-clear_prev_assets": True,
        }
        if format == "hls":
            stream_params.update(
                {
                    "-hls_base_url": return_assets_path(
                        False if format == "dash" else True
                    )
                    + os.sep
                }
            )
        streamer = StreamGear(
            output=assets_file_path, format=format, logging=True, **stream_params
        )
        streamer.transcode_source()
        streamer.terminate()
        if format == "dash":
            assert check_valid_mpd(assets_file_path), "Test Failed!"
        else:
            assert extract_meta_video(assets_file_path), "Test Failed!"
    except Exception as e:
        pytest.fail(str(e))


@pytest.mark.parametrize("format", ["dash", "hls"])
def test_ss_livestream(format):
    """
    Testing Single-Source Mode with livestream.
    """
    assets_file_path = os.path.join(
        return_assets_path(False if format == "dash" else True),
        "format_test{}".format(".mpd" if format == "dash" else ".m3u8"),
    )
    try:
        stream_params = {
            "-video_source": return_testvideo_path(),
            "-livestream": True,
            "-remove_at_exit": 1,
        }
        streamer = StreamGear(
            output=assets_file_path, format=format, logging=True, **stream_params
        )
        streamer.transcode_source()
        streamer.terminate()
    except Exception as e:
        pytest.fail(str(e))


@pytest.mark.parametrize(
    "conversion, format",
    [(None, "dash"), ("COLOR_BGR2GRAY", "hls"), ("COLOR_BGR2BGRA", "dash")],
)
def test_rtf_stream(conversion, format):
    """
    Testing Real-Time Frames Mode
    """
    assets_file_path = return_assets_path(False if format == "dash" else True)

    try:
        # Open stream
        options = {"THREAD_TIMEOUT": 300}
        stream = CamGear(
            source=return_testvideo_path(), colorspace=conversion, **options
        ).start()
        stream_params = {
            "-clear_prev_assets": True,
            "-input_framerate": "invalid",
        }
        if format == "hls":
            stream_params.update(
                {
                    "-hls_base_url": return_assets_path(
                        False if format == "dash" else True
                    )
                    + os.sep
                }
            )
        streamer = StreamGear(output=assets_file_path, format=format, **stream_params)
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
        asset_file = [
            os.path.join(assets_file_path, f)
            for f in os.listdir(assets_file_path)
            if f.endswith(".mpd" if format == "dash" else ".m3u8")
        ]
        assert len(asset_file) == 1, "Failed to create asset file!"
        if format == "dash":
            assert check_valid_mpd(asset_file[0]), "Test Failed!"
        else:
            assert extract_meta_video(asset_file[0]), "Test Failed!"
    except Exception as e:
        if not isinstance(e, queue.Empty):
            pytest.fail(str(e))


@pytest.mark.parametrize("format", ["dash", "hls"])
def test_rtf_livestream(format):
    """
    Testing Real-Time Frames Mode with livestream.
    """
    assets_file_path = return_assets_path(False if format == "dash" else True)

    try:
        # Open stream
        options = {"THREAD_TIMEOUT": 300}
        stream = CamGear(source=return_testvideo_path(), **options).start()
        stream_params = {
            "-livestream": True,
        }
        streamer = StreamGear(output=assets_file_path, format=format, **stream_params)
        while True:
            frame = stream.read()
            # check if frame is None
            if frame is None:
                break
            streamer.stream(frame)
        stream.stop()
        streamer.terminate()
    except Exception as e:
        if not isinstance(e, queue.Empty):
            pytest.fail(str(e))


@pytest.mark.parametrize("format", ["dash", "hls"])
def test_input_framerate_rtf(format):
    """
    Testing "-input_framerate" parameter provided by StreamGear
    """
    try:
        assets_file_path = os.path.join(
            return_assets_path(False if format == "dash" else True),
            "format_test{}".format(".mpd" if format == "dash" else ".m3u8"),
        )
        stream = cv2.VideoCapture(return_testvideo_path())  # Open stream
        test_framerate = stream.get(cv2.CAP_PROP_FPS)
        stream_params = {
            "-clear_prev_assets": True,
            "-input_framerate": test_framerate,
        }
        if format == "hls":
            stream_params.update(
                {
                    "-hls_base_url": return_assets_path(
                        False if format == "dash" else True
                    )
                    + os.sep
                }
            )
        streamer = StreamGear(
            output=assets_file_path, format=format, logging=True, **stream_params
        )
        while True:
            (grabbed, frame) = stream.read()
            if not grabbed:
                break
            streamer.stream(frame)
        stream.release()
        streamer.terminate()
        if format == "dash":
            meta_data = extract_meta_mpd(assets_file_path)
            assert meta_data and len(meta_data) > 0, "Test Failed!"
            framerate_mpd = string_to_float(meta_data[0]["framerate"])
            assert framerate_mpd > 0.0 and isinstance(
                framerate_mpd, float
            ), "Test Failed!"
            assert round(framerate_mpd) == round(test_framerate), "Test Failed!"
        else:
            meta_data = extract_meta_video(assets_file_path)
            assert meta_data and "framerate" in meta_data, "Test Failed!"
            framerate_m3u8 = float(meta_data["framerate"])
            assert framerate_m3u8 > 0.0 and isinstance(
                framerate_m3u8, float
            ), "Test Failed!"
            assert round(framerate_m3u8) == round(test_framerate), "Test Failed!"
    except Exception as e:
        pytest.fail(str(e))


@pytest.mark.parametrize(
    "stream_params, format",
    [
        (
            {
                "-clear_prev_assets": True,
                "-bpp": 0.2000,
                "-gop": 125,
                "-vcodec": "libx265",
            },
            "hls",
        ),
        (
            {
                "-clear_prev_assets": True,
                "-bpp": "unknown",
                "-gop": "unknown",
                "-s:v:0": "unknown",
                "-b:v:0": "unknown",
                "-b:a:0": "unknown",
            },
            "hls",
        ),
        (
            {
                "-clear_prev_assets": True,
                "-bpp": 0.2000,
                "-gop": 125,
                "-vcodec": "libx265",
            },
            "dash",
        ),
        (
            {
                "-clear_prev_assets": True,
                "-bpp": "unknown",
                "-gop": "unknown",
                "-s:v:0": "unknown",
                "-b:v:0": "unknown",
                "-b:a:0": "unknown",
            },
            "dash",
        ),
    ],
)
def test_params(stream_params, format):
    """
    Testing "-stream_params" parameters by StreamGear
    """
    try:
        assets_file_path = os.path.join(
            return_assets_path(False if format == "dash" else True),
            "format_test{}".format(".mpd" if format == "dash" else ".m3u8"),
        )
        if format == "hls":
            stream_params.update(
                {
                    "-hls_base_url": return_assets_path(
                        False if format == "dash" else True
                    )
                    + os.sep
                }
            )
        stream = cv2.VideoCapture(return_testvideo_path())  # Open stream
        streamer = StreamGear(
            output=assets_file_path, format=format, logging=True, **stream_params
        )
        while True:
            (grabbed, frame) = stream.read()
            if not grabbed:
                break
            streamer.stream(frame)
        stream.release()
        streamer.terminate()
        if format == "dash":
            assert check_valid_mpd(assets_file_path), "Test Failed!"
        else:
            assert extract_meta_video(assets_file_path), "Test Failed!"
    except Exception as e:
        pytest.fail(str(e))


@pytest.mark.parametrize(
    "stream_params, format",
    [
        (
            {
                "-clear_prev_assets": True,
                "-video_source": return_testvideo_path(fmt="vo"),
                "-audio": "https://gitlab.com/abhiTronix/Imbakup/-/raw/master/Images/invalid.aac",
            },
            "dash",
        ),
        (
            {
                "-clear_prev_assets": True,
                "-video_source": return_testvideo_path(fmt="vo"),
                "-audio": return_testvideo_path(fmt="ao"),
            },
            "dash",
        ),
        (
            {
                "-clear_prev_assets": True,
                "-video_source": return_testvideo_path(fmt="vo"),
                "-audio": "https://gitlab.com/abhiTronix/Imbakup/-/raw/master/Images/big_buck_bunny_720p_1mb_ao.aac",
            },
            "dash",
        ),
        (
            {
                "-clear_prev_assets": True,
                "-video_source": return_testvideo_path(fmt="vo"),
                "-audio": "https://gitlab.com/abhiTronix/Imbakup/-/raw/master/Images/invalid.aac",
            },
            "hls",
        ),
        (
            {
                "-clear_prev_assets": True,
                "-video_source": return_testvideo_path(fmt="vo"),
                "-audio": return_testvideo_path(fmt="ao"),
            },
            "hls",
        ),
        (
            {
                "-clear_prev_assets": True,
                "-video_source": "https://gitlab.com/abhiTronix/Imbakup/-/raw/master/Images/input.mp4",
                "-audio": "https://gitlab.com/abhiTronix/Imbakup/-/raw/master/Images/noise.wav",
            },
            "hls",
        ),
    ],
)
def test_audio(stream_params, format):
    """
    Testing external and audio audio for stream.
    """
    assets_file_path = os.path.join(
        return_assets_path(False if format == "dash" else True),
        "format_test{}".format(".mpd" if format == "dash" else ".m3u8"),
    )
    try:
        if format == "hls":
            stream_params.update(
                {
                    "-hls_base_url": return_assets_path(
                        False if format == "dash" else True
                    )
                    + os.sep
                }
            )
        streamer = StreamGear(
            output=assets_file_path, format=format, logging=True, **stream_params
        )
        streamer.transcode_source()
        streamer.terminate()
        if format == "dash":
            assert check_valid_mpd(assets_file_path), "Test Failed!"
        else:
            assert extract_meta_video(assets_file_path), "Test Failed!"
    except Exception as e:
        pytest.fail(str(e))


@pytest.mark.parametrize(
    "format, stream_params",
    [
        (
            "dash",
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
        ),
        (
            "hls",
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
        ),
        (
            "dash",
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
        ),
        (
            "hls",
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
        ),
        (
            "dash",
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
        ),
        (
            "hls",
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
        ),
    ],
)
def test_multistreams(format, stream_params):
    """
    Testing Support for additional Secondary Streams of variable bitrates or spatial resolutions.
    """
    assets_file_path = os.path.join(
        return_assets_path(False if format == "dash" else True),
        "asset_test.{}".format("mpd" if format == "dash" else "m3u8"),
    )
    results = extract_resolutions(
        stream_params["-video_source"], stream_params["-streams"]
    )
    try:
        streamer = StreamGear(
            output=assets_file_path, format=format, logging=True, **stream_params
        )
        streamer.transcode_source()
        streamer.terminate()
        if format == "dash":
            metadata = extract_meta_mpd(assets_file_path)
            meta_videos = [x for x in metadata if x["mime_type"].startswith("video")]
            assert meta_videos and (len(meta_videos) <= len(results)), "Test Failed!"
            if len(meta_videos) == len(results):
                for m_v, s_v in zip(meta_videos, list(results.values())):
                    assert int(m_v["width"]) == int(
                        s_v["resolution"][0]
                    ), "Width check failed!"
                    assert int(m_v["height"]) == int(
                        s_v["resolution"][1]
                    ), "Height check failed!"
            else:
                valid_widths = [int(x["resolution"][0]) for x in list(results.values())]
                valid_heights = [
                    int(x["resolution"][1]) for x in list(results.values())
                ]
                for m_v in meta_videos:
                    assert int(m_v["width"]) in valid_widths, "Width check failed!"
                    assert int(m_v["height"]) in valid_heights, "Height check failed!"
        else:
            meta_videos = check_valid_m3u8(assets_file_path)
            assert meta_videos and (len(meta_videos) <= len(results)), "Test Failed!"
            if len(meta_videos) == len(results):
                for m_v, s_v in zip(meta_videos, list(results.values())):
                    assert int(m_v["resolution"][0]) == int(
                        s_v["resolution"][0]
                    ), "Width check failed!"
                    assert int(m_v["resolution"][1]) == int(
                        s_v["resolution"][1]
                    ), "Height check failed!"
            else:
                valid_widths = [int(x["resolution"][0]) for x in list(results.values())]
                valid_heights = [
                    int(x["resolution"][1]) for x in list(results.values())
                ]
                for m_v in meta_videos:
                    assert (
                        int(m_v["resolution"][0]) in valid_widths
                    ), "Width check failed!"
                    assert (
                        int(m_v["resolution"][1]) in valid_heights
                    ), "Height check failed!"
    except Exception as e:
        pytest.fail(str(e))
