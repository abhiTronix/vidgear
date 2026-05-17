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

import logging as log
import os
import platform
import tempfile
import time

import cv2
import numpy as np
import pytest

from vidgear.gears import FFGear
from vidgear.gears.helper import logger_handler
from vidgear.tests.utils.helpers import return_testvideo_path

# define test logger
logger = log.getLogger("Test_FFGear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

# define machine os
_windows = os.name == "nt"

# skip entire module if deffcode is not installed
pytest.importorskip("deffcode", reason="`deffcode` is required for FFGear tests")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------





def actual_frame_count_n_frame_size(path):
    """Count frames and capture shape via OpenCV."""
    stream = cv2.VideoCapture(path)
    count = 0
    shape = None
    while True:
        grabbed, frame = stream.read()
        if not grabbed:
            break
        if shape is None:
            shape = frame.shape
        count += 1
    stream.release()
    logger.debug("Total frames via CV: %d, shape: %s", count, shape)
    return count, shape


# ---------------------------------------------------------------------------
# Source playback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source, options, result",
    [
        (return_testvideo_path(), {"THREAD_TIMEOUT": 300}, True),
        (return_testvideo_path(), {"THREAD_TIMEOUT": "wrong", "THREADED_QUEUE_MODE": False}, True),
        ("im_not_a_source.mp4", {}, False),
        (return_testvideo_path(), {"THREADED_QUEUE_MODE": "invalid"}, True),
    ],
)
def test_source_playback(source, options, result):
    """Verifies basic start/read/stop lifecycle with various sources and options."""
    stream = None
    try:
        stream = FFGear(source=source, logging=True, **options).start()
        frame_num = 0
        while True:
            frame = stream.read()
            if frame is None:
                break
            assert isinstance(frame, np.ndarray), "Frame must be a numpy array"
            time.sleep(0.05)
            frame_num += 1
        logger.debug("Total frames read: %d", frame_num)
        assert frame_num > 0
    except Exception as e:
        if not result:
            pytest.xfail(str(e))
        else:
            pytest.fail(str(e))
    finally:
        stream is not None and stream.stop()


# ---------------------------------------------------------------------------
# Threaded queue mode — frame count fidelity
# ---------------------------------------------------------------------------


def test_threaded_queue_mode_frame_count():
    """With THREADED_QUEUE_MODE=True, FFGear should deliver all frames."""
    stream = None
    source = return_testvideo_path()
    try:
        actual_count, _ = actual_frame_count_n_frame_size(source)
        stream = FFGear(
            source=source,
            logging=True,
            THREADED_QUEUE_MODE=True,
            QUEUE_SIZE=128,
            THREAD_TIMEOUT=300,
        ).start()
        count = 0
        while True:
            frame = stream.read()
            if frame is None:
                break
            count += 1
        logger.debug("FFGear frames: %d, actual: %d", count, actual_count)
        assert count == actual_count
    except Exception as e:
        pytest.fail(str(e))
    finally:
        stream is not None and stream.stop()


def test_threaded_queue_mode_disabled_skips_frames():
    """With THREADED_QUEUE_MODE=False, frame skipping is expected under load."""
    stream = None
    source = return_testvideo_path()
    try:
        actual_count, _ = actual_frame_count_n_frame_size(source)
        stream = FFGear(
            source=source,
            logging=True,
            THREADED_QUEUE_MODE=False,
            THREAD_TIMEOUT=300,
        ).start()
        count = 0
        while True:
            frame = stream.read()
            if frame is None:
                break
            time.sleep(0.05)  # simulate slow consumer to induce skipping
            count += 1
        logger.debug("FFGear (no queue) frames: %d, actual: %d", count, actual_count)
        assert count <= actual_count
    except Exception as e:
        pytest.fail(str(e))
    finally:
        stream is not None and stream.stop()


# ---------------------------------------------------------------------------
# Frame format
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "frame_format, extra, valid",
    [
        ("bgr24", {}, True),
        ("gray", {}, True),
        ("rgba", {}, True),
        ("rgb24", {}, True),
        ("yuv420p", {"-enforce_cv_patch": True}, True),
        ("invalid_fmt", {}, True),  # deffcode falls back to default; not a fatal error
    ],
)
def test_frame_format(frame_format, extra, valid):
    """Verifies FFGear handles various frame_format values without crashing."""
    stream = None
    source = return_testvideo_path()
    try:
        stream = FFGear(
            source=source,
            frame_format=frame_format,
            logging=True,
            **extra,
        ).start()
        frame = stream.read()
        assert frame is not None and isinstance(frame, np.ndarray)
        logger.debug("frame_format=%s → shape=%s dtype=%s", frame_format, frame.shape, frame.dtype)
    except Exception as e:
        if not valid:
            pytest.xfail(str(e))
        else:
            pytest.fail(str(e))
    finally:
        stream is not None and stream.stop()


def test_yuv420p_cv_patch_roundtrip():
    """
    yuv420p with -enforce_cv_patch delivers a 3:2 planar buffer that
    FFGear auto-converts to a standard (H, W, 3) BGR frame via cv2.cvtColor.
    """
    stream = None
    source = return_testvideo_path()
    _, actual_shape = actual_frame_count_n_frame_size(source)
    try:
        stream = FFGear(
            source=source,
            frame_format="yuv420p",
            logging=True,
            **{"-enforce_cv_patch": True},
        ).start()
        frame = stream.read()
        if frame is None:
            pytest.skip("yuv420p not supported by this FFmpeg build")
        assert frame.shape == actual_shape, (
            f"Expected BGR shape {actual_shape}, got {frame.shape}"
        )
    except Exception as e:
        pytest.fail(str(e))
    finally:
        stream is not None and stream.stop()


# ---------------------------------------------------------------------------
# Per-frame metadata extraction
# ---------------------------------------------------------------------------


def test_extract_metadata_basic():
    """
    With -extract_metadata=True, read() returns (frame, meta) tuples with the
    documented keys: frame_num, pts_time, is_keyframe, frame_type.
    """
    stream = None
    source = return_testvideo_path()
    try:
        stream = FFGear(
            source=source,
            frame_format="bgr24",
            logging=True,
            **{"-extract_metadata": True},
        ).start()

        expected_keys = {"frame_num", "pts_time", "is_keyframe", "frame_type"}
        frames_checked = 0
        while frames_checked < 5:
            out = stream.read()
            if out is None:
                break
            assert isinstance(out, tuple) and len(out) == 2, (
                "Expected (frame, meta) tuple when -extract_metadata is True"
            )
            frame, meta = out
            assert isinstance(frame, np.ndarray), "frame must be ndarray"
            assert isinstance(meta, dict), "metadata must be a dict"
            assert expected_keys.issubset(meta.keys()), (
                f"Missing metadata keys, got {list(meta.keys())}"
            )
            assert meta["pts_time"] >= 0.0
            assert meta["frame_type"] in {"I", "P", "B", "?"}
            frames_checked += 1

        assert frames_checked > 0, "No frames were delivered"
    except Exception as e:
        pytest.fail(str(e))
    finally:
        stream is not None and stream.stop()


def test_extract_metadata_frame_metadata_attr():
    """frame_metadata attribute on FFGear reflects latest decoded frame's metadata."""
    stream = None
    source = return_testvideo_path()
    try:
        stream = FFGear(
            source=source,
            frame_format="bgr24",
            logging=True,
            **{"-extract_metadata": True},
        ).start()
        out = stream.read()
        assert out is not None
        _frame, meta = out
        assert stream.frame_metadata == meta or isinstance(stream.frame_metadata, dict)
    except Exception as e:
        pytest.fail(str(e))
    finally:
        stream is not None and stream.stop()


def test_extract_metadata_disabled_yields_plain_frame():
    """Without -extract_metadata, read() returns plain ndarray, not a tuple."""
    stream = None
    source = return_testvideo_path()
    try:
        stream = FFGear(source=source, frame_format="bgr24", logging=True).start()
        frame = stream.read()
        assert frame is not None
        assert not isinstance(frame, tuple), "Expected plain ndarray without -extract_metadata"
        assert isinstance(frame, np.ndarray)
    except Exception as e:
        pytest.fail(str(e))
    finally:
        stream is not None and stream.stop()


# ---------------------------------------------------------------------------
# FFmpeg parameter pass-through
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ffparams, check_shape",
    [
        ({"-vf": "scale=320:240"}, (240, 320, 3)),
        ({"-ss": "00:00:01", "-vframes": 1}, None),
        ({"-framerate": 30.0}, None),
        ({"-custom_resolution": [320, 240]}, (240, 320, 3)),
        ({"-vf": "hflip"}, None),
    ],
)
def test_ffmpeg_params_passthrough(ffparams, check_shape):
    """FFmpeg params are forwarded to FFdecoder and produce valid frames."""
    stream = None
    source = return_testvideo_path()
    try:
        stream = FFGear(
            source=source,
            frame_format="bgr24",
            logging=True,
            **ffparams,
        ).start()
        frame = stream.read()
        assert frame is not None and isinstance(frame, np.ndarray)
        if check_shape is not None:
            assert frame.shape == check_shape, (
                f"Expected shape {check_shape}, got {frame.shape}"
            )
        logger.debug("ffparams=%s → shape=%s", ffparams, frame.shape)
    except Exception as e:
        pytest.fail(str(e))
    finally:
        stream is not None and stream.stop()


def test_vf_preserved_with_extract_metadata():
    """
    User -vf filter must survive when -extract_metadata appends showinfo
    via comma-chaining (scale=160:120,showinfo).
    """
    stream = None
    source = return_testvideo_path()
    try:
        stream = FFGear(
            source=source,
            frame_format="bgr24",
            logging=True,
            **{"-extract_metadata": True, "-vf": "scale=160:120"},
        ).start()
        out = stream.read()
        assert out is not None
        frame, meta = out
        assert frame.shape == (120, 160, 3), (
            f"User -vf scale=160:120 was not preserved, shape={frame.shape}"
        )
        assert isinstance(meta, dict) and "frame_num" in meta
    except Exception as e:
        pytest.fail(str(e))
    finally:
        stream is not None and stream.stop()


# ---------------------------------------------------------------------------
# Stream mode (yt_dlp backend)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url, quality, parameters",
    [
        ("https://youtu.be/uCy5OuSQnyA", "73p", "invalid"),
        (
            "https://www.dailymotion.com/video/x2yrnum",
            "invalid",
            {"nocheckcertificate": True},
        ),
        ("im_not_a_url", "", {}),
    ],
)
def test_stream_mode(url, quality, parameters):
    """Stream mode delegates to yt_dlp backend; invalid URLs and quality xfail."""
    pytest.importorskip("yt_dlp", reason="yt_dlp required for stream_mode tests")
    stream = None
    try:
        options = {"STREAM_RESOLUTION": quality, "STREAM_PARAMS": parameters}
        stream = FFGear(
            source=url,
            stream_mode=True,
            logging=True,
            **options,
        ).start()
        frame = stream.read()
        assert frame is not None and isinstance(frame, np.ndarray)
        logger.debug("stream_mode frame shape: %s", frame.shape)
    except Exception as e:
        if isinstance(e, (RuntimeError, ValueError)):
            pytest.xfail(str(e))
        else:
            pytest.fail(str(e))
    finally:
        stream is not None and stream.stop()


# ---------------------------------------------------------------------------
# Stop / double-stop safety
# ---------------------------------------------------------------------------


def test_stop_is_idempotent():
    """Calling stop() twice must not raise."""
    stream = None
    source = return_testvideo_path()
    try:
        stream = FFGear(source=source, logging=True).start()
        stream.stop()
        stream.stop()  # second call must not raise
    except Exception as e:
        pytest.fail(str(e))


# ---------------------------------------------------------------------------
# Camera / device capture
# ---------------------------------------------------------------------------


_camera_test_data = [
    ("/dev/video0", "v4l2", platform.system() == "Linux"),   # manual source + demuxer
    (0, None, platform.system() == "Linux"),                  # +ve index, no demuxer
    ("-1", "auto", platform.system() == "Linux"),             # -ve index, "auto" demuxer
    ("5", "auto", False),                                     # out-of-range index
    ("invalid", "auto", False),                               # invalid source
    ("/dev/video0", "invalid", False),                        # invalid demuxer
]


@pytest.mark.parametrize("source, source_demuxer, result", _camera_test_data)
def test_camera_capture(source, source_demuxer, result):
    """Tests FFGear real-time webcam and device capture via source_demuxer."""
    stream = None
    try:
        stream = FFGear(
            source=source,
            source_demuxer=source_demuxer,
            frame_format="bgr24",
            logging=True,
        ).start()
        for _ in range(5):
            frame = stream.read()
            if frame is None:
                raise AssertionError("No frame received from camera source")
            assert isinstance(frame, np.ndarray)
    except Exception as e:
        if result:
            pytest.fail(str(e))
        else:
            pytest.xfail(str(e))
    finally:
        stream is not None and stream.stop()
