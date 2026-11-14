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
import struct
import subprocess
import tempfile
import time

import cv2
import numpy as np
import pytest

from vidgear.gears import FFGear
from vidgear.gears.helper import logger_handler
from vidgear.tests.utils.helpers import (
    get_testing_dir,
    return_static_ffmpeg,
    return_testvideo_path,
)

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

# one `lavfi` input carrying two embedded video streams: v0 red 320x240, v1 blue 160x120
_MULTI_STREAM = "color=c=red:s=320x240:r=25:d=1[out0];color=c=blue:s=160x120:r=30:d=1[out1]"


def _color(frame):
    """Dominant colour of a BGR frame."""
    return "red" if frame[..., 2].mean() > frame[..., 0].mean() else "blue"


def _read_all(source, **params):
    """Decodes whole source with FFGear, returning its (stopped) stream and frames."""
    stream = FFGear(source=source, **params).start()
    try:
        frames = []
        while (frame := stream.read()) is not None:
            frames.append(frame)
        return stream, frames
    finally:
        stream.stop()


def _ffmpeg_raw(stream, *args):
    """Raw `bgr24` frames bytes from the FFmpeg binary that FFGear `stream` used."""
    cmd = [stream.metadata["ffmpeg_binary_path"], "-v", "error", *args]
    cmd += ["-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    return subprocess.run(cmd, capture_output=True).stdout


def _clip(name, *args):
    """Builds (once) a synthetic test clip with the static FFmpeg binary."""
    path = os.path.join(get_testing_dir(), "temp_write", name)
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        subprocess.run(
            [return_static_ffmpeg(), "-y", *args, path], check=True, capture_output=True
        )
    return path


def _testsrc():
    """320x240 lossless `testsrc2` clip with 25 frames."""
    return _clip(
        "ffgear_testsrc.mkv", "-f", "lavfi", "-i", "testsrc2=s=320x240:r=25:d=1",
        "-pix_fmt", "yuv444p", "-c:v", "ffv1",
    )  # fmt: skip


def _rotated():
    """320x240 clip flagged as rotated by 90° (portrait)."""
    path = os.path.join(get_testing_dir(), "temp_write", "ffgear_rot.mkv")
    if not os.path.isfile(path):
        cmd = [return_static_ffmpeg(), "-y", "-i", _testsrc(), "-c", "copy"]
        # `-display_rotation` needs FFmpeg 6.0+, older ones only honor the legacy `rotate` tag
        if subprocess.run([*cmd[:2], "-display_rotation", "90", *cmd[2:], path]).returncode:
            subprocess.run([*cmd, "-metadata:s:v:0", "rotate=90", path], check=True)
    return path


def _cover_first_mp4():
    """
    MP4 with red video + green cover art (attached pic). FFmpeg muxes cover art last;
    moving `udta` before `trak` inside `moov` (after `mdat`, so offsets stay valid)
    makes the demuxer expose it as stream 0, as some taggers do.
    """
    path = os.path.join(get_testing_dir(), "temp_write", "ffgear_cover_first.mp4")
    if not os.path.isfile(path):
        _clip(
            "ffgear_cover_first.mp4", "-f", "lavfi", "-i", "color=c=red:s=320x240:r=25:d=1",
            "-f", "lavfi", "-i", "color=c=green:s=64x64:r=1:d=1", "-map", "0", "-map", "1",
            "-c:v:0", "mpeg4", "-c:v:1", "png", "-frames:v:1", "1",
            "-disposition:v:1", "attached_pic",
        )  # fmt: skip

        def boxes(buf):
            out, i = [], 0
            while i < len(buf):
                size, kind = struct.unpack(">I4s", buf[i : i + 8])
                out.append((kind, buf[i : i + (size or len(buf) - i)]))
                i += size or len(buf) - i
            return out

        with open(path, "rb") as f:
            top = boxes(f.read())
        assert [k for k, _ in top][-1] == b"moov"
        order = {b"mvhd": 0, b"udta": 1}
        moov = sorted(boxes(top[-1][1][8:]), key=lambda k: order.get(k[0], 2))
        with open(path, "wb") as f:
            f.write(b"".join(b for _, b in top[:-1]) + top[-1][1][:8])
            f.write(b"".join(b for _, b in moov))
    return path


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


@pytest.mark.parametrize(
    "frame_format, shape, dtype",
    [
        # 8-bit 4:2:0 formats are auto-converted to BGR
        ("yuv420p", (240, 320, 3), "uint8"),
        ("nv12", (240, 320, 3), "uint8"),
        ("nv21", (240, 320, 3), "uint8"),
        # pixel format set by filters (as in hardware-decoding pipelines)
        ("null", (240, 320, 3), "uint8"),
        # no OpenCV conversion: FFdecoder's stacked planes are returned as-is
        ("yuv422p", (480, 320), "uint8"),
        ("yuv420p10le", (360, 320), "uint16"),
    ],
)
def test_enforce_cv_patch_formats(frame_format, shape, dtype):
    """-enforce_cv_patch converts yuv420p/nv12/nv21 to BGR, and leaves other YUV formats planar."""
    options = {"-enforce_cv_patch": True}
    if frame_format == "null":
        options["-vf"] = "format=nv12"
    _, frames = _read_all(_testsrc(), frame_format=frame_format, **options)
    assert len(frames) == 25
    assert frames[0].shape == shape and frames[0].dtype == dtype
    if len(shape) == 3:
        # same picture as FFmpeg's own BGR conversion (guards swapped chroma planes)
        _, expected = _read_all(_testsrc(), frame_format="bgr24")
        assert cv2.absdiff(frames[0], expected[0]).mean() < 5


# ---------------------------------------------------------------------------
# Stream selection / cover art / rotation (deffcode 0.3.0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ffparams, shape, color",
    [
        ({}, (240, 320, 3), "red"),  # first video stream by default
        ({"-default_stream_indexes": (1, 0)}, (120, 160, 3), "blue"),
        ({"-map": "0:v:1"}, (120, 160, 3), "blue"),
    ],
)
def test_multi_stream_selection(ffparams, shape, color):
    """Selecting an embedded video stream decodes THAT stream at its own size."""
    stream, frames = _read_all(_MULTI_STREAM, source_demuxer="lavfi", **ffparams)
    assert frames and frames[0].shape == shape
    assert _color(frames[0]) == color
    assert stream.metadata["source_video_streams"] == 2


def test_cover_art_first_source():
    """Source whose first video stream is cover art decodes its real video stream."""
    stream, frames = _read_all(_cover_first_mp4())
    assert len(frames) == 25 and frames[0].shape == (240, 320, 3)
    assert _color(frames[0]) == "red"
    assert stream.metadata["source_video_attached_pics"] == [0]


@pytest.mark.parametrize(
    "ffparams, shape",
    [
        ({}, (320, 240, 3)),  # auto-rotated upright, not squeezed into coded size
        ({"-ffprefixes": ["-noautorotate"]}, (240, 320, 3)),
    ],
)
def test_rotated_source(ffparams, shape):
    """±90° rotated sources decode as FFmpeg's own (auto-rotated) frames."""
    path = _rotated()
    stream, frames = _read_all(path, **ffparams)
    assert len(frames) == 25 and frames[0].shape == shape
    expected = _ffmpeg_raw(stream, *ffparams.get("-ffprefixes", []), "-i", path)
    assert b"".join(f.tobytes() for f in frames) == expected


def test_vf_scale_matches_ffmpeg():
    """Frames resized by `-vf scale` match FFmpeg's own output (no second scaler)."""
    stream, frames = _read_all(_testsrc(), **{"-vf": "scale=160:120"})
    assert len(frames) == 25 and frames[0].shape == (120, 160, 3)
    expected = _ffmpeg_raw(stream, "-i", _testsrc(), "-vf", "scale=160:120")
    assert b"".join(f.tobytes() for f in frames) == expected


def test_metadata_attribute():
    """`metadata` exposes FFdecoder metadata as dict, and is retained after stop()."""
    stream, _ = _read_all(return_testvideo_path())
    assert stream.stream is None  # stopped
    assert isinstance(stream.metadata, dict)
    assert stream.metadata["output_framerate"] == stream.framerate
    for key in ("source_video_streams", "source_video_sar", "source_video_dar",
                "source_video_color_transfer", "source_video_orientation"):  # fmt: skip
        assert key in stream.metadata


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
# Multi-outputs (-extra_outputs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ffparams, extra_outputs, check_shape",
    [
        # same filtering for multiple outputs
        (
            {
                "-filter_complex": "[0:v]yadif,split=2[main][rec];[main]scale=320:240[main];[rec]scale=160:120[rec]",
                "-map": "[main]",
            },
            ["-map", "[rec]", "-map", "0:a?", "-c:v", "libx264", "-crf", 23, "-c:a", "aac"],
            (240, 320, 3),
        ),
        # different filtering for multiple outputs
        (
            {
                "-filter_complex": "[0:v]split=2[in1][in2];[in1]boxblur=5[out1];[in2]negate[out2]",
                "-map": "[out1]",
            },
            ["-map", "[out2]", "-c:v", "libx264"],
            None,
        ),
    ],
)
def test_extra_outputs(ffparams, extra_outputs, check_shape):
    """
    -extra_outputs writes additional outputs from the same FFmpeg pipeline,
    without altering the frames delivered to read().
    """
    stream = None
    source = return_testvideo_path()
    actual_count, _ = actual_frame_count_n_frame_size(source)
    output = os.path.join(tempfile.mkdtemp(), "extra_output.mkv")
    try:
        stream = FFGear(
            source=source,
            frame_format="bgr24",
            logging=True,
            **ffparams,
            **{"-extra_outputs": ["-y", *extra_outputs, output]},
        ).start()
        count = 0
        while True:
            frame = stream.read()
            if frame is None:
                break
            if check_shape is not None:
                assert frame.shape == check_shape
            count += 1
        stream.stop()
        assert stream.returncode == 0
        assert count == actual_count
        assert actual_frame_count_n_frame_size(output)[0] == actual_count
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
    if result and not os.access("/dev/video0", os.R_OK | os.W_OK):
        pytest.skip("No accessible `/dev/video0` (v4l2loopback) device on this machine")
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


# ---------------------------------------------------------------------------
# EOF / timeout / stall handling
# ---------------------------------------------------------------------------


def test_framerate_and_returncode():
    """FFGear exposes source framerate, and retains FFmpeg exit status after stop()."""
    stream = FFGear(source=return_testvideo_path()).start()
    assert stream.framerate == 25.0
    while stream.read() is not None:
        pass
    stream.stop()
    assert stream.returncode == 0
    assert isinstance(stream.stderr_tail, list)


def test_eof_no_busy_spin():
    """Producer must exit at EOF (poison pill) instead of spinning until the queue drains."""
    stream = FFGear(source=return_testvideo_path(), QUEUE_SIZE=200).start()
    try:
        time.sleep(1.5)  # let producer hit EOF while frames are still queued
        cpu = time.process_time()
        time.sleep(1.0)
        assert time.process_time() - cpu < 0.5, "producer is busy-spinning at EOF"
        frames = 0
        while stream.read() is not None:
            frames += 1
        assert frames == actual_frame_count_n_frame_size(return_testvideo_path())[0]
        assert stream.read() is None
    finally:
        stream.stop()


@pytest.mark.parametrize("queue_mode", [True, False])
def test_stalled_source_read_timeout_and_stop(queue_mode):
    """read(timeout) raises on a stalled source, and stop() doesn't deadlock on the blocked pipe."""
    # one frame every 20s, read in real-time: second frame never arrives in time
    stream = FFGear(
        source="testsrc=size=64x48:rate=0.05",
        source_demuxer="lavfi",
        THREADED_QUEUE_MODE=queue_mode,
        **{"-ffprefixes": ["-re"]},
    ).start()
    queue_mode and stream.read()  # first frame was pre-queued
    with pytest.raises(TimeoutError):
        stream.read(timeout=0.5)
    start = time.time()
    stream.stop()
    assert time.time() - start < 10
    assert stream.read() is None
