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
================================================
"""

# import the necessary packages

import gc
import os
import platform
import subprocess
import time

import cv2
import numpy as np
import pytest
from deffcode import FFdecoder

from vidgear.gears import WriteGear
from vidgear.gears.helper import delete_file_safe
from vidgear.tests.utils.helpers import (
    decode_frames,
    get_testing_dir,
    probe_video,
    return_static_ffmpeg,
    return_testvideo_path,
)


def test_download_ffmpeg():
    """
    Auxilary test to simply delete old ffmpeg binaries.
    """
    try:
        import glob
        import shutil

        found = glob.glob(os.path.join(get_testing_dir(), "ffmpeg-static*"))
        if found and os.path.isdir(found[0]):
            shutil.rmtree(found[0])
    except Exception as e:
        if not isinstance(e, PermissionError):
            pytest.fail(str(e))


@pytest.mark.parametrize(
    "framerate, expected", [(30, 30.0), (29.97, 29.97), ("wrong_input", 25.0)]
)
def test_input_framerate(tmp_path, framerate, expected):
    """
    Testing "-input_framerate" parameter (defaults to 25 fps when invalid/omitted)
    """
    output_path = str(tmp_path / "Output_tif.mp4")
    with WriteGear(
        output=output_path,
        custom_ffmpeg=return_static_ffmpeg(),
        logging=True,
        **{"-input_framerate": framerate},
    ) as writer:
        for _ in range(30):
            writer.write(np.zeros((48, 64, 3), np.uint8))
    assert probe_video(output_path)["source_video_framerate"] == expected


@pytest.mark.parametrize(
    "pixfmts", ["bgr24", "rgba", "invalid", "invalid2", "yuv444p", "bgr48be"]
)
def test_write(tmp_path, pixfmts):
    """
    Testing `-input_pixfmt` with different pixel formats.
    """
    source = return_testvideo_path(fmt="vo")
    if pixfmts != "invalid2":
        decoder = FFdecoder(
            source, frame_format=pixfmts, custom_ffmpeg=return_static_ffmpeg()
        ).formulate()
        output_params = {"-input_pixfmt": pixfmts}
    else:
        # `rgb24` frames declared as `yuvj422p`, which must be rejected
        decoder = FFdecoder(source, custom_ffmpeg=return_static_ffmpeg()).formulate()
        output_params = {"-input_pixfmt": "yuvj422p"}
    output_path = str(tmp_path / "Output_tw.mp4")
    writer = WriteGear(
        output=output_path, custom_ffmpeg=return_static_ffmpeg(), **output_params
    )
    count = 0
    try:
        for frame in decoder.generateFrame():
            writer.write(frame)
            count += 1
    except ValueError as e:
        assert pixfmts == "invalid2" and "don't match" in str(e)
    finally:
        decoder.terminate()
        writer.close()
    if pixfmts != "invalid2":
        meta = probe_video(output_path)
        assert meta["source_video_resolution"] == [1280, 720]
        assert meta["approx_video_nframes"] == count


@pytest.mark.parametrize(
    "dimensions, expected", [((160, 120), [160, 120]), ((161, 121), [162, 122])]
)
def test_output_dimensions(tmp_path, dimensions, expected):
    """
    Testing "-output_dimensions" resizes (odd sizes rounded up for yuv420p), not mis-slices frames
    """
    output_params = {"-output_dimensions": dimensions}
    if platform.system() == "Windows":
        output_params.update(
            {
                "-ffmpeg_download_path": get_testing_dir(),
                "-disable_ffmpeg_window": True,
            }
        )
    output_path = str(tmp_path / "Output_tod.mp4")
    with WriteGear(
        output=output_path,
        custom_ffmpeg=return_static_ffmpeg(),
        logging=True,
        **output_params,
    ) as writer:
        for _ in range(50):
            writer.write(np.zeros((240, 320, 3), np.uint8))
    meta = probe_video(output_path)
    assert meta["source_video_resolution"] == expected
    assert meta["source_video_pixfmt"] == "yuv420p"
    assert len(decode_frames(output_path)) == 50


@pytest.mark.parametrize(
    "dtype, rgb_mode, size",
    [
        (np.uint8, False, (240, 320)),
        (np.uint8, True, (240, 320)),
        (np.uint16, False, (240, 320)),
        (np.uint8, False, (241, 321)),
    ],
)
def test_frame_roundtrip(tmp_path, dtype, rgb_mode, size):
    """
    Testing written frames decode back to same content (channel order, 16-bit, odd sizes)
    """
    height, width = size
    # blue frame with red top half (BGR)
    frame = np.zeros((height, width, 3), np.uint8)
    frame[..., 0] = 200
    frame[: height // 2, :, 2] = 100
    data = frame.astype(dtype) * (257 if dtype == np.uint16 else 1)
    if rgb_mode:
        data = np.ascontiguousarray(data[..., ::-1])
    output_path = str(tmp_path / "Output_rt.mp4")
    with WriteGear(output=output_path, custom_ffmpeg=return_static_ffmpeg()) as writer:
        for _ in range(10):
            writer.write(data, rgb_mode=rgb_mode)
    frames = decode_frames(output_path)
    assert len(frames) == 10
    assert frames[0].shape == (height + height % 2, width + width % 2, 3)
    assert np.abs(frames[0][:height, :width].astype(int) - frame).mean() < 8


@pytest.mark.parametrize(
    "output_params",
    [
        {"-i": return_testvideo_path(fmt="ao")},
        {"-i": return_testvideo_path(fmt="av")},
        {"-i": return_testvideo_path(fmt="ao"), "-acodec": "copy"},
        {"-clones": ["-f", "lavfi", "-i", "anullsrc"]},
        {"-ffpreheaders": ["-f", "lavfi", "-i", "anullsrc"]},
    ],
)
def test_extra_input(tmp_path, output_params):
    """
    Testing extra (audio) inputs keep piped video and every frame, and never hang `close()`
    """
    output_path = str(tmp_path / "Output_ei.mp4")
    writer = WriteGear(
        output=output_path, custom_ffmpeg=return_static_ffmpeg(), **output_params
    )
    # 6s of video, longer than the 5.3s test audio
    for _ in range(150):
        writer.write(np.zeros((240, 320, 3), np.uint8))
    start = time.time()
    writer.close(timeout=10)
    assert time.time() - start < 5 and writer.returncode == 0
    meta = probe_video(output_path)
    assert meta["source_video_resolution"] == [320, 240]
    assert meta["source_has_audio"]
    assert len(decode_frames(output_path)) == 150


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_invalid_dtype(tmp_path, dtype):
    """
    Testing non `uint8`/`uint16` frames are rejected instead of written as garbage
    """
    with (
        WriteGear(output=str(tmp_path / "Output.mp4")) as writer,
        pytest.raises(ValueError, match="uint8"),
    ):
        writer.write(np.zeros((48, 64, 3), dtype))


test_data_class = [
    ("", "", {}, False),
    (os.path.join(get_testing_dir(), "Output1.mp4"), "", {}, True),
    (os.path.join(get_testing_dir(), "temp_write"), "", {}, True),
    (
        os.path.join(get_testing_dir(), "Output2.mp4"),
        "",
        {
            "-vcodec": "libx264",
            "-crf": 0,
            "-preset": "fast",
            "-ffpreheaders": False,
            "-disable_ffmpeg_window": True,
        },
        True,
    ),
    (
        os.path.join(get_testing_dir(), "Output3.mp4"),
        return_static_ffmpeg(),
        {
            "-c:v": "libx265",
            "-vcodec": "libx264",
            "-crf": 0,
            "-preset": "veryfast",
            "-ffpreheaders": ["-re"],
            "-disable_ffmpeg_window": "Invalid",
        },
        True,
    ),
]


@pytest.mark.parametrize("f_name, c_ffmpeg, output_params, result", test_data_class)
def test_WriteGear_compression(f_name, c_ffmpeg, output_params, result):
    """
    Testing WriteGear Compression-Mode(FFmpeg) with different parameters
    """
    if not result:
        with pytest.raises(ValueError):
            WriteGear(output=f_name, custom_ffmpeg=c_ffmpeg, **output_params)
        return
    stream = cv2.VideoCapture(return_testvideo_path())  # Open stream
    with WriteGear(
        output=f_name, compression_mode=True, custom_ffmpeg=c_ffmpeg, **output_params
    ) as writer:
        while True:
            (grabbed, frame) = stream.read()
            if not grabbed:
                break
            writer.write(frame)
        stream.release()
    assert writer.returncode == 0
    delete_file_safe(f_name)


@pytest.mark.parametrize("case", ["extract_audio", "empty", "failing", "str", "no_compression"])
def test_WriteGear_customFFmpeg(tmp_path, case):
    """
    Testing WriteGear Compression-Mode(FFmpeg) custom FFmpeg Pipeline by seperating audio from video
    """
    audio = str(tmp_path / "input_audio.aac")
    commands = {
        "extract_audio": ["-y", "-i", return_testvideo_path(), "-vn", "-acodec", "copy", audio],
        "empty": None,
        "failing": ["wrong_input", "invalid_flag", "break_things"],
        "str": "wrong_input",
        "no_compression": ["invalid"],
    }
    writer = WriteGear(
        output=str(tmp_path / "Output.mp4"),
        compression_mode=case != "no_compression",
        logging=case == "empty",
        # deprecated, accepted as no-op
        **{"-disable_force_termination": True},
    )
    if case == "str":
        with pytest.raises(ValueError):
            writer.execute_ffmpeg_cmd(commands[case])
    elif case == "no_compression":
        with pytest.raises(RuntimeError):
            writer.execute_ffmpeg_cmd(commands[case])
    else:
        result = writer.execute_ffmpeg_cmd(commands[case])
        if case == "empty":
            assert result is None
        elif case == "failing":
            assert result.returncode != 0
        else:
            assert result.returncode == 0 and os.path.isfile(audio)
    writer.close()


@pytest.mark.parametrize("vcodec_key", ["-vcodec", "-c:v"])
def test_skip_vcodec(tmp_path, vcodec_key):
    """
    Testing `-vcodec`/`-c:v` set to `None` to discard `-vcodec` FFmpeg parameter
    entirely from the encoding pipeline (e.g. for GIF output, letting FFmpeg
    auto-select the best available video encoder).
    """
    stream = cv2.VideoCapture(return_testvideo_path())  # Open stream
    output_path = str(tmp_path / "Output_skipv.gif")
    with WriteGear(
        output=output_path,
        custom_ffmpeg=return_static_ffmpeg(),
        logging=True,
        **{vcodec_key: None},
    ) as writer:
        while True:
            (grabbed, frame) = stream.read()
            if not grabbed:
                break
            writer.write(frame)
        stream.release()
    # assert output file is created successfully
    assert os.path.isfile(output_path) and os.path.getsize(output_path) > 0


def test_execute_ffmpeg_cmd_result(tmp_path):
    """
    Testing `execute_ffmpeg_cmd` returns exit status/stderr, and honors `check` and `timeout`
    """
    writer = WriteGear(output=str(tmp_path / "Output.mp4"))
    assert writer.execute_ffmpeg_cmd([]) is None
    result = writer.execute_ffmpeg_cmd(
        ["-f", "lavfi", "-i", "testsrc=d=1", "-f", "null", "-"]
    )
    assert result.returncode == 0
    with pytest.raises(subprocess.CalledProcessError) as e:
        writer.execute_ffmpeg_cmd(["-i", "non_existent_file.mp4"], check=True)
    assert "non_existent_file.mp4" in e.value.stderr
    with pytest.raises(subprocess.TimeoutExpired):
        writer.execute_ffmpeg_cmd(
            ["-re", "-f", "lavfi", "-i", "testsrc", "-f", "null", "-"], timeout=1
        )
    writer.close()


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.parametrize("custom_ffmpeg", ["", return_static_ffmpeg()])
def test_ffmpeg_death(tmp_path, custom_ffmpeg):
    """
    Testing FFmpeg's stderr surfaces on early encoder death, `close()` doesn't raise
    or leak the dead pipe, and writes after `close()` fail clearly
    """
    writer = WriteGear(
        output=str(tmp_path / "Output.mp4"),
        custom_ffmpeg=custom_ffmpeg,
        **{"-invalid_flag": "1"},
    )
    frame = np.zeros((48, 64, 3), np.uint8)
    with pytest.raises(ValueError, match="invalid_flag"):
        for _ in range(500):
            writer.write(frame)
            time.sleep(0.01)
    assert "invalid_flag" in "\n".join(writer.stderr_tail)
    writer.close()
    gc.collect()
    assert writer.returncode not in (None, 0)
    with pytest.raises(RuntimeError, match="closed"):
        writer.write(frame)


def test_close_timeout(tmp_path):
    """
    Testing `close(timeout)` stops an FFmpeg that won't finish in time
    """
    # `-re` at 1 fps makes FFmpeg take seconds to drain the buffered frames
    writer = WriteGear(
        output=str(tmp_path / "Output.mp4"),
        **{"-input_framerate": 1, "-ffpreheaders": ["-re"]},
    )
    for _ in range(6):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    start = time.time()
    writer.close(timeout=0.5)
    assert time.time() - start < 4
    assert writer.returncode is not None
