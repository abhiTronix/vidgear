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
import logging as log
import os
import queue
import subprocess
import threading
import time
import warnings

import cv2
import m3u8
import numpy as np
import pytest
from mpegdash.parser import MPEGDASHParser

import vidgear.gears.streamgear as streamgear_module
from vidgear.gears import CamGear, StreamGear
from vidgear.gears.helper import logger_handler, validate_video
from vidgear.tests.utils.helpers import (
    make_clip,
    probe_video,
    return_static_ffmpeg,
    return_testvideo_path,
    stream_lines,
)

# define test logger
logger = log.getLogger("Test_Streamgear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


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
    return validate_video(return_static_ffmpeg(), str(file), logging=True)


def output_path(tmp_path, format, name="output"):
    """
    returns manifest path for format
    """
    return str(tmp_path / "{}.{}".format(name, "mpd" if format == "dash" else "m3u8"))


def check_valid_output(path, format):
    """
    checks manifest is valid
    """
    if format == "dash":
        assert check_valid_mpd(path), "Invalid MPD!"
    else:
        assert extract_meta_video(path), "Invalid M3U8!"


def video_lines(path):
    """
    `Stream #` banner lines of every video rendition
    """
    return [x for x in stream_lines(path) if "Video:" in x]


def audio_lines(path):
    """
    `Stream #` banner lines of every audio rendition
    """
    return [x for x in stream_lines(path) if "Audio:" in x]


def capture_commands(monkeypatch):
    """
    records Single-Source Mode FFmpeg commands without running them
    """
    commands = []
    monkeypatch.setattr(
        streamgear_module,
        "run_ffmpeg",
        lambda cmd, **kw: commands.append(cmd) or subprocess.CompletedProcess(cmd, 0, None, ""),
    )
    return commands


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    """
    Synthetic sources: cover art, PCM/Opus audio, rotated, anamorphic and HDR clips
    """
    d = tmp_path_factory.mktemp("clips")
    video = ["-f", "lavfi", "-i", "testsrc2=s=320x240:d=2"]
    audio = ["-f", "lavfi", "-i", "sine=d=2"]
    h264 = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    make_clip(d / "cover.png", "-f", "lavfi", "-i", "color=red:s=64x64", "-frames:v", "1")
    make_clip(d / "plain.mp4", *video, *h264)
    return {
        "cover": make_clip(
            d / "cover.mp4", "-i", d / "cover.png", *video, *audio, "-map", "0", "-map", "1", "-map", "2",
            "-c:v:0", "png", "-c:v:1", "libx264", "-pix_fmt:v:1", "yuv420p", "-c:a", "aac", "-disposition:v:0", "attached_pic",
        ),
        "subtitle": make_clip(
            d / "subtitle.mp4", *video, *audio, "-f", "lavfi", "-i", "anullsrc=d=2", *h264, "-map", "0", "-map", "1",
            "-c:a", "aac", "-map", "2", "-c:a:1", "aac", "-metadata:s:a:1", "language=fre",
        ),
        "pcm": make_clip(d / "pcm.mov", *video, *audio, *h264, "-c:a", "pcm_s16le"),
        "opus": make_clip(d / "opus.mkv", *video, *audio, *h264, "-c:a", "libopus"),
        "rotated": make_clip(d / "rotated.mp4", "-display_rotation", "90", "-i", d / "plain.mp4", "-c", "copy"),
        "anamorphic": make_clip(d / "anamorphic.mp4", "-f", "lavfi", "-i", "testsrc2=s=480x360:d=2", "-vf", "setsar=4/3", *h264),
        "hdr": make_clip(
            d / "hdr.mp4", *video, "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-color_primaries", "bt2020",
            "-color_trc", "smpte2084", "-colorspace", "bt2020nc",
        ),
    }


@pytest.mark.parametrize("format", ["dash", "hls"])
def test_ss_stream(format, tmp_path):
    """
    Testing Single-Source Mode
    """
    assets_file_path = output_path(tmp_path, format)
    stream_params = {
        "-video_source": return_testvideo_path(),
        "-vcodec": "copy",
        "-aspect": "4:3",
        "-vf": "format=yuv420p",
        "-streams": "invalid",
        "-clear_prev_assets": True,
    }
    if format == "hls":
        stream_params["-hls_base_url"] = str(tmp_path) + os.sep
    streamer = StreamGear(output=assets_file_path, format=format, logging=True, **stream_params)
    streamer.transcode_source()
    streamer.close()
    check_valid_output(assets_file_path, format)


@pytest.mark.parametrize("format", ["dash", "hls"])
def test_ss_livestream(format, tmp_path):
    """
    Testing Single-Source Mode with livestream writes live manifests
    """
    assets_file_path = output_path(tmp_path, format)
    with StreamGear(
        output=assets_file_path,
        format=format,
        logging=True,
        **{"-video_source": return_testvideo_path(), "-livestream": True, "-clear_prev_assets": "invalid"}
    ) as streamer:
        streamer.transcode_source()
    with open(assets_file_path) as f:
        manifest = f.read()
    if format == "dash":
        # FFmpeg finalizes live MPDs as `static`, but keeps live `-use_timeline 0`
        assert "SegmentTimeline" not in manifest
    else:
        assert "#EXT-X-PLAYLIST-TYPE:VOD" not in manifest


@pytest.mark.parametrize(
    "conversion, format",
    [(None, "dash"), ("COLOR_BGR2GRAY", "hls"), ("COLOR_BGR2RGBA", "dash")],
)
def test_rtf_stream(conversion, format, tmp_path):
    """
    Testing Real-Time Frames Mode
    """
    stream = CamGear(source=return_testvideo_path(), colorspace=conversion, THREAD_TIMEOUT=300).start()
    stream_params = {"-clear_prev_assets": True, "-input_framerate": "invalid"}
    if format == "hls":
        stream_params["-hls_base_url"] = str(tmp_path) + os.sep
    with StreamGear(output=str(tmp_path), format=format, **stream_params) as streamer:
        while True:
            frame = stream.read()
            # check if frame is None
            if frame is None:
                break
            if conversion == "COLOR_BGR2RGBA":
                with pytest.warns(DeprecationWarning):
                    streamer.stream(frame, rgb_mode=True)
            else:
                streamer.stream(frame)
        stream.stop()
    asset_file = [
        os.path.join(tmp_path, f)
        for f in os.listdir(tmp_path)
        if f.endswith(".mpd" if format == "dash" else ".m3u8")
    ]
    assert len(asset_file) == 1, "Failed to create asset file!"
    check_valid_output(asset_file[0], format)
    # browsers can't play High 4:4:4 (former `libx264rgb`)
    assert all("yuv420p" in x and "4:4:4" not in x for x in video_lines(asset_file[0]))


@pytest.mark.parametrize("format", ["dash", "hls"])
def test_rtf_livestream(format, tmp_path):
    """
    Testing Real-Time Frames Mode with livestream.
    """
    stream = CamGear(source=return_testvideo_path(), THREAD_TIMEOUT=300).start()
    assets_file_path = output_path(tmp_path, format)
    streamer = StreamGear(
        output=assets_file_path, format=format, **{"-livestream": True, "-enable_force_termination": True}
    )
    while True:
        frame = stream.read()
        # check if frame is None
        if frame is None:
            break
        streamer.stream(frame)
    stream.stop()
    with pytest.warns(DeprecationWarning):
        streamer.terminate()
    assert os.path.isfile(assets_file_path)


@pytest.mark.parametrize("format", ["dash", "hls"])
def test_input_framerate_rtf(format, tmp_path):
    """
    Testing "-input_framerate" parameter provided by StreamGear
    """
    assets_file_path = output_path(tmp_path, format)
    stream_params = {
        "-clear_prev_assets": True,
        "-input_framerate": 30,  # test clip is 25fps, the default
        "-vcodec": "copy",
        "-vf": "format=yuv420p",
        "-aspect": "4:3",
    }
    stream = cv2.VideoCapture(return_testvideo_path())  # Open stream
    streamer = StreamGear(output=assets_file_path, format=format, logging=True, **stream_params)
    while True:
        (grabbed, frame) = stream.read()
        if not grabbed:
            break
        streamer.stream(frame)
    stream.release()
    streamer.close()
    assert all("30 fps" in x for x in video_lines(assets_file_path))


@pytest.mark.parametrize(
    "stream_params, format",
    [
        (
            {
                "-clear_prev_assets": True,
                "-bpp": 0.2000,
                "-gop": 125,
                "-vcodec": "libx265",
                "-hls_segment_type": "invalid",
                "-hls_init_time": -223.2,
                "-hls_flags": 94884,
                "-hls_list_size": -4.3,
                "-hls_time": -4758.56,
                "-remove_at_exit": 4.56,
                "-livestream": True,
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
                "-enable_force_termination": "invalid",
            },
            "hls",
        ),
        (
            {
                "-clear_prev_assets": True,
                "-bpp": 0.2000,
                "-gop": 125,
                "-vcodec": "libx265",
                "-window_size": -456.4,
                "-extra_window_size": -354.45,
                "-remove_at_exit": -34.34,
                "-seg_duration": -334.23,
                "-livestream": True,
            },
            "dash",
        ),
        (
            {
                "-clear_prev_assets": True,
                "-seg_duration": 2.5,  # fractional durations are valid
                "-bpp": "unknown",
                "-gop": 0,
                "-s:v:0": "unknown",
                "-b:v:0": "unknown",
                "-b:a:0": "unknown",
            },
            "dash",
        ),
    ],
)
def test_params(stream_params, format, tmp_path):
    """
    Testing "-stream_params" parameters by StreamGear
    """
    assets_file_path = output_path(tmp_path, format)
    if format == "hls":
        stream_params = dict(stream_params, **{"-hls_base_url": str(tmp_path) + os.sep})
    stream = cv2.VideoCapture(return_testvideo_path())  # Open stream
    streamer = StreamGear(output=assets_file_path, format=format, logging=True, **stream_params)
    while True:
        (grabbed, frame) = stream.read()
        if not grabbed:
            break
        streamer.stream(frame)
    stream.release()
    streamer.close()
    assert streamer.returncode == 0
    if not stream_params.get("-livestream"):
        check_valid_output(assets_file_path, format)


def test_rtf_user_primary_stream(tmp_path):
    """
    Testing user-defined `-s:v:0`/`-b:v:0` are honoured for the primary stream
    """
    assets_file_path = output_path(tmp_path, "dash")
    with StreamGear(output=assets_file_path, **{"-s:v:0": "320x240", "-b:v:0": "200k"}) as streamer:
        for _ in range(25):
            streamer.stream(np.zeros((480, 640, 3), np.uint8))
    _, reps = check_valid_mpd(assets_file_path)
    assert (reps[0].width, reps[0].height, reps[0].bandwidth) == (320, 240, 200000)


@pytest.mark.parametrize(
    "stream_params, format",
    [
        ({"-video_source": return_testvideo_path(fmt="vo"), "-audio": return_testvideo_path(fmt="ao")}, "dash"),
        ({"-video_source": return_testvideo_path(fmt="vo"), "-audio": return_testvideo_path(fmt="ao")}, "hls"),
        # A/V source with external audio: external replaces source audio in both formats
        ({"-video_source": return_testvideo_path(), "-audio": return_testvideo_path(fmt="ao")}, "dash"),
        ({"-video_source": return_testvideo_path(), "-audio": return_testvideo_path(fmt="ao")}, "hls"),
    ],
)
def test_audio(stream_params, format, tmp_path):
    """
    Testing external audio for stream.
    """
    assets_file_path = output_path(tmp_path, format)
    streamer = StreamGear(output=assets_file_path, format=format, logging=True, **stream_params)
    streamer.transcode_source()
    streamer.close()
    check_valid_output(assets_file_path, format)
    assert len(audio_lines(assets_file_path)) == 1


def test_rtf_external_audio(tmp_path):
    """
    Testing Real-time Frames Mode with an infinite external audio source: every frame is kept and `close()` returns
    """
    assets_file_path = output_path(tmp_path, "dash")
    streamer = StreamGear(output=assets_file_path, **{"-audio": ["-f", "lavfi", "-i", "sine"]})
    for i in range(50):
        streamer.stream(np.full((240, 320, 3), i * 5, np.uint8))
    start = time.time()
    streamer.close(timeout=10)
    assert time.time() - start < 5 and streamer.returncode == 0
    meta = probe_video(assets_file_path)
    assert meta["source_has_audio"]
    assert meta["source_duration_sec"] == pytest.approx(2.0, abs=0.1)


@pytest.mark.parametrize(
    "format, stream_params",
    [
        (
            "dash",
            {
                "-clear_prev_assets": True,
                "-video_source": return_testvideo_path(fmt="vo"),
                "-streams": [
                    {"-video_bitrate": "unknown"},  # Invalid Stream 1
                    {"-resolution": "unxun"},  # Invalid Stream 2
                    {"-resolution": "640x480", "-video_bitrate": "unknown"},  # Invalid Stream 3
                    {"-resolution": "640x480", "-framerate": "unknown"},  # Invalid Stream 4
                    {"-resolution": "320x240", "-framerate": 20.0},  # Stream: 320x240 at 20fps framerate
                ],
            },
        ),
        (
            "hls",
            {
                "-clear_prev_assets": True,
                "-video_source": return_testvideo_path(fmt="vo"),
                "-streams": [
                    {"-video_bitrate": "unknown"},  # Invalid Stream 1
                    {"-resolution": "unxun"},  # Invalid Stream 2
                    {"-resolution": "640x480", "-video_bitrate": "unknown"},  # Invalid Stream 3
                    {"-resolution": "640x480", "-framerate": "unknown"},  # Invalid Stream 4
                    {"-resolution": "320x240", "-framerate": 20.0},  # Stream: 320x240 at 20fps framerate
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
                    {"-resolution": "640x480", "-video_bitrate": "850k", "-audio_bitrate": "128k"},
                    {"-resolution": "320x240", "-framerate": 20.0, "-audio_bitrate": 64000},  # invalid audio bitrate
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
                    {"-resolution": "640x480", "-video_bitrate": "850k", "-audio_bitrate": "128k"},
                    {"-resolution": "320x240", "-framerate": 20.0},
                ],
            },
        ),
        (
            "dash",
            {
                "-clear_prev_assets": True,
                "-video_source": return_testvideo_path(),
                "-streams": [{"-resolution": "960x540", "-video_bitrate": "1350k"}],
            },
        ),
        (
            "hls",
            {
                "-clear_prev_assets": True,
                "-video_source": return_testvideo_path(),
                "-streams": [{"-resolution": "960x540", "-video_bitrate": "1350k"}],
            },
        ),
    ],
)
def test_multistreams(format, stream_params, tmp_path):
    """
    Testing Support for additional Secondary Streams of variable bitrate or spatial resolutions.
    """
    assets_file_path = output_path(tmp_path, format, name="asset_test")
    expected = ["1280x720"] + [
        s["-resolution"] for s in stream_params["-streams"] if s.get("-resolution", "").startswith(("960", "320"))
    ] + [
        s["-resolution"] for s in stream_params["-streams"] if s.get("-video_bitrate") == "850k"
    ]
    streamer = StreamGear(output=assets_file_path, format=format, logging=True, **stream_params)
    streamer.transcode_source()
    streamer.close()
    resolutions = [x for line in video_lines(assets_file_path) for x in expected if x in line]
    assert sorted(resolutions) == sorted(expected)
    # streams with own framerate get it applied
    assert all("20 fps" in x for x in video_lines(assets_file_path) if "320x240" in x)


@pytest.mark.parametrize("format", ["dash", "hls"])
def test_ladder_bitrates(format, tmp_path):
    """
    Testing every rendition stays within its declared bitrate (no default `-crf` overriding them),
    and `-streams` dicts are not mutated, so they can be reused.
    """
    source = return_testvideo_path(fmt="vo")
    streams = [
        {"-resolution": "640x360", "-video_bitrate": "150k"},
        {"-resolution": "320x180", "-framerate": 10},  # bitrate computed from the framerate
    ]
    snapshot = [dict(s) for s in streams]
    for run in range(2):
        out = tmp_path / str(run)
        out.mkdir()
        assets_file_path = output_path(out, format)
        streamer = StreamGear(output=assets_file_path, format=format, **{"-video_source": source, "-streams": streams})
        streamer.transcode_source()
        streamer.close()
        assert streams == snapshot
        lines = video_lines(assets_file_path)
        assert len(lines) == 3 and "10 fps" in lines[-1]
    duration = probe_video(source)["source_duration_sec"]
    if format == "dash":
        declared = [rep.bandwidth for rep in check_valid_mpd(assets_file_path)[1]]
        prefixes = ["init-stream{}".format(i) for i in range(3)], ["chunk-stream{}-".format(i) for i in range(3)]
        prefixes = list(zip(*prefixes, strict=True))
    else:
        with open(assets_file_path) as f:
            declared = [pl.stream_info.bandwidth for pl in m3u8.loads(f.read()).playlists]
        prefixes = [("chunk-stream{}-".format(i),) for i in range(3)]
    for bandwidth, prefix in zip(declared, prefixes, strict=True):
        size = sum(os.path.getsize(out / f) for f in os.listdir(out) if f.startswith(prefix))
        actual = size * 8 / duration
        assert actual < 1.25 * bandwidth, "{}: actual {} vs declared {}".format(prefix, actual, bandwidth)


@pytest.mark.parametrize("format", ["dash", "hls"])
@pytest.mark.parametrize("clip", ["cover", "subtitle", "pcm", "opus"])
def test_source_tracks(clip, format, clips, tmp_path):
    """
    Testing cover art, extra audio tracks and non-AAC audio sources: one video + one AAC audio per rendition
    """
    assets_file_path = output_path(tmp_path, format)
    with StreamGear(output=assets_file_path, format=format, **{"-video_source": clips[clip]}) as streamer:
        streamer.transcode_source()
    lines = stream_lines(assets_file_path)
    assert [x for x in lines if "Video:" in x and "320x240" in x and "h264" in x] == video_lines(assets_file_path)
    assert len(video_lines(assets_file_path)) == 1
    assert len(audio_lines(assets_file_path)) == 1 and "Audio: aac" in audio_lines(assets_file_path)[0]


def test_rotated_source(clips, tmp_path):
    """
    Testing rotated (portrait) sources keep their display orientation
    """
    assets_file_path = output_path(tmp_path, "dash")
    with StreamGear(output=assets_file_path, **{"-video_source": clips["rotated"], "-streams": [{"-resolution": "120x160", "-video_bitrate": "100k"}]}) as streamer:
        streamer.transcode_source()
    lines = video_lines(assets_file_path)
    assert "240x320" in lines[0] and "120x160" in lines[1]


def test_odd_resolution(tmp_path):
    """
    Testing odd `-resolution` is rounded up to even, instead of failing the transcode
    """
    assets_file_path = output_path(tmp_path, "dash")
    with StreamGear(output=assets_file_path, **{"-video_source": return_testvideo_path(fmt="vo"), "-streams": [{"-resolution": "427x240", "-video_bitrate": "300k"}]}) as streamer:
        streamer.transcode_source()
    assert "428x240" in video_lines(assets_file_path)[1]


@pytest.mark.parametrize(
    "clip, resolution, warned",
    [("anamorphic", "640x360", False), ("anamorphic", "480x360", True), ("hdr", None, True)],
)
def test_source_warnings(clip, resolution, warned, clips, tmp_path, monkeypatch):
    """
    Testing aspect check uses display aspect (anamorphic sources), and HDR sources warn
    """
    messages = []
    monkeypatch.setattr(streamgear_module.logger, "warning", lambda msg, *a, **k: messages.append(msg))
    commands = capture_commands(monkeypatch)
    stream_params = {"-video_source": clips[clip]}
    resolution and stream_params.update({"-streams": [{"-resolution": resolution, "-video_bitrate": "300k"}]})
    StreamGear(output=output_path(tmp_path, "dash"), **stream_params).transcode_source()
    assert commands
    keyword = "aspect ratio" if resolution else "HDR"
    assert any(keyword in m for m in messages) == warned


def test_ss_command(monkeypatch, tmp_path):
    """
    Testing Single-Source Mode FFmpeg command: maps, capped bitrates, x265 params, tuning
    """
    commands = capture_commands(monkeypatch)
    StreamGear(
        output=output_path(tmp_path, "dash"),
        **{
            "-video_source": return_testvideo_path(),
            "-audio": return_testvideo_path(fmt="ao"),
            "-vcodec": "libx265",
            "-x265-params": "crf=28",
            "-streams": [{"-resolution": "640x360", "-video_bitrate": "500k", "-framerate": 10}],
        }
    ).transcode_source()
    cmd = commands[-1]
    joined = " ".join(cmd)
    assert cmd.count("-x265-params") == 1 and cmd[cmd.index("-x265-params") + 1] == "crf=28:keyint=50:min-keyint=50"
    assert "-crf" not in cmd and "-tune" not in cmd and "-re" not in cmd
    assert joined.count("-map 0:V:0") == 2 and joined.count("-map 1:a:0") == 2
    assert "-maxrate:v:1 500k -bufsize:v:1 1000k" in joined and "-r:v:1 10" in joined
    assert "-force_key_frames expr:gte(t,n_forced*2)" in joined
    assert "-af apad" in joined and "-shortest" in cmd
    for dead in ["-movflags", "-allowed_extensions", "-hls_allow_cache"]:
        assert dead not in cmd

    # user `-crf` is kept (capped by maxrate); live file sources are read at native speed
    StreamGear(
        output=output_path(tmp_path, "hls"),
        format="hls",
        **{"-video_source": return_testvideo_path(), "-crf": 23, "-livestream": True}
    ).transcode_source()
    cmd = commands[-1]
    assert cmd[cmd.index("-crf") + 1] == "23" and "-maxrate:v:0" in cmd
    assert "-re" in cmd and "-tune" in cmd and "-hls_playlist_type" not in cmd
    assert "split_by_time" not in cmd[cmd.index("-hls_flags") + 1]


def test_ss_rerun(tmp_path):
    """
    Testing `transcode_source()` builds the same command when run again
    """
    streamer = StreamGear(
        output=output_path(tmp_path, "dash"),
        **{"-video_source": return_testvideo_path(), "-streams": [{"-resolution": "320x180", "-video_bitrate": "100k"}]}
    )
    for _ in range(2):
        streamer.transcode_source()
        assert len(video_lines(output_path(tmp_path, "dash"))) == 2
    streamer.close()


def test_ss_progress_callback(tmp_path):
    """
    Testing Single-Source Mode `-progress_callback` and `returncode`
    """
    progress = []
    streamer = StreamGear(
        output=output_path(tmp_path, "dash"),
        **{
            "-video_source": return_testvideo_path(),
            "-vcodec": "copy",
            "-progress_callback": lambda done, total: progress.append((done, total)),
        },
    )
    streamer.transcode_source()
    streamer.close()
    assert streamer.returncode == 0
    assert progress and progress[-1][1] > 0
    assert progress[-1][0] == pytest.approx(progress[-1][1], abs=0.5)


def test_ss_progress_callback_raises(tmp_path):
    """
    Testing a raising `-progress_callback` is dropped, and the transcode completes
    """
    calls = []

    def callback(done, total):
        calls.append(done)
        raise RuntimeError("boom")

    streamer = StreamGear(
        output=output_path(tmp_path, "dash"),
        **{"-video_source": return_testvideo_path(), "-vcodec": "copy", "-progress_callback": callback},
    )
    streamer.transcode_source()
    streamer.close()
    assert streamer.returncode == 0 and len(calls) == 1
    check_valid_output(output_path(tmp_path, "dash"), "dash")


def test_ss_close_cancels_transcode(tmp_path):
    """
    Testing `close()` from another thread stops a running `transcode_source()` without raising
    """
    streamer = StreamGear(
        output=output_path(tmp_path, "dash"),
        **{
            "-video_source": return_testvideo_path(),
            "-vf": "realtime,format=yuv420p",  # throttle 4sec clip to wall-clock speed
        },
    )
    errors = queue.Queue()

    def transcode():
        try:
            streamer.transcode_source()
        except Exception as e:
            errors.put(e)

    worker = threading.Thread(target=transcode)
    worker.start()
    time.sleep(2)
    start = time.time()
    streamer.close()
    worker.join(10)
    assert not worker.is_alive() and errors.empty()
    assert time.time() - start < 10
    assert streamer.returncode not in (None, 0)


def test_rtf_window_sizes(monkeypatch, tmp_path):
    """
    Testing Real-time Frames Mode passes `-window_size` & `-extra_window_size` as given
    """
    commands = []
    popen = subprocess.Popen
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: commands.append(cmd) or popen(cmd, **kw))
    streamer = StreamGear(
        output=output_path(tmp_path, "dash"),
        **{"-livestream": True, "-window_size": 3, "-extra_window_size": 7},
    )
    streamer.stream(np.zeros((48, 64, 3), np.uint8))
    streamer.close()
    cmd = commands[-1]
    assert cmd[cmd.index("-window_size") + 1] == "3"
    assert cmd[cmd.index("-extra_window_size") + 1] == "7"
    assert cmd[cmd.index("-seg_duration") + 1] == "4"


@pytest.mark.parametrize("stream_params", [{"-capture_stderr": True}, {}])
def test_rtf_ffmpeg_death(stream_params, tmp_path):
    """
    Testing Real-time Frames Mode surfaces FFmpeg's stderr on early death, and `close()` neither raises
    nor leaks `BrokenPipeError`
    """
    streamer = StreamGear(output=output_path(tmp_path, "dash"), **{"-invalid_flag": "1"}, **stream_params)
    with pytest.raises(ValueError, match="invalid_flag"):
        for _ in range(500):
            streamer.stream(np.zeros((48, 64, 3), np.uint8))
            time.sleep(0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        streamer.close()
        del streamer
    import gc

    gc.collect()
