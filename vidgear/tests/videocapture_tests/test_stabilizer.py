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
from itertools import pairwise

import cv2
import numpy as np
import pytest

from vidgear.gears.helper import logger_handler
from vidgear.gears.stabilizer import (
    ASWStabilizer,
    KalmanStabilizer,
    Stabilizer,
    StabilizerMode,
)

# define test logger
logger = log.getLogger("Test_stabilizer")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


def test_factory_default_is_asw():
    """`Stabilizer()` with no args returns ASWStabilizer."""
    s = Stabilizer()
    assert isinstance(s, ASWStabilizer)


def test_factory_explicit_asw():
    """Explicit `mode=StabilizerMode.ASW` returns ASWStabilizer and forwards kwargs."""
    s = Stabilizer(mode=StabilizerMode.ASW, smoothing_radius=5, border_size=10)
    assert isinstance(s, ASWStabilizer)


def test_factory_kalman():
    """`StabilizerMode.KALMAN` returns KalmanStabilizer and forwards kwargs."""
    s = Stabilizer(
        mode=StabilizerMode.KALMAN, process_noise=1e-3, measurement_noise=0.5
    )
    assert isinstance(s, KalmanStabilizer)


def test_factory_bad_mode_type():
    """Non-enum `mode` argument raises TypeError."""
    with pytest.raises(TypeError):
        Stabilizer(mode="asw")


def test_asw_stabilizes_frames():
    """ASW produces stabilized frames once smoothing window fills."""
    s = Stabilizer(smoothing_radius=5)
    rng = np.random.default_rng(0)
    out = []
    for _ in range(12):
        r = s.stabilize(
            rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
        )
        if r is not None:
            out.append(r)
    assert len(out) > 0
    assert out[0].shape == (120, 160, 3)
    s.clean()


def test_asw_border_padding_shape():
    """`border_size` enlarges output dims by 2 * border_size on each axis."""
    s = Stabilizer(smoothing_radius=5, border_size=10, border_type="reflect")
    rng = np.random.default_rng(1)
    got_frame = False
    for _ in range(12):
        r = s.stabilize(
            rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
        )
        if r is not None:
            assert r.shape == (140, 180, 3)
            got_frame = True
    assert got_frame
    s.clean()


def test_asw_transforms_buffer_is_bounded():
    """
    Regression for issue #363: `__transforms` must stay bounded over a long
    stream. Previously a plain list that grew unbounded with frame count.
    Bound is `2 * smoothing_radius + 1` by design.
    """
    smoothing_radius = 5
    s = Stabilizer(smoothing_radius=smoothing_radius)
    rng = np.random.default_rng(3)
    # name-mangled private attrs
    transforms = s._ASWStabilizer__transforms
    frame_queue = s._ASWStabilizer__frame_queue
    for _ in range(500):
        s.stabilize(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
    assert len(transforms) <= 2 * smoothing_radius + 1
    assert transforms.maxlen == 2 * smoothing_radius + 1
    assert len(frame_queue) <= smoothing_radius
    s.clean()


def test_asw_long_stream_emits_continuously():
    """
    After the smoothing window fills, every subsequent `stabilize` call must
    return a frame. Regression against the original counter-based buildup
    check that broke once the bounded refactor removed the global index.
    """
    smoothing_radius = 5
    s = Stabilizer(smoothing_radius=smoothing_radius)
    rng = np.random.default_rng(4)
    emitted = 0
    total = 50
    for _ in range(total):
        if s.stabilize(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)) is not None:
            emitted += 1
    # first `smoothing_radius` calls warm up, rest emit
    assert emitted == total - smoothing_radius
    s.clean()


def test_asw_crop_n_zoom_preserves_shape():
    """`crop_n_zoom=True` crops borders and resizes back to input dims."""
    s = Stabilizer(smoothing_radius=5, border_size=10, crop_n_zoom=True)
    rng = np.random.default_rng(2)
    got_frame = False
    for _ in range(12):
        r = s.stabilize(
            rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
        )
        if r is not None:
            assert r.shape == (120, 160, 3)
            got_frame = True
    assert got_frame
    s.clean()


def _jittered_frames(n, seed=5):
    """Textured scene shifted by random per-frame jitter (simulated shaky cam)."""
    rng = np.random.default_rng(seed)
    scene = cv2.GaussianBlur(
        rng.integers(0, 255, (240, 320, 3), dtype=np.uint8), (5, 5), 0
    )
    for _ in range(n):
        tx, ty = rng.uniform(-6, 6, 2)
        yield cv2.warpAffine(
            scene,
            np.float32([[1, 0, tx], [0, 1, ty]]),
            (320, 240),
            borderMode=cv2.BORDER_REFLECT,
        )


def _mean_shift(frames):
    """Mean frame-to-frame translation magnitude via phase correlation."""
    gray = [np.float32(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames]
    return np.mean([np.hypot(*cv2.phaseCorrelate(a, b)[0]) for a, b in pairwise(gray)])


@pytest.mark.parametrize(
    "kwargs, shape",
    [
        ({}, (240, 320, 3)),
        ({"border_size": 10, "border_type": "reflect"}, (260, 340, 3)),
        ({"border_size": 10, "crop_n_zoom": True}, (240, 320, 3)),
    ],
)
def test_kalman_zero_latency(kwargs, shape):
    """Kalman emits a frame on every call (no warm-up) with expected shape."""
    s = Stabilizer(mode=StabilizerMode.KALMAN, **kwargs)
    for f in _jittered_frames(20):
        r = s.stabilize(f)
        assert r is not None and r.shape == shape
    assert s.stabilize(None) is None
    s.clean()


def test_kalman_reduces_jitter():
    """Stabilized output must shake noticeably less than jittered input."""
    frames = list(_jittered_frames(60))
    s = KalmanStabilizer()
    out = [s.stabilize(f) for f in frames]
    s.clean()
    # skip filter warm-up
    assert _mean_shift(out[10:]) < 0.5 * _mean_shift(frames[10:])
