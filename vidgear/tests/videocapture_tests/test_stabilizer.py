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

import numpy as np
import pytest

from vidgear.gears.helper import logger_handler
from vidgear.gears.stabilizer import ASWStabilizer, Stabilizer, StabilizerMode

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


def test_factory_kalman_not_implemented():
    """`StabilizerMode.KALMAN` is reserved; must raise NotImplementedError for now."""
    with pytest.raises(NotImplementedError):
        Stabilizer(mode=StabilizerMode.KALMAN)


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
