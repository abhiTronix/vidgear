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
from enum import Enum

from .asw import ASWStabilizer
from .base import _StabilizerBase


class StabilizerMode(Enum):
    """
    Selects the video stabilization algorithm.
    """

    ASW = "asw"  # Average Sliding-Window (default)
    KALMAN = "kalman"  # reserved; implementation coming in a future release


def Stabilizer(
    mode: StabilizerMode = StabilizerMode.ASW,
    **kwargs,
) -> _StabilizerBase:
    """
    Factory that returns a stabilizer instance for the requested `mode`.

    Parameters:
        mode (StabilizerMode): selects the stabilization algorithm.
            Defaults to `StabilizerMode.ASW`.
        **kwargs: forwarded to the selected backend's constructor (e.g.
            `smoothing_radius`, `border_type`, `border_size`, `crop_n_zoom`,
            `logging`).

    Returns:
        An instance of the chosen stabilizer (subclass of `_StabilizerBase`).
    """
    if not isinstance(mode, StabilizerMode):
        raise TypeError(
            "[Stabilizer:ERROR] :: `mode` must be a `StabilizerMode` enum member, got `{}`.".format(
                type(mode).__name__
            )
        )

    if mode is StabilizerMode.ASW:
        return ASWStabilizer(**kwargs)

    if mode is StabilizerMode.KALMAN:
        raise NotImplementedError(
            "[Stabilizer:ERROR] :: `KalmanStabilizer` is not yet implemented; "
            "use `StabilizerMode.ASW` for now."
        )

    raise ValueError(
        "[Stabilizer:ERROR] :: Unsupported stabilizer mode `{}`.".format(mode)
    )


__all__ = ["ASWStabilizer", "Stabilizer", "StabilizerMode"]
