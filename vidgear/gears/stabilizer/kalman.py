# This is a modified algorithm/code based on `videostabKalman.cpp` by Chen Jia
# (chenjia2013@foxmail.com), itself derived from `Simple video stabilization using OpenCV`
# published on February 20, 2014 by nghiaho12 (http://nghiaho.com/?p=2093)

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
import numpy as np

from .base import _StabilizerBase, logger


class KalmanStabilizer(_StabilizerBase):
    """
    Zero-Latency Kalman-filter Video Stabilizer.

    Accumulates per-frame rigid motion `[dx, dy, da]` into a camera trajectory and
    smooths it online with a vectorized constant-position Kalman filter (one
    independent filter per axis). Each incoming frame is warped immediately from
    its measured trajectory onto the filtered one, so no frames are buffered and
    every `stabilize()` call returns a frame - suitable for live streams.
    """

    def __init__(
        self,
        process_noise: float = 4e-3,
        measurement_noise: float = 0.25,
        border_type: str = "black",
        border_size: int = 0,
        crop_n_zoom: bool = False,
        logging: bool = False,
    ):
        """
        Parameters:
            process_noise (float): process noise covariance `Q`. Larger = follows camera motion faster, less smoothing.
            measurement_noise (float): measurement noise covariance `R`. Larger = heavier smoothing, more lag on panning.
            border_type (str): changes the extended border type.
            border_size (int): enables and set the value for extended border size to reduce the black borders.
            crop_n_zoom (bool): enables cropping and zooming of frames(to original size) to reduce the black borders.
            logging (bool): enables/disables logging.
        """
        super().__init__(
            border_type=border_type,
            border_size=border_size,
            crop_n_zoom=crop_n_zoom,
            logging=logging,
        )
        # scalar or per-axis `[x, y, a]` covariances; broadcast against state
        self.__Q = np.asarray(process_noise, dtype=np.float64)
        self.__R = np.asarray(measurement_noise, dtype=np.float64)
        self._logging and logger.debug(
            "Setting Kalman noise: Q={}, R={}".format(self.__Q, self.__R)
        )
        self.__reset()

    def __reset(self):
        """
        Resets tracking + filter state to initial guesses.
        """
        self.__previous_gray = None  # previous gray frame
        self.__previous_keypoints = None  # previous GFTT keypoints
        self.__trajectory = np.zeros(3, dtype=np.float64)  # measured [x, y, a]
        self.__X = np.zeros(3, dtype=np.float64)  # posteriori state estimate
        self.__P = np.ones(3, dtype=np.float64)  # posteriori error covariance

    def stabilize(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Takes an unstabilized video frame, and returns its stabilized version
        immediately (zero-latency).
        """
        if frame is None:
            return

        gray = self._to_gray(frame)

        if self.__previous_gray is None:
            # first frame: nothing to track against yet
            self._frame_height, self._frame_width = frame.shape[:2]
            if self._crop_n_zoom:
                self._frame_size = frame.shape[:2]
        else:
            # measurement: accumulated prev->cur motion
            self.__trajectory += self._estimate_motion(
                self.__previous_gray, self.__previous_keypoints, gray
            )
            # time update (prediction): X_ = X, P_ = P + Q
            p_prior = self.__P + self.__Q
            # measurement update (correction)
            gain = p_prior / (p_prior + self.__R)
            self.__X += gain * (self.__trajectory - self.__X)
            self.__P = (1.0 - gain) * p_prior

        self.__previous_gray = gray
        self.__previous_keypoints = self._detect_keypoints(gray)

        # warp current frame from measured onto smoothed trajectory
        dx, dy, da = self.__X - self.__trajectory
        return self._apply_warp(frame, dx, dy, da)

    def clean(self) -> None:
        """
        Cleans KalmanStabilizer resources.
        """
        super().clean()
        self.__reset()
