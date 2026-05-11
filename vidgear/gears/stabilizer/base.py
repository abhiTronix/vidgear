# This is a modified algorithm/code based on findings of `Simple video stabilization using OpenCV`
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
import logging as log

import cv2
import numpy as np

# import helper packages
from ..helper import (
    check_CV_version,
    logcurr_vidgear_ver,
    logger_handler,
    retrieve_best_interpolation,
)

# define logger
logger = log.getLogger("Stabilizer")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


class _StabilizerBase:
    """
    Internal base class shared by all Stabilizer implementations
    (ASW, Kalman, ...). Holds common config: border handling, crop+zoom,
    CLAHE, interpolation, OpenCV version, and the affine warp routine.
    Subclasses implement `stabilize()` and the per-frame transform estimation.
    """

    def __init__(
        self,
        border_type: str = "black",
        border_size: int = 0,
        crop_n_zoom: bool = False,
        logging: bool = False,
    ):
        # enable logging if specified
        self._logging = logging if isinstance(logging, bool) else False

        # print current version
        logcurr_vidgear_ver(logging=self._logging)

        # input frame dims placeholders
        self._frame_height, self._frame_width = 0, 0
        # crop margin (also flag); 0 = disabled
        self._crop_n_zoom = 0

        # if check if crop_n_zoom defined
        if crop_n_zoom and border_size:
            self._crop_n_zoom = border_size  # crops and zoom frame to original size
            self._border_size = 0  # zero out border size
            self._frame_size = None  # handles frame size for zooming
            self._logging and logger.debug(
                "Setting Cropping margin {} pixels".format(border_size)
            )
        else:
            # Add output borders to frame
            self._border_size = border_size
            self._logging and border_size and logger.debug(
                "Setting Border size {} pixels".format(border_size)
            )

        # define valid border modes
        border_modes = {
            "black": cv2.BORDER_CONSTANT,
            "reflect": cv2.BORDER_REFLECT,
            "reflect_101": cv2.BORDER_REFLECT_101,
            "replicate": cv2.BORDER_REPLICATE,
            "wrap": cv2.BORDER_WRAP,
        }
        # choose valid border_mode from border_type
        if border_type in border_modes:
            if not crop_n_zoom:
                self._border_mode = border_modes[border_type]
                self._logging and border_type != "black" and logger.info(
                    "Setting Border type: {}".format(border_type)
                )
            else:
                self._logging and border_type != "black" and logger.debug(
                    "Setting border type is disabled if cropping is enabled!"
                )
                self._border_mode = border_modes["black"]
        else:
            self._logging and logger.debug("Invalid input border type!")
            self._border_mode = border_modes["black"]  # reset to default mode

        # adaptive histogram equalization (shared by all modes)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # OpenCV major version
        self._cv2_version = check_CV_version()
        # best available interpolation for resize
        self._interpolation = retrieve_best_interpolation(
            ["INTER_LINEAR_EXACT", "INTER_LINEAR", "INTER_AREA"]
        )

    def _apply_warp(
        self, frame: np.ndarray, dx: float, dy: float, da: float
    ) -> np.ndarray:
        """
        Applies affine transform (translation `dx`,`dy`; rotation `da`) to `frame`,
        with border padding + optional crop-and-zoom. Shared across all modes.
        """
        # create border around extracted frame w.r.t border_size
        bordered_frame = cv2.copyMakeBorder(
            frame,
            top=self._border_size,
            bottom=self._border_size,
            left=self._border_size,
            right=self._border_size,
            borderType=self._border_mode,
            value=[0, 0, 0],
        )
        # create alpha channel
        alpha_bordered_frame = cv2.cvtColor(bordered_frame, cv2.COLOR_BGR2BGRA)
        alpha_bordered_frame[:, :, 3] = 0
        alpha_bordered_frame[
            self._border_size : self._border_size + self._frame_height,
            self._border_size : self._border_size + self._frame_width,
            3,
        ] = 255

        # build 2x3 affine transformation matrix
        transform = np.zeros((2, 3), np.float32)
        transform[0, 0] = np.cos(da)
        transform[0, 1] = -np.sin(da)
        transform[1, 0] = np.sin(da)
        transform[1, 1] = np.cos(da)
        transform[0, 2] = dx
        transform[1, 2] = dy

        # warp
        frame_wrapped = cv2.warpAffine(
            alpha_bordered_frame,
            transform,
            alpha_bordered_frame.shape[:2][::-1],
            borderMode=self._border_mode,
        )

        # drop alpha channel
        frame_stabilized = frame_wrapped[:, :, :3]

        # crop and zoom
        if self._crop_n_zoom:
            frame_cropped = frame_stabilized[
                self._crop_n_zoom : -self._crop_n_zoom,
                self._crop_n_zoom : -self._crop_n_zoom,
            ]
            frame_stabilized = cv2.resize(
                frame_cropped,
                self._frame_size[::-1],
                interpolation=self._interpolation,
            )
        return frame_stabilized

    def stabilize(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Subclasses must implement: takes an unstabilized frame, returns a
        stabilized one (or `None` while buffering/warming up).
        """
        raise NotImplementedError

    def clean(self) -> None:
        """
        Releases per-mode resources. Subclasses override to clear their state.
        """
        logger.debug("Closing the Stabilizer class")
