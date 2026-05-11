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
from collections import deque

import cv2
import numpy as np

from .base import _StabilizerBase, logger


class ASWStabilizer(_StabilizerBase):
    """
    Average Sliding-Window Video Stabilizer.

    Tracks a salient feature array over a fixed-size window of past frames and
    cancels perturbations relative to a normalized box-filter-smoothed path.
    Relies on **Threaded Queue mode** for error-free & ultra-fast frame handling.
    """

    def __init__(
        self,
        smoothing_radius: int = 25,
        border_type: str = "black",
        border_size: int = 0,
        crop_n_zoom: bool = False,
        logging: bool = False,
    ):
        """
        Parameters:
            smoothing_radius (int): alter averaging window size.
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

        # bounded buffers (size = smoothing window)
        self.__frame_queue = deque(maxlen=smoothing_radius)
        self.__frame_queue_indexes = deque(maxlen=smoothing_radius)

        # ASW-specific state
        self.__smoothing_radius = smoothing_radius  # averaging window
        self.__smoothed_path = None  # box-filter-smoothed path
        self.__path = None  # cumulative sum of previous_2_current transforms
        self.__transforms = []  # previous_2_current transforms [dx,dy,da]
        self.__frame_transforms_smoothed = None  # smoothed transforms per frame
        self.__previous_gray = None  # previous gray frame
        self.__previous_keypoints = None  # previous GFTT keypoints

        # normalized box filter
        self.__box_filter = np.ones(smoothing_radius) / smoothing_radius

    def stabilize(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Takes an unstabilized video frame, and returns a stabilized one
        (or `None` while the smoothing window is still filling).
        """
        if frame is None:
            return

        # save frame size for zooming
        if self._crop_n_zoom and self._frame_size is None:
            self._frame_size = frame.shape[:2]

        # initiate transformations capturing
        if not self.__frame_queue:
            # for first frame
            previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            previous_gray = self._clahe.apply(previous_gray)
            self.__previous_keypoints = cv2.goodFeaturesToTrack(
                previous_gray,
                maxCorners=200,
                qualityLevel=0.05,
                minDistance=30.0,
                blockSize=3,
                mask=None,
                useHarrisDetector=False,
                k=0.04,
            )
            self._frame_height, self._frame_width = frame.shape[:2]
            self.__frame_queue.append(frame)
            self.__frame_queue_indexes.append(0)
            self.__previous_gray = previous_gray[:]

        elif self.__frame_queue_indexes[-1] < self.__smoothing_radius - 1:
            # buffering remaining frames in window
            self.__frame_queue.append(frame)
            self.__frame_queue_indexes.append(self.__frame_queue_indexes[-1] + 1)
            self.__generate_transformations()
        else:
            # window full → start emitting stabilized frames
            self.__frame_queue.append(frame)
            self.__frame_queue_indexes.append(self.__frame_queue_indexes[-1] + 1)
            self.__generate_transformations()
            # smooth path with normalized box filter (per axis)
            for i in range(3):
                self.__smoothed_path[:, i] = self.__box_filter_convolve(
                    (self.__path[:, i]), window_size=self.__smoothing_radius
                )
            # deviation of path from smoothed path
            deviation = self.__smoothed_path - self.__path
            self.__frame_transforms_smoothed = self.frame_transform + deviation
            return self.__apply_transformations()

    def __generate_transformations(self):
        """
        Generates previous-to-current transformations [dx, dy, da] for the
        latest frame in the queue.
        """
        frame_gray = cv2.cvtColor(self.__frame_queue[-1], cv2.COLOR_BGR2GRAY)
        frame_gray = self._clahe.apply(frame_gray)

        transformation = None
        try:
            # Lucas-Kanade optical flow
            curr_kps, status, _error = cv2.calcOpticalFlowPyrLK(
                self.__previous_gray, frame_gray, self.__previous_keypoints, None
            )

            # keep only valid key-points
            valid_curr_kps = curr_kps[status == 1]
            valid_previous_keypoints = self.__previous_keypoints[status == 1]

            # affine estimate between previous_2_current key-points
            if self._cv2_version == 3:
                # backward compatibility with OpenCV3
                transformation = cv2.estimateRigidTransform(
                    valid_previous_keypoints, valid_curr_kps, False
                )
            else:
                transformation = cv2.estimateAffinePartial2D(
                    valid_previous_keypoints, valid_curr_kps
                )[0]
        except cv2.error:
            logger.warning("Video-Frame is too dark to generate any transformations!")
            transformation = None

        if transformation is not None:
            dx = transformation[0, 2]
            dy = transformation[1, 2]
            da = np.arctan2(transformation[1, 0], transformation[0, 0])
        else:
            dx = dy = da = 0

        self.__transforms.append([dx, dy, da])

        # cumulative path
        self.frame_transform = np.array(self.__transforms, dtype="float32")
        self.__path = np.cumsum(self.frame_transform, axis=0)
        self.__smoothed_path = np.copy(self.__path)

        # refresh GFTT keypoints for next iteration
        self.__previous_keypoints = cv2.goodFeaturesToTrack(
            frame_gray,
            maxCorners=200,
            qualityLevel=0.05,
            minDistance=30.0,
            blockSize=3,
            mask=None,
            useHarrisDetector=False,
            k=0.04,
        )
        self.__previous_gray = frame_gray[:]

    def __box_filter_convolve(self, path, window_size):
        """
        Applies *normalized linear box filter* to path w.r.t averaging window.
        """
        path_padded = np.pad(path, (window_size, window_size), "median")
        path_smoothed = np.convolve(path_padded, self.__box_filter, mode="same")
        path_smoothed = path_smoothed[window_size:-window_size]
        assert path.shape == path_smoothed.shape
        return path_smoothed

    def __apply_transformations(self):
        """
        Pops the oldest frame from the queue and applies its smoothed
        transformation via the shared affine warp.
        """
        queue_frame = self.__frame_queue.popleft()
        queue_frame_index = self.__frame_queue_indexes.popleft()

        # extracting Transformations w.r.t frame index
        dx = self.__frame_transforms_smoothed[queue_frame_index, 0]
        dy = self.__frame_transforms_smoothed[queue_frame_index, 1]
        da = self.__frame_transforms_smoothed[queue_frame_index, 2]

        return self._apply_warp(queue_frame, dx, dy, da)

    def clean(self) -> None:
        """
        Cleans ASWStabilizer resources.
        """
        if self.__frame_queue:
            logger.critical("Cleaning Resources...")
            self.__frame_queue.clear()
            self.__frame_queue_indexes.clear()
