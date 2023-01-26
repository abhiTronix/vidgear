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
import cv2
import numpy as np
import logging as log
from collections import deque

# import helper packages
from .helper import (
    logger_handler,
    check_CV_version,
    retrieve_best_interpolation,
    logcurr_vidgear_ver,
)

# define logger
logger = log.getLogger("Stabilizer")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


class Stabilizer:
    """
    This is an auxiliary class that enables Video Stabilization for vidgear with minimalistic latency, and at the expense
    of little to no additional computational requirements.

    The basic idea behind it is to tracks and save the salient feature array for the given number of frames and then uses
    these anchor point to cancel out all perturbations relative to it for the incoming frames in the queue. This class relies
    heavily on **Threaded Queue mode** for error-free & ultra-fast frame handling.
    """

    def __init__(
        self,
        smoothing_radius=25,
        border_type="black",
        border_size=0,
        crop_n_zoom=False,
        logging=False,
    ):

        """
        This constructor method initializes the object state and attributes of the Stabilizer class.

        Parameters:
            smoothing_radius (int): alter averaging window size.
            border_type (str): changes the extended border type.
            border_size (int): enables and set the value for extended border size to reduce the black borders.
            crop_n_zoom (bool): enables croping and zooming of frames(to original size) to reduce the black borders.
            logging (bool): enables/disables logging.
        """
        # print current version
        logcurr_vidgear_ver(logging=logging)

        # initialize deques for handling input frames and its indexes
        self.__frame_queue = deque(maxlen=smoothing_radius)
        self.__frame_queue_indexes = deque(maxlen=smoothing_radius)

        # enable logging if specified
        self.__logging = False
        if logging:
            self.__logging = logging

        # define and create Adaptive histogram equalization (AHE) object for optimizations
        self.__clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # initialize global vars
        self.__smoothing_radius = smoothing_radius  # averaging window, handles the quality of stabilization at expense of latency and sudden panning
        self.__smoothed_path = None  # handles the smoothed path with box filter
        self.__path = None  # handles path i.e cumulative sum of pevious_2_current transformations along a axis
        self.__transforms = []  # handles pevious_2_current transformations [dx,dy,da]
        self.__frame_transforms_smoothed = None  # handles smoothed array of pevious_2_current transformations w.r.t to frames
        self.__previous_gray = None  # handles previous gray frame
        self.__previous_keypoints = (
            None  # handles previous detect_GFTTed keypoints w.r.t previous gray frame
        )
        self.__frame_height, self.frame_width = (
            0,
            0,
        )  # handles width and height of input frames
        self.__crop_n_zoom = 0  # handles cropping and zooms frames to reduce the black borders from stabilization being too noticeable.

        # if check if crop_n_zoom defined
        if crop_n_zoom and border_size:
            self.__crop_n_zoom = border_size  # crops and zoom frame to original size
            self.__border_size = 0  # zero out border size
            self.__frame_size = None  # handles frame size for zooming
            if logging:
                logger.debug("Setting Cropping margin {} pixels".format(border_size))
        else:
            # Add output borders to frame
            self.__border_size = border_size
            if self.__logging and border_size:
                logger.debug("Setting Border size {} pixels".format(border_size))

        # define valid border modes
        border_modes = {
            "black": cv2.BORDER_CONSTANT,
            "reflect": cv2.BORDER_REFLECT,
            "reflect_101": cv2.BORDER_REFLECT_101,
            "replicate": cv2.BORDER_REPLICATE,
            "wrap": cv2.BORDER_WRAP,
        }
        # choose valid border_mode from border_type
        if border_type in ["black", "reflect", "reflect_101", "replicate", "wrap"]:
            if not crop_n_zoom:
                # initialize global border mode variable
                self.__border_mode = border_modes[border_type]
                if self.__logging and border_type != "black":
                    logger.debug("Setting Border type: {}".format(border_type))
            else:
                # log and reset to default
                if self.__logging and border_type != "black":
                    logger.debug(
                        "Setting border type is disabled if cropping is enabled!"
                    )
                self.__border_mode = border_modes["black"]
        else:
            # otherwise log if not
            if logging:
                logger.debug("Invalid input border type!")
            self.__border_mode = border_modes["black"]  # reset to default mode

        # define OpenCV version
        self.__cv2_version = check_CV_version()

        # retrieve best interpolation
        self.__interpolation = retrieve_best_interpolation(
            ["INTER_LINEAR_EXACT", "INTER_LINEAR", "INTER_AREA"]
        )

        # define normalized box filter
        self.__box_filter = np.ones(smoothing_radius) / smoothing_radius

    def stabilize(self, frame):
        """
        This method takes an unstabilized video frame, and returns a stabilized one.

        Parameters:
            frame (numpy.ndarray): inputs unstabilized video frames.
        """
        # check if frame is None
        if frame is None:
            # return if it does
            return

        # save frame size for zooming
        if self.__crop_n_zoom and self.__frame_size == None:
            self.__frame_size = frame.shape[:2]

        # initiate transformations capturing
        if not self.__frame_queue:
            # for first frame
            previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to gray
            previous_gray = self.__clahe.apply(previous_gray)  # optimize gray frame
            self.__previous_keypoints = cv2.goodFeaturesToTrack(
                previous_gray,
                maxCorners=200,
                qualityLevel=0.05,
                minDistance=30.0,
                blockSize=3,
                mask=None,
                useHarrisDetector=False,
                k=0.04,
            )  # track features using GFTT
            self.__frame_height, self.frame_width = frame.shape[
                :2
            ]  # save input frame height and width
            self.__frame_queue.append(frame)  # save frame to deque
            self.__frame_queue_indexes.append(0)  # save frame index to deque
            self.__previous_gray = previous_gray[
                :
            ]  # save gray frame clone for further processing

        elif self.__frame_queue_indexes[-1] <= self.__smoothing_radius - 1:
            # for rest of frames
            self.__frame_queue.append(frame)  # save frame to deque
            self.__frame_queue_indexes.append(
                self.__frame_queue_indexes[-1] + 1
            )  # save frame index
            self.__generate_transformations()  # generate transformations
            if self.__frame_queue_indexes[-1] == self.__smoothing_radius - 1:
                # calculate smooth path once transformation capturing is completed
                for i in range(3):
                    # apply normalized box filter to the path
                    self.__smoothed_path[:, i] = self.__box_filter_convolve(
                        (self.__path[:, i]), window_size=self.__smoothing_radius
                    )
                # calculate deviation of path from smoothed path
                deviation = self.__smoothed_path - self.__path
                # save smoothed transformation
                self.__frame_transforms_smoothed = self.frame_transform + deviation
        else:
            # start applying transformations
            self.__frame_queue.append(frame)  # save frame to deque
            self.__frame_queue_indexes.append(
                self.__frame_queue_indexes[-1] + 1
            )  # save frame index
            self.__generate_transformations()  # generate transformations
            # calculate smooth path once transformation capturing is completed
            for i in range(3):
                # apply normalized box filter to the path
                self.__smoothed_path[:, i] = self.__box_filter_convolve(
                    (self.__path[:, i]), window_size=self.__smoothing_radius
                )
            # calculate deviation of path from smoothed path
            deviation = self.__smoothed_path - self.__path
            # save smoothed transformation
            self.__frame_transforms_smoothed = self.frame_transform + deviation
            # return transformation applied stabilized frame
            return self.__apply_transformations()

    def __generate_transformations(self):
        """
        An internal method that generate previous-to-current transformations [dx,dy,da].
        """
        frame_gray = cv2.cvtColor(
            self.__frame_queue[-1], cv2.COLOR_BGR2GRAY
        )  # retrieve current frame and convert to gray
        frame_gray = self.__clahe.apply(frame_gray)  # optimize it

        transformation = None
        try:
            # calculate optical flow using Lucas-Kanade differential method
            curr_kps, status, error = cv2.calcOpticalFlowPyrLK(
                self.__previous_gray, frame_gray, self.__previous_keypoints, None
            )

            # select only valid key-points
            valid_curr_kps = curr_kps[status == 1]  # current
            valid_previous_keypoints = self.__previous_keypoints[
                status == 1
            ]  # previous

            # calculate optimal affine transformation between pevious_2_current key-points
            if self.__cv2_version == 3:
                # backward compatibility with OpenCV3
                transformation = cv2.estimateRigidTransform(
                    valid_previous_keypoints, valid_curr_kps, False
                )
            else:
                transformation = cv2.estimateAffinePartial2D(
                    valid_previous_keypoints, valid_curr_kps
                )[0]
        except cv2.error as e:
            # catch any OpenCV assertion errors and warn user
            logger.warning("Video-Frame is too dark to generate any transformations!")
            transformation = None

        # check if transformation is not None
        if not (transformation is None):
            # pevious_2_current translation in x direction
            dx = transformation[0, 2]
            # pevious_2_current translation in y direction
            dy = transformation[1, 2]
            # pevious_2_current rotation in angle
            da = np.arctan2(transformation[1, 0], transformation[0, 0])
        else:
            # otherwise zero it
            dx = dy = da = 0

        # save this transformation
        self.__transforms.append([dx, dy, da])

        # calculate path from cumulative transformations sum
        self.frame_transform = np.array(self.__transforms, dtype="float32")
        self.__path = np.cumsum(self.frame_transform, axis=0)
        # create smoothed path from a copy of path
        self.__smoothed_path = np.copy(self.__path)

        # re-calculate and save GFTT key-points for current gray frame
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
        # save this gray frame for further processing
        self.__previous_gray = frame_gray[:]

    def __box_filter_convolve(self, path, window_size):
        """
        An internal method that applies *normalized linear box filter* to path w.r.t averaging window

        Parameters:

        * path (numpy.ndarray): a cumulative sum of transformations
        * window_size (int): averaging window size
        """
        # pad path to size of averaging window
        path_padded = np.pad(path, (window_size, window_size), "median")
        # apply linear box filter to path
        path_smoothed = np.convolve(path_padded, self.__box_filter, mode="same")
        # crop the smoothed path to original path
        path_smoothed = path_smoothed[window_size:-window_size]
        # assert if cropping is completed
        assert path.shape == path_smoothed.shape
        # return smoothed path
        return path_smoothed

    def __apply_transformations(self):
        """
        An internal method that applies affine transformation to the given frame
        from previously calculated transformations
        """
        # extract frame and its index from deque
        queue_frame = self.__frame_queue.popleft()
        queue_frame_index = self.__frame_queue_indexes.popleft()

        # create border around extracted frame w.r.t border_size
        bordered_frame = cv2.copyMakeBorder(
            queue_frame,
            top=self.__border_size,
            bottom=self.__border_size,
            left=self.__border_size,
            right=self.__border_size,
            borderType=self.__border_mode,
            value=[0, 0, 0],
        )
        alpha_bordered_frame = cv2.cvtColor(
            bordered_frame, cv2.COLOR_BGR2BGRA
        )  # create alpha channel
        # extract alpha channel
        alpha_bordered_frame[:, :, 3] = 0
        alpha_bordered_frame[
            self.__border_size : self.__border_size + self.__frame_height,
            self.__border_size : self.__border_size + self.frame_width,
            3,
        ] = 255

        # extracting Transformations w.r.t frame index
        dx = self.__frame_transforms_smoothed[queue_frame_index, 0]  # x-axis
        dy = self.__frame_transforms_smoothed[queue_frame_index, 1]  # y-axis
        da = self.__frame_transforms_smoothed[queue_frame_index, 2]  # angle

        # building 2x3 transformation matrix from extracted transformations
        queue_frame_transform = np.zeros((2, 3), np.float32)
        queue_frame_transform[0, 0] = np.cos(da)
        queue_frame_transform[0, 1] = -np.sin(da)
        queue_frame_transform[1, 0] = np.sin(da)
        queue_frame_transform[1, 1] = np.cos(da)
        queue_frame_transform[0, 2] = dx
        queue_frame_transform[1, 2] = dy

        # Applying an affine transformation to the frame
        frame_wrapped = cv2.warpAffine(
            alpha_bordered_frame,
            queue_frame_transform,
            alpha_bordered_frame.shape[:2][::-1],
            borderMode=self.__border_mode,
        )

        # drop alpha channel
        frame_stabilized = frame_wrapped[:, :, :3]

        # crop and zoom
        if self.__crop_n_zoom:
            # crop stabilized frame
            frame_cropped = frame_stabilized[
                self.__crop_n_zoom : -self.__crop_n_zoom,
                self.__crop_n_zoom : -self.__crop_n_zoom,
            ]
            # zoom stabilized frame
            frame_stabilized = cv2.resize(
                frame_cropped,
                self.__frame_size[::-1],
                interpolation=self.__interpolation,
            )

        # finally return stabilized frame
        return frame_stabilized

    def clean(self):
        """
        Cleans Stabilizer resources
        """
        # check if deque present
        if self.__frame_queue:
            # clear frame deque
            self.__frame_queue.clear()
            # clear frame indexes deque
            self.__frame_queue_indexes.clear()
