import time
import numpy as np


class FPS:
    """
    Class to calculate average FPS based on time.time() python module
    """

    def __init__(self):
        # initiating FPS class and its variable
        self.__start = 0
        self.__numFrames = 0
        self.__average_fps = []

    def start(self):
        # start timer
        if not (self.__start):
            self.__start = time.time()
        return self

    def update(self):
        # calculate frames
        self.__numFrames += 1
        if (time.time() - self.__start) > 1.0:
            fps = self.__numFrames / (time.time() - self.__start)
            self.__numFrames = 0
            self.__average_fps.append(fps)
            self.__start = time.time()

    def average_fps(self):
        av_fps = np.average(self.__average_fps) if self.__average_fps else 0.0
        return av_fps
