# This class is a modified fps module from `imutils` python library.

import time


class FPS:
    """
    Class to calculate FPS based on time.time python module
    """

    def __init__(self):
        # initiating FPS class and its variable
        self.__start = 0
        self.__numFrames = 0
        self.__fps = 0.0

    def start(self):
        # start timer
        if not (self.__start):
            self.__start = time.time()
        return self

    def update(self):
        # calculate frames
        self.__numFrames += 1

    def fps(self):
        # return FPS
        if (time.time() - self.__start) > 1.0:
            self.__fps = self.__numFrames / (time.time() - self.__start)
            self.__numFrames = 0
            self.__start = time.time()
        return self.__fps
