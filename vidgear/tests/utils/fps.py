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

import time
import numpy as np
from threading import Thread


class FPS:
    """
    Threaded Class to calculate average FPS based on time.perf_counter() python module
    """

    def __init__(self):
        # initiating FPS class and its variable
        # intiate Time elasped hander
        self.__t_elasped = 0
        # intiate frame counter
        self.__numFrames = 0
        # intiate calculated fps holder
        self.__fps = []
        # initiate termination flag
        self.__terminate = False
        # intiate Timer thread
        self.__timer = None

    def start(self):
        """
        start and return Timer Threaded instance
        """
        self.__timer = Thread(target=self.__timeit, name="FPS_Timer", args=())
        self.__timer.daemon = True
        self.__timer.start()
        return self

    def __timeit(self):
        """
        Calculates Frames Per Second and resets variables
        """
        # assign current time
        self.__t_elasped = time.perf_counter()
        # loop until termainated
        while not (self.__terminate):
            # calulate frames elasped per second
            if time.perf_counter() - self.__t_elasped > 1.0:
                # calculate FPS
                fps = self.__numFrames / (time.perf_counter() - self.__t_elasped)
                # reset frames counter
                self.__numFrames = 0
                # append calulated FPS
                self.__fps.append(fps)
                # reset timer
                self.__t_elasped = time.perf_counter()

    def update(self):
        """
        counts frames
        """
        self.__numFrames += 1

    def average_fps(self):
        """
        calculates and return average FPS
        """
        self.__terminate = True
        if not (self.__timer is None):
            self.__timer.join()
            self.__timer = None
        av_fps = np.average(self.__fps) if self.__fps else 0.0
        return av_fps
