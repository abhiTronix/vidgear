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
import time
import queue
import logging as log
from threading import Thread, Event

# import helper packages
from .helper import (
    capPropId,
    logger_handler,
    check_CV_version,
    restore_levelnames,
    youtube_url_validator,
    get_supported_resolution,
    check_gstreamer_support,
    dimensions_to_resolutions,
    import_dependency_safe,
)

# define logger
logger = log.getLogger("CamGear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


class CamGear:
    """
    CamGear supports a diverse range of video streams which can handle/control video stream almost any IP/USB Cameras, multimedia video file format (upto 4k tested),
    any network stream URL such as http(s), rtp, rstp, rtmp, mms, etc. In addition to this, it also supports Gstreamer's RAW pipelines and various live video streaming
    sites like YouTube, Twitch, Dailymotion etc.

    CamGear API provides a flexible, high-level multi-threaded wrapper around OpenCV's VideoCapture API with direct access to almost all of its available parameters.
    It relies on Threaded Queue mode for threaded, error-free and synchronized frame handling.

    CamGear internally employs streamlink for piping live videos from various streaming services like Twitch, Livestream, Dailymotion etc., and also utilizies pafy
    with youtube-dl at its backend for seamless YouTube pipelining.
    """

    def __init__(
        self,
        source=0,
        stream_mode=False,
        backend=0,
        colorspace=None,
        logging=False,
        time_delay=0,
        **options
    ):

        """
        This constructor method initializes the object state and attributes of the CamGear class.

        Parameters:
            source (based on input): defines the source for the input stream.
            stream_mode (bool): controls the exclusive **Stream Mode** for handling streaming URLs.
            backend (int): selects the backend for OpenCV's VideoCapture class.
            colorspace (str): selects the colorspace of the input stream.
            logging (bool): enables/disables logging.
            time_delay (int): time delay (in sec) before start reading the frames.
            options (dict): provides ability to alter Source Tweak Parameters.
        """

        # enable logging if specified
        self.__logging = False
        if logging:
            self.__logging = logging

        # check if Stream-Mode is ON (True)
        if stream_mode:
            # check GStreamer backend support
            gst_support = check_gstreamer_support(logging=logging)
            # handle special Stream Mode parameters
            stream_resolution = get_supported_resolution(
                options.pop("STREAM_RESOLUTION", "best"), logging=logging
            )
            stream_params = options.pop("STREAM_PARAMS", {})
            if isinstance(stream_params, dict):
                stream_params = {str(k).strip(): v for k, v in stream_params.items()}
            else:
                stream_params = {}
            # handle Stream-Mode
            try:
                # detect whether a YouTube URL
                video_url = youtube_url_validator(source)
                if video_url:
                    # import backend library
                    pafy = import_dependency_safe("pafy")
                    logger.info("Using Youtube-dl Backend")
                    # create new pafy object
                    source_object = pafy.new(video_url, ydl_opts=stream_params)
                    # extract all valid video-streams
                    valid_streams = [
                        v_streams
                        for v_streams in source_object.videostreams
                        if v_streams.notes != "HLS" and v_streams.extension == "webm"
                    ] + [va_streams for va_streams in source_object.streams]
                    # extract available stream resolutions
                    available_streams = [qs.notes for qs in valid_streams]
                    # check whether YouTube-stream is live or not?
                    is_live = (
                        all(x == "HLS" or x == "" for x in available_streams)
                        if available_streams
                        else False
                    )
                    # validate streams
                    if valid_streams and (not is_live or gst_support):
                        # handle live-streams
                        if is_live:
                            # Enforce GStreamer backend for YouTube-livestreams
                            logger.critical(
                                "YouTube livestream URL detected. Enforcing GStreamer backend."
                            )
                            backend = cv2.CAP_GSTREAMER
                            # convert stream dimensions to streams resolutions
                            available_streams = dimensions_to_resolutions(
                                [qs.resolution for qs in valid_streams]
                            )
                        # check whether stream-resolution was specified and available
                        if stream_resolution in ["best", "worst"] or not (
                            stream_resolution in available_streams
                        ):
                            # not specified and unavailable
                            if not stream_resolution in ["best", "worst"]:
                                logger.warning(
                                    "Specified stream-resolution `{}` is not available. Reverting to `best`!".format(
                                        stream_resolution
                                    )
                                )
                                # revert to best
                                stream_resolution = "best"
                            # parse and select best or worst streams
                            valid_streams.sort(
                                key=lambda x: x.dimensions,
                                reverse=True if stream_resolution == "best" else False,
                            )
                            parsed_stream = valid_streams[0]
                        else:
                            # apply specfied valid stream-resolution
                            matched_stream = [
                                i
                                for i, s in enumerate(available_streams)
                                if stream_resolution in s
                            ]
                            parsed_stream = valid_streams[matched_stream[0]]
                        # extract stream URL as source
                        source = parsed_stream.url
                        # log progress
                        if self.__logging:
                            logger.debug(
                                "YouTube source ID: `{}`, Title: `{}`, Quality: `{}`".format(
                                    video_url, source_object.title, stream_resolution
                                )
                            )
                    else:
                        # raise error if Gstreamer backend unavailable for YouTube live-streams
                        # see issue: https://github.com/abhiTronix/vidgear/issues/133
                        raise RuntimeError(
                            "[CamGear:ERROR] :: Unable to find any OpenCV supported streams in `{}` YouTube URL. Kindly compile OpenCV with GSstreamer(>=v1.0.0) support or Use another URL!".format(
                                source
                            )
                        )
                else:
                    # import backend library
                    Streamlink = import_dependency_safe(
                        "from streamlink import Streamlink"
                    )
                    restore_levelnames()
                    logger.info("Using Streamlink Backend")
                    # check session
                    session = Streamlink()
                    # extract streams
                    streams = session.streams(source)
                    # set parameters
                    for key, value in stream_params.items():
                        session.set_option(key, value)
                    # select streams are available
                    assert (
                        streams
                    ), "[CamGear:ERROR] :: Invalid `{}` URL cannot be processed!".format(
                        source
                    )
                    # check whether stream-resolution was specified and available
                    if not stream_resolution in streams.keys():
                        # if unavailable
                        logger.warning(
                            "Specified stream-resolution `{}` is not available. Reverting to `best`!".format(
                                stream_resolution
                            )
                        )
                        # revert to best
                        stream_resolution = "best"
                    # extract url
                    source = streams[stream_resolution].url
            except Exception as e:
                # raise error if something went wrong
                raise ValueError(
                    "[CamGear:ERROR] :: Stream Mode is enabled but Input URL is invalid!"
                )

        # youtube mode variable initialization
        self.__youtube_mode = stream_mode

        # assigns special parameter to global variable and clear
        # Threaded Queue Mode
        self.__threaded_queue_mode = options.pop("THREADED_QUEUE_MODE", True)
        if not isinstance(self.__threaded_queue_mode, bool):
            # reset improper values
            self.__threaded_queue_mode = True
        # Thread Timeout
        self.__thread_timeout = options.pop("THREAD_TIMEOUT", None)
        if self.__thread_timeout and isinstance(self.__thread_timeout, (int, float)):
            # set values
            self.__thread_timeout = int(self.__thread_timeout)
        else:
            # defaults to 5mins timeout
            self.__thread_timeout = None

        self.__queue = None
        # initialize queue for video files only
        if self.__threaded_queue_mode and isinstance(source, str):
            # define queue and assign it to global var
            self.__queue = queue.Queue(maxsize=96)  # max len 96 to check overflow
            # log it
            if self.__logging:
                logger.debug(
                    "Enabling Threaded Queue Mode for the current video source!"
                )
        else:
            # otherwise disable it
            self.__threaded_queue_mode = False
            # log it
            if self.__logging:
                logger.warning(
                    "Threaded Queue Mode is disabled for the current video source!"
                )

        if self.__thread_timeout:
            logger.debug(
                "Setting Video-Thread Timeout to {}s.".format(self.__thread_timeout)
            )

        # stream variable initialization
        self.stream = None

        if backend and isinstance(backend, int):
            # add backend if specified and initialize the camera stream
            if check_CV_version() == 3:
                # Different OpenCV 3.4.x statement
                self.stream = cv2.VideoCapture(source + backend)
            else:
                # Two parameters are available since OpenCV 4+ (master branch)
                self.stream = cv2.VideoCapture(source, backend)
            logger.debug("Setting backend `{}` for this source.".format(backend))
        else:
            # initialize the camera stream
            self.stream = cv2.VideoCapture(source)

        # initializing colorspace variable
        self.color_space = None

        # apply attributes to source if specified
        options = {str(k).strip(): v for k, v in options.items()}
        for key, value in options.items():
            property = capPropId(key)
            if not (property is None):
                self.stream.set(property, value)

        # handle colorspace value
        if not (colorspace is None):
            self.color_space = capPropId(colorspace.strip())
            if self.__logging and not (self.color_space is None):
                logger.debug(
                    "Enabling `{}` colorspace for this video stream!".format(
                        colorspace.strip()
                    )
                )

        # initialize and assign frame-rate variable
        self.framerate = 0.0
        _fps = self.stream.get(cv2.CAP_PROP_FPS)
        if _fps > 1.0:
            self.framerate = _fps

        # applying time delay to warm-up webcam only if specified
        if time_delay and isinstance(time_delay, (int, float)):
            time.sleep(time_delay)

        # frame variable initialization
        (grabbed, self.frame) = self.stream.read()

        # check if valid stream
        if grabbed:
            # render colorspace if defined
            if not (self.color_space is None):
                self.frame = cv2.cvtColor(self.frame, self.color_space)

            if self.__threaded_queue_mode:
                # initialize and append to queue
                self.__queue.put(self.frame)
        else:
            raise RuntimeError(
                "[CamGear:ERROR] :: Source is invalid, CamGear failed to initialize stream on this source!"
            )

        # thread initialization
        self.__thread = None

        # initialize termination flag event
        self.__terminate = Event()

        # initialize stream read flag event
        self.__stream_read = Event()

    def start(self):
        """
        Launches the internal *Threaded Frames Extractor* daemon.

        **Returns:** A reference to the CamGear class object.
        """

        self.__thread = Thread(target=self.__update, name="CamGear", args=())
        self.__thread.daemon = True
        self.__thread.start()
        return self

    def __update(self):
        """
        A **Threaded Frames Extractor**, that keep iterating frames from OpenCV's VideoCapture API to a internal monitored queue,
        until the thread is terminated, or frames runs out.
        """

        # keep iterating infinitely
        # until the thread is terminated
        # or frames runs out
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.__terminate.is_set():
                break

            # stream not read yet
            self.__stream_read.clear()

            # otherwise, read the next frame from the stream
            (grabbed, frame) = self.stream.read()

            # stream read completed
            self.__stream_read.set()

            # check for valid frame if received
            if not grabbed:
                # no frames received, then safely exit
                if self.__threaded_queue_mode:
                    if self.__queue.empty():
                        break
                    else:
                        continue
                else:
                    break

            # apply colorspace to frames if valid
            if not (self.color_space is None):
                color_frame = None
                try:
                    if isinstance(self.color_space, int):
                        color_frame = cv2.cvtColor(frame, self.color_space)
                    else:
                        raise ValueError(
                            "Global color_space parameter value `{}` is not a valid!".format(
                                self.color_space
                            )
                        )
                except Exception as e:
                    # Catch if any error occurred
                    self.color_space = None
                    if self.__logging:
                        logger.exception(str(e))
                        logger.warning("Input colorspace is not a valid colorspace!")
                if not (color_frame is None):
                    self.frame = color_frame
                else:
                    self.frame = frame
            else:
                self.frame = frame

            # append to queue
            if self.__threaded_queue_mode:
                self.__queue.put(self.frame)

        # indicate immediate termination
        self.__threaded_queue_mode = False
        self.__terminate.set()
        self.__stream_read.set()
        # release resources
        self.stream.release()

    def read(self):
        """
        Extracts frames synchronously from monitored queue, while maintaining a fixed-length frame buffer in the memory,
        and blocks the thread if the queue is full.

        **Returns:** A n-dimensional numpy array.
        """
        while self.__threaded_queue_mode:
            return self.__queue.get(timeout=self.__thread_timeout)
        # return current frame
        # only after stream is read
        return (
            self.frame
            if not self.__terminate.is_set()  # check if already terminated
            and self.__stream_read.wait(timeout=self.__thread_timeout)  # wait for it
            else None
        )

    def stop(self):
        """
        Safely terminates the thread, and release the VideoStream resources.
        """
        if self.__logging:
            logger.debug("Terminating processes.")
        # terminate Threaded queue mode separately
        if self.__threaded_queue_mode:
            self.__threaded_queue_mode = False

        # indicate that the thread
        # should be terminated immediately
        self.__terminate.set()
        self.__stream_read.set()

        # wait until stream resources are released (producer thread might be still grabbing frame)
        if self.__thread is not None:
            if not (self.__queue is None):
                while not self.__queue.empty():
                    try:
                        self.__queue.get_nowait()
                    except queue.Empty:
                        continue
                    self.__queue.task_done()
            self.__thread.join()
