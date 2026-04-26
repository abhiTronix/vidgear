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

import logging as log
import queue
from threading import Event, Thread
from typing import Any

import cv2
from numpy.typing import NDArray

# import helper packages
from .helper import (
    get_supported_resolution,
    import_dependency_safe,
    logcurr_vidgear_ver,
    logger_handler,
)
from .yt_backend import YT_backend

yt_dlp = import_dependency_safe("yt_dlp", error="silent")

# define logger
logger = log.getLogger("FFGear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

# Safely import the deffcode library
deffcode = import_dependency_safe("deffcode", error="silent")
if deffcode is not None:
    from deffcode import FFdecoder


class FFGear:
    """
    FFGear API is a multi-threaded, high-performance wrapper around
    [DeFFcode's FFdecoder API](https://abhitronix.github.io/deffcode/latest/reference/ffdecoder/)
    that compiles and executes an FFmpeg pipeline inside a subprocess pipe for generating
    real-time, low-overhead, lightning-fast decoded video frames in Python.

    FFGear API provides direct, transparent access to the full FFdecoder feature-set, including:

    - **Hardware-Accelerated Decoding** — CUDA/CUVID and other `-hwaccel` backends.
    - **Flexible Pixel Formats** — any FFmpeg-supported `-pix_fmt` (e.g. `bgr24`, `yuv420p`, `gray`), with an optional OpenCV-compatibility patch (`-enforce_cv_patch`) for YUV/NV layouts.
    - **Per-Frame Metadata Extraction** — asynchronous `showinfo` filter integration via `-extract_metadata`, yielding `(frame, metadata)` tuples with `frame_num`, `pts_time`, `is_keyframe`, and `frame_type`.
    - **Complex Filtergraphs** — live simple (`-vf`) and complex (`-filter_complex`) FFmpeg filter pipelines.
    - **Multi-Input Sources** — multiple simultaneous inputs routed via `-map` or `-filter_complex`.
    - **Wide Source Support** — USB/Virtual/IP camera feeds, multimedia files, image sequences, screen recordings, and network protocols (HTTP(s), RTP/RTSP, etc.).

    Internally, FFGear employs a Producer-Consumer threaded queue (configurable via `QUEUE_SIZE`,
    `THREADED_QUEUE_MODE`, `THREAD_TIMEOUT`) for zero-bottleneck asynchronous frame delivery, and
    maintains the standard OpenCV-Python coding syntax for drop-in integration.

    Similar to CamGear, FFGear also supports the `yt_dlp` backend via `stream_mode=True` for
    seamlessly pipelining live video-frames and metadata from streaming services like YouTube,
    Dailymotion, Twitch, and
    [many more ➶](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md#supported-sites).
    """

    def __init__(
        self,
        source: Any = None,
        stream_mode: bool = False,
        source_demuxer: str | None = None,
        frame_format: str = "bgr24",
        custom_ffmpeg: str = "",
        logging: bool = False,
        **options: dict,
    ):
        """
        This constructor method initializes the object state and attributes of the FFGear class.

        Parameters:
            source (Any): defines the source for the input stream (URL, file path, or device index).
            stream_mode (bool): controls the exclusive **Stream Mode** for handling streaming URLs.
            source_demuxer (str): specifies the demuxer for the source (e.g., 'v4l2', 'dshow').
            frame_format (str): specifies the pixel layout for decoding (e.g., 'bgr24', 'rgb24', 'yuv420p', 'gray').
            custom_ffmpeg (str): specifies a custom FFmpeg executable path.
            logging (bool): enables/disables console logging.
            options (dict): FFdecoder configuration (like -hwaccel, -enforce_cv_patch, -extract_metadata)
                            and FFGear queue tweaks.
        """
        if deffcode is None:
            raise ImportError(
                "[FFGear:ERROR] :: `deffcode` is not installed! Please install it via `pip install deffcode`."
            )

        self.__logging = logging if isinstance(logging, bool) else False
        logcurr_vidgear_ver(logging=self.__logging)

        # initialize global
        self.ytv_metadata = {}

        # check if Stream-Mode is ON (True)
        if stream_mode:
            stream_resolution = get_supported_resolution(
                options.pop("STREAM_RESOLUTION", "best"), logging=self.__logging
            )
            if yt_dlp is not None:
                yt_stream_params = options.pop("STREAM_PARAMS", {})
                if isinstance(yt_stream_params, dict):
                    yt_stream_params = {
                        str(k).strip(): v for k, v in yt_stream_params.items()
                    }
                else:
                    yt_stream_params = {}
                try:
                    logger.info(
                        "Verifying Streaming URL using yt-dlp backend. Please wait..."
                    )
                    ytbackend = YT_backend(
                        source_url=source, logging=self.__logging, **yt_stream_params
                    )
                    if ytbackend:
                        self.ytv_metadata = ytbackend.meta_data
                        ytbackend.is_livestream and logger.warning(
                            "Livestream URL detected. Consider using a GStreamer-based pipeline for best results."
                        )
                        if stream_resolution not in ytbackend.streams:
                            logger.warning(
                                "Specified stream-resolution `{}` is not available. Reverting to `best`!".format(
                                    stream_resolution
                                )
                            )
                            stream_resolution = "best"
                        else:
                            self.__logging and logger.debug(
                                "Using `{}` resolution for streaming.".format(
                                    stream_resolution
                                )
                            )
                        source = ytbackend.streams[stream_resolution]
                        self.__logging and logger.debug(
                            "YouTube source ID: `{}`, Title: `{}`, Quality: `{}`".format(
                                self.ytv_metadata["id"],
                                self.ytv_metadata["title"],
                                stream_resolution,
                            )
                        )
                except Exception:
                    raise ValueError(
                        "[FFGear:ERROR] :: Stream Mode is enabled but Input URL is invalid!"
                    )
            else:
                import_dependency_safe("yt_dlp")

        self.source = source
        self.frame_format = frame_format.strip().lower() if frame_format else "bgr24"

        # -------------------------------------------------------------
        # Extract FFGear-specific properties from **options
        # -------------------------------------------------------------

        # Extract Threaded Queue Mode constraints
        self.__threaded_queue_mode = options.pop("THREADED_QUEUE_MODE", True)
        if not isinstance(self.__threaded_queue_mode, bool):
            self.__threaded_queue_mode = True

        self.__thread_timeout = options.pop("THREAD_TIMEOUT", None)
        if self.__thread_timeout and isinstance(self.__thread_timeout, (int, float)):
            self.__thread_timeout = float(self.__thread_timeout)
        else:
            self.__thread_timeout = None

        # Parse queue size (similar to WriteGear parameter parsing)
        q_size = options.pop("QUEUE_SIZE", 96)
        q_size = q_size if isinstance(q_size, int) else 96
        self.__queue = (
            queue.Queue(maxsize=q_size) if self.__threaded_queue_mode else None
        )

        if self.__threaded_queue_mode:
            self.__logging and logger.debug(
                f"Threaded Queue Mode enabled with Queue Size: {q_size}."
            )
        else:
            self.__logging and logger.warning(
                "Threaded Queue Mode disabled. High performance may be bottlenecked!"
            )

        # -------------------------------------------------------------
        # Parse specialized FFdecoder features / Frame format handling
        # -------------------------------------------------------------

        # Support metadata extraction recipe natively
        self.__extract_metadata = options.get("-extract_metadata", False)
        self.frame_metadata = {}

        # Handle OpenCV patch for YUV420p frames
        # (This yields a H*1.5, W frame layout ready for cv2.cvtColor)
        self.__enforce_cv_patch = options.get("-enforce_cv_patch", False)
        if self.frame_format == "yuv420p" and self.__enforce_cv_patch:
            self.__logging and logger.debug(
                "Enforcing CV patch for yuv420p format. Frames will require OpenCV conversion."
            )

        # -------------------------------------------------------------
        # Formulate FFdecoder Stream Pipeline
        # -------------------------------------------------------------
        try:
            self.stream = FFdecoder(
                source=self.source,
                source_demuxer=source_demuxer,
                frame_format=self.frame_format,
                custom_ffmpeg=custom_ffmpeg,
                verbose=self.__logging,
                **options,  # Pass remaining FFmpeg params (-hwaccel, -vcodec, etc.)
            ).formulate()
        except Exception as e:
            raise RuntimeError(
                f"[FFGear:ERROR] :: Failed to initialize FFdecoder pipeline: {e!s}"
            )

        # Initialize the generator
        self.__generator = self.stream.generateFrame()

        # Pull first frame to confirm stream validity
        self.frame = None
        self.__read_first_frame()

        if self.frame is not None:
            self.__threaded_queue_mode and self.__queue.put(self.frame)
        else:
            raise RuntimeError("[FFGear:ERROR] :: Source is invalid or unreadable!")

        # Synchronization primitives
        self.__thread = None
        self.__terminate = Event()
        self.__stream_read = Event()

    def __read_first_frame(self):
        """Helper to pull the first frame and process formatting/metadata."""
        try:
            out = next(self.__generator)
            # Route based on the '-extract_metadata' recipe output
            if self.__extract_metadata and isinstance(out, tuple):
                self.frame, self.frame_metadata = out[0], out[1]
            else:
                self.frame = out

            self.__process_frame_format()
        except StopIteration:
            self.frame = None

    def __process_frame_format(self):
        """
        Handles OpenCV conversion routines based on the user's chosen frame_format
        and patching parameters.
        """
        if self.frame is None:
            return

        # Automatically translate patched YUV420p data into standard BGR space via OpenCV
        # if the user requested the patch but wants a ready-to-display frame output
        if self.frame_format == "yuv420p" and self.__enforce_cv_patch:
            try:
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_YUV2BGR_I420)
            except Exception as e:
                self.__logging and logger.warning(
                    f"Failed to process patched YUV420p frame via OpenCV: {e!s}"
                )

    def start(self):
        """
        Starts the Threaded Video Streaming.
        """
        self.__thread = Thread(target=self.__update, name="FFGear", args=())
        self.__thread.daemon = True
        self.__thread.start()
        return self

    def __update(self):
        """
        Producer Thread: Iterates frames continuously from the FFdecoder generator,
        processes metadata/colorspace routines, and yields them to the bounded queue.
        """
        while not self.__terminate.is_set():
            self.__stream_read.clear()

            try:
                out = next(self.__generator)

                # Separate frame and metadata if recipe is activated
                if self.__extract_metadata and isinstance(out, tuple):
                    frame, self.frame_metadata = out[0], out[1]
                else:
                    frame = out

                grabbed = True
            except StopIteration:
                frame = None
                grabbed = False

            self.__stream_read.set()

            if not grabbed or frame is None:
                if self.__threaded_queue_mode:
                    if self.__queue.empty():
                        break
                    else:
                        continue
                else:
                    break

            self.frame = frame

            # Apply any OpenCV colorspace logic based on format choices
            self.__process_frame_format()

            # push the frame onto the producer queue
            self.__threaded_queue_mode and self.__queue.put(self.frame)

        # Poison pill for the queue signaling termination
        self.__threaded_queue_mode and self.__queue.put(None)
        self.__threaded_queue_mode = False

        self.__terminate.set()
        self.__stream_read.set()

        # Cleanup deffcode pipeline (similar to WriteGear close process)
        if self.stream is not None:
            self.stream.terminate()

    def read(self) -> NDArray | None:
        """
        Consumer Thread: Pops frames symmetrically from the queue block.
        **Returns:** N-dimensional numpy array of the frame.
        """
        while self.__threaded_queue_mode and not self.__terminate.is_set():
            try:
                return self.__queue.get(timeout=self.__thread_timeout)
            except queue.Empty:
                continue

        return (
            self.frame
            if not self.__terminate.is_set()
            and self.__stream_read.wait(timeout=self.__thread_timeout)
            else None
        )

    def stop(self) -> None:
        """
        Safely halts processes, unblocks memory, and flushes pipelines.
        """
        self.__logging and logger.debug("Terminating processes.")
        self.__threaded_queue_mode = False

        self.__stream_read.set()
        self.__terminate.set()

        # Flush the remainder of the queue to unblock the producer
        if self.__thread is not None:
            if self.__queue is not None:
                while not self.__queue.empty():
                    try:
                        self.__queue.get_nowait()
                    except queue.Empty:
                        continue
                    self.__queue.task_done()
            self.__thread.join()

        # Forcibly teardown FFdecoder process safely
        if self.stream is not None:
            self.stream.terminate()
