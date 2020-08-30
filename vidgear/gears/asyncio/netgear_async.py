"""
===============================================
vidgear library source-code is deployed under the Apache 2.0 License:

Copyright (c) 2019-2020 Abhishek Thakur(@abhiTronix) <abhi.una12@gmail.com>

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
import sys
import zmq
import numpy as np
import asyncio
import inspect
import logging as log
import msgpack
import platform
import zmq.asyncio
import msgpack_numpy as m
from collections import deque

from .helper import logger_handler
from ..videogear import VideoGear

# define logger
logger = log.getLogger("NetGear_Async")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


class NetGear_Async:
    """
    NetGear_Async is an asyncio videoframe messaging framework, built on `zmq.asyncio`, and powered by high-performance asyncio event loop
    called **`uvloop`** to achieve unmatchable high-speed and lag-free video streaming over the network with minimal resource constraints.
    Basically, this API is able to transfer thousands of frames in just a few seconds without causing any significant load on your system.

    NetGear_Async can generate double performance as compared to [NetGear API](#netgear) at about 1/3rd of memory consumption, and also
    provide complete server-client handling with various options to use variable protocols/patterns similar to NetGear, but it doesn't support
    any NetGear's Exclusive Modes yet.

    Furthermore, NetGear_Async allows us to  define our own custom Server Source to manipulate frames easily before sending them across the
    network. In addition to all this, NetGear_Async also **provides a special internal wrapper around VideoGear API]**, which itself provides
    internal access to both CamGear and PiGear APIs thereby granting it exclusive power for streaming frames incoming from any connected
    device/source to the network.

    NetGear_Async as of now supports four ZeroMQ messaging patterns:

    - `zmq.PAIR` _(ZMQ Pair Pattern)_
    - `zmq.REQ/zmq.REP` _(ZMQ Request/Reply Pattern)_
    - `zmq.PUB/zmq.SUB` _(ZMQ Publish/Subscribe Pattern)_
    - `zmq.PUSH/zmq.PULL` _(ZMQ Push/Pull Pattern)_

    Whereas supported protocol are: `tcp` and `ipc`.
    """

    def __init__(
        self,
        # NetGear_Async parameters
        address=None,
        port=None,
        protocol="tcp",
        pattern=0,
        receive_mode=False,
        timeout=0.0,
        # Videogear parameters
        enablePiCamera=False,
        stabilize=False,
        source=0,
        camera_num=0,
        y_tube=False,
        backend=0,
        colorspace=None,
        resolution=(640, 480),
        framerate=25,
        time_delay=0,
        # common parameters
        logging=False,
        **options
    ):
        # enable logging if specified
        self.__logging = logging

        # define valid messaging patterns => `0`: PAIR, `1`:(REQ, REP), `2`:(SUB, PUB), `3`:(PUSH, PULL)
        valid_messaging_patterns = {
            0: (zmq.PAIR, zmq.PAIR),
            1: (zmq.REQ, zmq.REP),
            2: (zmq.PUB, zmq.SUB),
            3: (zmq.PUSH, zmq.PULL),
        }

        # check whether user-defined messaging pattern is valid
        if isinstance(pattern, int) and pattern in valid_messaging_patterns:
            # assign value
            self.__msg_pattern = pattern
            self.__pattern = valid_messaging_patterns[pattern]
        else:
            # otherwise default to 0:`zmq.PAIR`
            self.__msg_pattern = 0
            self.__pattern = valid_messaging_patterns[self.__msg_pattern]
            if self.__logging:
                logger.warning(
                    "Invalid pattern {pattern}. Defaulting to `zmq.PAIR`!".format(
                        pattern=pattern
                    )
                )

        # check  whether user-defined messaging protocol is valid
        if isinstance(protocol, str) and protocol in ["tcp", "ipc"]:
            # assign value
            self.__protocol = protocol
        else:
            # else default to `tcp` protocol
            self.__protocol = "tcp"
            if self.__logging:
                logger.warning("Invalid protocol. Defaulting to `tcp`!")

        # initialize Termination flag
        self.__terminate = False
        # initialize and assign `Receive Mode`
        self.__receive_mode = receive_mode
        # initialize stream handler
        self.__stream = None
        # initialize Messaging Socket
        self.__msg_socket = None
        # initialize NetGear's configuration dictionary
        self.config = {}

        # assign timeout for Receiver end
        if timeout > 0 and isinstance(timeout, (int, float)):
            self.__timeout = float(timeout)
        else:
            self.__timeout = 15.0
        # define messaging asynchronous Context
        self.__msg_context = zmq.asyncio.Context()

        # check whether `Receive Mode` is enabled
        if receive_mode:
            # assign local ip address if None
            if address is None:
                self.__address = "*"  # define address
            else:
                self.__address = address
            # assign default port address if None
            if port is None:
                self.__port = "5555"
            else:
                self.__port = port
        else:
            # Handle video source if not None
            if source is None:
                self.config = {"generator": None}
                if self.__logging:
                    logger.warning("Given source is of NoneType!")
            else:
                # define stream with necessary params
                self.__stream = VideoGear(
                    enablePiCamera=enablePiCamera,
                    stabilize=stabilize,
                    source=source,
                    camera_num=camera_num,
                    y_tube=y_tube,
                    backend=backend,
                    colorspace=colorspace,
                    resolution=resolution,
                    framerate=framerate,
                    logging=logging,
                    time_delay=time_delay,
                    **options
                )
                # define default frame generator in configuration
                self.config = {"generator": self.__frame_generator()}
            # assign local ip address if None
            if address is None:
                self.__address = "localhost"
            else:
                self.__address = address
            # assign default port address if None
            if port is None:
                self.__port = "5555"
            else:
                self.__port = port
            # add server task handler
            self.task = None

        # Setup and assign event loop policy
        if platform.system() == "Windows":
            # On Windows, VidGear requires the ``WindowsSelectorEventLoop``, and this is
            # the default in Python 3.7 and older, but new Python 3.8, defaults to an
            # event loop that is not compatible with it. Thereby, we had to set it manually.
            if sys.version_info[:2] >= (3, 8):
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        else:
            # import library
            import uvloop

            # uvloop eventloop is only available for UNIX machines.
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

        # Retrieve event loop and assign it
        self.loop = asyncio.get_event_loop()

        # debugging
        logger.info(
            "Using `{}` event loop for this process.".format(
                self.loop.__class__.__name__
            )
        )

    def launch(self):
        """
        Launches an asynchronous generators and loop executors for respective task.
        """
        # check if receive mode enabled
        if self.__receive_mode:
            if self.__logging:
                logger.debug("Launching NetGear asynchronous generator!")
            # run loop executor for Receiver asynchronous generator
            self.loop.run_in_executor(None, self.recv_generator)
            # return instance
            return self
        else:
            # Otherwise launch Server handler
            if self.__logging:
                logger.debug("Creating NetGear asynchronous server handler!")
            # create task for Server Handler
            self.task = asyncio.ensure_future(self.__server_handler(), loop=self.loop)
            # return instance
            return self

    async def __server_handler(self):
        """
        Handles various Server-end processes/tasks.
        """
        # validate assigned frame generator in NetGear configuration
        if isinstance(self.config, dict) and "generator" in self.config:
            # check if its  assigned value is a asynchronous generator
            if self.config["generator"] is None or not inspect.isasyncgen(
                self.config["generator"]
            ):
                # otherwise raise error
                raise ValueError(
                    "[NetGear_Async:ERROR] :: Invalid configuration. Assigned generator must be a asynchronous generator function/method only!"
                )
        else:
            # raise error if validation fails
            raise RuntimeError(
                "[NetGear_Async:ERROR] :: Assigned NetGear configuration is invalid!"
            )

        # define our messaging socket
        self.__msg_socket = self.__msg_context.socket(self.__pattern[0])

        # if req/rep pattern, define additional flags
        if self.__msg_pattern == 1:
            self.__msg_socket.REQ_RELAXED = True
            self.__msg_socket.REQ_CORRELATE = True

        # if pub/sub pattern, define additional optimizer
        if self.__msg_pattern == 2:
            self.__msg_socket.set_hwm(1)

        # try connecting socket to assigned protocol, address and port
        try:
            self.__msg_socket.connect(
                self.__protocol + "://" + str(self.__address) + ":" + str(self.__port)
            )
            # finally log if successful
            if self.__logging:
                logger.debug(
                    "Successfully connected to address: {} with pattern: {}.".format(
                        (
                            self.__protocol
                            + "://"
                            + str(self.__address)
                            + ":"
                            + str(self.__port)
                        ),
                        self.__msg_pattern,
                    )
                )
                logger.debug(
                    "Send Mode is successfully activated and ready to send data!"
                )
        except Exception as e:
            # log ad raise error if failed
            logger.exception(str(e))
            raise ValueError(
                "[NetGear_Async:ERROR] :: Failed to connect address: {} and pattern: {}!".format(
                    (
                        self.__protocol
                        + "://"
                        + str(self.__address)
                        + ":"
                        + str(self.__port)
                    ),
                    self.__msg_pattern,
                )
            )

        # loop over our Asynchronous frame generator
        async for frame in self.config["generator"]:
            # check if retrieved frame is `CONTIGUOUS`
            if not (frame.flags["C_CONTIGUOUS"]):
                # otherwise make it
                frame = np.ascontiguousarray(frame, dtype=frame.dtype)
            # encode message
            msg_enc = msgpack.packb(frame, default=m.encode)
            # send it over network
            await self.__msg_socket.send_multipart([msg_enc])
            # check if bidirectional patterns
            if self.__msg_pattern < 2:
                # then receive and log confirmation
                recv_confirmation = await self.__msg_socket.recv_multipart()
                if self.__logging:
                    logger.debug(recv_confirmation)

        # send `exit` flag when done!
        await self.__msg_socket.send_multipart([b"exit"])
        # check if bidirectional patterns
        if self.__msg_pattern < 2:
            # then receive and log confirmation
            recv_confirmation = await self.__msg_socket.recv_multipart()
            if self.__logging:
                logger.debug(recv_confirmation)

    async def recv_generator(self):
        """
        A default Asynchronous Frame Generator for NetGear's Receiver-end.
        """
        # check whether `receive mode` is activated
        if not (self.__receive_mode):
            # raise Value error and exit
            self.__terminate = True
            raise ValueError(
                "[NetGear:ERROR] :: `recv_generator()` function cannot be accessed while `receive_mode` is disabled. Kindly refer vidgear docs!"
            )

        # initialize and define messaging socket
        self.__msg_socket = self.__msg_context.socket(self.__pattern[1])

        # define exclusive socket options for patterns
        if self.__msg_pattern == 2:
            self.__msg_socket.set_hwm(1)
            self.__msg_socket.setsockopt(zmq.SUBSCRIBE, b"")

        try:
            # bind socket to the assigned protocol, address and port
            self.__msg_socket.bind(
                self.__protocol + "://" + str(self.__address) + ":" + str(self.__port)
            )
            # finally log progress
            if self.__logging:
                logger.debug(
                    "Successfully Binded to address: {} with pattern: {}.".format(
                        (
                            self.__protocol
                            + "://"
                            + str(self.__address)
                            + ":"
                            + str(self.__port)
                        ),
                        self.__msg_pattern,
                    )
                )
                logger.debug("Receive Mode is activated successfully!")
        except Exception as e:
            logger.exception(str(e))
            raise ValueError(
                "[NetGear:ERROR] :: Failed to bind address: {} and pattern: {}!".format(
                    (
                        self.__protocol
                        + "://"
                        + str(self.__address)
                        + ":"
                        + str(self.__port)
                    ),
                    self.__msg_pattern,
                )
            )

        # loop until terminated
        while not self.__terminate:
            # get message withing timeout limit
            recvd_msg = await asyncio.wait_for(
                self.__msg_socket.recv_multipart(), timeout=self.__timeout
            )
            # check if bidirectional patterns
            if self.__msg_pattern < 2:
                # send confirmation
                await self.__msg_socket.send_multipart([b"Message Received!"])
            # terminate if exit` flag received
            if recvd_msg[0] == b"exit":
                break
            # retrieve frame from message
            frame = msgpack.unpackb(recvd_msg[0], object_hook=m.decode)
            # yield received frame
            yield frame
            # sleep for sometime
            await asyncio.sleep(0.00001)

    async def __frame_generator(self):
        """
        Returns a default frame-generator for NetGear's Server Handler.
        """
        # start stream
        self.__stream.start()
        # loop over stream until its terminated
        while not self.__terminate:
            # read frames
            frame = self.__stream.read()
            # break if NoneType
            if frame is None:
                break
            # yield frame
            yield frame
            # sleep for sometime
            await asyncio.sleep(0.00001)

    def close(self, skip_loop=False):
        """
        Terminates all NetGear Asynchronous processes gracefully.

        Parameters:
            skip_loop (Boolean): (optional)used only if closing executor loop throws an error.
        """
        # log termination
        if self.__logging:
            logger.debug(
                "Terminating various {} Processes.".format(
                    "Receive Mode" if self.__receive_mode else "Send Mode"
                )
            )
        #  whether `receive_mode` is enabled or not
        if self.__receive_mode:
            # indicate that process should be terminated
            self.__terminate = True
        else:
            # indicate that process should be terminated
            self.__terminate = True
            # terminate stream
            if not (self.__stream is None):
                self.__stream.stop()

        # close event loop if specified
        if not (skip_loop):
            self.loop.close()
