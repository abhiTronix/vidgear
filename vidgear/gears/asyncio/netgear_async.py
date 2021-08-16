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
import sys
import zmq
import numpy as np
import asyncio
import inspect
import logging as log
import msgpack
import string
import secrets
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
    NetGear_Async can generate the same performance as NetGear API at about one-third the memory consumption, and also provide complete server-client handling with various
    options to use variable protocols/patterns similar to NetGear, but only support bidirectional data transmission exclusive mode.

    NetGear_Async is built on `zmq.asyncio`, and powered by a high-performance asyncio event loop called uvloop to achieve unwatchable high-speed and lag-free video streaming
    over the network with minimal resource constraints. NetGear_Async can transfer thousands of frames in just a few seconds without causing any significant load on your
    system.

    NetGear_Async provides complete server-client handling and options to use variable protocols/patterns similar to NetGear API but doesn't support any NetGear's Exclusive
    Modes yet. Furthermore, NetGear_Async allows us to define our custom Server as source to manipulate frames easily before sending them across the network.

    In addition to all this, NetGear_Async API also provides internal wrapper around VideoGear, which itself provides internal access to both CamGear and PiGear APIs, thereby
    granting it exclusive power for transferring frames incoming from any source to the network.

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
        source=None,
        camera_num=0,
        stream_mode=False,
        backend=0,
        colorspace=None,
        resolution=(640, 480),
        framerate=25,
        time_delay=0,
        # common parameters
        logging=False,
        **options
    ):

        """
        This constructor method initializes the object state and attributes of the NetGear_Async class.

        Parameters:
            address (str): sets the valid network address of the Server/Client.
            port (str): sets the valid Network Port of the Server/Client.
            protocol (str): sets the valid messaging protocol between Server/Client.
            pattern (int): sets the supported messaging pattern(flow of communication) between Server/Client
            receive_mode (bool): select the NetGear_Async's Mode of operation.
            timeout (int/float): controls the maximum waiting time(in sec) after which Client throws `TimeoutError`.
            enablePiCamera (bool): provide access to PiGear(if True) or CamGear(if False) APIs respectively.
            stabilize (bool): enable access to Stabilizer Class for stabilizing frames.
            camera_num (int): selects the camera module index which will be used as Rpi source.
            resolution (tuple): sets the resolution (i.e. `(width,height)`) of the Rpi source.
            framerate (int/float): sets the framerate of the Rpi source.
            source (based on input): defines the source for the input stream.
            stream_mode (bool): controls the exclusive YouTube Mode.
            backend (int): selects the backend for OpenCV's VideoCapture class.
            colorspace (str): selects the colorspace of the input stream.
            logging (bool): enables/disables logging.
            time_delay (int): time delay (in sec) before start reading the frames.
            options (dict): provides ability to alter Tweak Parameters of NetGear_Async, CamGear, PiGear & Stabilizer.
        """

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
        # initialize NetGear_Async's configuration dictionary
        self.config = {}
        # asyncio queue handler
        self.__queue = None
        # define Bidirectional mode
        self.__bi_mode = False  # handles Bidirectional mode state

        # assign timeout for Receiver end
        if timeout > 0 and isinstance(timeout, (int, float)):
            self.__timeout = float(timeout)
        else:
            self.__timeout = 15.0

        # generate 8-digit random system id
        self.__id = "".join(
            secrets.choice(string.ascii_uppercase + string.digits) for i in range(8)
        )

        # Handle user-defined options dictionary values
        # reformat dictionary
        options = {str(k).strip(): v for k, v in options.items()}
        # handle bidirectional mode
        if "bidirectional_mode" in options:
            value = options["bidirectional_mode"]
            # also check if pattern and source is valid
            if isinstance(value, bool) and pattern < 2 and source is None:
                # activate Bidirectional mode if specified
                self.__bi_mode = value
            else:
                # otherwise disable it
                self.__bi_mode = False
                logger.warning("Bidirectional data transmission is disabled!")
            # handle errors and logging
            if pattern >= 2:
                # raise error
                raise ValueError(
                    "[NetGear_Async:ERROR] :: `{}` pattern is not valid when Bidirectional Mode is enabled. Kindly refer Docs for more Information!".format(
                        pattern
                    )
                )
            elif not (source is None):
                raise ValueError(
                    "[NetGear_Async:ERROR] :: Custom source must be used when Bidirectional Mode is enabled. Kindly refer Docs for more Information!".format(
                        pattern
                    )
                )
            elif isinstance(value, bool) and self.__logging:
                # log Bidirectional mode activation
                logger.debug(
                    "Bidirectional Data Transmission is {} for this connection!".format(
                        "enabled" if value else "disabled"
                    )
                )
            else:
                logger.error("`bidirectional_mode` value is invalid!")
            # clean
            del options["bidirectional_mode"]

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
            # Handle video source
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
                    stream_mode=stream_mode,
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
            try:
                # import library
                import uvloop

                # Latest uvloop eventloop is only available for UNIX machines & python>=3.7.
                asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            except ImportError:
                pass

        # Retrieve event loop and assign it
        self.loop = asyncio.get_event_loop()
        # create asyncio queue if bidirectional mode activated
        self.__queue = asyncio.Queue(loop=self.loop) if self.__bi_mode else None
        # log eventloop for debugging
        if self.__logging:
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
                logger.debug("Launching NetGear_Async asynchronous generator!")
            # run loop executor for Receiver asynchronous generator
            self.loop.run_in_executor(None, self.recv_generator)
        else:
            # Otherwise launch Server handler
            if self.__logging:
                logger.debug("Creating NetGear_Async asynchronous server handler!")
            # create task for Server Handler
            self.task = asyncio.ensure_future(self.__server_handler(), loop=self.loop)
        # return instance
        return self

    async def __server_handler(self):
        """
        Handles various Server-end processes/tasks.
        """
        # validate assigned frame generator in NetGear_Async configuration
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
                "[NetGear_Async:ERROR] :: Assigned NetGear_Async configuration is invalid!"
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
            logger.critical(
                "Send Mode is successfully activated and ready to send data!"
            )
        except Exception as e:
            # log ad raise error if failed
            logger.exception(str(e))
            if self.__bi_mode:
                logger.error(
                    "Failed to activate Bidirectional Mode for this connection!"
                )
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
        async for dataframe in self.config["generator"]:
            # extract data if Bidirection mode
            if self.__bi_mode and len(dataframe) == 2:
                (data, frame) = dataframe
                if not (data is None) and isinstance(data, np.ndarray):
                    logger.warning(
                        "Skipped unsupported `data` of datatype: {}!".format(
                            type(data).__name__
                        )
                    )
                    data = None
                assert isinstance(
                    frame, np.ndarray
                ), "[NetGear_Async:ERROR] :: Invalid data recieved from server end!"
            elif self.__bi_mode:
                # raise error for invaid data
                raise ValueError(
                    "[NetGear_Async:ERROR] :: Send Mode only accepts tuple(data, frame) as input in Bidirectional Mode. \
                    Kindly refer vidgear docs!"
                )
            else:
                # otherwise just make a copy of frame
                frame = np.copy(dataframe)

            # check if retrieved frame is `CONTIGUOUS`
            if not (frame.flags["C_CONTIGUOUS"]):
                # otherwise make it
                frame = np.ascontiguousarray(frame, dtype=frame.dtype)

            # create data dict
            data_dict = dict(
                terminate=False,
                bi_mode=self.__bi_mode,
                data=data if not (data is None) else "",
            )
            # encode it
            data_enc = msgpack.packb(data_dict)
            # send the encoded data with correct flags
            await self.__msg_socket.send(data_enc, flags=zmq.SNDMORE)

            # encode frame
            frame_enc = msgpack.packb(frame, default=m.encode)
            # send the encoded frame
            await self.__msg_socket.send_multipart([frame_enc])

            # check if bidirectional patterns used
            if self.__msg_pattern < 2:
                # handle birectional data transfer if enabled
                if self.__bi_mode:
                    # get reciever encoded message withing timeout limit
                    recvdmsg_encoded = await asyncio.wait_for(
                        self.__msg_socket.recv(), timeout=self.__timeout
                    )
                    # retrieve reciever data from encoded message
                    recvd_data = msgpack.unpackb(recvdmsg_encoded, use_list=False)
                    # check message type
                    if recvd_data["return_type"] == "ndarray":  # nummpy.ndarray
                        # get encoded frame from reciever
                        recvdframe_encooded = await asyncio.wait_for(
                            self.__msg_socket.recv_multipart(), timeout=self.__timeout
                        )
                        # retrieve frame and put in queue
                        await self.__queue.put(
                            msgpack.unpackb(
                                recvdframe_encooded[0],
                                use_list=False,
                                object_hook=m.decode,
                            )
                        )
                    else:
                        # otherwise put data directly in queue
                        await self.__queue.put(recvd_data["return_data"])
                else:
                    # otherwise log recieved confirmation
                    recv_confirmation = await asyncio.wait_for(
                        self.__msg_socket.recv(), timeout=self.__timeout
                    )
                    if self.__logging:
                        logger.debug(recv_confirmation)

    async def recv_generator(self):
        """
        A default Asynchronous Frame Generator for NetGear_Async's Receiver-end.
        """
        # check whether `receive mode` is activated
        if not (self.__receive_mode):
            # raise Value error and exit
            self.__terminate = True
            raise ValueError(
                "[NetGear_Async:ERROR] :: `recv_generator()` function cannot be accessed while `receive_mode` is disabled. Kindly refer vidgear docs!"
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
            logger.critical("Receive Mode is activated successfully!")
        except Exception as e:
            logger.exception(str(e))
            if self.__bi_mode:
                logger.error(
                    "Failed to activate Bidirectional Mode for this connection!"
                )
            raise RuntimeError(
                "[NetGear_Async:ERROR] :: Failed to bind address: {} and pattern: {}!".format(
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
            # get encoded data message from server withing timeout limit
            datamsg_encoded = await asyncio.wait_for(
                self.__msg_socket.recv(), timeout=self.__timeout
            )
            # retrieve data from message
            data = msgpack.unpackb(datamsg_encoded, use_list=False)

            # terminate if exit` flag received from server
            if data["terminate"]:
                # send confirmation message to server if birectional patterns
                if self.__msg_pattern < 2:
                    # create termination confirmation message
                    return_dict = dict(
                        terminated="Device-`{}` successfully terminated!".format(
                            self.__id
                        ),
                    )
                    # encode message
                    retdata_enc = msgpack.packb(return_dict)
                    # send message back to server
                    await self.__msg_socket.send(retdata_enc)
                # break loop
                break

            # get encoded frame message from server withing timeout limit
            framemsg_encoded = await asyncio.wait_for(
                self.__msg_socket.recv_multipart(), timeout=self.__timeout
            )
            # retrieve frame from message
            frame = msgpack.unpackb(
                framemsg_encoded[0], use_list=False, object_hook=m.decode
            )

            # check if bidirectional patterns
            if self.__msg_pattern < 2:
                # handle birectional data transfer if enabled
                if self.__bi_mode and data["bi_mode"]:
                    # handle empty queue
                    if not self.__queue.empty():
                        return_data = await self.__queue.get()
                        self.__queue.task_done()
                    else:
                        return_data = None
                    # check if we are returning `ndarray` frames
                    if not (return_data is None) and isinstance(
                        return_data, np.ndarray
                    ):
                        # check whether the incoming frame is contiguous
                        if not (return_data.flags["C_CONTIGUOUS"]):
                            return_data = np.ascontiguousarray(
                                return_data, dtype=return_data.dtype
                            )

                        # create return type dict without data
                        rettype_dict = dict(
                            return_type=(type(return_data).__name__),
                            return_data=None,
                        )
                        # encode it
                        rettype_enc = msgpack.packb(rettype_dict)
                        # send it to server with correct flags
                        await self.__msg_socket.send(rettype_enc, flags=zmq.SNDMORE)

                        # encode return ndarray data
                        retframe_enc = msgpack.packb(return_data, default=m.encode)
                        # send it over network to server
                        await self.__msg_socket.send_multipart([retframe_enc])
                    else:
                        # otherwise create type and data dict
                        return_dict = dict(
                            return_type=(type(return_data).__name__),
                            return_data=return_data
                            if not (return_data is None)
                            else "",
                        )
                        # encode it
                        retdata_enc = msgpack.packb(return_dict)
                        # send it over network to server
                        await self.__msg_socket.send(retdata_enc)
                elif self.__bi_mode or data["bi_mode"]:
                    # raise error if bidirectional mode is disabled at server or client but not both
                    raise RuntimeError(
                        "[NetGear_Async:ERROR] :: Invalid configuration! Bidirectional Mode is not activate on {} end.".format(
                            "client" if self.__bi_mode else "server"
                        )
                    )
                else:
                    # otherwise just send confirmation message to server
                    await self.__msg_socket.send(
                        bytes(
                            "Data received on device: {} !".format(self.__id), "utf-8"
                        )
                    )
            # yield received tuple(data-frame) if birectional mode or else just frame
            yield (data["data"], frame) if self.__bi_mode else frame
            # sleep for sometime
            await asyncio.sleep(0.00001)

    async def __frame_generator(self):
        """
        Returns a default frame-generator for NetGear_Async's Server Handler.
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

    async def transceive_data(self, data=None):
        """
        Bidirectional Mode exculsive method to Transmit _(in Recieve mode)_ and Receive _(in Send mode)_ data in real-time.

        Parameters:
            data (any): inputs data _(of any datatype)_ for sending back to Server.
        """
        recvd_data = None
        if not self.__terminate:
            if self.__bi_mode:
                if self.__receive_mode:
                    await self.__queue.put(data)
                elif not self.__receive_mode and not self.__queue.empty():
                    recvd_data = await self.__queue.get()
                    self.__queue.task_done()
                else:
                    pass
            else:
                logger.error(
                    "`transceive_data()` function cannot be used when Bidirectional Mode is disabled."
                )
        return recvd_data

    async def __terminate_connection(self):
        """
        Internal asyncio method to safely terminate ZMQ connection and queues
        """
        # check whether `receive_mode` is enabled or not
        if self.__receive_mode:
            # indicate that process should be terminated
            self.__terminate = True
        else:
            # indicate that process should be terminated
            self.__terminate = True
            # terminate stream
            if not (self.__stream is None):
                self.__stream.stop()

            # signal `exit` flag for termination!
            data_dict = dict(terminate=True)
            data_enc = msgpack.packb(data_dict)
            await self.__msg_socket.send(data_enc)
            # check if bidirectional patterns
            if self.__msg_pattern < 2:
                # then receive and log confirmation
                recv_confirmation = await asyncio.wait_for(
                    self.__msg_socket.recv(), timeout=self.__timeout
                )
                recvd_conf = msgpack.unpackb(recv_confirmation, use_list=False)
                if self.__logging and "terminated" in recvd_conf:
                    logger.debug(recvd_conf["terminated"])

        # handle asyncio queues in bidirectional mode
        if self.__bi_mode:
            # empty queue if not
            while not self.__queue.empty():
                try:
                    self.__queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue
                self.__queue.task_done()
            # join queues
            await self.__queue.join()

        logger.critical(
            "{} successfully terminated!".format(
                "Receive Mode" if self.__receive_mode else "Send Mode"
            )
        )

    def close(self, skip_loop=False):
        """
        Terminates all NetGear_Async Asynchronous processes gracefully.

        Parameters:
            skip_loop (Boolean): (optional)used only if closing executor loop throws an error.
        """
        # log termination
        if self.__logging:
            logger.debug(
                "Terminating various {} Processes. Please wait...".format(
                    "Receive Mode" if self.__receive_mode else "Send Mode"
                )
            )

        # close connection gracefully
        self.loop.run_until_complete(self.__terminate_connection())

        # close event loop if specified
        if not (skip_loop):
            self.loop.close()
