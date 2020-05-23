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
import asyncio
import inspect
import logging as log
import platform

import cv2
import msgpack
import msgpack_numpy as m
import numpy as np
import zmq
import zmq.asyncio

from collections import deque
from ..videogear import VideoGear
from .helper import logger_handler


# define logger
logger = log.getLogger("NetGear_Async")
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


class NetGear_Async:
    """
    NetGear_Async is NetGear API on steroids. NetGear_Async is invented for uninterrupted network performance with excellent system-resources 
    handling, insignificant latency and no fancy stuff.

    NetGear_Async is asynchronous I/O API built on AsyncIO ZmQ API and powered by state-of-the-art asyncio event loop called `uvloop`, to 
    achieve unmatchable high-speed and lag-free video streaming over the network with minimal resource constraint. 

    NetGear_Async provides complete server-client handling and options to use variable protocols/patterns but no support for any NetGear API modes.
    It supports  all four ZeroMQ messaging patterns: i.e zmq.PAIR, zmq.REQ/zmq.REP, zmq.PUB/zmq.SUB, and zmq.PUSH/zmq.PULL , whereas supported 
    protocol are: 'tcp' and 'ipc'. 


    NetGear_Async specific parameters:

    :param address(string): sets the valid network address of the server/client. Network addresses unique identifiers across the 
                            network. Its default value of this parameter is based on mode it is working, 'localhost' for Send Mode
                            and `*` for Receive Mode.

    :param port(string): sets the valid network port of the server/client. A network port is a number that identifies one side 
                            of a connection between two devices on network. It is used determine to which process or application 
                            a message should be delivered. Its default value is '5555'. 

    :param protocol(string): sets the valid messaging protocol between server and client. A network protocol is a set of established rules
                             that dictates how to format, transmit and receive data so computer network devices - from servers and 
                             routers to endpoints - can communicate regardless of the differences in their underlying infrastructures, 
                             designs or standards. As of now, supported protocol are: 'tcp', and 'ipc'. Its default value is `tcp`.

    :param pattern(int): sets the supported messaging pattern(flow of communication) between server and client. Messaging patterns are network 
                            oriented architectural pattern that describes the flow of communication between interconnecting systems.
                            vidgear provides access to ZeroMQ's pre-optimized sockets which enables you to take advantage of these patterns.
                            The supported patterns are:
                                0: zmq.PAIR -> In this, the communication is bidirectional. There is no specific state stored within the socket. 
                                                There can only be one connected peer.The server listens on a certain port and a client connects to it.
                                1. zmq.REQ/zmq.REP -> In this, ZMQ REQ sockets can connect to many servers. The requests will be
                                                        interleaved or distributed to both the servers. socket zmq.REQ will block 
                                                        on send unless it has successfully received a reply back and socket zmq.REP 
                                                        will block on recv unless it has received a request.
                                2. zmq.PUB,zmq.SUB -> Publish/Subscribe is another classic pattern where senders of messages, called publishers, 
                                                        do not program the messages to be sent directly to specific receivers, called subscribers. 
                                                        Messages are published without the knowledge of what or if any subscriber of that knowledge exists.
                                3. zmq.PUSH, zmq.PULL - > Push and Pull sockets let you distribute messages to multiple workers, arranged in a pipeline. 
                                                        A Push socket will distribute sent messages to its Pull clients evenly. This is equivalent to 
                                                        producer/consumer model but the results computed by consumer are not sent upstream but downstream 
                                                        to another pull/consumer socket.
                            Its default value is `0`(i.e zmq.PAIR).

    :param (boolean) receive_mode: set this flag to select the Netgear's Mode of operation. This basically activates `Receive Mode`(if True) and `Send Mode`(if False). 
                                    Furthermore `recv()` function will only works when this flag is enabled(i.e. `Receive Mode`) and `send()` function will only works 
                                    when this flag is disabled(i.e.`Send Mode`). Checkout VidGear docs for usage details.
                                    Its default value is False(i.e. Send Mode is activated by default).


    VideoGear Specific parameters for NetGear_Async:
    
        :param (boolean) enablePiCamera: set this flag to access PiGear or CamGear class respectively. 
                                        / This means the if enablePiCamera flag is `True`, PiGear class will be accessed 
                                        / and if `False`, the camGear Class will be accessed. Its default value is False.

        :param (boolean) stabilize: set this flag to enable access to VidGear's Stabilizer Class. This basically enables(if True) or disables(if False) 
                                        video stabilization in VidGear. Its default value is False.

    CamGear Specific supported parameters for NetGear_Async:

        :param source : take the source value for CamGear Class. Its default value is 0. Valid Inputs are:
            - Index(integer): Valid index of the video device.
            - YouTube Url(string): Youtube URL as input.
            - Network_Stream_Address(string): Incoming Stream Valid Network address. 
            - GStreamer (string) videostream Support
        :param (boolean) y_tube: enables YouTube Mode in CamGear Class, i.e If enabled the class will interpret the given source string as YouTube URL. 
                                / Its default value is False.
        :param (int) backend: set the backend of the video stream (if specified). Its default value is 0.


    PiGear Specific supported parameters for NetGear_Async:
        :param (integer) camera_num: selects the camera module index that will be used by API. 
                                /   Its default value is 0 and shouldn't be altered until unless 
                                /   if you using Raspberry Pi 3/3+ compute module in your project along with multiple camera modules. 
                                /   Furthermore, Its value can only be greater than zero, otherwise, it will throw ValueError for any negative value.
        :param (tuple) resolution: sets the resolution (width,height) in Picamera class. Its default value is (640,480).
        :param (integer) framerate: sets the framerate in Picamera class. Its default value is 25.


    Common parameters for NetGear_Async: 
        :param (string) colorspace: set colorspace of the video stream. Its default value is None.
        :param (dict) **options: parameter supported by various API (whichever being accessed).
        :param (boolean) logging: set this flag to enable/disable error logging essential for debugging. Its default value is False.
        :param (integer) time_delay: sets time delay(in seconds) before start reading the frames. 
                            / This delay is essentially required for camera to warm-up. 
                            / Its default value is 0.
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
                logger.warning("Invalid pattern {pattern}. Defaulting to `zmq.PAIR`!".format(pattern=pattern))

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
            self.__timeout = 10.0
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
            if not (source is None) and source:
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
            else:
                # else set it to None
                self.config = {"generator": None}
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

        # Setup and assign uvloop event loop policy
        if platform.system() != "Windows":
            import uvloop

            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        # Retrieve event loop and assign it
        self.loop = asyncio.get_event_loop()

    def launch(self):
        """
        Launches asynchronous loop executors for various tasks
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
            self.task = asyncio.ensure_future(self.__server_handler(), loop = self.loop)
            # return instance
            return self

    async def __server_handler(self):
        """
        Handles various Server-end  processes
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

        # try connecting socket to assigned protocol, address and port
        try:
            self.__msg_socket.connect(
                self.__protocol + "://" + str(self.__address) + ":" + str(self.__port)
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
        finally:
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

    async def recv_generator(self):
        """
        Asynchronous Frame Generator for NetGear's receiver-end  
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

        try:
            # define exclusive socket options for patterns
            if self.__msg_pattern == 2:
                self.__msg_socket.setsockopt(zmq.SUBSCRIBE, b"")
            # bind socket to the assigned protocol, address and port
            self.__msg_socket.bind(
                self.__protocol + "://" + str(self.__address) + ":" + str(self.__port)
            )
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
        finally:
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
        Default frame generator for NetGear's Server Handler 
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
        Terminates the NetGear Asynchronous processes safely

        :param: skip_loop(Boolean) => used if closing loop throws error
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
