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
import numpy as np
import logging as log
import os
import random
import time

from collections import deque
from threading import Thread
from pkg_resources import parse_version
from .helper import generate_auth_certificates, logger_handler

# define logger
logger = log.getLogger("NetGear")
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


class NetGear:

    """
    NetGear is exclusively designed to transfer video frames synchronously and asynchronously between interconnecting systems over the network in real-time.

    NetGear implements a high-level wrapper around `PyZmQ` python library that contains python bindings for [ZeroMQ](http://zeromq.org/) - a high-performance 
    asynchronous distributed messaging library that provides a message queue, but unlike message-oriented middle-ware, its system can run without a dedicated 
    message broker. 

    NetGear also supports real-time *Frame Compression capabilities* for optimizing performance while sending the frames directly over the network, by encoding 
    the frame before sending it and decoding it on the client's end automatically in real-time.
    
    !!! info
        NetGear API now internally implements robust *Lazy Pirate pattern* (auto-reconnection) for its synchronous messaging patterns _(i.e. `zmq.PAIR` & `zmq.REQ/zmq.REP`)_ 
        at both Server and Client ends, where its API instead of doing a blocking receive, will:

        * Poll the socket and receive from it only when it's sure a reply has arrived.
        * Attempt to reconnect, if no reply has arrived within a timeout period.
        * Abandon the connection if there is still no reply after several requests.

    NetGear as of now seamlessly supports three ZeroMQ messaging patterns:

    - `zmq.PAIR` _(ZMQ Pair Pattern)_
    - `zmq.REQ/zmq.REP` _(ZMQ Request/Reply Pattern)_
    - `zmq.PUB/zmq.SUB` _(ZMQ Publish/Subscribe Pattern)_ 

    _whereas the supported protocol are: `tcp` and `ipc`_.

    ??? tip "Modes of Operation"

        * **Primary Modes**

            NetGear API primarily has two modes of operations:
              
            * **Send Mode:** _which employs `send()` function to send video frames over the network in real-time._
              
            * **Receive Mode:** _which employs `recv()` function to receive frames, sent over the network with *Send Mode* in real-time. The mode sends back confirmation when the 
            frame is received successfully in few patterns._

        * **Exclusive Modes**

            In addition to these primary modes, NetGear API offers applications-specific Exclusive Modes:

            * **Multi-Servers Mode:** _In this exclusive mode, NetGear API robustly **handles multiple servers at once**, thereby providing seamless access to frames and unidirectional 
            data transfer from multiple Servers/Publishers across the network in real-time._

            * **Multi-Clients Mode:** _In this exclusive mode, NetGear API robustly **handles multiple clients at once**, thereby providing seamless access to frames and unidirectional 
            data transfer to multiple Client/Consumers across the network in real-time._

            * **Bidirectional Mode:** _This exclusive mode **provides seamless support for bidirectional data transmission between between Server and Client along with video frames**._

            * **Secure Mode:** _In this exclusive mode, NetGear API **provides easy access to powerful, smart & secure ZeroMQ's Security Layers** that enables strong encryption on 
            data, and unbreakable authentication between the Server and Client with the help of custom certificates/keys that brings cheap, standardized privacy and authentication 
            for distributed systems over the network._
    """

    def __init__(
        self,
        address=None,
        port=None,
        protocol=None,
        pattern=0,
        receive_mode=False,
        logging=False,
        **options
    ):

        try:
            # import PyZMQ library
            import zmq
            from zmq.error import ZMQError

            # assign values to global variable for further use
            self.__zmq = zmq
            self.__ZMQError = ZMQError

        except ImportError as error:
            # raise error
            raise ImportError(
                "[NetGear:ERROR] :: pyzmq python library not installed. Kindly install it with `pip install pyzmq` command."
            )

        # enable logging if specified
        self.__logging = True if logging else False

        # define valid messaging patterns => `0`: zmq.PAIR, `1`:(zmq.REQ,zmq.REP), and `1`:(zmq.SUB,zmq.PUB)
        valid_messaging_patterns = {
            0: (zmq.PAIR, zmq.PAIR),
            1: (zmq.REQ, zmq.REP),
            2: (zmq.PUB, zmq.SUB),
        }

        # Handle messaging pattern
        msg_pattern = None
        # check whether user-defined messaging pattern is valid
        if isinstance(pattern, int) and pattern in valid_messaging_patterns.keys():
            # assign value
            msg_pattern = valid_messaging_patterns[pattern]
        else:
            # otherwise default to 0:`zmq.PAIR`
            pattern = 0
            msg_pattern = valid_messaging_patterns[pattern]
            if self.__logging:
                logger.warning(
                    "Wrong pattern value, Defaulting to `zmq.PAIR`! Kindly refer Docs for more Information."
                )
        # assign pattern to global parameter for further use
        self.__pattern = pattern

        # Handle messaging protocol
        if protocol is None or not (protocol in ["tcp", "ipc"]):
            # else default to `tcp` protocol
            protocol = "tcp"
            # log it
            if self.__logging:
                logger.warning(
                    "Protocol is not supported or not provided. Defaulting to `tcp` protocol!"
                )

        # Handle connection params

        self.__msg_flag = 0  # handles connection flags
        self.__msg_copy = False  # handles whether to copy data
        self.__msg_track = False  # handles whether to track packets

        # Handle NetGear's internal exclusive modes and params

        # define Multi-Server mode
        self.__multiserver_mode = False  # handles multi-server_mode state

        # define Multi-Client mode
        self.__multiclient_mode = False  # handles multi-client_mode state

        # define Bi-directional mode
        self.__bi_mode = False  # handles bi-directional mode state

        # define Secure mode
        valid_security_mech = {0: "Grasslands", 1: "StoneHouse", 2: "IronHouse"}
        self.__secure_mode = 0  # handles ZMQ security layer status
        auth_cert_dir = ""  # handles valid ZMQ certificates dir
        self.__auth_publickeys_dir = ""  # handles valid ZMQ public certificates dir
        self.__auth_secretkeys_dir = ""  # handles valid ZMQ private certificates dir
        overwrite_cert = False  # checks if certificates overwriting allowed
        custom_cert_location = ""  # handles custom ZMQ certificates path

        # define frame-compression handler
        self.__compression = (
            ".jpg" if not (address is None) else ""
        )  # disabled by default for local connections
        self.__compression_params = (
            cv2.IMREAD_COLOR
            if receive_mode
            else [
                cv2.IMWRITE_JPEG_QUALITY,
                85,
                cv2.IMWRITE_JPEG_PROGRESSIVE,
                False,
                cv2.IMWRITE_JPEG_OPTIMIZE,
                True,
            ]
        )
        # defines frame compression on return data
        self.__ex_compression_params = None

        # define receiver return data handler
        self.__return_data = None

        # generate random system id
        self.__id = "".join(random.choice("0123456789ABCDEF") for i in range(5))

        # define termination flag
        self.__terminate = False

        # additional settings for reliability
        if pattern < 2:
            # define zmq poller for reliable transmission
            self.__poll = zmq.Poller()
            # define max retries
            self.__max_retries = 3
            # request timeout
            self.__request_timeout = 4000  # 4 secs

        # Handle user-defined options dictionary values

        # reformat dictionary
        options = {str(k).strip(): v for k, v in options.items()}

        # loop over dictionary key & values and assign to global variables if valid
        for key, value in options.items():

            if key == "multiserver_mode" and isinstance(value, bool):
                # check if valid pattern assigned
                if pattern > 0:
                    # activate Multi-server mode
                    self.__multiserver_mode = value
                else:
                    # otherwise disable it and raise error
                    self.__multiserver_mode = False
                    logger.critical("Multi-Server Mode is disabled!")
                    raise ValueError(
                        "[NetGear:ERROR] :: `{}` pattern is not valid when Multi-Server Mode is enabled. Kindly refer Docs for more Information.".format(
                            pattern
                        )
                    )

            if key == "multiclient_mode" and isinstance(value, bool):
                # check if valid pattern assigned
                if pattern > 0:
                    # activate Multi-client mode
                    self.__multiclient_mode = value
                else:
                    # otherwise disable it and raise error
                    self.__multiclient_mode = False
                    logger.critical("Multi-Client Mode is disabled!")
                    raise ValueError(
                        "[NetGear:ERROR] :: `{}` pattern is not valid when Multi-Client Mode is enabled. Kindly refer Docs for more Information.".format(
                            pattern
                        )
                    )

            elif (
                key == "secure_mode"
                and isinstance(value, int)
                and (value in valid_security_mech)
            ):
                # check if installed libzmq version is valid
                assert zmq.zmq_version_info() >= (
                    4,
                    0,
                ), "[NetGear:ERROR] :: ZMQ Security feature is not supported in libzmq version < 4.0."
                # assign valid mode
                self.__secure_mode = value

            elif key == "custom_cert_location" and isinstance(value, str):
                # verify custom auth certificates path for secure mode
                assert os.access(
                    value, os.W_OK
                ), "[NetGear:ERROR] :: Permission Denied!, cannot write ZMQ authentication certificates to '{}' directory!".format(
                    value
                )
                assert os.path.isdir(
                    os.path.abspath(value)
                ), "[NetGear:ERROR] :: `custom_cert_location` value must be the path to a valid directory!"
                custom_cert_location = os.path.abspath(value)

            elif key == "overwrite_cert" and isinstance(value, bool):
                # enable/disable auth certificate overwriting in secure mode
                overwrite_cert = value

            elif key == "bidirectional_mode" and isinstance(value, bool):
                # check if pattern is valid
                if pattern < 2:
                    # activate bi-directional mode if specified
                    self.__bi_mode = value
                else:
                    # otherwise disable it and raise error
                    self.__bi_mode = False
                    logger.critical("Bi-Directional data transmission is disabled!")
                    raise ValueError(
                        "[NetGear:ERROR] :: `{}` pattern is not valid when Bi-Directional Mode is enabled. Kindly refer Docs for more Information!".format(
                            pattern
                        )
                    )

            elif (
                key == "compression_format"
                and isinstance(value, str)
                and value.lower().strip() in [".jpg", ".jpeg", ".bmp", ".png"]
            ):
                # assign frame-compression encoding value
                self.__compression = value.lower().strip()
            elif key == "compression_param":
                # assign encoding/decoding params/flags for frame-compression if valid
                if receive_mode and isinstance(value, int):
                    self.__compression_params = value
                elif not (receive_mode) and isinstance(value, list):
                    self.__compression_params = value
                elif isinstance(value, tuple) and len(value) == 2:
                    if receive_mode:
                        self.__compression_params = [
                            x for x in value if isinstance(x, int)
                        ][0]
                        self.__ex_compression_params = [
                            x for x in value if (value and isinstance(x, list))
                        ][0]
                    else:
                        self.__compression_params = [
                            x for x in value if (value and isinstance(x, list))
                        ][0]
                        self.__ex_compression_params = [
                            x for x in value if isinstance(x, int)
                        ][0]
                else:
                    logger.warning(
                        "Invalid compression parameters: {} skipped!".format(value)
                    )

            # assign maximum retries in synchronous patterns
            elif key == "max_retries" and isinstance(value, int) and pattern < 2:
                if value >= 0:
                    self.__max_retries = value
                else:
                    logger.warning("Invalid `max_retries` value skipped!")
            # assign request timeout in synchronous patterns
            elif key == "request_timeout" and isinstance(value, int) and pattern < 2:
                if value >= 4:
                    self.__request_timeout = value * 1000  # covert to milliseconds
                else:
                    logger.warning("Invalid `request_timeout` value skipped!")

            # assign ZMQ flags
            elif key == "flag" and isinstance(value, int):
                self.__msg_flag = value
            elif key == "copy" and isinstance(value, bool):
                self.__msg_copy = value
            elif key == "track" and isinstance(value, bool):
                self.__msg_track = value
            else:
                pass

        # Handle Secure mode
        if self.__secure_mode:

            # import required libs
            import zmq.auth
            from zmq.auth.thread import ThreadAuthenticator

            # activate and log if overwriting is enabled
            if overwrite_cert:
                if not receive_mode:
                    if self.__logging:
                        logger.warning(
                            "Overwriting ZMQ Authentication certificates over previous ones!"
                        )
                else:
                    overwrite_cert = False
                    if self.__logging:
                        logger.critical(
                            "Overwriting ZMQ Authentication certificates is disabled for Client's end!"
                        )

            # Validate certificate generation paths
            try:
                # check if custom certificates path is specified
                if custom_cert_location:
                    (
                        auth_cert_dir,
                        self.__auth_secretkeys_dir,
                        self.__auth_publickeys_dir,
                    ) = generate_auth_certificates(
                        custom_cert_location, overwrite=overwrite_cert, logging=logging
                    )
                else:
                    # otherwise auto-generate suitable path
                    from os.path import expanduser

                    (
                        auth_cert_dir,
                        self.__auth_secretkeys_dir,
                        self.__auth_publickeys_dir,
                    ) = generate_auth_certificates(
                        os.path.join(expanduser("~"), ".vidgear"),
                        overwrite=overwrite_cert,
                        logging=logging,
                    )
                # log it
                if self.__logging:
                    logger.debug(
                        "`{}` is the default location for storing ZMQ authentication certificates/keys.".format(
                            auth_cert_dir
                        )
                    )
            except Exception as e:
                # catch if any error occurred and disable Secure mode
                logger.exception(str(e))
                self.__secure_mode = 0
                logger.critical(
                    "ZMQ Security Mechanism is disabled for this connection due to errors!"
                )

        # Handle multiple exclusive modes if enabled

        if self.__multiclient_mode and self.__multiserver_mode:
            raise ValueError(
                "[NetGear:ERROR] :: Multi-Client and Multi-Server Mode cannot be enabled simultaneously!"
            )
        elif self.__multiserver_mode or self.__multiclient_mode:
            # check if Bi-directional Mode also enabled
            if self.__bi_mode:
                # disable bi_mode if enabled
                self.__bi_mode = False
                logger.warning(
                    "Bi-Directional Data Transmission is disabled when {} Mode is Enabled due to incompatibility!".format(
                        "Multi-Server" if self.__multiserver_mode else "Multi-Client"
                    )
                )
        elif self.__bi_mode:
            # log Bi-directional mode activation
            if self.__logging:
                logger.debug(
                    "Bi-Directional Data Transmission is enabled for this connection!"
                )

        # handle frame compression on return data
        if (
            (self.__bi_mode or self.__multiclient_mode)
            and self.__compression
            and self.__ex_compression_params is None
        ):
            # define exclusive compression params
            self.__ex_compression_params = (
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    85,
                    cv2.IMWRITE_JPEG_PROGRESSIVE,
                    False,
                    cv2.IMWRITE_JPEG_OPTIMIZE,
                    True,
                ]
                if receive_mode
                else cv2.IMREAD_COLOR
            )

        # define messaging context instance
        self.__msg_context = zmq.Context.instance()

        # initialize and assign receive mode to global variable
        self.__receive_mode = receive_mode

        # check whether `receive_mode` is enabled
        if self.__receive_mode:

            # define connection address
            if address is None:
                address = "*"  # define address

            # check if multiserver_mode is enabled
            if self.__multiserver_mode:
                # check if unique server port address list/tuple is assigned or not in multiserver_mode
                if port is None or not isinstance(port, (tuple, list)):
                    # raise error if not
                    raise ValueError(
                        "[NetGear:ERROR] :: Incorrect port value! Kindly provide a list/tuple of Server ports while Multi-Server mode is enabled. For more information refer VidGear docs."
                    )
                else:
                    # otherwise log it
                    logger.debug(
                        "Enabling Multi-Server Mode at PORTS: {}!".format(port)
                    )
                # create port address buffer for keeping track of connected client's port(s)
                self.__port_buffer = []
            # check if multiclient_mode is enabled
            elif self.__multiclient_mode:
                # check if unique server port address is assigned or not in multiclient_mode
                if port is None:
                    # raise error if not
                    raise ValueError(
                        "[NetGear:ERROR] :: Kindly provide a unique & valid port value at Client-end. For more information refer VidGear docs."
                    )
                else:
                    # otherwise log it
                    logger.debug(
                        "Enabling Multi-Client Mode at PORT: {} on this device!".format(
                            port
                        )
                    )
                # assign value to global variable
                self.__port = port
            else:
                # otherwise assign local port address if None
                if port is None:
                    port = "5555"

            try:
                # activate secure_mode threaded authenticator
                if self.__secure_mode > 0:
                    # start an authenticator for this context
                    auth = ThreadAuthenticator(self.__msg_context)
                    auth.start()
                    auth.allow(str(address))  # allow current address

                    # check if `IronHouse` is activated
                    if self.__secure_mode == 2:
                        # tell authenticator to use the certificate from given valid dir
                        auth.configure_curve(
                            domain="*", location=self.__auth_publickeys_dir
                        )
                    else:
                        # otherwise tell the authenticator how to handle the CURVE requests, if `StoneHouse` is activated
                        auth.configure_curve(
                            domain="*", location=zmq.auth.CURVE_ALLOW_ANY
                        )

                # define thread-safe messaging socket
                self.__msg_socket = self.__msg_context.socket(msg_pattern[1])

                # define pub-sub flag
                if self.__pattern == 2:
                    self.__msg_socket.set_hwm(1)

                # enable specified secure mode for the socket
                if self.__secure_mode > 0:
                    # load server key
                    server_secret_file = os.path.join(
                        self.__auth_secretkeys_dir, "server.key_secret"
                    )
                    server_public, server_secret = zmq.auth.load_certificate(
                        server_secret_file
                    )
                    # load  all CURVE keys
                    self.__msg_socket.curve_secretkey = server_secret
                    self.__msg_socket.curve_publickey = server_public
                    # enable CURVE connection for this socket
                    self.__msg_socket.curve_server = True

                # define exclusive socket options for patterns
                if self.__pattern == 2:
                    self.__msg_socket.setsockopt_string(zmq.SUBSCRIBE, "")

                # if multiserver_mode is enabled, then assign port addresses to zmq socket
                if self.__multiserver_mode:
                    # bind socket to given server protocol, address and ports
                    for pt in port:
                        self.__msg_socket.bind(
                            protocol + "://" + str(address) + ":" + str(pt)
                        )
                else:
                    # bind socket to given protocol, address and port normally
                    self.__msg_socket.bind(
                        protocol + "://" + str(address) + ":" + str(port)
                    )

                # additional settings
                if pattern < 2:
                    self.__connection_address = (
                        protocol + "://" + str(address) + ":" + str(port)
                    )
                    self.__msg_pattern = msg_pattern[1]
                    self.__poll.register(self.__msg_socket, zmq.POLLIN)

                    if self.__logging:
                        logger.debug(
                            "Reliable transmission is enabled for this pattern with max-retries: {} and timeout: {} secs.".format(
                                self.__max_retries, self.__request_timeout / 1000
                            )
                        )

            except Exception as e:
                # otherwise log and raise error
                logger.exception(str(e))
                if self.__secure_mode:
                    logger.critical(
                        "Failed to activate Secure Mode: `{}` for this connection!".format(
                            valid_security_mech[self.__secure_mode]
                        )
                    )
                if self.__multiserver_mode or self.__multiclient_mode:
                    raise RuntimeError(
                        "[NetGear:ERROR] :: Receive Mode failed to activate {} Mode at address: {} with pattern: {}! Kindly recheck all parameters.".format(
                            "Multi-Server"
                            if self.__multiserver_mode
                            else "Multi-Client",
                            (protocol + "://" + str(address) + ":" + str(port)),
                            pattern,
                        )
                    )
                else:
                    if self.__bi_mode:
                        logger.critical(
                            "Failed to activate Bi-Directional Mode for this connection!"
                        )
                    raise RuntimeError(
                        "[NetGear:ERROR] :: Receive Mode failed to bind address: {} and pattern: {}! Kindly recheck all parameters.".format(
                            (protocol + "://" + str(address) + ":" + str(port)), pattern
                        )
                    )

            # Handle threaded queue mode
            if self.__logging:
                logger.debug(
                    "Threaded Queue Mode is enabled by default for this connection."
                )

            # define deque and assign it to global var
            self.__queue = deque(maxlen=96)  # max len 96 to check overflow

            # initialize and start threaded recv_handler
            self.__thread = Thread(target=self.__recv_handler, name="NetGear", args=())
            self.__thread.daemon = True
            self.__thread.start()

            if self.__logging:
                # finally log progress
                logger.debug(
                    "Successfully Binded to address: {} with pattern: {}.".format(
                        (protocol + "://" + str(address) + ":" + str(port)), pattern
                    )
                )
                if self.__compression:
                    logger.debug(
                        "Optimized `{}` Frame Compression is enabled with decoding flag:`{}` for this connection.".format(
                            self.__compression, self.__compression_params
                        )
                    )
                if self.__secure_mode:
                    logger.debug(
                        "Successfully enabled ZMQ Security Mechanism: `{}` for this connection.".format(
                            valid_security_mech[self.__secure_mode]
                        )
                    )
                logger.debug("Multi-threaded Receive Mode is successfully enabled.")
                logger.debug("Unique System ID is {}.".format(self.__id))
                logger.debug("Receive Mode is now activated.")

        else:

            # otherwise default to `Send Mode`

            # define connection address
            if address is None:
                address = "localhost"

            # check if multiserver_mode is enabled
            if self.__multiserver_mode:
                # check if unique server port address is assigned or not in multiserver_mode
                if port is None:
                    # raise error if not
                    raise ValueError(
                        "[NetGear:ERROR] :: Kindly provide a unique & valid port value at Server-end. For more information refer VidGear docs."
                    )
                else:
                    # otherwise log it
                    logger.debug(
                        "Enabling Multi-Server Mode at PORT: {} on this device!".format(
                            port
                        )
                    )
                # assign value to global variable
                self.__port = port
            # check if multiclient_mode is enabled
            elif self.__multiclient_mode:
                # check if unique client port address list/tuple is assigned or not in multiclient_mode
                if port is None or not isinstance(port, (tuple, list)):
                    # raise error if not
                    raise ValueError(
                        "[NetGear:ERROR] :: Incorrect port value! Kindly provide a list/tuple of Client ports while Multi-Client mode is enabled. For more information refer VidGear docs."
                    )
                else:
                    # otherwise log it
                    logger.debug(
                        "Enabling Multi-Client Mode at PORTS: {}!".format(port)
                    )
                # create port address buffer for keeping track of connected client ports
                self.__port_buffer = []
            else:
                # otherwise assign local port address if None
                if port is None:
                    port = "5555"

            try:
                # activate secure_mode threaded authenticator
                if self.__secure_mode > 0:
                    # start an authenticator for this context
                    auth = ThreadAuthenticator(self.__msg_context)
                    auth.start()
                    auth.allow(str(address))  # allow current address

                    # check if `IronHouse` is activated
                    if self.__secure_mode == 2:
                        # tell authenticator to use the certificate from given valid dir
                        auth.configure_curve(
                            domain="*", location=self.__auth_publickeys_dir
                        )
                    else:
                        # otherwise tell the authenticator how to handle the CURVE requests, if `StoneHouse` is activated
                        auth.configure_curve(
                            domain="*", location=zmq.auth.CURVE_ALLOW_ANY
                        )

                # define thread-safe messaging socket
                self.__msg_socket = self.__msg_context.socket(msg_pattern[0])

                # if req/rep pattern, define additional flags
                if self.__pattern == 1:
                    self.__msg_socket.REQ_RELAXED = True
                    self.__msg_socket.REQ_CORRELATE = True

                # if pub/sub pattern, define additional optimizer
                if self.__pattern == 2:
                    self.__msg_socket.set_hwm(1)

                # enable specified secure mode for the socket
                if self.__secure_mode > 0:
                    # load client key
                    client_secret_file = os.path.join(
                        self.__auth_secretkeys_dir, "client.key_secret"
                    )
                    client_public, client_secret = zmq.auth.load_certificate(
                        client_secret_file
                    )
                    # load  all CURVE keys
                    self.__msg_socket.curve_secretkey = client_secret
                    self.__msg_socket.curve_publickey = client_public
                    # load server key
                    server_public_file = os.path.join(
                        self.__auth_publickeys_dir, "server.key"
                    )
                    server_public, _ = zmq.auth.load_certificate(server_public_file)
                    # inject public key to make a CURVE connection.
                    self.__msg_socket.curve_serverkey = server_public

                # check if multi-client_mode is enabled
                if self.__multiclient_mode:
                    # bind socket to given server protocol, address and ports
                    for pt in port:
                        self.__msg_socket.connect(
                            protocol + "://" + str(address) + ":" + str(pt)
                        )
                else:
                    # connect socket to given protocol, address and port
                    self.__msg_socket.connect(
                        protocol + "://" + str(address) + ":" + str(port)
                    )

                # additional settings
                if pattern < 2 and not self.__multiclient_mode:
                    self.__connection_address = (
                        protocol + "://" + str(address) + ":" + str(port)
                    )
                    self.__msg_pattern = msg_pattern[0]
                    self.__poll.register(self.__msg_socket, zmq.POLLIN)

                    if self.__logging:
                        logger.debug(
                            "Reliable transmission is enabled for this pattern with max-retries: {} and timeout: {} secs.".format(
                                self.__max_retries, self.__request_timeout / 1000
                            )
                        )

            except Exception as e:
                # otherwise log and raise error
                logger.exception(str(e))
                if self.__secure_mode:
                    logger.critical(
                        "Failed to activate Secure Mode: `{}` for this connection!".format(
                            valid_security_mech[self.__secure_mode]
                        )
                    )
                if self.__multiserver_mode or self.__multiclient_mode:
                    raise RuntimeError(
                        "[NetGear:ERROR] :: Send Mode failed to activate {} Mode at address: {} with pattern: {}! Kindly recheck all parameters.".format(
                            "Multi-Server"
                            if self.__multiserver_mode
                            else "Multi-Client",
                            (protocol + "://" + str(address) + ":" + str(port)),
                            pattern,
                        )
                    )
                else:
                    if self.__bi_mode:
                        logger.critical(
                            "Failed to activate Bi-Directional Mode for this connection!"
                        )
                    raise RuntimeError(
                        "[NetGear:ERROR] :: Send Mode failed to connect address: {} and pattern: {}! Kindly recheck all parameters.".format(
                            (protocol + "://" + str(address) + ":" + str(port)), pattern
                        )
                    )

            if self.__logging:
                # finally log progress
                logger.debug(
                    "Successfully connected to address: {} with pattern: {}.".format(
                        (protocol + "://" + str(address) + ":" + str(port)), pattern
                    )
                )
                if self.__compression:
                    logger.debug(
                        "Optimized `{}` Frame Compression is enabled with encoding params:`{}` for this connection.".format(
                            self.__compression, self.__compression_params
                        )
                    )
                if self.__secure_mode:
                    logger.debug(
                        "Enabled ZMQ Security Mechanism: `{}` for this connection.".format(
                            valid_security_mech[self.__secure_mode]
                        )
                    )
                logger.debug("Unique System ID is {}.".format(self.__id))
                logger.debug(
                    "Send Mode is successfully activated and ready to send data."
                )

    def __recv_handler(self):

        """
        A threaded receiver handler, that keep iterating data from ZMQ socket to a internally monitored deque, 
        until the thread is terminated, or socket disconnects.
        """
        # initialize frame variable
        frame = None

        # keep looping infinitely until the thread is terminated
        while not self.__terminate:

            # check queue buffer for overflow
            if len(self.__queue) >= 96:
                # stop iterating if overflowing occurs
                time.sleep(0.000001)
                continue

            if self.__pattern < 2:
                socks = dict(self.__poll.poll(self.__request_timeout * 3))
                if socks.get(self.__msg_socket) == self.__zmq.POLLIN:
                    msg_json = self.__msg_socket.recv_json(
                        flags=self.__msg_flag | self.__zmq.DONTWAIT
                    )
                else:
                    logger.critical("No response from Server(s), Reconnecting again...")
                    self.__msg_socket.close(linger=0)
                    self.__poll.unregister(self.__msg_socket)
                    self.__max_retries -= 1

                    if not (self.__max_retries):
                        if self.__multiserver_mode:
                            logger.error("All Servers seems to be offline, Abandoning!")
                        else:
                            logger.error("Server seems to be offline, Abandoning!")
                        self.__terminate = True
                        continue

                    # Create new connection
                    try:
                        self.__msg_socket = self.__msg_context.socket(
                            self.__msg_pattern
                        )
                        self.__msg_socket.bind(self.__connection_address)
                    except Exception as e:
                        logger.exception(str(e))
                        self.__terminate = True
                        raise RuntimeError("API failed to restart the Client-end!")
                    self.__poll.register(self.__msg_socket, self.__zmq.POLLIN)

                    continue
            else:
                msg_json = self.__msg_socket.recv_json(flags=self.__msg_flag)

            # check if terminate_flag` received
            if msg_json["terminate_flag"]:
                # if multiserver_mode is enabled
                if self.__multiserver_mode:
                    # check and remove from which ports signal is received
                    if msg_json["port"] in self.__port_buffer:
                        # if pattern is 1, then send back server the info about termination
                        if self.__pattern == 1:
                            self.__msg_socket.send_string(
                                "Termination signal successfully received at client!"
                            )
                        self.__port_buffer.remove(msg_json["port"])
                        if self.__logging:
                            logger.warning(
                                "Termination signal received from Server at port: {}!".format(
                                    msg_json["port"]
                                )
                            )
                    # if termination signal received from all servers then exit client.
                    if not self.__port_buffer:
                        logger.critical(
                            "Termination signal received from all Servers!!!"
                        )
                        self.__terminate = True  # termination
                else:
                    # if pattern is 1, then send back server the info about termination
                    if self.__pattern == 1:
                        self.__msg_socket.send_string(
                            "Termination signal successfully received at Client's end!"
                        )
                    # termination
                    self.__terminate = True
                    # notify client
                    if self.__logging:
                        logger.critical("Termination signal received from server!")
                continue

            msg_data = self.__msg_socket.recv(
                flags=self.__msg_flag | self.__zmq.DONTWAIT,
                copy=self.__msg_copy,
                track=self.__msg_track,
            )

            if self.__pattern < 2:

                if self.__bi_mode or self.__multiclient_mode:

                    if not (self.__return_data is None) and isinstance(
                        self.__return_data, np.ndarray
                    ):

                        # handle return data
                        return_data = self.__return_data[:]

                        # check whether exit_flag is False
                        if not (return_data.flags["C_CONTIGUOUS"]):
                            # check whether the incoming frame is contiguous
                            return_data = np.ascontiguousarray(
                                return_data, dtype=return_data.dtype
                            )

                        # handle encoding
                        if self.__compression:
                            retval, return_data = cv2.imencode(
                                self.__compression,
                                return_data,
                                self.__ex_compression_params,
                            )
                            # check if it works
                            if not (retval):
                                # otherwise raise error and exit
                                self.__terminate = True
                                raise RuntimeError(
                                    "[NetGear:ERROR] :: Return frame compression failed with encoding: {} and parameters: {}.".format(
                                        self.__compression, self.__ex_compression_params
                                    )
                                )

                        if self.__bi_mode:
                            return_dict = dict(
                                return_type=(type(return_data).__name__),
                                compression=str(self.__compression),
                                array_dtype=str(return_data.dtype),
                                array_shape=return_data.shape,
                                data=None,
                            )
                        else:
                            return_dict = dict(
                                port=self.__port,
                                return_type=(type(return_data).__name__),
                                compression=str(self.__compression),
                                array_dtype=str(return_data.dtype),
                                array_shape=return_data.shape,
                                data=None,
                            )
                        # send the json dict
                        self.__msg_socket.send_json(
                            return_dict, self.__msg_flag | self.__zmq.SNDMORE
                        )
                        # send the array with correct flags
                        self.__msg_socket.send(
                            return_data,
                            flags=self.__msg_flag,
                            copy=self.__msg_copy,
                            track=self.__msg_track,
                        )
                    else:
                        if self.__bi_mode:
                            return_dict = dict(
                                return_type=(type(self.__return_data).__name__),
                                data=self.__return_data,
                            )
                        else:
                            return_dict = dict(
                                port=self.__port,
                                return_type=(type(self.__return_data).__name__),
                                data=self.__return_data,
                            )
                        self.__msg_socket.send_json(return_dict, self.__msg_flag)
                else:
                    # send confirmation message to server
                    self.__msg_socket.send_string(
                        "Data received on device: {} !".format(self.__id)
                    )
            else:
                if self.__return_data and self.__logging:
                    logger.warning("`return_data` is disabled for this pattern!")

            # recover and reshape frame from buffer
            frame_buffer = np.frombuffer(msg_data, dtype=msg_json["dtype"])
            frame = frame_buffer.reshape(msg_json["shape"])

            # check if encoding was enabled
            if msg_json["compression"]:
                frame = cv2.imdecode(frame, self.__compression_params)
                # check if valid frame returned
                if frame is None:
                    self.__terminate = True
                    # otherwise raise error and exit
                    raise RuntimeError(
                        "[NetGear:ERROR] :: Received compressed frame `{}` decoding failed with flag: {}.".format(
                            msg_json["compression"], self.__compression_params
                        )
                    )

            # check if multiserver_mode
            if self.__multiserver_mode:
                # save the unique port addresses
                if not msg_json["port"] in self.__port_buffer:
                    self.__port_buffer.append(msg_json["port"])
                # extract if any message from server and display it
                if msg_json["message"]:
                    self.__queue.append((msg_json["port"], msg_json["message"], frame))
                else:
                    # append recovered unique port and frame to queue
                    self.__queue.append((msg_json["port"], frame))
            # extract if any message from server if Bi-Directional Mode is enabled
            elif self.__bi_mode:
                if msg_json["message"]:
                    # append grouped frame and data to queue
                    self.__queue.append((msg_json["message"], frame))
                else:
                    self.__queue.append((None, frame))
            else:
                # otherwise append recovered frame to queue
                self.__queue.append(frame)

    def recv(self, return_data=None):
        """
        A Receiver end method, that extracts received frames synchronously from monitored deque, while maintaining a 
        fixed-length frame buffer in the memory, and blocks the thread if the deque is full.

        Parameters:
            return_data (any): inputs return data _(of any datatype)_, for sending back to Server. 

        **Returns:** A n-dimensional numpy array. 
        """
        # check whether `receive mode` is activated
        if not (self.__receive_mode):
            # raise value error and exit
            self.__terminate = True
            raise ValueError(
                "[NetGear:ERROR] :: `recv()` function cannot be used while receive_mode is disabled. Kindly refer vidgear docs!"
            )

        # handle bi-directional return data
        if (self.__bi_mode or self.__multiclient_mode) and not (return_data is None):
            self.__return_data = return_data

        # check whether or not termination flag is enabled
        while not self.__terminate:
            try:
                # check if queue is empty
                if len(self.__queue) > 0:
                    return self.__queue.popleft()
                else:
                    time.sleep(0.00001)
                    continue
            except KeyboardInterrupt:
                self.__terminate = True
                break
        # otherwise return NoneType
        return None

    def send(self, frame, message=None):
        """
        A Server end method, that sends the data and frames over the network to Client(s).

        Parameters:
            frame (numpy.ndarray): inputs numpy array(frame).
            message (any): input for sending additional data _(of any datatype except `numpy.ndarray`)_ to Client(s).

        **Returns:** A n-dimensional numpy array in selected modes, otherwise None-type.
        
        """
        # check whether `receive_mode` is disabled
        if self.__receive_mode:
            # raise value error and exit
            self.__terminate = True
            raise ValueError(
                "[NetGear:ERROR] :: `send()` function cannot be used while receive_mode is enabled. Kindly refer vidgear docs!"
            )

        if not (message is None) and isinstance(message, np.ndarray):
            logger.warning(
                "Skipped unsupported `message` of datatype: {}!".format(
                    type(message).__name__
                )
            )
            message = None

        # define exit_flag and assign value
        exit_flag = True if (frame is None or self.__terminate) else False

        # check whether exit_flag is False
        if not (exit_flag) and not (frame.flags["C_CONTIGUOUS"]):
            # check whether the incoming frame is contiguous
            frame = np.ascontiguousarray(frame, dtype=frame.dtype)

        # handle encoding
        if self.__compression:
            retval, frame = cv2.imencode(
                self.__compression, frame, self.__compression_params
            )
            # check if it works
            if not (retval):
                # otherwise raise error and exit
                self.__terminate = True
                raise ValueError(
                    "[NetGear:ERROR] :: Frame compression failed with encoding: {} and parameters: {}.".format(
                        self.__compression, self.__compression_params
                    )
                )

        # check if multiserver_mode is activated
        if self.__multiserver_mode:
            # prepare the exclusive json dict and assign values with unique port
            msg_dict = dict(
                terminate_flag=exit_flag,
                compression=str(self.__compression),
                port=self.__port,
                pattern=str(self.__pattern),
                message=message,
                dtype=str(frame.dtype),
                shape=frame.shape,
            )
        else:
            # otherwise prepare normal json dict and assign values
            msg_dict = dict(
                terminate_flag=exit_flag,
                compression=str(self.__compression),
                message=message,
                pattern=str(self.__pattern),
                dtype=str(frame.dtype),
                shape=frame.shape,
            )

        # send the json dict
        self.__msg_socket.send_json(msg_dict, self.__msg_flag | self.__zmq.SNDMORE)
        # send the frame array with correct flags
        self.__msg_socket.send(
            frame, flags=self.__msg_flag, copy=self.__msg_copy, track=self.__msg_track
        )

        # check if synchronous patterns, then wait for confirmation
        if self.__pattern < 2:
            # check if bi-directional data transmission is enabled
            if self.__bi_mode or self.__multiclient_mode:

                # handles return data
                recvd_data = None

                if self.__multiclient_mode:
                    recv_json = self.__msg_socket.recv_json(flags=self.__msg_flag)
                else:
                    socks = dict(self.__poll.poll(self.__request_timeout))
                    if socks.get(self.__msg_socket) == self.__zmq.POLLIN:
                        # handle return data
                        recv_json = self.__msg_socket.recv_json(flags=self.__msg_flag)
                    else:
                        logger.critical(
                            "No response from Client, Reconnecting again..."
                        )
                        # Socket is confused. Close and remove it.
                        self.__msg_socket.setsockopt(self.__zmq.LINGER, 0)
                        self.__msg_socket.close()
                        self.__poll.unregister(self.__msg_socket)
                        self.__max_retries -= 1

                        if not (self.__max_retries):
                            if self.__multiclient_mode:
                                logger.error(
                                    "All Clients failed to respond on multiple attempts."
                                )
                            else:
                                logger.error(
                                    "Client failed to respond on multiple attempts."
                                )
                            self.__terminate = True
                            raise RuntimeError(
                                "[NetGear:ERROR] :: Client(s) seems to be offline, Abandoning."
                            )

                        # Create new connection
                        self.__msg_socket = self.__msg_context.socket(
                            self.__msg_pattern
                        )
                        self.__msg_socket.connect(self.__connection_address)
                        self.__poll.register(self.__msg_socket, self.__zmq.POLLIN)

                        return None

                # save the unique port addresses
                if (
                    self.__multiclient_mode
                    and not recv_json["port"] in self.__port_buffer
                ):
                    self.__port_buffer.append(recv_json["port"])

                if recv_json["return_type"] == "ndarray":
                    recv_array = self.__msg_socket.recv(
                        flags=self.__msg_flag,
                        copy=self.__msg_copy,
                        track=self.__msg_track,
                    )
                    recvd_data = np.frombuffer(
                        recv_array, dtype=recv_json["array_dtype"]
                    ).reshape(recv_json["array_shape"])

                    # check if encoding was enabled
                    if recv_json["compression"]:
                        recvd_data = cv2.imdecode(
                            recvd_data, self.__ex_compression_params
                        )
                        # check if valid frame returned
                        if recvd_data is None:
                            self.__terminate = True
                            # otherwise raise error and exit
                            raise RuntimeError(
                                "[NetGear:ERROR] :: Received compressed frame `{}` decoding failed with flag: {}.".format(
                                    recv_json["compression"],
                                    self.__ex_compression_params,
                                )
                            )
                else:
                    recvd_data = recv_json["data"]

                return (
                    (recv_json["port"], recvd_data)
                    if self.__multiclient_mode
                    else recvd_data
                )
            else:
                if self.__multiclient_mode:
                    recv_confirmation = self.__msg_socket.recv()
                else:
                    # otherwise log normally
                    socks = dict(self.__poll.poll(self.__request_timeout))
                    if socks.get(self.__msg_socket) == self.__zmq.POLLIN:
                        recv_confirmation = self.__msg_socket.recv()
                    else:
                        logger.critical(
                            "No response from Client, Reconnecting again..."
                        )
                        # Socket is confused. Close and remove it.
                        self.__msg_socket.setsockopt(self.__zmq.LINGER, 0)
                        self.__msg_socket.close()
                        self.__poll.unregister(self.__msg_socket)
                        self.__max_retries -= 1

                        if not (self.__max_retries):
                            logger.error(
                                "Client failed to respond on repeated attempts."
                            )
                            self.__terminate = True
                            raise RuntimeError(
                                "[NetGear:ERROR] :: Client seems to be offline, Abandoning!"
                            )

                        # Create new connection
                        self.__msg_socket = self.__msg_context.socket(
                            self.__msg_pattern
                        )
                        self.__msg_socket.connect(self.__connection_address)
                        self.__poll.register(self.__msg_socket, self.__zmq.POLLIN)

                        return None

                # log confirmation
                if self.__logging:
                    logger.debug(recv_confirmation)

    def close(self):
        """
        Safely terminates the threads, and NetGear resources.
        """
        if self.__logging:
            # log it
            logger.debug(
                "Terminating various {} Processes.".format(
                    "Receive Mode" if self.__receive_mode else "Send Mode"
                )
            )
        #  whether `receive_mode` is enabled or not
        if self.__receive_mode:
            # check whether queue mode is empty
            if not (self.__queue is None) and self.__queue:
                self.__queue.clear()
            # call immediate termination
            self.__terminate = True
            # wait until stream resources are released (producer thread might be still grabbing frame)
            if self.__thread is not None:
                # properly handle thread exit
                self.__thread.join()
                self.__thread = None
            if self.__logging:
                logger.debug("Terminating. Please wait...")
            # properly close the socket
            self.__msg_socket.close(linger=0)
            if self.__logging:
                logger.debug("Terminated Successfully!")

        else:
            # indicate that process should be terminated
            self.__terminate = True

            # check if all attempts of reconnecting failed, then skip to closure
            if (self.__pattern < 2 and not self.__max_retries) or (
                self.__multiclient_mode and not self.__port_buffer
            ):
                try:
                    # properly close the socket
                    self.__msg_socket.setsockopt(self.__zmq.LINGER, 0)
                    self.__msg_socket.close()
                except self.__ZMQError:
                    pass
                finally:
                    return

            if self.__multiserver_mode:
                # check if multiserver_mode
                # send termination flag to client with its unique port
                term_dict = dict(terminate_flag=True, port=self.__port)
            else:
                # otherwise send termination flag to client
                term_dict = dict(terminate_flag=True)

            try:
                if self.__multiclient_mode:
                    if self.__port_buffer:
                        for _ in self.__port_buffer:
                            self.__msg_socket.send_json(term_dict)

                        # check for confirmation if available within half timeout
                        if self.__pattern < 2:
                            if self.__logging:
                                logger.debug("Terminating. Please wait...")
                            if self.__msg_socket.poll(
                                self.__request_timeout // 5, self.__zmq.POLLIN
                            ):
                                self.__msg_socket.recv()
                else:
                    self.__msg_socket.send_json(term_dict)

                    # check for confirmation if available within half timeout
                    if self.__pattern < 2:
                        if self.__logging:
                            logger.debug("Terminating. Please wait...")
                        if self.__msg_socket.poll(
                            self.__request_timeout // 5, self.__zmq.POLLIN
                        ):
                            self.__msg_socket.recv()
            except Exception as e:
                if not isinstance(e, self.__ZMQError):
                    logger.exception(str(e))
            finally:
                # properly close the socket
                self.__msg_socket.setsockopt(self.__zmq.LINGER, 0)
                self.__msg_socket.close()
                if self.__logging:
                    logger.debug("Terminated Successfully!")
