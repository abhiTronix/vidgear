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
import json
import logging as log
import os
import queue
import secrets
import string
import time
from os.path import expanduser
from threading import Thread
from typing import Any

import numpy as np
from numpy.typing import NDArray

# import helper packages
from .helper import (
    check_open_port,
    check_WriteAccess,
    import_dependency_safe,
    load_curve_keys,
    logcurr_vidgear_ver,
    logger_handler,
    zmq_endpoint,
)

# safe import critical Class modules
zmq = import_dependency_safe("zmq", pkg_name="pyzmq", error="silent", min_version="4.0")
if zmq is not None:
    from zmq import auth, ssh
    from zmq.auth.thread import ThreadAuthenticator
simplejpeg = import_dependency_safe("simplejpeg", error="silent", min_version="1.9.0")

# define logger
logger = log.getLogger("NetGear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

# largest decoded JPEG accepted from a peer (8K x 8K), guards against decompression bombs
_JPEG_MAX_PIXELS = 2**26
# channels expected by each JPEG colorspace
_JPEG_CHANNELS = {
    "GRAY": 1,
    "RGB": 3,
    "BGR": 3,
    "RGBX": 4,
    "BGRX": 4,
    "XBGR": 4,
    "XRGB": 4,
    "RGBA": 4,
    "BGRA": 4,
    "ABGR": 4,
    "ARGB": 4,
    "CMYK": 4,
}
_JPEG_SUBSAMPLINGS = ("444", "422", "420", "440", "411")
# receiver poll slice (ms): bounds how long `close()` waits for the receive thread
_POLL_MS = 100
# end-of-stream marker for the receive queue
_END = object()


class NetGear:
    """
    NetGear is exclusively designed to transfer video frames synchronously and asynchronously between interconnecting systems over the network in real-time.

    NetGear implements a high-level wrapper around PyZmQ python library that contains python bindings for ZeroMQ - a high-performance asynchronous distributed messaging library
    that provides a message queue, but unlike message-oriented middleware, its system can run without a dedicated message broker.

    NetGear also supports real-time Frame Compression capabilities for optimizing performance while sending the frames directly over the network, by encoding the frame before sending
    it and decoding it on the client's end automatically in real-time.

    !!! info
        NetGear API now internally implements robust *Lazy Pirate pattern* (auto-reconnection) for its synchronous messaging patterns _(i.e. `zmq.PAIR` & `zmq.REQ/zmq.REP`)_
        at both Server and Client ends, where its API instead of doing a blocking receive, will:

        * Poll the socket and receive from it only when it's sure a reply has arrived.
        * Attempt to reconnect, if no reply has arrived within a timeout period.
        * Abandon the connection if there is still no reply after several requests.

    NetGear as of now seamlessly supports four ZeroMQ messaging patterns:

    - `zmq.PAIR` _(ZMQ Pair Pattern)_
    - `zmq.REQ/zmq.REP` _(ZMQ Request/Reply Pattern)_
    - `zmq.PUB/zmq.SUB` _(ZMQ Publish/Subscribe Pattern)_
    - `zmq.PUSH/zmq.PULL` _(ZMQ Push/Pull Pattern)_

    _whereas the supported protocol are: `tcp` and `ipc`_.

    In every pattern the **receiver (Client) binds** and the **sender (Server) connects**, so `address` is always the receiver's address.

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
        address: str | None = None,
        port: str | None = None,
        protocol: str | None = None,
        pattern: int = 0,
        receive_mode: bool = False,
        logging: bool = False,
        **options: Any,
    ):
        """
        This constructor method initializes the object state and attributes of the NetGear class.

        Parameters:
            address (str): sets the receiver's network address (or `ipc` path).
            port (str): sets the valid Network Port of the Server/Client.
            protocol (str): sets the valid messaging protocol between Server/Client.
            pattern (int): sets the supported messaging pattern(flow of communication) between Server/Client
            receive_mode (bool): select the Netgear's Mode of operation.
            logging (bool): enables/disables logging.
            options (dict): provides the flexibility to alter various NetGear internal properties.
        """
        # enable logging if specified
        self.__logging = logging if isinstance(logging, bool) else False

        # print current version
        logcurr_vidgear_ver(logging=self.__logging)

        # raise error(s) for critical Class imports
        import_dependency_safe(
            "zmq" if zmq is None else "", min_version="4.0", pkg_name="pyzmq"
        )
        import_dependency_safe(
            "simplejpeg" if simplejpeg is None else "", error="log", min_version="1.9.0"
        )

        # define valid messaging patterns => `0`: PAIR, `1`:(REQ, REP), `2`:(PUB, SUB), `3`:(PUSH, PULL)
        valid_messaging_patterns = {
            0: (zmq.PAIR, zmq.PAIR),
            1: (zmq.REQ, zmq.REP),
            2: (zmq.PUB, zmq.SUB),
            3: (zmq.PUSH, zmq.PULL),
        }

        # Handle messaging pattern
        if not (isinstance(pattern, int) and pattern in valid_messaging_patterns):
            logger.warning(
                "Invalid pattern `{}`, Defaulting to `zmq.PAIR`! Kindly refer Docs for more Information.".format(
                    pattern
                )
            )
            pattern = 0
        self.__pattern = pattern

        # Handle messaging protocol
        if protocol not in ["tcp", "ipc"]:
            protocol is not None and logger.warning(
                "Protocol `{}` is not supported. Defaulting to `tcp` protocol!".format(
                    protocol
                )
            )
            protocol = "tcp"

        # Handle connection params
        self.__msg_flag = 0  # handles connection flags
        self.__msg_copy = False  # handles whether to copy data
        self.__msg_track = False  # handles whether to track packets

        # define Secure Mode
        self.__z_auth = None
        valid_security_mech = {0: "Grasslands", 1: "StoneHouse", 2: "IronHouse"}
        self.__secure_mode = 0  # handles ZMQ security layer status
        self.__curve_keys = None  # (secret, public, server_public) CURVE keys
        allowed_ips = None  # optional ZAP IP whitelist (receiver only)
        overwrite_cert = False  # checks if certificates overwriting allowed
        custom_cert_location = ""  # handles custom ZMQ certificates path

        # define SSH Tunneling Mode
        self.__ssh_tunnel_mode = None  # handles ssh_tunneling mode state
        self.__ssh_tunnel_pwd = None
        self.__ssh_tunnel_keyfile = None

        # define exclusive modes
        self.__multiserver_mode = False
        self.__multiclient_mode = False
        self.__bi_mode = False

        # define frame-compression handler (enabled by default if simplejpeg is installed)
        self.__jpeg_compression = simplejpeg is not None
        self.__jpeg_compression_quality = 90  # 90% quality
        self.__jpeg_compression_fastdct = True  # fastest DCT on by default
        self.__jpeg_compression_fastupsample = False  # fastupsample off by default
        self.__jpeg_compression_colorspace = "BGR"  # use BGR colorspace by default
        self.__jpeg_compression_colorsubsampling = "422"  # chroma subsampling
        self.__raw_fallback_warned = False  # warn once about uncompressible frames

        # define receiver return data handler
        self.__return_data = None

        # generate 8-digit random system id
        self.__id = "".join(
            secrets.choice(string.ascii_uppercase + string.digits) for i in range(8)
        )

        # define termination flags
        self.__terminate = False
        self.__terminate_sent = False
        self.__terminate_delivered = False  # terminate flag queued for sending
        self.__abandoned = False  # all reconnection attempts failed

        # reliability settings
        self.__max_retries = 3  # synchronous patterns only
        self.__request_timeout = 4000  # 4 secs (also `send()` timeout for pattern 3)
        self.__receive_timeout = None  # patterns 2 & 3 receiver only
        self.__send_hwm = 4  # pattern 3 sender queue length

        # Handle user-defined options dictionary values
        # reformat dictionary
        options = {str(k).strip(): v for k, v in options.items()}

        # loop over dictionary key & values and assign to global variables if valid
        for key, value in options.items():
            # handle multi-server mode
            if key == "multiserver_mode" and isinstance(value, bool):
                if value and pattern == 0:
                    logger.critical("Multi-Server Mode is disabled!")
                    raise ValueError(
                        "[NetGear:ERROR] :: `{}` pattern is not valid when Multi-Server Mode is enabled. Kindly refer Docs for more Information.".format(
                            pattern
                        )
                    )
                self.__multiserver_mode = value

            # handle multi-client mode
            elif key == "multiclient_mode" and isinstance(value, bool):
                if value and pattern == 0:
                    logger.critical("Multi-Client Mode is disabled!")
                    raise ValueError(
                        "[NetGear:ERROR] :: `{}` pattern is not valid when Multi-Client Mode is enabled. Kindly refer Docs for more Information.".format(
                            pattern
                        )
                    )
                self.__multiclient_mode = value

            # handle bidirectional mode
            elif key == "bidirectional_mode" and isinstance(value, bool):
                if value and pattern >= 2:
                    logger.warning("Bidirectional data transmission is disabled!")
                    raise ValueError(
                        "[NetGear:ERROR] :: `{}` pattern is not valid when Bidirectional Mode is enabled. Kindly refer Docs for more Information!".format(
                            pattern
                        )
                    )
                self.__bi_mode = value

            # handle secure mode
            elif (
                key == "secure_mode"
                and isinstance(value, int)
                and (value in valid_security_mech)
            ):
                self.__secure_mode = value
            elif key == "custom_cert_location" and isinstance(value, str):
                # verify custom auth certificates path for secure mode
                custom_cert_location = os.path.abspath(value)
                assert os.path.isdir(
                    custom_cert_location
                ), "[NetGear:ERROR] :: `custom_cert_location` value must be the path to a valid directory!"
                assert check_WriteAccess(
                    custom_cert_location,
                    is_windows=(os.name == "nt"),
                    logging=self.__logging,
                ), "[NetGear:ERROR] :: Permission Denied!, cannot write ZMQ authentication certificates to '{}' directory!".format(
                    value
                )
            elif key == "overwrite_cert" and isinstance(value, bool):
                # enable/disable auth certificate overwriting in secure mode
                overwrite_cert = value
            elif key == "allowed_ips" and isinstance(value, (list, tuple)):
                # optional IP whitelist for Secure Mode (receiver only)
                allowed_ips = [str(ip) for ip in value]

            # handle ssh-tunneling mode
            elif key == "ssh_tunnel_mode" and isinstance(value, str):
                self.__ssh_tunnel_mode = value.strip()
            elif key == "ssh_tunnel_pwd" and isinstance(value, str):
                self.__ssh_tunnel_pwd = value
            elif key == "ssh_tunnel_keyfile" and isinstance(value, str):
                # user's SSH private key (`ssh -i`)
                self.__ssh_tunnel_keyfile = value if os.path.isfile(value) else None
                if self.__ssh_tunnel_keyfile is None:
                    logger.warning(
                        "Discarded invalid or non-existential SSH Tunnel Key-file at {}!".format(
                            value
                        )
                    )

            # handle jpeg compression
            elif key == "jpeg_compression" and simplejpeg is not None:
                if isinstance(value, bool):
                    self.__jpeg_compression = value
                elif (
                    isinstance(value, str) and value.strip().upper() in _JPEG_CHANNELS
                ):
                    # set encoding colorspace and enable frame-compression
                    self.__jpeg_compression_colorspace = value.strip().upper()
                    self.__jpeg_compression = True
                else:
                    logger.warning(
                        "Skipped invalid `jpeg_compression` value: {}!".format(value)
                    )
            elif key == "jpeg_compression_quality" and isinstance(value, (int, float)):
                # set valid jpeg quality
                if value >= 10 and value <= 100:
                    self.__jpeg_compression_quality = int(value)
                else:
                    logger.warning("Skipped invalid `jpeg_compression_quality` value!")
            elif key == "jpeg_compression_fastdct" and isinstance(value, bool):
                self.__jpeg_compression_fastdct = value
            elif key == "jpeg_compression_fastupsample" and isinstance(value, bool):
                self.__jpeg_compression_fastupsample = value
            elif key == "jpeg_compression_colorsubsampling":
                if str(value).strip() in _JPEG_SUBSAMPLINGS:
                    self.__jpeg_compression_colorsubsampling = str(value).strip()
                else:
                    logger.warning(
                        "Skipped invalid `jpeg_compression_colorsubsampling` value!"
                    )

            # assign maximum retries in synchronous patterns
            elif key == "max_retries" and isinstance(value, int) and pattern < 2:
                if value >= 1:
                    self.__max_retries = value
                else:
                    logger.warning("Invalid `max_retries` value skipped!")

            # assign request timeout in synchronous patterns, and send timeout in pattern 3
            elif (
                key == "request_timeout"
                and isinstance(value, (int, float))
                and pattern in (0, 1, 3)
            ):
                if value >= 4:
                    self.__request_timeout = int(value * 1000)  # to milliseconds
                else:
                    logger.warning("Invalid `request_timeout` value skipped!")

            # assign receiver timeout for asynchronous patterns
            elif (
                key in ["receive_timeout", "subscriber_timeout"]
                and isinstance(value, (int, float))
                and pattern >= 2
            ):
                if value > 0:
                    self.__receive_timeout = int(value * 1000)  # to milliseconds
                else:
                    logger.warning("Invalid `{}` value skipped!".format(key))

            # assign sender queue length for pattern 3
            elif key == "send_hwm" and isinstance(value, int) and pattern == 3:
                if value >= 1:
                    self.__send_hwm = value
                else:
                    logger.warning("Invalid `send_hwm` value skipped!")

            # handle ZMQ flags
            elif key == "flag" and isinstance(value, int):
                self.__msg_flag = value
                self.__msg_flag and logger.warning(
                    "The flag optional value is set to `1` (NOBLOCK) for this run. `send()` may raise `zmq.Again`, and NetGear might not terminate gracefully."
                )
            elif key == "copy" and isinstance(value, bool):
                self.__msg_copy = value
            elif key == "track" and isinstance(value, bool):
                self.__msg_track = value
                self.__msg_copy and self.__msg_track and logger.info(
                    "The `track` optional value will be ignored for this run because `copy=True` is also defined."
                )

        # Handle ssh tunneling if enabled
        if self.__ssh_tunnel_mode is not None:
            if receive_mode:
                # SSH Tunnel Mode only available for server mode
                logger.error("SSH Tunneling cannot be enabled for Client-end!")
                self.__ssh_tunnel_mode = None
            elif protocol != "tcp":
                raise ValueError(
                    "[NetGear:ERROR] :: SSH Tunneling Mode only supports `tcp` protocol!"
                )
            else:
                # check if SSH server is reachable
                ssh_address, _, ssh_port = self.__ssh_tunnel_mode.partition(":")
                ssh_ip = ssh_address.rpartition("@")[2]
                assert check_open_port(
                    ssh_ip, port=int(ssh_port or 22)
                ), "[NetGear:ERROR] :: Host `{}` is not available for SSH Tunneling at port-{}!".format(
                    ssh_address, ssh_port or 22
                )

        # Handle multiple exclusive modes if enabled
        if self.__multiclient_mode and self.__multiserver_mode:
            raise ValueError(
                "[NetGear:ERROR] :: Multi-Client and Multi-Server Mode cannot be enabled simultaneously!"
            )
        elif (self.__multiserver_mode or self.__multiclient_mode) and (
            self.__ssh_tunnel_mode
        ):
            raise ValueError(
                "[NetGear:ERROR] :: SSH Tunneling and {} Mode cannot be enabled simultaneously. Kindly refer docs!".format(
                    "Multi-Server" if self.__multiserver_mode else "Multi-Client"
                )
            )
        self.__bi_mode and self.__logging and logger.debug(
            "Bidirectional Data Transmission is enabled for this connection!"
        )
        self.__ssh_tunnel_mode and self.__logging and logger.debug(
            "SSH Tunneling is enabled for host:`{}`.".format(self.__ssh_tunnel_mode)
        )

        # initialize and assign receive mode to global variable
        self.__receive_mode = receive_mode
        self.__zmq_type = valid_messaging_patterns[pattern][
            1 if receive_mode else 0
        ]

        # validate ports for exclusive modes:
        # list of ports where the socket binds/connects to many peers, single port otherwise
        many_ports = (self.__multiserver_mode and receive_mode) or (
            self.__multiclient_mode and not receive_mode
        )
        if self.__multiserver_mode or self.__multiclient_mode:
            mode_name = "Multi-Server" if self.__multiserver_mode else "Multi-Client"
            if many_ports and not isinstance(port, (tuple, list)):
                raise ValueError(
                    "[NetGear:ERROR] :: Incorrect port value! Kindly provide a list/tuple of {} ports while {} mode is enabled. For more information refer VidGear docs.".format(
                        "Server" if self.__multiserver_mode else "Client", mode_name
                    )
                )
            if not many_ports and (port is None or isinstance(port, (tuple, list))):
                raise ValueError(
                    "[NetGear:ERROR] :: Kindly provide a unique & valid port value at {}-end. For more information refer VidGear docs.".format(
                        "Client" if receive_mode else "Server"
                    )
                )
            logger.debug("Enabling {} Mode at PORT(s): {}!".format(mode_name, port))
            # track unique ports of connected peers
            self.__port_buffer = []
            self.__port = port
        else:
            port = "5555" if port is None else port

        # build endpoint(s): the receiver binds, the sender connects
        suffix_port = self.__multiserver_mode or self.__multiclient_mode
        self.__endpoints = [
            zmq_endpoint(protocol, address, pt, bind=receive_mode, suffix_port=suffix_port)
            for pt in (port if many_ports else [port])
        ]

        # define ZMQ messaging context (per-instance, so `close()` can't
        # tear down sockets of other NetGear instances in this process)
        self.__msg_context = zmq.Context()
        self.__msg_context.linger = 0

        # Handle Secure mode
        if self.__secure_mode > 0:
            if not zmq.has("curve"):
                raise RuntimeError(
                    "[NetGear:ERROR] :: Secure Mode requires libzmq built with CURVE support!"
                )
            if receive_mode and overwrite_cert:
                overwrite_cert = False
                logger.warning(
                    "Overwriting ZMQ Authentication certificates is disabled for Client's end!"
                )
            else:
                overwrite_cert and self.__logging and logger.info(
                    "Overwriting ZMQ Authentication certificates over previous ones!"
                )
            self.__curve_keys, auth_publickeys_dir = load_curve_keys(
                custom_cert_location or os.path.join(expanduser("~"), ".vidgear"),
                receive_mode,
                overwrite=overwrite_cert,
                logging=logging,
            )
            if receive_mode:
                # the receiver binds and is the CURVE server: ZAP authentication only applies here
                self.__z_auth = ThreadAuthenticator(self.__msg_context)
                self.__z_auth.start()
                if allowed_ips:
                    # whitelist peer (sender) IPs
                    self.__z_auth.allow(*allowed_ips)
                self.__z_auth.configure_curve(
                    domain="*",
                    location=(
                        auth_publickeys_dir
                        if self.__secure_mode == 2
                        else auth.CURVE_ALLOW_ANY
                    ),
                )
            else:
                allowed_ips and logger.warning(
                    "`allowed_ips` is only used at the receiver's (Client) end!"
                )

        # create the socket
        try:
            self.__msg_socket = self.__make_socket()
        except Exception as e:
            logger.exception(str(e))
            self.__secure_mode and logger.critical(
                "Failed to activate Secure Mode: `{}` for this connection!".format(
                    valid_security_mech[self.__secure_mode]
                )
            )
            self.__z_auth is not None and self.__z_auth.stop()
            self.__msg_context.destroy(linger=0)
            raise RuntimeError(
                "[NetGear:ERROR] :: {} Mode failed to {} address: {} with pattern: {}! Kindly recheck all parameters.".format(
                    "Receive" if receive_mode else "Send",
                    "bind" if receive_mode else "connect",
                    self.__endpoints,
                    pattern,
                )
            ) from e

        # log progress
        if self.__logging:
            logger.debug(
                "Successfully {} to address: {} with pattern: {}.".format(
                    "bound" if receive_mode else "connected",
                    self.__endpoints,
                    pattern,
                )
            )
            if pattern < 2:
                logger.debug(
                    "Reliable transmission is enabled for this pattern with max-retries: {} and timeout: {} secs.".format(
                        self.__max_retries, self.__request_timeout / 1000
                    )
                )
            elif self.__receive_timeout and receive_mode:
                logger.debug(
                    "Timeout: {} secs is enabled for this system.".format(
                        self.__receive_timeout / 1000
                    )
                )
            self.__jpeg_compression and logger.debug(
                "JPEG Frame-Compression is activated for this connection with Colorspace:`{}`, Quality:`{}`%, Subsampling:`{}`, Fastdct:`{}`, and Fastupsample:`{}`.".format(
                    self.__jpeg_compression_colorspace,
                    self.__jpeg_compression_quality,
                    self.__jpeg_compression_colorsubsampling,
                    self.__jpeg_compression_fastdct,
                    self.__jpeg_compression_fastupsample,
                )
            )
            self.__secure_mode and logger.debug(
                "Enabled ZMQ Security Mechanism: `{}` for this connection.".format(
                    valid_security_mech[self.__secure_mode]
                )
            )
            logger.debug("Unique System ID is {}.".format(self.__id))

        # retry budget, reset after every successful exchange
        self.__retries_left = self.__max_retries

        if receive_mode:
            # threaded queue mode: the handler thread owns the socket from now on
            self.__queue = queue.Queue(maxsize=96)
            self.__thread = Thread(target=self.__recv_handler, name="NetGear", daemon=True)
            self.__thread.start()
            self.__logging and logger.debug("Receive Mode is now activated.")
        else:
            self.__logging and logger.debug(
                "Send Mode is successfully activated and ready to send data."
            )

    def __make_socket(self):
        """
        Creates the messaging socket with every socket option applied, and binds/connects it.
        Used by both initial setup and every Lazy-Pirate reconnect, so options (CURVE, HWM, ...) never get lost.
        """
        sock = self.__msg_context.socket(self.__zmq_type)
        try:
            if self.__pattern == 1 and not self.__receive_mode:
                sock.REQ_RELAXED = True
                sock.REQ_CORRELATE = True
            if self.__pattern == 2:
                sock.set_hwm(1)
                self.__receive_mode and sock.setsockopt(zmq.SUBSCRIBE, b"")
            if self.__pattern == 3 and not self.__receive_mode:
                # bounded queue + timeout, so `send()` raises instead of blocking forever,
                # and nothing queues for clients that aren't connected
                sock.sndhwm = self.__send_hwm
                sock.sndtimeo = self.__request_timeout
                sock.immediate = True
            if self.__curve_keys is not None:
                secret, public, server_public = self.__curve_keys
                sock.curve_secretkey = secret
                sock.curve_publickey = public
                if self.__receive_mode:
                    sock.curve_server = True
                else:
                    sock.curve_serverkey = server_public
            for endpoint in self.__endpoints:
                if self.__receive_mode:
                    sock.bind(endpoint)
                elif self.__ssh_tunnel_mode:
                    ssh.tunnel_connection(
                        sock,
                        endpoint,
                        self.__ssh_tunnel_mode,
                        keyfile=self.__ssh_tunnel_keyfile,
                        password=self.__ssh_tunnel_pwd,
                    )
                else:
                    sock.connect(endpoint)
        except Exception:
            sock.close(linger=0)
            raise
        return sock

    def __jpeg_encodable(self, frame: NDArray) -> bool:
        """
        Checks whether simplejpeg can encode this frame with the selected colorspace.
        """
        channels = frame.shape[2] if frame.ndim == 3 else (1 if frame.ndim == 2 else -1)
        return (
            frame.dtype == np.uint8
            and channels == _JPEG_CHANNELS[self.__jpeg_compression_colorspace]
        )

    def __encode(self, frame: NDArray) -> tuple[dict, Any]:
        """
        Encodes a frame for the wire.

        **Returns:** a tuple of header fields and the payload buffer.
        """
        if self.__jpeg_compression:
            if self.__jpeg_encodable(frame):
                if frame.ndim == 2:
                    # simplejpeg needs 3-D input (GRAY decodes to `(h, w, 1)`)
                    frame = frame[:, :, np.newaxis]
                # rows may be strided (simplejpeg >= 1.8.0), but columns and channels must be contiguous
                if frame.strides[1] != frame.shape[2] or (
                    frame.shape[2] > 1 and frame.strides[2] != 1
                ):
                    frame = np.ascontiguousarray(frame)
                payload = simplejpeg.encode_jpeg(
                    frame,
                    quality=self.__jpeg_compression_quality,
                    colorspace=self.__jpeg_compression_colorspace,
                    colorsubsampling=self.__jpeg_compression_colorsubsampling,
                    fastdct=self.__jpeg_compression_fastdct,
                )
                return {
                    "compression": {
                        "dct": self.__jpeg_compression_fastdct,
                        "ups": self.__jpeg_compression_fastupsample,
                        "colorspace": self.__jpeg_compression_colorspace,
                    },
                    "dtype": "",
                    "shape": "",
                }, payload
            if not self.__raw_fallback_warned:
                self.__raw_fallback_warned = True
                logger.warning(
                    "Frame of dtype `{}` and shape `{}` cannot be JPEG compressed with `{}` colorspace (needs uint8 with matching channels). Sending such frames uncompressed.".format(
                        frame.dtype, frame.shape, self.__jpeg_compression_colorspace
                    )
                )
        frame = np.ascontiguousarray(frame)
        return {
            "compression": False,
            "dtype": str(frame.dtype),
            "shape": frame.shape,
        }, frame

    def __decode(self, buffer: Any, compression: Any, dtype: Any, shape: Any) -> NDArray:
        """
        Decodes a frame received from a peer. Raises on invalid or oversized data.
        """
        if not compression:
            return np.frombuffer(buffer, dtype=dtype).reshape(shape)
        if simplejpeg is None:
            raise ValueError("`simplejpeg` is required to decode JPEG compressed frames!")
        # check dimensions before decoding, since the decoder allocates from the header
        height, width, _, _ = simplejpeg.decode_jpeg_header(buffer)
        if height * width > _JPEG_MAX_PIXELS:
            raise ValueError(
                "JPEG frame of {}x{} exceeds the {} pixels limit!".format(
                    width, height, _JPEG_MAX_PIXELS
                )
            )
        # decode with this end's own flags
        frame = simplejpeg.decode_jpeg(
            buffer,
            colorspace=compression["colorspace"],
            fastdct=self.__jpeg_compression_fastdct,
            fastupsample=self.__jpeg_compression_fastupsample,
        )
        if compression["colorspace"] == "GRAY":
            # GRAY decodes to `(h, w, 1)`
            frame = np.squeeze(frame, axis=2)
        return frame

    def __put(self, item: Any) -> None:
        """
        Puts an item in the receive queue, waiting while it is full unless NetGear is closing.
        """
        while not self.__terminate:
            try:
                self.__queue.put(item, timeout=_POLL_MS / 1000)
                return
            except queue.Full:
                continue

    def __recv_handler(self):
        """
        A threaded receiver handler, that keep iterating data from ZMQ socket to a internally monitored queue,
        until the thread is terminated, or socket disconnects. This thread owns the socket.
        """
        result = _END
        try:
            self.__recv_loop()
        except Exception as e:
            logger.exception(str(e))
            result = e
        finally:
            self.__msg_socket.close(linger=0)
            # signal end of stream (or error) after all queued frames
            self.__put(result)

    def __recv_loop(self):
        """
        Receives messages until terminated, the peer(s) signal termination, or the connection times out.
        """
        # silence timeout: reconnect (synchronous patterns) or exit (asynchronous patterns)
        timeout = (
            self.__request_timeout * 3 if self.__pattern < 2 else self.__receive_timeout
        )
        last_msg = time.monotonic()

        while not self.__terminate:
            if not self.__msg_socket.poll(_POLL_MS, zmq.POLLIN):
                if timeout and (time.monotonic() - last_msg) * 1000 >= timeout:
                    if self.__pattern >= 2:
                        logger.warning("Connection Timeout. Exiting!")
                        return
                    logger.warning("No response from Server(s), Reconnecting again...")
                    self.__msg_socket.close(linger=0)
                    self.__retries_left -= 1
                    if self.__retries_left <= 0:
                        logger.error(
                            "All Servers seems to be offline, Abandoning!"
                            if self.__multiserver_mode
                            else "Server seems to be offline, Abandoning!"
                        )
                        self.__abandoned = True
                        return
                    self.__msg_socket = self.__make_socket()
                    last_msg = time.monotonic()
                continue

            try:
                parts = self.__msg_socket.recv_multipart(
                    flags=zmq.NOBLOCK, copy=self.__msg_copy, track=self.__msg_track
                )
            except zmq.Again:
                # nothing to read yet
                continue
            last_msg = time.monotonic()
            self.__retries_left = self.__max_retries

            msg_json = json.loads(bytes(parts[0]))

            # check if terminate_flag` received
            if msg_json.get("terminate_flag"):
                if self.__pattern == 1:
                    # send back server the info about termination
                    self.__msg_socket.send_string(
                        "Termination signal successfully received at Client's end!"
                    )
                if self.__multiserver_mode:
                    if msg_json.get("port") in self.__port_buffer:
                        self.__port_buffer.remove(msg_json["port"])
                        self.__logging and logger.warning(
                            "Termination signal received from Server at port: {}!".format(
                                msg_json["port"]
                            )
                        )
                    # exit when termination signal received from all servers
                    if not self.__port_buffer:
                        self.__logging and logger.info(
                            "Termination signal received from all Servers!!!"
                        )
                        return
                else:
                    self.__logging and logger.info("Termination signal received from server!")
                    return
                continue

            # handle replies in synchronous patterns (before decoding, REP must answer every request)
            if self.__pattern < 2:
                if self.__bi_mode or self.__multiclient_mode:
                    self.__send_return_data()
                else:
                    self.__msg_socket.send_string(
                        "Data received on device: {} !".format(self.__id)
                    )
            elif self.__return_data is not None:
                logger.warning("`return_data` is disabled for this pattern!")

            # decode frame, one bad frame costs one frame
            try:
                frame = self.__decode(
                    parts[1],
                    msg_json["compression"],
                    msg_json["dtype"],
                    msg_json["shape"],
                )
            except Exception as e:
                logger.error("Dropped an invalid frame: {}".format(e))
                continue

            if self.__multiserver_mode:
                # save the unique port addresses
                if msg_json["port"] not in self.__port_buffer:
                    self.__port_buffer.append(msg_json["port"])
                self.__put((msg_json["port"], msg_json["message"], frame))
            elif self.__bi_mode:
                self.__put((msg_json["message"], frame))
            else:
                self.__put(frame)

    def __send_return_data(self):
        """
        Sends `return_data` (or an empty reply) back to the Server in Bidirectional/Multi-Client mode.
        """
        return_data = self.__return_data
        return_dict = {"port": self.__port} if self.__multiclient_mode else {}
        return_dict["return_type"] = type(return_data).__name__
        if isinstance(return_data, np.ndarray):
            # copy guards against the user mutating `return_data` meanwhile
            header, payload = self.__encode(np.copy(return_data))
            return_dict.update(
                {
                    "compression": header["compression"],
                    "array_dtype": header["dtype"],
                    "array_shape": header["shape"],
                    "data": None,
                }
            )
            self.__msg_socket.send_multipart(
                [json.dumps(return_dict).encode(), payload],
                flags=self.__msg_flag,
                copy=self.__msg_copy,
                track=self.__msg_track,
            )
        else:
            return_dict["data"] = return_data
            self.__msg_socket.send_json(return_dict, self.__msg_flag)

    def recv(self, return_data: Any = None) -> Any:
        """
        A Receiver end method, that extracts received frames synchronously from monitored queue, while maintaining a
        fixed-length frame buffer in the memory, and blocks the thread if the queue is full.

        Parameters:
            return_data (any): inputs JSON-serializable return data _(or a `numpy.ndarray` frame)_, for sending back to Server
                in Bidirectional/Multi-Client mode. It stays set until replaced, and is sent with the reply to the **next** received frame.

        **Returns:** A n-dimensional numpy array, `(message, frame)` tuple in Bidirectional Mode, `(port, message, frame)` tuple in
        Multi-Server Mode, or `None` once the stream has ended. Raises `RuntimeError` if the receive thread failed.
        """
        # check whether `receive mode` is activated
        if not (self.__receive_mode):
            # raise value error and exit
            self.__terminate = True
            raise ValueError(
                "[NetGear:ERROR] :: `recv()` function cannot be used while receive_mode is disabled. Kindly refer vidgear docs!"
            )

        # handle Bidirectional return data
        if (self.__bi_mode or self.__multiclient_mode) and return_data is not None:
            self.__return_data = return_data

        try:
            while True:
                try:
                    item = self.__queue.get(timeout=_POLL_MS / 1000)
                except queue.Empty:
                    if self.__terminate or not self.__thread.is_alive():
                        return None
                    continue
                if item is _END:
                    self.__terminate = True
                    return None
                if isinstance(item, BaseException):
                    self.__terminate = True
                    raise RuntimeError(
                        "[NetGear:ERROR] :: Receive thread failed: {}".format(item)
                    ) from item
                return item
        except KeyboardInterrupt:
            self.__terminate = True
            return None

    def send(self, frame: NDArray | None, message: Any = None) -> Any:
        """
        A Server end method, that sends the data and frames over the network to Client(s).

        Parameters:
            frame (numpy.ndarray): inputs numpy array(frame). `None` signals end of stream to Client(s).
            message (any): input for sending additional JSON-serializable data _(of any datatype except `numpy.ndarray`)_ to Client(s).

        **Returns:** Data from Client(s) in Bidirectional/Multi-Client modes, otherwise None-type.
        """
        # check whether `receive_mode` is disabled
        if self.__receive_mode:
            # raise value error and exit
            self.__terminate = True
            raise ValueError(
                "[NetGear:ERROR] :: `send()` function cannot be used while receive_mode is enabled. Kindly refer vidgear docs!"
            )

        # `None` frame means end of stream
        if frame is None:
            self.__send_terminate()
            return None

        if message is not None and isinstance(message, np.ndarray):
            logger.warning(
                "Skipped unsupported `message` of datatype: {}!".format(
                    type(message).__name__
                )
            )
            message = None

        header, payload = self.__encode(frame)
        # prepare the exclusive json dict
        msg_dict = {"port": self.__port} if self.__multiserver_mode else {}
        msg_dict.update(
            {
                "terminate_flag": False,
                "compression": header["compression"],
                "message": message,
                "pattern": str(self.__pattern),
                "dtype": header["dtype"],
                "shape": header["shape"],
            }
        )
        try:
            self.__msg_socket.send_multipart(
                [json.dumps(msg_dict).encode(), payload],
                flags=self.__msg_flag,
                copy=self.__msg_copy,
                track=self.__msg_track,
            )
        except zmq.Again as e:
            if self.__pattern == 3 and not self.__msg_flag:
                raise RuntimeError(
                    "[NetGear:ERROR] :: Client(s) seems to be offline, `send()` timed out!"
                ) from e
            raise

        # asynchronous patterns don't wait for confirmation
        if self.__pattern >= 2:
            return None

        if not self.__msg_socket.poll(self.__request_timeout, zmq.POLLIN):
            self.__reconnect()
            return None
        self.__retries_left = self.__max_retries

        # check if Bidirectional data transmission is enabled
        if not (self.__bi_mode or self.__multiclient_mode):
            recv_confirmation = self.__msg_socket.recv()
            self.__logging and logger.debug(recv_confirmation)
            return None

        parts = self.__msg_socket.recv_multipart(
            flags=self.__msg_flag, copy=self.__msg_copy, track=self.__msg_track
        )
        recv_json = json.loads(bytes(parts[0]))
        # save the unique port addresses
        if self.__multiclient_mode and recv_json["port"] not in self.__port_buffer:
            self.__port_buffer.append(recv_json["port"])
        if recv_json["return_type"] == "ndarray":
            recvd_data = self.__decode(
                parts[1],
                recv_json["compression"],
                recv_json["array_dtype"],
                recv_json["array_shape"],
            )
        else:
            recvd_data = recv_json["data"]
        return (recv_json["port"], recvd_data) if self.__multiclient_mode else recvd_data

    def __reconnect(self):
        """
        Lazy-Pirate: rebuilds the sender socket after a missed reply, raising when retries are exhausted.
        """
        logger.warning("No response from Client(s), Reconnecting again...")
        # Socket is confused. Close and remove it.
        self.__msg_socket.close(linger=0)
        self.__retries_left -= 1
        if self.__retries_left <= 0:
            logger.error("Client(s) failed to respond on repeated attempts.")
            self.__terminate = True
            self.__abandoned = True
            raise RuntimeError(
                "[NetGear:ERROR] :: Client(s) seems to be offline, Abandoning!"
            )
        self.__msg_socket = self.__make_socket()

    def __send_terminate(self):
        """
        Sends the termination flag to Client(s), best-effort and bounded in time.
        """
        if self.__terminate_sent or self.__abandoned:
            return
        self.__terminate_sent = True

        term_dict = {"terminate_flag": True}
        if self.__multiserver_mode:
            term_dict["port"] = self.__port
        # number of termination messages to send
        if not self.__multiclient_mode or self.__pattern == 2:
            count = 1  # single peer, or PUB broadcast
        elif self.__pattern == 3:
            count = len(self.__port)  # PUSH round-robins one to each client
        else:
            count = len(self.__port_buffer)  # REQ: only clients that ever replied
        try:
            for _ in range(count):
                # wait (bounded) for queue space, a dead peer must not block `close()`
                if not self.__msg_socket.poll(self.__request_timeout // 5, zmq.POLLOUT):
                    raise zmq.Again()
                self.__msg_socket.send_json(term_dict, zmq.NOBLOCK)
                self.__terminate_delivered = True
                # check for confirmation if available within 1/5 timeout
                if self.__pattern == 1 and self.__msg_socket.poll(
                    self.__request_timeout // 5, zmq.POLLIN
                ):
                    self.__msg_socket.recv()
        except zmq.ZMQError as e:
            self.__logging and logger.debug("Termination signal not delivered: {}".format(e))

    def close(self, kill: bool = False) -> None:
        """
        Safely terminates the threads, and NetGear resources.

        Parameters:
            kill (bool): don't wait for the receive thread to finish before releasing the ZMQ context _(receive mode only)_.
        """
        self.__logging and logger.debug(
            "Terminating various {} Processes.".format(
                "Receive Mode" if self.__receive_mode else "Send Mode"
            )
        )
        if self.__receive_mode:
            # signal the receive thread, which owns and closes the socket
            self.__terminate = True
            if self.__z_auth is not None:
                self.__logging and logger.debug("Terminating Authenticator Thread.")
                self.__z_auth.stop()
                self.__z_auth = None
            if self.__thread is not None:
                self.__thread.join(timeout=_POLL_MS / 1000 if kill else 5.0)
                self.__thread.is_alive() and logger.warning(
                    "Receive thread still running, releasing ZMQ context anyway!"
                )
                self.__thread = None
        else:
            # send termination flag to Client(s), unless already done
            self.__send_terminate()
            self.__terminate = True
            # give queued messages (terminate flag included) a bounded time to flush
            self.__msg_socket.close(
                linger=self.__request_timeout if self.__terminate_delivered else 0
            )
        # release ZMQ context
        self.__msg_context.destroy(linger=0)
        self.__logging and logger.debug("Terminated Successfully!")
