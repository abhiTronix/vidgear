<!--
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
-->

# NetGear API Parameters 

&thinsp;

## **`address`**

This parameter sets the valid Network IP address for Server/Client. Network addresses are unique identifiers across the network. 

!!! info "In every [messaging pattern](#pattern), the **receiver (Client) binds** and the **sender (Server) connects**. Hence, this parameter **MUST** always be the IP address of the receiver (Client) machine at both ends _(or `'*'` at the Client end to bind on all interfaces)_."

!!! note "With [`protocol="ipc"`](#protocol), this parameter is instead a filesystem path _(e.g. `/tmp/netgear.ipc`)_, and **MUST** be same at both ends. Its default path is `<tempdir>/vidgear-<port>.ipc` at both ends. In Multi-Servers/Multi-Clients Mode, `-<port>` is appended to an explicit path for each port."

**Data-Type:** String

**Default Value:** Its default value is based on selected [primary mode](./#primary-modes), _i.e `'localhost'` for Send Mode and `'*'` for Receive Mode_ on a local machine.

**Usage:**

```python
NetGear(address="192.168.0.145")
```

&nbsp; 


## **`port`**

This parameter sets the valid Network Port for Server/Client. Network port is a number that identifies one side of a connection between two devices on the network and is used determine to which process or application a message should be delivered.

!!! danger "Exception for Exclusive Modes"
    
    [**In Multi-Servers Mode**](advanced/multi_server/):

      * A unique port number **MUST** be assigned to each Server on the network using this parameter. 
      * At Client end, a List/Tuple of all available Server(s) ports **MUST** be assigned using this same parameter. 
      * See its usage example [here ➶](advanced/multi_server/#bare-minimum-usage).

    [**In Multi-Client Mode**](advanced/multi_client/):

      * A unique port number **MUST** be assigned to each Client on the network using this parameter. 
      * At Server end, a List/Tuple of all available Client(s) ports **MUST** be assigned using this same parameter. 
      * See its usage example [here ➶](advanced/multi_client/#bare-minimum-usage).

**Data-Type:** String or List/Tuple

**Default Value:** Its default value is `'5555'`

**Usage:**

```python
NetGear(port="5575")
```

&nbsp; 


## **`protocol`** 


This parameter sets the valid messaging protocol between server and client. A network protocol is a set of established rules that dictates how to format, transmit and receive data so computer network devices - from servers and routers to endpoints - can communicate regardless of the differences in their underlying infrastructures, designs or standards. Supported protocol are: `'tcp'` and `'ipc'`. Any other value is skipped with a warning, and NetGear falls back to `'tcp'`.

!!! warning "`ipc` protocol needs UNIX domain sockets, and works only between processes on the same machine. Its [`address`](#address) is a filesystem path. Also, [SSH Tunneling Mode](advanced/ssh_tunnel/) only supports `tcp` _(raises `ValueError` otherwise)_."

**Data-Type:** String

**Default Value:** Its default value is `'tcp'`

**Usage:**

```python
NetGear(protocol="ipc")
```

&nbsp; 


## **`pattern`**

This parameter sets the supported messaging pattern(flow of communication) between server and client. Messaging patterns are the network-oriented architectural pattern that describes the flow of communication between interconnecting systems. NetGear provides access to ZeroMQ's pre-optimized sockets which enables you to take advantage of these patterns.

**Data-Type:** Integer

**Default Value:** Its default value is `0` (_i.e `zmq.PAIR`_). Any invalid value is skipped with a warning, and NetGear falls back to `0`.

!!! info "Supported ZMQ patterns"

    **All supported ZMQ patterns for NetGear are:**

    * **`0` (_.i.e. zmq.PAIR_):** In this pattern, the communication is bidirectional. There is no specific state stored within the socket. There can only be one connected peer. The client (receiver) binds on a certain port and the server (sender) connects to it.
    * **`1` (_.i.e. zmq.REQ/zmq.REP_):** In this pattern, it employs `ZMQ REQ` sockets that can connect to many peers. The requests will be interleaved or distributed (round-robin) between them. socket `zmq.REQ` will block send unless it has successfully received a reply back and socket `zmq.REP` will block on recv() unless it has received a request.
    * **`2` (_.i.e. zmq.PUB/zmq.SUB_):** It is an another classic pattern where senders of messages, called _publishers_, do not program the messages to be sent directly to specific receivers, called _subscribers_. Messages are published without the knowledge of what or if any subscriber of that knowledge exists. A `ZMQ.SUB` can receive from multiple `ZMQ.PUB` (publishers). No single publisher overwhelms the subscriber. The messages from both publishers are interleaved. Frames are dropped while no subscriber is connected.
    * **`3` (_.i.e. zmq.PUSH/zmq.PULL_):** In this one-way pipeline pattern, the `ZMQ.PUSH` sender distributes frames to its connected `ZMQ.PULL` receivers in round-robin _(load-balancing, not broadcast)_, and a `ZMQ.PULL` receiver fair-queues frames from all its senders. Frame order is preserved per-sender only. The sender buffers up to [`send_hwm`](#options) frames, and its `send()` raises `RuntimeError` after [`request_timeout`](#options) if no client takes them. [Bidirectional Mode](advanced/bidirectional_mode/) is not supported.

    In every pattern, the receiver (Client) binds and the sender (Server) connects. [Multi-Servers](advanced/multi_server/) and [Multi-Clients](advanced/multi_client/) Modes need pattern `1`, `2` or `3`, whereas [Bidirectional Mode](advanced/bidirectional_mode/) needs pattern `0` or `1`.

**Usage:**

```python
NetGear(pattern=1) # sets zmq.REQ/zmq.REP pattern
```

&nbsp; 


## **`receive_mode`** 

This parameter select the Netgear's Mode of operation. It basically activates `Receive Mode`(_if `True`_) and `Send Mode`(_if `False`_). Furthermore, `recv()` method will only work when this flag is enabled(_i.e. `Receive Mode`_), whereas `send()` method will only work when this flag is disabled(_i.e.`Send Mode`_). 

**Data-Type:** Boolean

**Default Value:** Its default value is `False`(_i.e. Send Mode is activated by default_).

**Usage:**

```python
NetGear(receive_mode=True) # activates Recieve Mode
```

&nbsp; 

## **`options`** 

This parameter provides the flexibility to alter various NetGear API's internal properties, modes, and some PyZMQ flags.

**Data-Type:** Dictionary

**Default Value:** Its default value is `{}`

**Usage:**

!!! abstract "Supported dictionary attributes for NetGear API"

    * **`multiserver_mode`** (_boolean_) : This internal attribute activates the exclusive [**Multi-Servers Mode**](advanced/multi_server/), if enabled(`True`).

    * **`multiclient_mode`** (_boolean_) : This internal attribute activates the exclusive [**Multi-Clients Mode**](advanced/multi_client/), if enabled(`True`).

    * **`secure_mode`** (_integer_) : This internal attribute selects the exclusive [**Secure Mode**](advanced/secure_mode/). Its possible values are: `0`_(i.e. Grassland(no security))_ or `1`_(i.e. StoneHouse)_ or `2`_(i.e. IronHouse)_.

    * **`bidirectional_mode`** (_boolean_) : This internal attribute activates the exclusive [**Bidirectional Mode**](advanced/bidirectional_mode/), if enabled(`True`).

    * **`ssh_tunnel_mode`** (_string_) : This internal attribute activates the exclusive [**SSH Tunneling Mode**](advanced/ssh_tunnel/) ==at the Server-end only==, and sets the SSH server to tunnel through _(formatted as `user@server:port`, default port `22`)_.

    * **`ssh_tunnel_pwd`** (_string_): In SSH Tunneling Mode, This internal attribute sets the password required to authorize Host for SSH Connection ==at the Server-end only==. More information can be found [here ➶](advanced/ssh_tunnel/#exclusive-attributes)

    * **`ssh_tunnel_keyfile`** (_string_): In SSH Tunneling Mode, This internal attribute sets path to the user's SSH private key _(similar to `ssh -i`)_ that provide another way to authenticate with the SSH server ==at the Server-end only==. An invalid path is discarded with a warning. More information can be found [here ➶](advanced/ssh_tunnel/#exclusive-attributes)

    * **`custom_cert_location`** (_string_) : In Secure Mode, This internal attribute assigns user-defined location/path to directory for generating/storing Public+Secret Keypair necessary for encryption. This directory **MUST** already exist and be writable, otherwise `AssertionError` is raised. More information can be found [here ➶](advanced/secure_mode/#exclusive-attributes)

    * **`overwrite_cert`** (_boolean_) : In Secure Mode, This internal attribute decides whether to overwrite existing Public+Secret Keypair/Certificates or not, ==at the Server-end only==. More information can be found [here ➶](advanced/secure_mode/#exclusive-attributes)

    * **`allowed_ips`** (_list of strings_) : In Secure Mode, This optional internal attribute whitelists IP addresses of the **Server(s)** _(senders)_ allowed to connect, ==at the Client-end only==. By default, there's no IP whitelist, and any peer with valid CURVE credentials is accepted. More information can be found [here ➶](advanced/secure_mode/#exclusive-attributes)

    * **`jpeg_compression`**(_bool/str_): This internal attribute is used to activate(if `True`)/deactivate(if `False`) JPEG Frame Compression as well as to specify incoming frames colorspace with compression. By default colorspace is `BGR` and compression is enabled(`True`), but only if [simplejpeg](https://gitlab.com/jfolz/simplejpeg) is installed _(otherwise this attribute is silently ignored, and frames are sent uncompressed)_. Invalid values are skipped with a warning. More information can be found [here ➶](advanced/compression/#exclusive-attributes)

    * **`jpeg_compression_quality`**(_int/float_): This internal attribute controls the JPEG quantization factor in JPEG Frame Compression. Its value varies from `10` to `100` (the higher is the better quality but performance will be lower). Its default value is `90`. More information can be found [here ➶](advanced/compression/#exclusive-attributes)

    * **`jpeg_compression_fastdct`**(_bool_): This internal attributee if True, use fastest DCT method that speeds up encoding/decoding by 4-5% for a minor loss in quality in JPEG Frame Compression. It affects each end independently _(i.e. encoding at the Server, and decoding at the Client)_. Its default value is also `True`. More information can be found [here ➶](advanced/compression/#exclusive-attributes)

    * **`jpeg_compression_fastupsample`**(_bool_): This internal attribute if True, use fastest color upsampling method while decoding at the end it is set on. Its default value is `False`. More information can be found [here ➶](advanced/compression/#exclusive-attributes)

    * **`jpeg_compression_colorsubsampling`**(_str_): This internal attribute sets the chroma subsampling used while encoding at the Server end. Its possible values are: `"444"`, `"422"`, `"420"`, `"440"` and `"411"`, and its default value is `"422"`. More information can be found [here ➶](advanced/compression/#exclusive-attributes)

    * **`max_retries`**(_integer_): This internal attribute controls the maximum consecutive reconnection attempts before Server/Client exit itself, if it's unable to get any response/reply from the socket before a certain amount of time, when synchronous messaging patterns like (`zmq.PAIR` & `zmq.REQ/zmq.REP`) are being used. This counter is reset after every successful exchange. The Server raises `RuntimeError` once all retries fail. It's value can be any integer greater than or equal to `1`, and its default value is `3`.

    * **`request_timeout`**(_int/float_): This internal attribute controls the timeout value _(in seconds)_ for getting a response/reply from the socket, when synchronous messaging patterns like (`zmq.PAIR` & `zmq.REQ/zmq.REP`) are being used. The Client waits 3× this value before reconnecting, whereas the Server reconnects after this value and raises `RuntimeError` after [`max_retries`](#options) failed attempts. In `zmq.PUSH/zmq.PULL` pattern, it is also the `send()` timeout at the Server end. It's value can be any number greater than or equal to `4`, and its default value is `4` seconds.

    * **`receive_timeout`**(_int/float_): Similar to `request_timeout`, this internal attribute also controls the timeout value _(in seconds)_ but for non-synchronous `zmq.PUB/zmq.SUB` and `zmq.PUSH/zmq.PULL` patterns at the Client end only, after which the Client exits itself _(i.e. `recv()` returns `None`)_ if nothing arrives from the socket for that long. It's value can be any number greater than `0`, and its disabled by default _(meaning the client will wait forever for response)_. **Recommended** for these patterns, since their termination signal is best-effort. Its older name **`subscriber_timeout`** is also accepted as an alias.

    * **`send_hwm`**(_integer_): This internal attribute sets the maximum number of frames buffered at the Server end in `zmq.PUSH/zmq.PULL` pattern, while no Client takes them. Once full, `send()` waits up to `request_timeout` and then raises `RuntimeError`. It's value can be any integer greater than or equal to `1`, and its default value is `4`.

    * **`flag`**(_integer_): This PyZMQ attribute value can be either `0` or `zmq.NOBLOCK`_( i.e. 1)_, and applies to both sending and receiving. Its default value is `0`. More information can be found [here ➶](https://pyzmq.readthedocs.io/en/latest/api/zmq.html#zmq.Socket.recv).

    !!! warning "With flags=1 (i.e. `NOBLOCK`), `send()` may raise `zmq.Again` if the message cannot be queued right away, and NetGear might not terminate gracefully."

    * **`copy`**(_boolean_): This PyZMQ attribute selects if message be sent/received in a copying or non-copying _(zero-copy)_ manner. Its default value is `False` _(i.e. zero-copy)_.

    !!! danger "With `copy=False`, do not modify a raw frame's buffer that you've passed to `send()` until it is sent. JPEG compressed frames are always encoded into fresh buffers, so they're safe."

    * **`track`**(_boolean_): This PyZMQ attribute check if the message is tracked for notification that ZMQ has finished with it _(ignored if `copy=True`)_. Its default value is `False`.


The desired attributes can be passed to NetGear API as follows:

```python
# formatting parameters as dictionary attributes
options = {
    "secure_mode": 2,
    "custom_cert_location": "/home/foo/foo1/foo2", # must be an existing writable directory
    "overwrite_cert": True,
    "flag": 0, 
    "copy": True, 
    "track": False
}
# assigning it
NetGear(logging=True, **options)
```

&nbsp;


## **`logging`**

This parameter enables logging _(if `True`)_, essential for debugging. 

**Data-Type:** Boolean

**Default Value:** Its default value is `False`.

**Usage:**

```python
NetGear(logging=True)
```

&nbsp;