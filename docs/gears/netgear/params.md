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

**Data-Type:** String

**Default Value:** Its default value is based on selected [primary mode](../overview/#primary-modes), _i.e `'localhost'` for Send Mode and `'*'` for Receive Mode_ on a local machine.

**Usage:**

```python
NetGear(address="192.168.0.145")
```

&nbsp; 


## **`port`**

This parameter sets the valid Network Port for Server/Client. Network port is a number that identifies one side of a connection between two devices on the network and is used determine to which process or application a message should be delivered.

!!! danger "Exception for Exclusive Modes"
    
    [**In Multi-Servers Mode**](../advanced/multi_server/):

      * A unique port number **MUST** be assigned to each Server on the network using this parameter. 
      * At Client end, a List/Tuple of all available Server(s) ports **MUST** be assigned using this same parameter. 
      * See its usage example [here ➶](../advanced/multi_server/#bare-minimum-usage).

    [**In Multi-Client Mode**](../advanced/multi_client/):

      * A unique port number **MUST** be assigned to each Client on the network using this parameter. 
      * At Server end, a List/Tuple of all available Client(s) ports **MUST** be assigned using this same parameter. 
      * See its usage example [here ➶](../advanced/multi_client/#bare-minimum-usage).

**Data-Type:** String or List/Tuple

**Default Value:** Its default value is `'5555'`

**Usage:**

```python
NetGear(port="5575")
```

&nbsp; 


## **`protocol`** 


This parameter sets the valid messaging protocol between server and client. A network protocol is a set of established rules that dictates how to format, transmit and receive data so computer network devices - from servers and routers to endpoints - can communicate regardless of the differences in their underlying infrastructures, designs or standards. Supported protocol are: `'tcp'` and `'ipc'`.

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

**Default Value:** Its default value is `0` (_i.e `zmq.PAIR`_). 

!!! info "Supported ZMQ patterns"

    **All supported ZMQ patterns for NetGear are:**

    * **`0` (_.i.e. zmq.PAIR_):** In this pattern, the communication is bidirectional. There is no specific state stored within the socket. There can only be one connected peer. The server listens on a certain port and a client connects to it.
    * **`1` (_.i.e. zmq.REQ/zmq.REP_):** In this pattern, it employs `ZMQ REQ` sockets that can connect to many servers. The requests will be interleaved or distributed to both the servers. socket `zmq.REQ` will block send unless it has successfully received a reply back and socket `zmq.REP` will block on recv() unless it has received a request.
    * **`2` (_.i.e. zmq.PUB/zmq.SUB_):** It is an another classic pattern where senders of messages, called _publishers_, do not program the messages to be sent directly to specific receivers, called _subscribers_. Messages are published without the knowledge of what or if any subscriber of that knowledge exists. A `ZMQ.SUB` can connect to multiple `ZMQ.PUB` (publishers). No single publisher overwhelms the subscriber. The messages from both publishers are interleaved.

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

    * **`multiserver_mode`** (_boolean_) : This internal attribute activates the exclusive [**Multi-Servers Mode**](../advanced/multi_server/), if enabled(`True`).

    * **`multiclient_mode`** (_boolean_) : This internal attribute activates the exclusive [**Multi-Clients Mode**](../advanced/multi_client/), if enabled(`True`).

    * **`secure_mode`** (_integer_) : This internal attribute selects the exclusive [**Secure Mode**](../advanced/secure_mode/). Its possible values are: `0`_(i.e. Grassland(no security))_ or `1`_(i.e. StoneHouse)_ or `2`_(i.e. IronHouse)_.

    * **`bidirectional_mode`** (_boolean_) : This internal attribute activates the exclusive [**Bidirectional Mode**](../advanced/bidirectional_mode/), if enabled(`True`).

    * **`ssh_tunnel_mode`** (_string_) : This internal attribute activates the exclusive [**SSH Tunneling Mode**](../advanced/ssh_tunnel/) ==at the Server-end only==.

    * **`ssh_tunnel_pwd`** (_string_): In SSH Tunneling Mode, This internal attribute sets the password required to authorize Host for SSH Connection ==at the Server-end only==. More information can be found [here ➶](../advanced/ssh_tunnel/#supported-attributes)

    * **`ssh_tunnel_keyfile`** (_string_): In SSH Tunneling Mode, This internal attribute sets path to Host key that provide another way to authenticate host for SSH Connection ==at the Server-end only==. More information can be found [here ➶](../advanced/ssh_tunnel/#supported-attributes)

    * **`custom_cert_location`** (_string_) : In Secure Mode, This internal attribute assigns user-defined location/path to directory for generating/storing Public+Secret Keypair necessary for encryption. More information can be found [here ➶](../advanced/secure_mode/#supported-attributes)

    * **`overwrite_cert`** (_boolean_) : In Secure Mode, This internal attribute decides whether to overwrite existing Public+Secret Keypair/Certificates or not, ==at the Server-end only==. More information can be found [here ➶](../advanced/secure_mode/#supported-attributes)

    * **`jpeg_compression`**(_bool/str_): This internal attribute is used to activate(if `True`)/deactivate(if `False`) JPEG Frame Compression as well as to specify incoming frames colorspace with compression. By default colorspace is `BGR` and compression is enabled(`True`). More information can be found [here ➶](../advanced/compression/#supported-attributes)

    * **`jpeg_compression_quality`**(_int/float_): This internal attribute controls the JPEG quantization factor in JPEG Frame Compression. Its value varies from `10` to `100` (the higher is the better quality but performance will be lower). Its default value is `90`. More information can be found [here ➶](../advanced/compression/#supported-attributes)

    * **`jpeg_compression_fastdct`**(_bool_): This internal attributee if True, use fastest DCT method that speeds up decoding by 4-5% for a minor loss in quality in JPEG Frame Compression. Its default value is also `True`. More information can be found [here ➶](../advanced/compression/#supported-attributes)

    * **`jpeg_compression_fastupsample`**(_bool_): This internal attribute if True, use fastest color upsampling method. Its default value is `False`. More information can be found [here ➶](../advanced/compression/#supported-attributes)

    * **`max_retries`**(_integer_): This internal attribute controls the maximum retries before Server/Client exit itself, if it's unable to get any response/reply from the socket before a certain amount of time, when synchronous messaging patterns like (`zmq.PAIR` & `zmq.REQ/zmq.REP`) are being used. It's value can anything greater than `0`, and its default value is `3`.

    * **`request_timeout`**(_integer_): This internal attribute controls the timeout value _(in seconds)_, after which the Server/Client exit itself if it's unable to get any response/reply from the socket, when synchronous messaging patterns like (`zmq.PAIR` & `zmq.REQ/zmq.REP`) are being used. It's value can anything greater than `0`, and its default value is `10` seconds.

    * **`flag`**(_integer_): This PyZMQ attribute value can be either `0` or `zmq.NOBLOCK`_( i.e. 1)_. More information can be found [here ➶](https://pyzmq.readthedocs.io/en/latest/api/zmq.html).

    * **`copy`**(_boolean_): This PyZMQ attribute selects if message be received in a copying or non-copying manner. If `False` a object is returned, if `True` a string copy of the message is returned.

    * **`track`**(_boolean_): This PyZMQ attribute check if the message is tracked for notification that ZMQ has finished with it. (_ignored if copy=True_).


The desired attributes can be passed to NetGear API as follows:

```python
# formatting parameters as dictionary attributes
options = {
    "secure_mode": 2,
    "custom_cert_location": "/home/foo/foo1/foo2",
    "overwrite_cert": True,
    "flag": 0,
    "copy": False,
    "track": False,
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
NetGear_Async(logging=True)
```

&nbsp;