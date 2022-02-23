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

# NetGear FAQs

&nbsp;

## What is NetGear API and what does it do?

**Answer:** NetGear is exclusively designed to transfer video frames & data synchronously (Pair & Request/Reply) as well as asynchronously (Publish/Subscribe) between various interconnecting systems over the network in real-time. _For more info. see [NetGear doc ➶](../../gears/netgear/overview/)_

&nbsp;

## How to get started with NetGear API?

**Answer:** See [NetGear doc ➶](../../gears/netgear/overview/). Still in doubt, then discuss on [Gitter ➶](https://gitter.im/vidgear/community) Community channel.

&nbsp;

## What Exclusive Modes are compatible with each other in NetGear API?

Here's the compatibility chart for NetGear's [Exclusive Modes](../../gears/netgear/overview/#exclusive-modes):


| Exclusive Modes | Multi-Servers | Multi-Clients | Secure | Bidirectional | SSH Tunneling |
| :-------------: | :-----------: | :-----------: | :----: | :-----------: | :-----------: |
| **Multi-Servers** | - | No _(throws error)_ | Yes | Yes | No _(throws error)_ |
| **Multi-Clients** |  No _(throws error)_ | - | Yes | Yes | No _(throws error)_ |
| **Secure** | Yes | Yes | - | Yes | Yes |
| **Bidirectional** | Yes | Yes | Yes | - | Yes |
| **SSH Tunneling** |  No _(throws error)_ | No _(throws error)_ | Yes | Yes | - |

&nbsp;


## Why NetGear is running slow?

**Answer:** Here are few tips to troubleshoot performance on your machine:

* **Update ZMQ to latest:** Update your `pyzmq` lib as follows:

    ```sh
    sudo pip3 install -U pyzmq
    ``` 

* **Install testing branch:** The [`testing`](https://github.com/abhiTronix/vidgear/tree/testing) branch may contain many latest performance updates, which are not yet merged into master branch. Therefore, you can try them earlier, by [installing `testing` branch directly ➶](../../installation/source_install/#installation).

* **Use PUB/SUB pattern if you're live streaming**:  Try different [`pattern`](../../gears/netgear/params/#pattern) values, as each of them suits different settings. For example, you can use its [**Publisher/Subscriber pattern**](https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/pubsub.html) _(i.e. `pattern=2`)_ for asynchronous high-speed transmission over real-time streams, and it works faster than other synchronous patterns for this scenario.

* **Use Wired connection instead of Wireless connection**: Remember typical 802.11g Wireless has a theoretical maximum of 54Mbps. Typical wired 10/100/1000 Ethernet has a theoretical maximum of 100 Gbps. So in theory wired is faster. However, these speeds are only on your local network. So chose your network configuration wisely.

* **Enable all Performance Attributes with Frame Compression**: You can also try enabling [Frame Compression](../../gears/netgear/advanced/compression/) with its all [Performance Attributes](../../gears/netgear/advanced/compression/#performance-attributes) for NetGear API.

* **Reduce Frame Size:** Use VidGear's real-time _Frame-Size Reducer_(`reducer`) method for reducing frame-size on-the-go for additional performance _(see [this usage example ➶](../../gears/netgear/advanced/bidirectional_mode/#using-bidirectional-mode-for-video-frames-transfer-with-frame-compression))_. Remember, sending large HQ video-frames may required more network bandwidth and packet size, which can lead to additional latency!

* Systematically, **check for [Hardware/Network Issues ➶](https://github.com/abhiTronix/vidgear/issues/137)**

* Finally, if nothing works then, **checkout [**NetGear_Async API ➶**](../../gears/netgear_async/overview/)**

&nbsp;

## How to find local IP-address on different OS platforms?

**Answer:** For finding local IP-address of your machine:

=== "On Linux OS"

    - [x] **Follow [this tutorial ➶](https://linuxize.com/post/how-to-find-ip-address-linux/#find-your-private-ip-address)**

=== "On Windows OS"

    - [x] **Follow [this tutorial ➶](https://www.avast.com/c-how-to-find-ip-address)**

=== "On MAC OS"
    
    - [x] **Follow [this tutorial ➶](https://www.avast.com/c-how-to-find-ip-address)**

&nbsp;

## How to send data along with frames in Multi-Servers and Multi-Clients Modes?

**Answer:** See [Multi-Servers usage example ➶](../../gears/netgear/advanced/multi_server/#using-multi-servers-mode-with-custom-data-transfer) and [Multi-Clients usage example ➶](../../gears/netgear/advanced/multi_client/#using-multi-clients-mode-with-custom-data-transfer)

&nbsp;

## How to use enable Encryption and Authentication in NetGear API?

**Answer:** See its [Secure Mode doc ➶](../../gears/netgear/advanced/secure_mode/).

&nbsp;

## How to send custom data along with frames bidirectionally in NetGear API?

**Answer:** See its [Bidirectional Mode doc ➶](../../gears/netgear/advanced/bidirectional_mode/).

&nbsp;


## How to access NetGear API outside network or remotely?

**Answer:** See its [SSH Tunneling Mode doc ➶](../../gears/netgear/advanced/ssh_tunnel/).

&nbsp;

## Are there any side-effect of sending data with frames?

**Answer:** Yes, it may lead to additional **LATENCY** depending upon the size/amount of the data being transferred. User discretion is advised.

&nbsp;


## Why NetGear API not working correctly?

**Answer:** First, carefully go through [NetGear doc ➶](../../gears/netgear/overview/) that contains detailed information. Also, checkout [PyZmq Docs ➶](https://pyzmq.readthedocs.io/en/latest/) for its various settings/parameters. If still it doesn't work for you, then let us know on [Gitter ➶](https://gitter.im/vidgear/community)

&nbsp;

## How to solve `zmq.error.ZMQError` errors?

**Answer:** For those used to the idea that a "server" provides their address to a client, then you should *recheck your preconceptions*! Please read the [Netgear instructions](https://abhitronix.github.io/vidgear/latest/gears/netgear/usage/#using-netgear-with-variable-parameters) carefully, and you will note that it is the client device that defines the IP that is provided to the server config. If you get this the wrong way (using the server IP on the client), then you will get a `zmq.error.ZMQError` error. Make sure it is the **client's IP** shared across the two systems.

&nbsp;
