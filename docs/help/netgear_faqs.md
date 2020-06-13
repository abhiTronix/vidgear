<!--
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
-->

# NetGear FAQs

&nbsp;

## What is NetGear API and what does it do?

**Answer:** NetGear is exclusively designed to transfer video frames & data synchronously (Pair & Request/Reply) as well as asynchronously (Publish/Subscribe) between various interconnecting systems over the network in real-time. _For more info. see [NetGear doc ➶](../../gears/netgear/overview/)_

&nbsp;

## How to get started with NetGear API?

**Answer:** See [NetGear doc ➶](../../gears/netgear/overview/). Still in doubt, then ask us on [Gitter ➶](https://gitter.im/vidgear/community) Community channel.

&nbsp;

## What Exclusive Modes are compatible with each other in NetGear API?

Here's the compatibility chart for NetGear's [Exclusive Modes](../../gears/netgear/overview/#exclusive-modes):


| Exclusive Modes | Multi-Servers | Multi-Clients | Secure | Bidirectional |
| :-------------: | :-----------: | :-----------: | :----: | :-----------: |
| **Multi-Servers** | - | No _(throws error)_ | Yes | No _(disables it)_ |
| **Multi-Clients** |  No _(throws error)_ | - | Yes | No _(disables it)_ |
| **Secure** | Yes | Yes | - | Yes |
| **Bidirectional** | No _(disabled)_ | No _(disabled)_ | Yes | - |

&nbsp;

## How to receive frames from multiple Servers and multiple Clients through NetGear API?

**Answer:** See [Multi-Servers Mode doc ➶](../../gears/netgear/advanced/multi_server/) and [Multi-Clients Mode doc ➶](../../gears/netgear/advanced/multi_client/)

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

## Are there any side-effect of sending data with frames?

**Answer:** Yes, it may lead to additional **LATENCY** depending upon the size/amount of the data being transferred. User discretion is advised.

&nbsp;


## How can I compress frames before sending them to Client(s) in NetGear API?

**Answer:** See [Frame Compression doc ➶](../../gears/netgear/advanced/compression/)

&nbsp;

## Which compression format is the fastest for NetGear API?

**Answer:** According to an [answer](https://answers.opencv.org/question/207286/why-imencode-taking-so-long/?answer=211496#post-id-211496):

The time varies differently for different encoding/decoding format as follows:

| Encoding format | Time taken _(in milliseconds)_ |
| :---------: | :-------: |
| bmp | 20-40 |
| jpg | 50-70 |
| png | 200-250 | 

Despite `bmp` being the fasted, using `jpg` is more suitable for encoding, since highly-optimized `turbojpeg` JPEG library is now a part of OpenCV binaries. But you can choose whatever suits you.

&nbsp;

## Why NetGear API not working correctly?

**Answer:** First, carefully go through [NetGear doc ➶](../../gears/netgear/overview/) that contains detailed information. Also, checkout [PyZmq Docs ➶](https://zeromq.github.io/pyzmq/) for its various settings/parameters. If still it doesn't work for you, then [tell us on Gitter ➶](https://gitter.im/vidgear/community), and if that doesn't help, then finally [report an issue ➶](../../contribution/issue/)

&nbsp;

## Why NetGear is slow?

**Answer:** Here are few tips that may help you in improving speed significantly:

* **Update ZMQ to latest:** Update your `pyzmq` lib as follows:

    ```sh
    sudo pip3 install -U pyzmq
    ``` 

* **Use PUB/SUB pattern if you're live streaming**: Remember to use [**Publisher/Subscriber pattern**](https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/pubsub.html) only for asynchronous high-speed transmission over live streams. Other messaging patterns such as Pair & Client/Server are only useful for slower synchronous transmission. You can set parameter [`pattern=2` during NetGear class initialization](../../gears/netgear/params/#pattern) in your code to activate Publisher/Subscriber pattern.

* **Use Wired connection instead of Wireless connection**: Remember typical 802.11g Wireless has a theoretical maximum of 54Mbps. Typical wired 10/100/1000 Ethernet has a theoretical maximum of 100 Gbps. So in theory wired is faster. However, these speeds are only on your local network. So chose your network configuration wisely.

* **Compress your image/frame before transmission:** Try [Frame Encoding/Decoding Compression capabilities for NetGear API ➶](../../gears/netgear/advanced/compression/).

* **Reduce Frame Size:** Use VidGear's real-time _Frame-Size Reducer_(`reducer`) method for reducing frame-size on-the-go for additional performance _(see [this usage example ➶](../../gears/netgear/advanced/bidirectional_mode/#using-bidirectional-mode-for-video-frames-transfer-with-frame-compression))_. Remember, sending large HQ video-frames may required more network bandwidth and packet size, which may add to video latency!

* _Finally, if nothing works, then, switch to more faster and efficient [**NetGear_Async API ➶**](../../gears/netgear_async/overview/)_


&nbsp;