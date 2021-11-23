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

# Secure Mode for NetGear API 

<figure>
  <img src="../../../../assets/images/secure_mode.webp" alt="Secure Mode" loading="lazy" title="NetGear's Secure Mode" class="center"/>
</figure>


## Overview

Secure Mode provides easy access to powerful, smart & secure ZeroMQ's Security Layers in NetGear API that enables strong encryption on data, and unbreakable authentication between the Server and the Client with the help of custom Certificates/keys and brings cheap, standardized privacy and authentication for distributed systems over the network.

Secure Mode uses a new wire protocol, [**ZMTP 3.0**](http://zmtp.org/) that adds a security handshake to all ZeroMQ connections and a new security protocol, [**CurveZMQ**](http://curvezmq.org/), that implements "perfect forward security" between two ZeroMQ peers over a TCP connection. 

Secure Mode can be easily activated in NetGear API through `secure_mode` attribute of its [`options`](../../params/#options) dictionary parameter, during initialization. Furthermore, for managing this mode, NetGear API provides additional `custom_cert_location` & `overwrite_cert` like attribute too.


&nbsp;

## Supported ZMQ Security Layers

Secure mode supports the two most powerful ZMQ security layers:

* **Stonehouse:** which switches to the [**_CURVE_**](http://curvezmq.org/) security protocol that provides strong encryption on data, and almost unbreakable authentication. Stonehouse is the minimum you would use over public networks and assures clients that they are speaking to an authentic server while allowing any client to connect. ==This security layer is less secure but at the same time faster than IronHouse security mechanism.==

* **Ironhouse:** which further extends *Stonehouse* layer with client public key authentication. This is the strongest security model present in ZeroMQ, protecting against every attack we know about except _end-point attacks_. ==This security layer enhanced security comes at a price of additional latency.==


&nbsp;


!!! danger "Important Information regarding Secure Mode"

    * The `secure_mode` attribute value at the Client's end **MUST** match exactly the Server's end _(i.e. **IronHouse** security layer is only compatible with **IronHouse**, and **NOT** with **StoneHouse**)_.

    * The Public+Secret Keypairs generated at the Server end **MUST** be made available at the Client's end too for successful authentication. If mismatched, connection failure will occur.

    * By Default, the Public+Secret Keypairs will be generated/stored at the `$HOME/.vidgear/keys` directory of your machine _(e.g. `/home/foo/.vidgear/keys` on Linux)_. But you can also use [`custom_cert_location`](../../params/#options) attribute to set your own Custom-Path for a directory to generate/store these Keypairs.

    * ==**DO NOT** share generated public+secret Keypairs with anyone else on the network to avoid any potential security breach.== At the Server End, You can easily use the [`'overwrite_cert'`](../../params/#options) attribute for regenerating New-Keypairs on initialization. But make sure those newly generated Keypairs at the Server-End **MUST** be made available at Client's End for successful authentication.

    * **IronHouse** is the strongest Security Layer available, but it involves certain security checks that lead to  **ADDITIONAL LATENCY**.

    * Secure Mode only supports `libzmq` library version `>= 4.0`.


&nbsp;


## Features

- [x] Supports the two most powerful **ZMQ security layers:** StoneHouse & IronHouse.

- [x] Auto-generates, auto-validates & auto-stores the required Public+Secret Keypairs safely.

- [x] Compatible with all messaging pattern, primary and exclusive modes.

- [x] Strong data encryption & Unbreakable authentication.

- [x] Able to protect against many **man-in-the-middle (MITM)** attacks.

- [x] Minimum hassle and very easy to enable and integrate.


&nbsp;

## Exclusive Attributes

For implementing Secure Mode, NetGear API currently provide following exclusive attribute for its [`options`](../../params/#options) dictionary parameter:


* `secure_mode` (_integer_) : This attribute activates and sets the ZMQ security Mechanism. Its possible values are: `1`(_StoneHouse_) & `2`(_IronHouse_), and its default value is `0`(_Grassland(no security)_). Its usage is as follows:
  
    ```python
    #activates IronHouse Security Mechanism
    options = {'secure_mode':2}
    ```

* **`custom_cert_location`** (_string_): This attribute sets a custom location/path to directory to generate/store Public+Secret Keypair/Certificates for enabling encryption. This attribute will force NetGear to create `.vidgear` folder _(only if not available)_ at the assigned custom path _(instead of home directory)_, and then use that directory for storing new Keypairs/Certificates. It can be used as follows:

    ```python
    # set custom Keypair location to '/home/foo/foo1/foo2'
    options = {
        "secure_mode": 2,
        "custom_cert_location": "/home/foo/foo1/foo2",
    } 
    ```

* **`overwrite_cert`** (_bool_): **[For Server-end only]** This attribute sets whether to overwrite existing Public+Secret Keypair/Certificates and re-generate new ones, to protect against any potential security breach. If set to `True` a new Keypair/Certificates will be generated during NetGear initialization in place of old ones. Its usage is as follows:
    
    !!! warning "`overwrite_cert` parameter is disabled for Client's end!"

    ```python
    # a new Keypair will be generated
    options = {"secure_mode": 2, "overwrite_cert": True} 
    ```


&nbsp;


## Usage Examples


### Bare-Minimum Usage

Following is the bare-minimum code you need to get started with Secure Mode in NetGear API:

#### Server's End

Open your favorite terminal and execute the following python code:

!!! tip "You can terminate both sides anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="9"
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import NetGear

# open any valid video stream(for e.g `test.mp4` file)
stream = VideoGear(source="test.mp4").start()

# activate StoneHouse security mechanism
options = {"secure_mode": 1}

# Define NetGear Server with defined parameters
server = NetGear(pattern=1, logging=True, **options)

# loop over until KeyBoard Interrupted
while True:

    try:
        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # send frame to server
        server.send(frame)

    except KeyboardInterrupt:
        break

# safely close video stream
stream.stop()

# safely close server
server.close()
```

#### Client's End

Then open another terminal on the same system and execute the following python code and see the output:

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="6"
# import required libraries
from vidgear.gears import NetGear
import cv2

# activate StoneHouse security mechanism
options = {"secure_mode": 1}

# define NetGear Client with `receive_mode = True` and defined parameter
client = NetGear(pattern=1, receive_mode=True, logging=True, **options)

# loop over
while True:

    # receive frames from network
    frame = client.recv()

    # check for received frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # Show output window
    cv2.imshow("Output Frame", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close client
client.close()
```

&nbsp; 

&nbsp;


### Using Secure Mode with Variable Parameters


#### Client's End

Open a terminal on Client System _(where you want to display the input frames received from the Server)_ and execute the following python code: 

!!! info "Note down the local IP-address of this system(required at Server's end) and also replace it in the following code. You can follow [this FAQ](../../../../help/netgear_faqs/#how-to-find-local-ip-address-on-different-os-platforms) for this purpose."

!!! danger "You need to paste the Public+Secret Keypairs _(generated at the Server End)_ at the `$HOME/.vidgear/keys` directory of your Client machine for a successful authentication!"

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="6 11-17"
# import required libraries
from vidgear.gears import NetGear
import cv2

# activate IronHouse security mechanism
options = {"secure_mode": 2}

# Define NetGear Client at given IP address and define parameters 
# !!! change following IP address '192.168.x.xxx' with yours !!!
client = NetGear(
    address="192.168.x.xxx",
    port="5454",
    protocol="tcp",
    pattern=2,
    receive_mode=True,
    logging=True,
    **options
)

# loop over
while True:

    # receive frames from network
    frame = client.recv()

    # check for received frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # Show output window
    cv2.imshow("Output Frame", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close client
client.close()
```

&nbsp;

#### Server's End

Now, Open the terminal on another Server System _(with a webcam connected to it at index `0`)_, and execute the following python code: 

!!! info "Replace the IP address in the following code with Client's IP address you noted earlier."

!!! danger "You also need to copy the Public+Secret Keypairs _(generated on running this example code)_ present in the `$HOME/.vidgear/keys` directory, and make available at Client's end for a successful authentication."

!!! tip "You can terminate stream on both side anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="7 15-20"
# import required libraries
from vidgear.gears import VideoGear
from vidgear.gears import NetGear

# activate IronHouse security mechanism, and 
# [BEWARE!!!] generating new Keypairs for this example !!!
options = {"secure_mode": 2, "overwrite_cert": True}

# Open live video stream on webcam at first index(i.e. 0) device
stream = VideoGear(source=0).start()

# Define NetGear server at given IP address and define parameters 
# !!! change following IP address '192.168.x.xxx' with client's IP address !!!
server = NetGear(
    address="192.168.x.xxx",
    port="5454",
    protocol="tcp",
    pattern=2,
    logging=True,
    **options
)

# loop over until KeyBoard Interrupted
while True:

    try:
        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break

        # {do something with the frame here}

        # send frame to server
        server.send(frame)

    except KeyboardInterrupt:
        break

# safely close video stream
stream.stop()

# safely close server
server.close()
```

&nbsp; 