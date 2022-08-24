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

# NetGear_Async API Usage Examples:


!!! tip "Helpful Tips"

    * It is advised to enable logging(`logging = True`) on the first run for easily identifying any runtime errors.

    * It is advised to comprehend [NetGear API](../../netgear/overview/) before using this API.

!!! example "After going through following Usage Examples, Checkout more bonus examples [here ➶](../../../help/netgear_async_ex/)"


## Requirement

NetGear_Async API is the part of `asyncio` package of VidGear, thereby you need to install VidGear with asyncio support as follows:

```sh
pip install vidgear[asyncio]
```

&thinsp;


## Bare-Minimum Usage

Following is the bare-minimum code you need to get started with NetGear_Async API:

### Server's End

Open your favorite terminal and execute the following python code:

!!! tip "You can terminate stream on both side anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python
# import libraries
from vidgear.gears.asyncio import NetGear_Async
import asyncio

# initialize Server with suitable source
server = NetGear_Async(source="/home/foo/foo1.mp4").launch()

if __name__ == "__main__":
    # set event loop
    asyncio.set_event_loop(server.loop)
    try:
        # run your main function task until it is complete
        server.loop.run_until_complete(server.task)
    except (KeyboardInterrupt, SystemExit):
        # wait for interrupts
        pass
    finally:
        # finally close the server
        server.close()
```

### Client's End

Then open another terminal on the same system and execute the following python code and see the output:

!!! warning "Client will throw TimeoutError if it fails to connect to the Server in given [`timeout`](../params/#timeout) value!"

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python
# import libraries
from vidgear.gears.asyncio import NetGear_Async
import cv2, asyncio

# define and launch Client with `receive_mode=True`
client = NetGear_Async(receive_mode=True).launch()


# Create a async function where you want to show/manipulate your received frames
async def main():
    # loop over Client's Asynchronous Frame Generator
    async for frame in client.recv_generator():

        # do something with received frames here

        # Show output window
        cv2.imshow("Output Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # await before continuing
        await asyncio.sleep(0)


if __name__ == "__main__":
    # Set event loop to client's
    asyncio.set_event_loop(client.loop)
    try:
        # run your main function task until it is complete
        client.loop.run_until_complete(main())
    except (KeyboardInterrupt, SystemExit):
        # wait for interrupts
        pass

    # close all output window
    cv2.destroyAllWindows()
    # safely close client
    client.close()
```


&nbsp; 

## Using NetGear_Async with Variable Parameters

### Client's End

Open a terminal on Client System _(where you want to display the input frames received from the Server)_ and execute the following python code: 

!!! info "Note down the local IP-address of this system(required at Server's end) and also replace it in the following code. You can follow [this FAQ](../../../help/netgear_faqs/#how-to-find-local-ip-address-on-different-os-platforms) for this purpose."

!!! warning "Client will throw TimeoutError if it fails to connect to the Server in given [`timeout`](../params/#timeout) value!"

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="7-12"
# import libraries
from vidgear.gears.asyncio import NetGear_Async
import cv2, asyncio

# define and launch Client with `receive_mode=True`. #change following IP address '192.168.x.xxx' with yours
client = NetGear_Async(
    address="192.168.x.xxx",
    port="5454",
    protocol="tcp",
    pattern=2,
    receive_mode=True,
    logging=True,
).launch()


# Create a async function where you want to show/manipulate your received frames
async def main():
    # loop over Client's Asynchronous Frame Generator
    async for frame in client.recv_generator():

        # do something with received frames here

        # Show output window
        cv2.imshow("Output Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # await before continuing
        await asyncio.sleep(0)


if __name__ == "__main__":
    # Set event loop to client's
    asyncio.set_event_loop(client.loop)
    try:
        # run your main function task until it is complete
        client.loop.run_until_complete(main())
    except (KeyboardInterrupt, SystemExit):
        # wait for interrupts
        pass

    # close all output window
    cv2.destroyAllWindows()
    # safely close client
    client.close()
```

### Server's End

Now, Open the terminal on another Server System _(with a webcam connected to it at index `0`)_, and execute the following python code: 

!!! info "Replace the IP address in the following code with Client's IP address you noted earlier."

!!! tip "You can terminate stream on both side anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="7-12"
# import libraries
from vidgear.gears.asyncio import NetGear_Async
import asyncio

# initialize Server with suitable source
server = NetGear_Async(
    source=0,
    address="192.168.x.xxx",
    port="5454",
    protocol="tcp",
    pattern=2,
    logging=True,
).launch()

if __name__ == "__main__":
    # set event loop
    asyncio.set_event_loop(server.loop)
    try:
        # run your main function task until it is complete
        server.loop.run_until_complete(server.task)
    except (KeyboardInterrupt, SystemExit):
        # wait for interrupts
        pass
    finally:
        # finally close the server
        server.close()
```

&nbsp; 


## Using NetGear_Async with a Custom Source(OpenCV)

NetGear_Async allows you to easily define your own custom Source at Server-end that you want to use to transform your frames before sending them onto the network. 

Let's implement a bare-minimum example with a Custom Source using NetGear_Async API and OpenCV:

### Server's End

Open your favorite terminal and execute the following python code:

!!! tip "You can terminate stream on both side anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="14-31 38"
# import library
from vidgear.gears.asyncio import NetGear_Async
import cv2, asyncio

# initialize Server without any source
server = NetGear_Async(source=None, logging=True)

# !!! define your own video source here !!!
# Open any video stream such as live webcam
# video stream on first index(i.e. 0) device
stream = cv2.VideoCapture(0)

# Create a async frame generator as custom source
async def my_frame_generator():

    # loop over stream until its terminated
    while True:

        # read frames
        (grabbed, frame) = stream.read()

        # check if frame empty
        if not grabbed:
            break

        # do something with the frame to be sent here

        # yield frame
        yield frame
        # sleep for sometime
        await asyncio.sleep(0)


if __name__ == "__main__":
    # set event loop
    asyncio.set_event_loop(server.loop)
    # Add your custom source generator to Server configuration
    server.config["generator"] = my_frame_generator()
    # Launch the Server
    server.launch()
    try:
        # run your main function task until it is complete
        server.loop.run_until_complete(server.task)
    except (KeyboardInterrupt, SystemExit):
        # wait for interrupts
        pass
    finally:
        # close stream
        stream.release()
        # finally close the server
        server.close()
```

### Client's End

Then open another terminal on the same system and execute the following python code and see the output:

!!! warning "Client will throw TimeoutError if it fails to connect to the Server in given [`timeout`](../params/#timeout) value!"

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python
# import libraries
from vidgear.gears.asyncio import NetGear_Async
import cv2, asyncio

# define and launch Client with `receive_mode=True`
client = NetGear_Async(receive_mode=True, logging=True).launch()


# Create a async function where you want to show/manipulate your received frames
async def main():
    # loop over Client's Asynchronous Frame Generator
    async for frame in client.recv_generator():

        # {do something with received frames here}

        # Show output window
        cv2.imshow("Output Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # await before continuing
        await asyncio.sleep(0)


if __name__ == "__main__":
    # Set event loop to client's
    asyncio.set_event_loop(client.loop)
    try:
        # run your main function task until it is complete
        client.loop.run_until_complete(main())
    except (KeyboardInterrupt, SystemExit):
        # wait for interrupts
        pass

    # close all output window
    cv2.destroyAllWindows()
    # safely close client
    client.close()
```

&nbsp; 

## Using NetGear_Async with Other Gears

NetGear_Async can be used with any other Gears without any compatibility issues. 

Let's implement a bare-minimum example where we are sending [Stabilized](../../stabilizer/overview/) frames from Server-end and saving them at Client's end with [WriteGear](../../writegear/introduction/) as follows:

### Server's End

Open your favorite terminal and execute the following python code:

!!! tip "You can terminate stream on both side anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python hl_lines="7"
# import libraries
from vidgear.gears.asyncio import NetGear_Async
import asyncio

# initialize Server with suitable source and enable stabilization
server = NetGear_Async(
    source="/home/foo/foo1.mp4", stabilize=True, logging=True
).launch()

if __name__ == "__main__":
    # set event loop
    asyncio.set_event_loop(server.loop)
    try:
        # run your main function task until it is complete
        server.loop.run_until_complete(server.task)
    except (KeyboardInterrupt, SystemExit):
        # wait for interrupts
        pass
    finally:
        # finally close the server
        server.close()
```

### Client's End

Then open another terminal on the same system and execute the following python code and see the output:

!!! warning "Client will throw TimeoutError if it fails to connect to the Server in given [`timeout`](../params/#timeout) value!"

!!! tip "You can terminate client anytime by pressing ++ctrl+"C"++ on your keyboard!"

```python
# import libraries
from vidgear.gears.asyncio import NetGear_Async
from vidgear.gears import WriteGear
import cv2, asyncio

# define and launch Client with `receive_mode=True`
client = NetGear_Async(receive_mode=True).launch()
# Define writer with output filename 'Output.mp4'
writer = WriteGear(output="Output.mp4", logging=True)


# Create a async function where you want to show/manipulate your received frames
async def main():
    # loop over Client's Asynchronous Frame Generator
    async for frame in client.recv_generator():

        # {do something with received frames here}

        # write a modified frame to writer
        writer.write(frame)

        # Show output window
        cv2.imshow("Output Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # await before continuing
        await asyncio.sleep(0)


if __name__ == "__main__":
    # Set event loop to client's
    asyncio.set_event_loop(client.loop)
    try:
        # run your main function task until it is complete
        client.loop.run_until_complete(main())
    except (KeyboardInterrupt, SystemExit):
        # wait for interrupts
        pass

    # close all output window
    cv2.destroyAllWindows()
    # safely close client
    client.close()
    # safely close writer
    writer.close()
```

&nbsp; 
