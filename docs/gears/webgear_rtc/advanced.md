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

# WebGear_RTC API Advanced Usage:

!!! note "This is a continuation of the [WebGear_RTC doc ➶](./#webgear_rtc-api). Thereby, It's advised to first get familiarize with this API, and its [requirements](usage/#requirements)."


!!! example "After going through following Usage Examples, Checkout more bonus examples [here ➶](../../help/webgear_rtc_ex/)"

&thinsp;


## Using WebGear_RTC as Real-time Broadcaster 

WebGear_RTC streams every frame to every connected peer consumer/client _(through aiortc's `MediaRelay`)_. Additionally, you can use [`enable_live_broadcast`](params/#webgear_rtc-specific-attributes) boolean attribute through its options dictionary parameter to turn it into a never-ending live broadcast, that keeps streaming _(with "No Input" frames)_ even when the source disconnects, so peers joining later always get a stream.

!!! warning "Before `v0.3.7`, multiple consumers without `enable_live_broadcast` silently split frames among themselves _(each one got only a fraction of frames)_. This is fixed now, so `enable_live_broadcast` is no longer needed for multiple consumers."

Let's implement a bare-minimum example using WebGear_RTC as Real-time Broadcaster:

!!! info "[`enable_infinite_frames`](params/#webgear_rtc-specific-attributes) is enforced by default with this(`enable_live_broadcast`) attribute."

!!! tip "For accessing WebGear_RTC on different Client Devices on the network, we use `"0.0.0.0"` as host value instead of `"localhost"` on Host Machine. More information can be found [here ➶](../../help/webgear_rtc_faqs/#is-it-possible-to-stream-on-a-different-device-on-the-network-with-webgear_rtc)"

```python linenums="1" hl_lines="8"
# import required libraries
import uvicorn
from vidgear.gears.asyncio import WebGear_RTC

# various performance tweaks and enable live broadcasting
options = {
    "frame_size_reduction": 25,
    "enable_live_broadcast": True,
}

# initialize WebGear_RTC app
web = WebGear_RTC(source="foo.mp4", logging=True, **options)

# run this app on Uvicorn server at address http://0.0.0.0:8000/
uvicorn.run(web(), host="0.0.0.0", port=8000)

# close app safely
web.shutdown()
```

**And that's all, Now you can see output at [`http://localhost:8000/`](http://localhost:8000/) address on your local machine.**

&nbsp;


## Using WebGear_RTC with a Custom Source(OpenCV)

WebGear_RTC provides [`custom_stream`](params/#webgear_rtc-specific-attributes) attribute with its `options` parameter that allows you to easily define your own Custom Streaming Class with suitable source that you want to use to transform your frames before sending them onto the browser. 

Let's implement a bare-minimum example with a Custom Source using WebGear_RTC API and OpenCV:

??? new "New in v0.2.4" 
    This implementation was added in `v0.2.4`.

!!! success "Auto-Reconnection or Auto-Refresh works out-of-the-box with this implementation: Default Theme closes its own peer (`/close_connection`) when the page is hidden/reloaded, and reconnects with backoff when connection fails or the stream stalls, while WebGear_RTC also drops failed/closed peers by itself."

!!! danger "Make sure your Custom Streaming Class at-least implements `read()` and `stop()` methods as shown in following example, otherwise WebGear_RTC will throw ValueError!"

???+ tip "Using Vidgear's VideoCapture APIs instead of OpenCV"
    You can directly replace Custom Streaming Class(`Custom_Stream_Class` in following example) with any [VideoCapture APIs](../#a-videocapture-gears). These APIs implements `read()` and `stop()` methods by-default, so they're also supported out-of-the-box. 

    See this [example ➶](../../help/screengear_ex/#using-screengear-with-webgear_rtc) for more information.


```python linenums="1" hl_lines="6-54 58"
# import necessary libs
import uvicorn, cv2
from vidgear.gears.asyncio import WebGear_RTC

# create your own custom streaming class
class Custom_Stream_Class:
    """
    Custom Streaming using OpenCV
    """

    def __init__(self, source=0):

        # !!! define your own video source here !!!
        self.source = cv2.VideoCapture(source)

        # define running flag
        self.running = True

    def read(self):

        # don't forget this function!!!

        # check if source was initialized or not
        if self.source is None:
            return None
        # check if we're still running
        if self.running:
            # read frame from provided source
            (grabbed, frame) = self.source.read()
            # check if frame is available
            if grabbed:

                # do something with your OpenCV frame here

                # lets convert frame to gray for this example
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # return our gray frame
                return gray
            else:
                # signal we're not running now
                self.running = False
        # return None-type
        return None

    def stop(self):

        # don't forget this function!!!

        # flag that we're not running
        self.running = False
        # close stream
        if not self.source is None:
            self.source.release()

# assign your Custom Streaming Class with adequate source (for e.g. foo.mp4) 
# to `custom_stream` attribute in options parameter
options = {"custom_stream": Custom_Stream_Class(source="foo.mp4")}

# initialize WebGear_RTC app without any source
web = WebGear_RTC(logging=True, **options)

# run this app on Uvicorn server at address http://localhost:8000/
uvicorn.run(web(), host="localhost", port=8000)

# close app safely
web.shutdown()
```

**And that's all, Now you can see output at [`http://localhost:8000/`](http://localhost:8000/) address.**


&nbsp;


## Using WebGear_RTC with STUN/TURN Servers

On localhost and LAN, WebGear_RTC needs no ICE servers. But when peers are behind different NATs _(e.g. WebGear_RTC server behind a router, accessed over Internet)_, you can define STUN and TURN servers with [`ice_servers`](params/#webgear_rtc-specific-attributes) attribute, which are used both by WebGear_RTC's server-side peers and the browser _(Default Theme reads them from the homepage)_:

??? new "New in v0.3.7" 
    This example was added in `v0.3.7`.

!!! danger "ICE servers including TURN `username`/`credential` are sent to every browser visiting the homepage, so only use credentials meant for clients."

```python linenums="1" hl_lines="7-11"
# import required libraries
import uvicorn
from vidgear.gears.asyncio import WebGear_RTC

# define STUN & TURN servers, and force H.264 codec
options = {
    "ice_servers": [
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "turn:turn.example.org:3478", "username": "foo", "credential": "bar"},
    ],
    "codec": "H264",
}

# initialize WebGear_RTC app
web = WebGear_RTC(source="foo.mp4", logging=True, **options)

# run this app on Uvicorn server at address http://0.0.0.0:8000/
uvicorn.run(web(), host="0.0.0.0", port=8000)

# close app safely
web.shutdown()
```

&nbsp;


## Using WebGear_RTC with Native(Non-Browser) Clients

WebGear_RTC's signaling is plain HTTP+JSON, so any WebRTC client can connect to it, using following endpoints:

| Endpoint | Request _(JSON body)_ | Response |
|:---|:---|:---|
| `POST /offer` | `{"sdp": "...", "type": "offer"}`, with a `recvonly` _(or `sendrecv`)_ video transceiver | `200`: `{"sdp": "...", "type": "answer", "id": "<peer id>"}`, `400`: `{"error": "..."}` for invalid offers, `503`: `{"error": "..."}` when stream has ended |
| `POST /close_connection` | `{"id": "<peer id>"}` | `OK` if that peer was closed, otherwise `DISABLED` |

!!! warning "These endpoints are unauthenticated, like the homepage. Use a Starlette [authentication middleware](https://www.starlette.io/authentication/) or a reverse-proxy if your stream isn't public. A peer can only be closed by whoever knows its `id`."

For example, a Python client using [aiortc](https://aiortc.readthedocs.io/en/latest/) and [httpx](https://www.python-httpx.org/) is as follows:

??? new "New in v0.3.7" 
    The peer `id` and `400`/`503` responses were added in `v0.3.7`.

```python linenums="1"
# import required libraries
import asyncio
import httpx
from aiortc import RTCPeerConnection, RTCSessionDescription


async def main(url="http://localhost:8000"):
    pc = RTCPeerConnection()

    @pc.on("track")
    def on_track(track):
        async def show():
            while True:
                frame = await track.recv()  # av.VideoFrame
                print(frame.width, frame.height, frame.pts)

        asyncio.ensure_future(show())

    # create offer with a receive-only video transceiver
    pc.addTransceiver("video", direction="recvonly")
    await pc.setLocalDescription(await pc.createOffer())
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url + "/offer",
            json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
        )
        answer = response.raise_for_status().json()
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )
        # receive frames for 10 seconds
        await asyncio.sleep(10)
        # close our peer on server too
        await client.post(url + "/close_connection", json={"id": answer["id"]})
    await pc.close()


asyncio.run(main())
```

&nbsp;


## Using WebGear_RTC with Custom Mounting Points

With our highly extensible WebGear_RTC API, you can add your own mounting points, where additional files located, as follows:

```python linenums="1" hl_lines="18-20"
# import libs
import uvicorn
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from vidgear.gears.asyncio import WebGear_RTC

# various performance tweaks
options = {
    "frame_size_reduction": 25,
}

# initialize WebGear_RTC app
web = WebGear_RTC(
    source="foo.mp4", logging=True, **options
)  # enable source i.e. `test.mp4` and enable `logging` for debugging

# append new route i.e. mount another folder called `test` located at `/home/foo/.vidgear/test` directory
web.routes.append(
    Mount("/test", app=StaticFiles(directory="/home/foo/.vidgear/test"), name="test")
)

# run this app on Uvicorn server at address http://localhost:8000/
uvicorn.run(web(), host="localhost", port=8000)

# close app safely
web.shutdown()
```

Then you can use this folder in your HTML page, to host data-files. For example, if we have jQuery script `jquery-3.3.1.slim.min.js` in this folder and  want to integrate it, then, we can do something like this:

```html
<script src="{{ url_for('test', path='jquery-3.3.1.slim.min.js') }}"></script>
```

&nbsp;


## Using WebGear_RTC with Custom Webpage Routes

With Webgear_RTC's flexible API, you can even add your additional HTML Static webpages without any extra efforts.

Suppose we want to add a simple **`hello world` webpage** to our WebGear_RTC server. So let's create a bare-minimum `hello.html` file with HTML code as follows:

```html
<html>
   <header>
      <title>This is Hello world page</title>
   </header>
   <body>
      <h1>Hello World</h1>
      <p>how ya doing?</p>
   </body>
</html>
``` 
 
Then in our application code, we can integrate this webpage route, as follows:

```python linenums="1" hl_lines="11-12 26"
# import libs
import uvicorn, asyncio
from starlette.templating import Jinja2Templates
from starlette.routing import Route
from vidgear.gears.asyncio import WebGear_RTC

# Build out Jinja2 template render at `/home/foo/.vidgear/custom_template` path in which our `hello.html` file is located
template = Jinja2Templates(directory="/home/foo/.vidgear/custom_template")

# render and return our webpage template
async def hello_world(request):
    return template.TemplateResponse(request, "hello.html")


# add various performance tweaks as usual
options = {
    "frame_size_reduction": 25,
}

# initialize WebGear_RTC app with a valid source
web = WebGear_RTC(
    source="/home/foo/foo1.mp4", logging=True, **options
)  # enable source i.e. `test.mp4` and enable `logging` for debugging

# append new route to point our rendered webpage
web.routes.append(Route("/hello", endpoint=hello_world))

# run this app on Uvicorn server at address http://localhost:8000/
uvicorn.run(web(), host="localhost", port=8000)

# close app safely
web.shutdown()
```
**And that's all, Now you can see output at [`http://localhost:8000/hello`](http://localhost:8000/hello) address.**

&nbsp;

## Using WebGear_RTC with MiddleWares

WebGear_RTC also natively supports ASGI middleware classes with Starlette for implementing behavior that is applied across your entire ASGI application easily.

??? new "New in v0.2.2" 
    This example was added in `v0.2.2`.

!!! info "All supported middlewares can be found [here ➶](https://www.starlette.io/middleware/)"

For this example, let's use [`CORSMiddleware`](https://www.starlette.io/middleware/#corsmiddleware) for implementing appropriate [CORS headers](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS) to outgoing responses in our application in order to allow cross-origin requests from browsers, as follows:

!!! danger "The default parameters used by the CORSMiddleware implementation are restrictive by default, so you'll need to explicitly enable particular origins, methods, or headers, in order for browsers to be permitted to use them in a Cross-Domain context."

!!! tip "Starlette provides several arguments for enabling origins, methods, or headers for CORSMiddleware API. More information can be found [here ➶](https://www.starlette.io/middleware/#corsmiddleware)"

```python linenums="1" hl_lines="18-26"
# import libs
import uvicorn, asyncio
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from vidgear.gears.asyncio import WebGear_RTC

# add various performance tweaks as usual
options = {
    "frame_size_reduction": 25,
}

# initialize WebGear_RTC app with a valid source
web = WebGear_RTC(
    source="/home/foo/foo1.mp4", logging=True, **options
)  # enable source i.e. `test.mp4` and enable `logging` for debugging

# define and assign suitable cors middlewares
web.middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

# run this app on Uvicorn server at address http://localhost:8000/
uvicorn.run(web(), host="localhost", port=8000)

# close app safely
web.shutdown()
```

**And that's all, Now you can see output at [`http://localhost:8000`](http://localhost:8000) address.**

&nbsp;

## Rules for Altering WebGear_RTC Files and Folders

WebGear_RTC gives us complete freedom of altering data files generated in [**Auto-Generation Process**](./#auto-generation-process), But you've to  keep the following rules in mind:

### Rules for Altering Data Files
  
- [x] You allowed to alter/change code in all existing [default downloaded files](./#auto-generation-process) at your convenience without any restrictions.
- [x] You allowed to delete/rename all existing data files, except remember **NOT** to delete/rename three critical data-files (i.e `index.html`, `404.html` & `500.html`) present in `templates` folder, and `custom.js` present in `static/js` folder inside the `webgear_rtc` directory at the [default location](./#default-location), otherwise, it will trigger [Auto-generation process](./#auto-generation-process), and it will overwrite the existing files with Server ones.
- [x] You're allowed to add your own additional `.html`, `.css`, `.js`, etc. files in the respective folders at the [**default location**](./#default-location) and [custom mounted Data folders](#using-webgear_rtc-with-custom-mounting-points).

### Rules for Altering Data Folders 
  
- [x] You're allowed to add/mount any number of additional folder as shown in [this example above](#using-webgear_rtc-with-custom-mounting-points).
- [x] You're allowed to delete/rename existing folders at the [**default location**](./#default-location) except remember **NOT** to delete/rename `templates` folder in the `webgear_rtc` directory where critical data-files (i.e `index.html`, `404.html` & `500.html`) are located, otherwise, it will trigger [Auto-generation process](./#auto-generation-process).

&nbsp;

## Bonus Examples

!!! example "Checkout more advanced WebGear_RTC examples with unusual configuration [here ➶](../../help/webgear_rtc_ex/)"

&nbsp;