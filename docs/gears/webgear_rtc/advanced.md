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

!!! note "This is a continuation of the [WebGear_RTC doc ➶](../overview/#webgear_rtc-api). Thereby, It's advised to first get familiarize with this API, and its [requirements](../usage/#requirements)."


&thinsp;


## Using WebGear_RTC as Real-time Broadcaster 

WebGear_RTC by default only supports one-to-one peer connection with a single consumer or client. But you can use [`enable_live_broadcast`](../params/#webgear_rtc-specific-attributes) boolean attribute through its options dictionary parameter to easily enable live broadcast/stream to multiple peer consumers/clients at the same time.

Let's implement a bare-minimum example using WebGear_RTC as Real-time Broadcaster:

!!! info "[`enable_infinite_frames`](../params/#webgear_rtc-specific-attributes) is enforced by default with this(`enable_live_broadcast`) attribute."

!!! tip "For accessing WebGear_RTC on different Client Devices on the network, we use `"0.0.0.0"` as host value instead of `"localhost"` on Host Machine. More information can be found [here ➶](../../../help/webgear_rtc_faqs/#is-it-possible-to-stream-on-a-different-device-on-the-network-with-webgear_rtc)"

```python
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

WebGear_RTC allows you to easily define your own Custom Media Server with a custom source that you want to use to manipulate your frames before sending them onto the browser. 

Let's implement a bare-minimum example with a Custom Source using WebGear_RTC API and OpenCV:


!!! danger "Make sure your Custom Media Server Class is inherited from aiortc's [VideoStreamTrack](https://github.com/aiortc/aiortc/blob/a270cd887fba4ce9ccb680d267d7d0a897de3d75/src/aiortc/mediastreams.py#L109) only and at-least implements `recv()` and `terminate()` methods as shown in following example, otherwise WebGear_RTC will throw ValueError!"


```python
# import necessary libs
import uvicorn, asyncio, cv2
from av import VideoFrame
from aiortc import VideoStreamTrack
from vidgear.gears.asyncio import WebGear_RTC
from vidgear.gears.asyncio.helper import reducer

# initialize WebGear_RTC app without any source
web = WebGear_RTC(logging=True)

# create your own Bare-Minimum Custom Media Server
class Custom_RTCServer(VideoStreamTrack):
    """
    Custom Media Server using OpenCV, an inherit-class
    to aiortc's VideoStreamTrack.
    """

    def __init__(self, source=None):

        # don't forget this line!
        super().__init__()

        # initialize global params
        self.stream = cv2.VideoCapture(source)

    async def recv(self):
        """
        A coroutine function that yields `av.frame.Frame`.
        """
        # don't forget this function!!!

        # get next timestamp
        pts, time_base = await self.next_timestamp()

        # read video frame
        (grabbed, frame) = self.stream.read()

        # if NoneType
        if not grabbed:
            return None

        # reducer frames size if you want more performance otherwise comment this line
        frame = await reducer(frame, percentage=30)  # reduce frame by 30%

        # contruct `av.frame.Frame` from `numpy.nd.array`
        av_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        av_frame.pts = pts
        av_frame.time_base = time_base

        # return `av.frame.Frame`
        return av_frame

    def terminate(self):
        """
        Gracefully terminates VideoGear stream
        """
        # don't forget this function!!!

        # terminate
        if not (self.stream is None):
            self.stream.release()
            self.stream = None


# assign your custom media server to config with adequate source (for e.g. foo.mp4)
web.config["server"] = Custom_RTCServer(source="foo.mp4")

# run this app on Uvicorn server at address http://localhost:8000/
uvicorn.run(web(), host="localhost", port=8000)

# close app safely
web.shutdown()

```

**And that's all, Now you can see output at [`http://localhost:8000/`](http://localhost:8000/) address.**

&nbsp;


## Using WebGear_RTC with Custom Mounting Points

With our highly extensible WebGear_RTC API, you can add your own mounting points, where additional files located, as follows:

```python
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

```python
# import libs
import uvicorn, asyncio
from starlette.templating import Jinja2Templates
from starlette.routing import Route
from vidgear.gears.asyncio import WebGear_RTC

# Build out Jinja2 template render at `/home/foo/.vidgear/custom_template` path in which our `hello.html` file is located
template = Jinja2Templates(directory="/home/foo/.vidgear/custom_template")

# render and return our webpage template
async def hello_world(request):
    page = "hello.html"
    context = {"request": request}
    return template.TemplateResponse(page, context)


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

!!! new "New in v0.2.2" 
    This example was added in `v0.2.2`.

!!! info "All supported middlewares can be [here ➶](https://www.starlette.io/middleware/)"

For this example, let's use [`CORSMiddleware`](https://www.starlette.io/middleware/#corsmiddleware) for implementing appropriate [CORS headers](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS) to outgoing responses in our application in order to allow cross-origin requests from browsers, as follows:

!!! danger "The default parameters used by the CORSMiddleware implementation are restrictive by default, so you'll need to explicitly enable particular origins, methods, or headers, in order for browsers to be permitted to use them in a Cross-Domain context."

!!! tip "Starlette provides several arguments for enabling origins, methods, or headers for CORSMiddleware API. More information can be found [here ➶](https://www.starlette.io/middleware/#corsmiddleware)"

```python
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

WebGear_RTC gives us complete freedom of altering data files generated in [**Auto-Generation Process**](../overview/#auto-generation-process), But you've to  keep the following rules in mind:

### Rules for Altering Data Files
  
- [x] You allowed to alter/change code in all existing [default downloaded files](../overview/#auto-generation-process) at your convenience without any restrictions.
- [x] You allowed to delete/rename all existing data files, except remember **NOT** to delete/rename three critical data-files (i.e `index.html`, `404.html` & `500.html`) present in `templates` folder inside the `webgear_rtc` directory at the [default location](../overview/#default-location), otherwise, it will trigger [Auto-generation process](../overview/#auto-generation-process), and it will overwrite the existing files with Server ones.
- [x] You're allowed to add your own additional `.html`, `.css`, `.js`, etc. files in the respective folders at the [**default location**](../overview/#default-location) and [custom mounted Data folders](#using-webgear_rtc-with-custom-mounting-points).

### Rules for Altering Data Folders 
  
- [x] You're allowed to add/mount any number of additional folder as shown in [this example above](#using-webgear_rtc-with-custom-mounting-points).
- [x] You're allowed to delete/rename existing folders at the [**default location**](../overview/#default-location) except remember **NOT** to delete/rename `templates` folder in the `webgear_rtc` directory where critical data-files (i.e `index.html`, `404.html` & `500.html`) are located, otherwise, it will trigger [Auto-generation process](../overview/#auto-generation-process).

&nbsp;

## Bonus Usage Examples

Because of WebGear_RTC API's flexible internal wapper around [VideoGear](../../videogear/overview/), it can easily access any parameter of [CamGear](#camgear) and [PiGear](#pigear) videocapture APIs.

!!! info "Following usage examples are just an idea of what can be done with WebGear_RTC API, you can try various [VideoGear](../../videogear/params/), [CamGear](../../camgear/params/) and [PiGear](../../pigear/params/) parameters directly in WebGear_RTC API in the similar manner."

### Using WebGear_RTC with Pi Camera Module
 
Here's a bare-minimum example of using WebGear_RTC API with the Raspberry Pi camera module while tweaking its various properties in just one-liner:

```python
# import libs
import uvicorn
from vidgear.gears.asyncio import WebGear_RTC

# various webgear_rtc performance and Raspberry Pi camera tweaks
options = {
    "frame_size_reduction": 25,
    "hflip": True,
    "exposure_mode": "auto",
    "iso": 800,
    "exposure_compensation": 15,
    "awb_mode": "horizon",
    "sensor_mode": 0,
}

# initialize WebGear_RTC app
web = WebGear_RTC(
    enablePiCamera=True, resolution=(640, 480), framerate=60, logging=True, **options
)

# run this app on Uvicorn server at address http://localhost:8000/
uvicorn.run(web(), host="localhost", port=8000)

# close app safely
web.shutdown()
```

&nbsp;

### Using WebGear_RTC with real-time Video Stabilization enabled
 
Here's an example of using WebGear_RTC API with real-time Video Stabilization enabled:

```python
# import libs
import uvicorn
from vidgear.gears.asyncio import WebGear_RTC

# various webgear_rtc performance tweaks
options = {
    "frame_size_reduction": 25,
}

# initialize WebGear_RTC app  with a raw source and enable video stabilization(`stabilize=True`)
web = WebGear_RTC(source="foo.mp4", stabilize=True, logging=True, **options)

# run this app on Uvicorn server at address http://localhost:8000/
uvicorn.run(web(), host="localhost", port=8000)

# close app safely
web.shutdown()
```

&nbsp;
 