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

# WebGear API Advanced Usage:

!!! note "This is a continuation of the [WebGear doc ➶](../overview/#webgear-api). Thereby, It's advised to first get familiarize with this API, and its [requirements](../usage/#requirements)."

!!! example "After going through following Usage Examples, Checkout more bonus examples [here ➶](../../../help/webgear_ex/)"


&thinsp;


### Using WebGear with Variable Colorspace

WebGear by default only supports "BGR" colorspace frames as input, but you can use [`jpeg_compression_colorspace`](../params/#webgear-specific-attributes) string attribute through its options dictionary parameter to specify incoming frames colorspace. 

Let's implement a bare-minimum example using WebGear, where we will be sending [**GRAY**](https://en.wikipedia.org/wiki/Grayscale) frames to client browser:

??? new "New in v0.2.2" 
    This example was added in `v0.2.2`.

!!! example "This example works in conjunction with [Source ColorSpace manipulation for VideoCapture Gears ➶](../../../bonus/colorspace_manipulation/#source-colorspace-manipulation)"

!!! info "Supported `jpeg_compression_colorspace` colorspace values are `RGB`, `BGR`, `RGBX`, `BGRX`, `XBGR`, `XRGB`, `GRAY`, `RGBA`, `BGRA`, `ABGR`, `ARGB`, `CMYK`. More information can be found [here ➶](https://gitlab.com/jfolz/simplejpeg)"

```python hl_lines="8" 
# import required libraries
import uvicorn
from vidgear.gears.asyncio import WebGear

# various performance tweaks and enable grayscale input
options = {
    "frame_size_reduction": 25,
    "jpeg_compression_colorspace": "GRAY",  # set grayscale
    "jpeg_compression_quality": 90,
    "jpeg_compression_fastdct": True,
    "jpeg_compression_fastupsample": True,
}

# initialize WebGear app and change its colorspace to grayscale
web = WebGear(
    source="foo.mp4", colorspace="COLOR_BGR2GRAY", logging=True, **options
)

# run this app on Uvicorn server at address http://0.0.0.0:8000/
uvicorn.run(web(), host="0.0.0.0", port=8000)

# close app safely
web.shutdown()
```

**And that's all, Now you can see output at [`http://localhost:8000/`](http://localhost:8000/) address on your local machine.**

&thinsp;

## Using WebGear with a Custom Source(OpenCV)

??? new "New in v0.2.1" 
    This example was added in `v0.2.1`.

WebGear allows you to easily define your own custom Source that you want to use to transform your frames before sending them onto the browser. 

!!! warning "JPEG Frame-Compression and all of its [performance enhancing attributes](../usage/#performance-enhancements) are disabled with a Custom Source!"

Let's implement a bare-minimum example with a Custom Source using WebGear API and OpenCV:


```python hl_lines="10-34 38"
# import necessary libs
import uvicorn, asyncio, cv2
from vidgear.gears.asyncio import WebGear
from vidgear.gears.asyncio.helper import reducer

# initialize WebGear app without any source
web = WebGear(logging=True)

# create your own custom frame producer
async def my_frame_producer():

    # !!! define your own video source here !!!
    # Open any video stream such as live webcam 
    # video stream on first index(i.e. 0) device
    stream = cv2.VideoCapture(0)
    # loop over frames
    while True:
        # read frame from provided source
        (grabbed, frame) = stream.read()
        # break if NoneType
        if not grabbed:
            break

        # do something with your OpenCV frame here

        # reducer frames size if you want more performance otherwise comment this line
        frame = await reducer(frame, percentage=30, interpolation=cv2.INTER_AREA)  # reduce frame by 30%
        # handle JPEG encoding
        encodedImage = cv2.imencode(".jpg", frame)[1].tobytes()
        # yield frame in byte format
        yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" + encodedImage + b"\r\n")
        await asyncio.sleep(0)
    # close stream
    stream.release()


# add your custom frame producer to config
web.config["generator"] = my_frame_producer

# run this app on Uvicorn server at address http://localhost:8000/
uvicorn.run(web(), host="localhost", port=8000)

# close app safely
web.shutdown()
```

**And that's all, Now you can see output at [`http://localhost:8000/`](http://localhost:8000/) address.**

&nbsp;


## Using WebGear with Custom Mounting Points

With our highly extensible WebGear API, you can add your own mounting points, where additional files located, as follows:

```python hl_lines="21-23"
# import libs
import uvicorn
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from vidgear.gears.asyncio import WebGear

# various performance tweaks
options = {
    "frame_size_reduction": 40,
    "jpeg_compression_quality": 80,
    "jpeg_compression_fastdct": True,
    "jpeg_compression_fastupsample": False,
}

# initialize WebGear app
web = WebGear(
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


## Using WebGear with Custom Webpage Routes

With Webgear's flexible API, you can even add your additional HTML Static webpages without any extra efforts.

Suppose we want to add a simple **`hello world` webpage** to our WebGear server. So let's create a bare-minimum `hello.html` file with HTML code as follows:

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

```python hl_lines="11-14 31"
# import libs
import uvicorn, asyncio
from starlette.templating import Jinja2Templates
from starlette.routing import Route
from vidgear.gears.asyncio import WebGear

# Build out Jinja2 template render at `/home/foo/.vidgear/custom_template` path in which our `hello.html` file is located
template = Jinja2Templates(directory="/home/foo/.vidgear/custom_template")

# render and return our webpage template
async def hello_world(request):
    page = "hello.html"
    context = {"request": request}
    return template.TemplateResponse(page, context)


# add various performance tweaks as usual
options = {
    "frame_size_reduction": 40,
    "jpeg_compression_quality": 80,
    "jpeg_compression_fastdct": True,
    "jpeg_compression_fastupsample": False,
}

# initialize WebGear app with a valid source
web = WebGear(
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

## Using WebGear with MiddleWares

WebGear natively supports ASGI middleware classes with Starlette for implementing behavior that is applied across your entire ASGI application easily.

??? new "New in v0.2.2" 
    This example was added in `v0.2.2`.

!!! info "All supported middlewares can be found [here ➶](https://www.starlette.io/middleware/)"

For this example, let's use [`CORSMiddleware`](https://www.starlette.io/middleware/#corsmiddleware) for implementing appropriate [CORS headers](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS) to outgoing responses in our application in order to allow cross-origin requests from browsers, as follows:

!!! danger "The default parameters used by the CORSMiddleware implementation are restrictive by default, so you'll need to explicitly enable particular origins, methods, or headers, in order for browsers to be permitted to use them in a Cross-Domain context."

!!! tip "Starlette provides several arguments for enabling origins, methods, or headers for CORSMiddleware API. More information can be found [here ➶](https://www.starlette.io/middleware/#corsmiddleware)"

```python hl_lines="21-29"
# import libs
import uvicorn, asyncio
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from vidgear.gears.asyncio import WebGear

# add various performance tweaks as usual
options = {
    "frame_size_reduction": 40,
    "jpeg_compression_quality": 80,
    "jpeg_compression_fastdct": True,
    "jpeg_compression_fastupsample": False,
}

# initialize WebGear app with a valid source
web = WebGear(
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

## Rules for Altering WebGear Files and Folders

WebGear gives us complete freedom of altering data files generated in [**Auto-Generation Process**](../overview/#auto-generation-process), But you've to  keep the following rules in mind:

### Rules for Altering Data Files
  
- [x] You allowed to alter/change code in all existing [default downloaded files](../overview/#auto-generation-process) at your convenience without any restrictions.
- [x] You allowed to delete/rename all existing data files, except remember **NOT** to delete/rename three critical data-files (i.e `index.html`, `404.html` & `500.html`) present in `templates` folder inside the `webgear` directory at the [default location](../overview/#default-location), otherwise, it will trigger [Auto-generation process](../overview/#auto-generation-process), and it will overwrite the existing files with Server ones.
- [x] You're allowed to add your own additional `.html`, `.css`, `.js`, etc. files in the respective folders at the [**default location**](../overview/#default-location) and [custom mounted Data folders](#using-webgear-with-custom-mounting-points).

### Rules for Altering Data Folders 
  
- [x] You're allowed to add/mount any number of additional folder as shown in [this example above](#using-webgear-with-custom-mounting-points).
- [x] You're allowed to delete/rename existing folders at the [**default location**](../overview/#default-location) except remember **NOT** to delete/rename `templates` folder in the `webgear` directory where critical data-files (i.e `index.html`, `404.html` & `500.html`) are located, otherwise, it will trigger [Auto-generation process](../overview/#auto-generation-process).

&nbsp;

## Bonus Examples

!!! example "Checkout more advanced WebGear examples with unusual configuration [here ➶](../../../help/webgear_ex/)"

&nbsp;
