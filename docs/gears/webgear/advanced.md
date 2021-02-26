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

# WebGear API Advanced Usage:

!!! note "This is a continuation of the [WebGear doc ➶](../overview/#webgear-api). Thereby, It's advised to first get familiarize with this API, and its [requirements](../usage/#requirements)."


&nbsp;

## Using WebGear with a Custom Source(OpenCV)

WebGear allows you to easily define your own custom Source that you want to use to manipulate your frames before sending them onto the browser. Let's implement a bare-minimum example with a Custom Source using WebGear API and OpenCV:


```python
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
        frame = await reducer(frame, percentage=30)  # reduce frame by 30%
        # handle JPEG encoding
        encodedImage = cv2.imencode(".jpg", frame)[1].tobytes()
        # yield frame in byte format
        yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" + encodedImage + b"\r\n")
        await asyncio.sleep(0.00001)


# add your custom frame producer to config
web.config["generator"] = my_frame_producer()

# run this app on Uvicorn server at address http://localhost:8000/
uvicorn.run(web(), host="localhost", port=8000)

# close app safely
web.shutdown()
```

**And that's all, Now you can see output at [`http://localhost:8000/`](http://localhost:8000/) address.**

&nbsp;


## Using WebGear with Custom Mounting Points

With our highly extensible WebGear API, you can add your own mounting points, where additional files located, as follows:

```python
# import libs
import uvicorn
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from vidgear.gears.asyncio import WebGear

# various performance tweaks
options = {
    "frame_size_reduction": 40,
    "frame_jpeg_quality": 80,
    "frame_jpeg_optimize": True,
    "frame_jpeg_progressive": False,
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

```python
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
    "frame_jpeg_quality": 80,
    "frame_jpeg_optimize": True,
    "frame_jpeg_progressive": False,
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

## Rules for Altering WebGear Files and Folders

WebGear gives us complete freedom of altering data files generated in [**Auto-Generation Process**](../overview/#auto-generation-process), But you've to  keep the following rules in mind:

### Rules for Altering Data Files
  
- [x] You allowed to alter/change code in all existing [default downloaded files](../overview/#auto-generation-process) at your convenience without any restrictions.
- [x] You allowed to delete/rename all existing data files, except remember **NOT** to delete/rename three critical data-files i.e `index.html`, `404.html` & `500.html` present in `templates` folder at the [default location](../overview/#default-location), otherwise, it will trigger [Auto-generation process](../overview/#auto-generation-process), and it will overwrite the existing files with Server ones.
- [x] You're allowed to add your own additional `.html`, `.css`, `.js`, etc. files in the respective folders at the [**default location**](../overview/#default-location) and [custom mounted Data folders](#using-webgear-with-custom-mounting-points).

### Rules for Altering Data Folders 
  
- [x] You're allowed to add/mount any number of additional folder as shown in [this example above](#using-webgear-with-custom-mounting-points).
- [x] You're allowed to delete/rename existing folders at the [**default location**](../overview/#default-location) except remember **NOT** to delete/rename `templates` folder where critical data-files i.e `index.html`, `404.html` & `500.html` are located, otherwise, it will trigger [Auto-generation process](../overview/#auto-generation-process).

&nbsp;

## Bonus Usage Examples

Because of WebGear API's flexible internal wapper around [VideoGear](../../videogear/overview/), it can easily access any parameter of [CamGear](#camgear) and [PiGear](#pigear) videocapture APIs.

!!! info "Following usage examples are just an idea of what can be done with WebGear API, you can try various [VideoGear](../../videogear/params/), [CamGear](../../camgear/params/) and [PiGear](../../pigear/params/) parameters directly in WebGear API in the similar manner."

### Using WebGear with Pi Camera Module
 
Here's a bare-minimum example of using WebGear API with the Raspberry Pi camera module while tweaking its various properties in just one-liner:

```python
# import libs
import uvicorn
from vidgear.gears.asyncio import WebGear

# various webgear performance and Rasbperry camera tweaks
options = {
    "frame_size_reduction": 40,
    "frame_jpeg_quality": 80,
    "frame_jpeg_optimize": True,
    "frame_jpeg_progressive": False,
    "hflip": True,
    "exposure_mode": "auto",
    "iso": 800,
    "exposure_compensation": 15,
    "awb_mode": "horizon",
    "sensor_mode": 0,
}

# initialize WebGear app
web = WebGear(
    enablePiCamera=True, resolution=(640, 480), framerate=60, logging=True, **options
)

# run this app on Uvicorn server at address http://localhost:8000/
uvicorn.run(web(), host="localhost", port=8000)

# close app safely
web.shutdown()
```

&nbsp;

### Using WebGear with real-time Video Stabilization enabled
 
Here's an example of using WebGear API with real-time Video Stabilization enabled:

```python
# import libs
import uvicorn
from vidgear.gears.asyncio import WebGear

# various webgear performance tweaks
options = {
    "frame_size_reduction": 40,
    "frame_jpeg_quality": 80,
    "frame_jpeg_optimize": True,
    "frame_jpeg_progressive": False,
}

# initialize WebGear app  with a raw source and enable video stabilization(`stabilize=True`)
web = WebGear(source="foo.mp4", stabilize=True, logging=True, **options)

# run this app on Uvicorn server at address http://localhost:8000/
uvicorn.run(web(), host="localhost", port=8000)

# close app safely
web.shutdown()
```

&nbsp;
 