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



## Performance Enhancements :fire:

Previously, on running [bare-minimum usage example](../usage/#bare-minimum-usage), you will notice a significant performance throttling, lag and frame drop in output Stream on the browser. To cope with this throttling problem, WebGear provides certain performance enhancing attributes for its [`option`](../params/#options) dictionary parameter.


??? tip "Performance Enhancing Attributes"

    * **`frame_size_reduction`**: _(int/float)_ _This attribute controls the size reduction(in percentage) of the frame to be streamed on Server._ Its value has the most significant effect on WebGear performance: More its value, smaller will be frame size and faster will be live streaming. The value defaults to `20`, and must be no higher than `90` _(fastest, max compression, Barely Visible frame-size)_ and no lower than `0` _(slowest, no compression, Original frame-size)_. Its recommended value is between `40~60`. Its usage is as follows:

        ```python
        options={"frame_size_reduction": 50} #frame-size will be reduced by 50%
        ```
     
    * **Various Encoding Parameters:**

        In WebGear API, the input video frames are first encoded into [**Motion JPEG (M-JPEG or MJPEG**)](https://en.docpedia.org/doc/Motion_JPEG) compression format, in which each video frame or interlaced field of a digital video sequence is compressed separately as a JPEG image, before sending onto a server. Therefore, WebGear API provides various attributes to have full control over JPEG encoding performance and quality, which are as follows:

        *  **`frame_jpeg_quality`**: _(int)_ It controls the JPEG encoder quality. Its value varies from `0` to `100` (the higher is the better quality but performance will be lower). Its default value is `95`. Its usage is as follows:

            ```python
            options={"frame_jpeg_quality": 80} #JPEG will be encoded at 80% quality.
            ```

        * **`frame_jpeg_optimize`**: _(bool)_ It enables various JPEG compression optimizations such as Chroma sub-sampling, Quantization table, etc. These optimizations based on JPEG libs which are used while compiling OpenCV binaries, and recent versions of OpenCV uses [**TurboJPEG library**](https://libjpeg-turbo.org/), which is highly recommended for performance. Its default value is `False`. Its usage is as follows:

            ```python
            options={"frame_jpeg_optimize": True} #JPEG optimizations are enabled.
            ```

        * **`frame_jpeg_progressive`**: _(bool)_ It enables **Progressive** JPEG encoding instead of the **Baseline**.   Progressive Mode, displays an image in such a way that it shows a blurry/low-quality photo in its entirety, and then becomes clearer as the image downloads, whereas in Baseline Mode, an image created using the JPEG compression algorithm that will start to display the image as the data is made available, line by line. Progressive Mode, can drastically improve the performance in WebGear but at the expense of additional CPU load, thereby suitable for powerful systems only. Its default value is `False` meaning baseline mode is in-use. Its usage is as follows:

            ```python
            options={"frame_jpeg_progressive": True} #Progressive JPEG encoding enabled.
            ```

### Bare-Minimum Usage with Performance Enhancements

Let's re-implement our previous Bare-Minimum usage example with these Performance Enhancing Attributes:

#### Running Programmatically

You can access and run WebGear VideoStreamer Server programmatically in your python script in just a few lines of code, as follows:

```python
# import required libraries
import uvicorn
from vidgear.gears.asyncio import WebGear

#various performance tweaks
options={"frame_size_reduction": 40, "frame_jpeg_quality": 80, "frame_jpeg_optimize": True, "frame_jpeg_progressive": False}

#initialize WebGear app  
web=WebGear(source="foo.mp4", logging=True, **options)

#run this app on Uvicorn server at address http://0.0.0.0:8000/
uvicorn.run(web(), host='0.0.0.0', port=8000)

#close app safely
web.shutdown()
```

which can be accessed on any browser on the network at http://0.0.0.0:8000/.


#### Running from Terminal

Now lets, run this same example directly through the terminal commandline:

!!! warning "If you're using `--options/-op` flag, then kindly wrap your dictionary value in single `''` quotes."


```sh
python3 -m vidgear.gears.asyncio --source test.avi --logging True --options '{"frame_size_reduction": 50, "frame_jpeg_quality": 80, "frame_jpeg_optimize": True, "frame_jpeg_progressive": False}'
```

which can also be accessed on any browser on the network at http://0.0.0.0:8000/.

&nbsp;


## Using WebGear with Custom Mounting Points

With our highly extensible WebGear API, you can add your own mounting points, where additional files located, as follows:

```python
#import libs
import uvicorn
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from vidgear.gears.asyncio import WebGear

#various performance tweaks
options={"frame_size_reduction": 40, "frame_jpeg_quality": 80, "frame_jpeg_optimize": True, "frame_jpeg_progressive": False}

#initialize WebGear app  
web=WebGear(source="foo.mp4", logging=True, **options) #enable source i.e. `test.mp4` and enable `logging` for debugging 

#append new route i.e. mount another folder called `test` located at `/home/foo/.vidgear/test` directory
web.routes.append(Mount('/test', app=StaticFiles(directory='/home/foo/.vidgear/test'), name="test")) 

#run this app on Uvicorn server at address http://0.0.0.0:8000/
uvicorn.run(web(), host='0.0.0.0', port=8000)

#close app safely
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
#import libs
import uvicorn, asyncio
from starlette.templating import Jinja2Templates
from starlette.routing import Route
from vidgear.gears.asyncio import WebGear

#Build out Jinja2 template render at `/home/foo/.vidgear/custom_template` path in which our `hello.html` file is located
template=Jinja2Templates(directory='/home/foo/.vidgear/custom_template')

#render and return our webpage template
async def hello_world(request):
    page="hello.html"
    context={"request": request}
    return template.TemplateResponse(page, context)

#add various performance tweaks as usual
options={"frame_size_reduction": 40, "frame_jpeg_quality": 80, "frame_jpeg_optimize": True, "frame_jpeg_progressive": False}

#initialize WebGear app with a valid source
web=WebGear(source="/home/foo/foo1.mp4", logging=True, **options) #enable source i.e. `test.mp4` and enable `logging` for debugging 

#append new route to point our rendered webpage 
web.routes.append(Route('/hello', endpoint=hello_world)) 

#run this app on Uvicorn server at address http://0.0.0.0:8000/
uvicorn.run(web(), host='0.0.0.0', port=8000)

#close app safely
web.shutdown()
```
**And that's all, Now you can see output at [`http://0.0.0.0:8000/hello`](http://0.0.0.0:8000/hello) address.**

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
#import libs
import uvicorn
from vidgear.gears.asyncio import WebGear

#various webgear performance and Rasbperry camera tweaks  
options={"frame_size_reduction": 40, "frame_jpeg_quality": 80, "frame_jpeg_optimize": True, "frame_jpeg_progressive": False, "hflip": True, "exposure_mode": "auto", "iso": 800, "exposure_compensation": 15, "awb_mode": "horizon", "sensor_mode": 0}

#initialize WebGear app  
web=WebGear(enablePiCamera=True, resolution=(640, 480), framerate=60, logging=True, **options)

#run this app on Uvicorn server at address http://0.0.0.0:8000/
uvicorn.run(web(), host='0.0.0.0', port=8000)

#close app safely
web.shutdown()
```

&nbsp;

### Using WebGear with real-time Video Stabilization enabled
 
Here's an example of using WebGear API with real-time Video Stabilization enabled:

```python
#import libs
import uvicorn
from vidgear.gears.asyncio import WebGear

#various webgear performance tweaks  
options={"frame_size_reduction": 40, "frame_jpeg_quality": 80, "frame_jpeg_optimize": True, "frame_jpeg_progressive": False}

#initialize WebGear app  with a raw source and enable video stabilization(`stabilize=True`)
web=WebGear(source="foo.mp4", stabilize=True, logging=True, **options)

#run this app on Uvicorn server at address http://0.0.0.0:8000/
uvicorn.run(web(), host='0.0.0.0', port=8000)

#close app safely
web.shutdown()
```

&nbsp;
 