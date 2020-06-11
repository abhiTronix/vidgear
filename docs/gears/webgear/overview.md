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

# WebGear API 


## Overview

<p align="center">
  <img src="/assets/gifs/webgear.gif" alt="WebGear in action!" width=120%/>
  <br>
  <sub><i>WebGear Video Server at <a href="http://0.0.0.0:8000/" title="default address">http://0.0.0.0:8000/</a> address.</i></sub>
</p>

WebGear is a powerful [ASGI](https://asgi.readthedocs.io/en/latest/) Video-streamer API, that is built upon [Starlette](https://www.starlette.io/) - a lightweight ASGI framework/toolkit, which is ideal for building high-performance asyncio services.

WebGear API provides a highly extensible and flexible asyncio wrapper around Starlette ASGI application, and provides easy access to its complete framework. Thereby, WebGear API can flexibly interact with the Starlette's ecosystem of shared middleware and mountable applications, and its various [Response classes](https://www.starlette.io/responses/), [Routing tables](https://www.starlette.io/routing/), [Static Files](https://www.starlette.io/staticfiles/), [Templating engine(with Jinja2)](https://www.starlette.io/templates/), etc. 

In layman's terms, WebGear can acts as powerful **Video Streaming Server** that transfers live video-frames to any web browser on a network. It addition to this, WebGear API also provides a special internal wrapper around [VideoGear API](/gears/videogear/overview/), which itself provides internal access to both [CamGear](/gears/camgear/overview/) and [PiGear](/gears/pigear/overview/) APIs thereby granting it exclusive power for streaming frames incoming from any device/source, such as streaming [Stabilization enabled Video](/gears/webgear/advanced/#using-webgear-with-real-time-video-stabilization-enabled) in real-time.


&nbsp; 


## Data-Files Auto-Generation WorkFlow

On initializing WebGear API, it automatically checks for three critical data-files i.e `index.html`, `404.html` & `500.html` inside `templates` folder at the [*default location*](#default-location), which give rise to possible scenario:

* **If data-files found:** it will proceed normally for instantiating the Starlette application.
* **If data-files not found:** it will trigger the [**Auto-Generation process**](#auto-generation-process)

### Default Location

* A _default location_ is the path of the directory where data files/folders are downloaded/generated/saved.
* By default, the `.vidgear` the folder at the home directory of your machine _(for e.g `/home/foo/.vidgear` on Linux)_ serves as the _default location_.
* But you can also use WebGear's [`custom_data_location`](/gears/webgear/params/#webgear-specific-attributes) dictionary attribute to change/alter *default location* path to somewhere else.

	!!! tip
			You can set [`logging=True`](/gears/webgear/params/#logging) during initialization, for easily identifying the selected _default location_, which will be something like this _(on a Linux machine)_:

		  ```sh
		  WebGear :: DEBUG :: `/home/foo/.vidgear` is the default location for saving WebGear data-files.
		  ```

### Auto-Generation process

* On triggering this process, WebGear API creates `templates` and `static` folders along with `js`, `css`, `img` sub-folders at the assigned [*default location*](#default-location).
* Thereby at this [*default location*](#default-location), the necessary default data files will be downloaded from a dedicated [**Github Server**](https://github.com/abhiTronix/webgear_data) inside respective folders in the following order:

	```sh
		.vidgear
		├── static
		│   ├── css
		│   │   ├── bootstrap.min.css
		│   │   └── cover.css
		│   ├── img
		│   │   └── favicon-32x32.png
		│   └── js
		│       ├── bootstrap.min.js
		│       ├── jquery-3.4.1.slim.min.js
		│       └── popper.min.js
		└── templates
		    ├── 404.html
		    ├── 500.html
		    ├── base.html
		    └── index.html
		5 directories, 10 files
	```

* Finally these downloaded files thereby are verified for errors and API proceeds for instantiating the Starlette application normally.


&nbsp;

!!! tip "Important Tips"

		* You can also force trigger the Auto-generation process to overwrite existing data-files using [`overwrite_default_files`](/gears/webgear/params/#webgear-specific-attributes) dictionary attribute. Remember, only downloaded default data files(given above) will be overwritten in this process, and any other file/folder will NOT be affected.

		* It is advised to enable logging(`logging=True`) on the first run for easily identifying any runtime errors

&nbsp; 

## Importing

You can import WebGear API in your program as follows:

```python
from vidgear.gears import WebGear
```

&nbsp; 

## WebGear's Default Template

The WebGear API by default uses simple & elegant **Bootstrap's [Cover template](https://github.com/twbs/bootstrap/blob/master/site/content/docs/4.3/examples/cover/index.html), by [@mdo](https://twitter.com/mdo)**, which looks like something as follows:

### Index.html

*Can be accessed by visiting WebGear app server, running at http://0.0.0.0:8000/:*

<h2 align="center">
  <img src="assets/images/webgear_temp_index.jpg" alt="WebGear default Index page"/>
</h2>


### 404.html

*Appears when respective URL is not found, for example http://0.0.0.0:8000/ok:*

<h2 align="center">
  <img src="assets/images/webgear_temp_404.jpg" alt="WebGear default 404 page"/>
</h2>


### 500.html

*Appears when an API Error is encountered:*

!!! warning "If [`logging`](/gears/webgear/params/#logging) is enabled and an error occurs, then instead of displaying this 500 handler, WebGear will respond with a traceback response."

<h2 align="center">
  <img src="assets/images/webgear_temp_500.jpg" alt="WebGear default 500 page"/>
</h2>


&nbsp; 