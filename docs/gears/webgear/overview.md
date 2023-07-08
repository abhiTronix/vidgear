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

# WebGear API

<figure>
  <img src="../../../assets/gifs/webgear.gif" alt="WebGear in action!" loading="lazy" width=100%/>
  <figcaption>WebGear API's Video Server running at <a href="http://localhost:8000/" title="default address">http://localhost:8000/</a> address.</figcaption>
</figure>

## Overview

> _WebGear is a powerful [ASGI](https://asgi.readthedocs.io/en/latest/) Video-Broadcaster API ideal for transmitting [Motion-JPEG](https://en.wikipedia.org/wiki/Motion_JPEG)-frames from a single source to multiple recipients via the browser._

WebGear API works on [**Starlette**](https://www.starlette.io/)'s ASGI application and provides a highly extensible and flexible async wrapper around its complete framework. WebGear can flexibly interact with Starlette's ecosystem of shared middleware, mountable applications, [Response classes](https://www.starlette.io/responses/), [Routing tables](https://www.starlette.io/routing/), [Static Files](https://www.starlette.io/staticfiles/), [Templating engine(with Jinja2)](https://www.starlette.io/templates/), etc.

WebGear API uses an intraframe-only compression scheme under the hood where the sequence of video-frames are first encoded as JPEG-DIB (JPEG with Device-Independent Bit compression) and then streamed over HTTP using Starlette's Multipart [Streaming Response](https://www.starlette.io/responses/#streamingresponse) and a [Uvicorn](https://www.uvicorn.org/#quickstart) ASGI Server. This method imposes lower processing and memory requirements, but the quality is not the best, since JPEG compression is not very efficient for motion video.

In layman's terms, WebGear acts as a powerful ==**Video Broadcaster**== that transmits live video-frames to any web-browser in the network. Additionally, WebGear API also provides internal wrapper around [VideoGear](../../videogear/overview/), which itself provides internal access to both [CamGear](../../camgear/overview/) and [PiGear](../../pigear/overview/) APIs, thereby granting it exclusive power for transferring frames incoming from any source to the network.

&thinsp;

## Data-Files Auto-Generation WorkFlow for WebGear

??? tip "Disabling Auto-Generation process in WebGear"

    Starting with vidgear `v0.3.0`, you can now completely disable Auto-Generation process in WebGear API using [`skip_generate_webdata`](../params/#webgear-specific-attributes) optional boolean attribute. When `{skip_generate_webdata:True}`, no default data files will be downloaded or validated during initialization.

    !!! warning "Only `/video` route is available when `{skip_generate_webdata:True}` in WebGear API. All other default routes will be JSONResponses with `404`/`500` status codes."

??? note "Customizing default video endpoint path"
	Starting with vidgear `v0.3.1`, you can change default `/video` video endpoint path to any alphanumeric string value, using [`custom_video_endpoint`](../params/#webgear-specific-attributes) optional string attribute. For example:

	!!! error "Only alphanumeric string with no space in between are allowed as `custom_video_endpoint` value. Any other value will be discarded."

	!!! warning "WebGear's Default Theme which expects only default `/video` video endpoint path, will fail to work, if it is customized to any other value using this `custom_video_endpoint` attribute."

	```py
	# custom alphanumeric video endpoint string
	options = {"custom_video_endpoint": "xyz"}

	# initialize WebGear app
	web = WebGear(source="foo.mp4", logging=True, **options)
	```
	Hence, default video endpoint will now be available at `/xyz` path.

On initializing WebGear API, it automatically checks for three critical **data files**(i.e `index.html`, `404.html` & `500.html`) inside the `templates` folder of the `webgear` directory at the [_default location_](#default-location) which gives rise to the following two possible scenario:

- [x] **If data-files found:** it will proceed normally for instantiating the Starlette application.
- [ ] **If data-files not found:** it will trigger the [**Auto-Generation process**](#auto-generation-process)

### Default Location

- A _default location_ is the path of the directory where data files/folders are downloaded/generated/saved.
- By default, the `.vidgear` the folder at the home directory of your machine _(for e.g `/home/foo/.vidgear` on Linux)_ serves as the _default location_.
- But you can also use WebGear's [`custom_data_location`](../params/#webgear-specific-attributes) dictionary attribute to change/alter _default location_ path to somewhere else.

	!!! tip
			You can set [`logging=True`](../params/#logging) during initialization, for easily identifying the selected _default location_, which will be something like this _(on a Linux machine)_

			```sh
			WebGear :: DEBUG :: `/home/foo/.vidgear` is the default location for saving WebGear data-files.
			```

### Auto-Generation process

!!! info

    * You can also force trigger the Auto-generation process to overwrite existing data-files using [`overwrite_default_files`](../params/#webgear-specific-attributes) dictionary attribute. Remember, only downloaded default data files(given above) will be overwritten in this process but any other file/folder will NOT be affected.

    * It is advised to enable logging(`logging=True`) on the first run for easily identifying any runtime errors

- On triggering this process, WebGear API creates `webgear` directory, and `templates` and `static` folders inside along with `js`, `css`, `img` sub-folders at the assigned [_default location_](#default-location).
- Thereby at this [_default location_](#default-location), the necessary default data files will be downloaded from a dedicated [**Github Server**](https://github.com/abhiTronix/vidgear-vitals) inside respective folders in the following order:

	```sh
	.vidgear
	└── webgear
	    ├── static
	    │   ├── css
	    │   │   └── custom.css
	    │   ├── img
	    │   │   └── favicon-32x32.png
	    │   └── js
	    │       └── custom.js
	    └── templates
	        ├── 404.html
	        ├── 500.html
	        ├── base.html
	        └── index.html
	6 directories, 7 files
	```

- Finally these downloaded files thereby are verified for errors and API proceeds for instantiating the Starlette application normally.

&nbsp;

&nbsp;

## Importing

You can import WebGear API in your program as follows:

```python
from vidgear.gears.asyncio import WebGear
```

&thinsp;

&nbsp;

## WebGear's Default Template

??? new "New in v0.2.1"
New Standalone **WebGear's Default Theme** was added in `v0.2.1`.

The WebGear API by default uses simple & elegant [**WebGear's Default Theme**](https://github.com/abhiTronix/vidgear-vitals#webgear-default-theme) which looks like something as follows:

### Index.html

_Can be accessed by visiting WebGear app server, running at http://localhost:8000/:_

<h2 align="center">
  <img src="../../../assets/images/webgear_temp_index.png" loading="lazy" alt="WebGear default Index page"/>
</h2>

### 404.html

_Appears when respective URL is not found, for example http://localhost:8000/ok:_

<h2 align="center">
  <img src="../../../assets/images/webgear_temp_404.png" loading="lazy" alt="WebGear default 404 page"/>
</h2>

### 500.html

_Appears when an API Error is encountered:_

!!! warning "If [`logging`](../params/#logging) is enabled and an error occurs, then instead of displaying this 500 handler, WebGear will respond with a traceback response."

<h2 align="center">
  <img src="../../../assets/images/webgear_temp_500.png" loading="lazy" alt="WebGear default 500 page"/>
</h2>

&nbsp;

## Usage Examples

<div>
<a href="../usage/">See here 🚀</a>
</div>

!!! example "After going through WebGear Usage Examples, Checkout more bonus examples [here ➶](../../../help/webgear_ex/)"

## Parameters

<div>
<a href="../params/">See here 🚀</a>
</div>

## References

<div>
<a href="../../../bonus/reference/webgear/">See here 🚀</a>
</div>

## FAQs

<div>
<a href="../../../help/webgear_faqs/">See here 🚀</a>
</div>

&thinsp;
