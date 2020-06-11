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

# Installation Overview

## Installation methods

* [Install using pip](/install/pip_install/) _(recommended)_
* [Install from source](/install/source_install/)

&nbsp;


## Supported Systems

VidGear is well-tested and supported on the following systems, with [python 3.6+]() and [pip]() installed:

* Any Linux distro released in 2016 or later
* Windows 7 or later
* macOS 10.12.6 (Sierra) or later

&nbsp;

## Supported Python legacies

**Python 3.6+** are only supported legacies for installing Vidgear v0.1.7 and above.

&nbsp;

## Testing the Latest Version

If you want to just install and try out the latest version of VidGear you can do so with the following command. This can be useful if you want to provide feedback for a new feature or want to confirm if a bug you have encountered is fixed in the `testing` branch. 

!!! tip "It is strongly recommended that you do this within a [virtualenv](https://virtualenv.pypa.io/en/latest/user_guide.html)."

```sh
pip install git+git://github.com/abhiTronix/vidgear@testing#egg=vidgear

# or with asyncio support
pip install git+git://github.com/abhiTronix/vidgear@testing#egg=vidgear[asyncio]
```

&nbsp;

<!--
External URLs
-->

[python]: https://www.python.org/doc/
[pip]: https://pip.pypa.io/en/stable/installing/#installation