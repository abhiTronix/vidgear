"""
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
"""

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.pages import Page

js_scripts = """
<script type="text/javascript" src="https://cdn.jsdelivr.net/npm/@clappr/player@latest/dist/clappr.min.js"></script>
<script type="text/javascript" src="https://cdn.jsdelivr.net/gh/clappr/clappr-level-selector-plugin@latest/dist/level-selector.min.js"></script>
<script type="text/javascript" src="https://cdn.jsdelivr.net/gh/clappr/dash-shaka-playback@latest/dist/dash-shaka-playback.js"></script>
<script type="text/javascript" src="https://cdn.jsdelivr.net/npm/@clappr/hlsjs-playback@latest/dist/hlsjs-playback.min.js"></script>
"""


# Add per-file custom javascripts.
def on_page_markdown(markdown: str, *, page: Page, config: MkDocsConfig, files):
    if not (
        page.file.src_uri
        in ["gears/stabilizer/overview.md", "gears/streamgear/introduction.md"]
    ):
        return

    # Replace markdown + js scripts
    comment, content = markdown.split("-->")
    modified_markdown = comment + "-->\n" + js_scripts + content

    # Return modified
    return modified_markdown
