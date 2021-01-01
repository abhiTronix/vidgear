/* 
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
*/

window.addEventListener("DOMContentLoaded", function() {
    // This is a bit hacky. Figure out the base URL from a known CSS file the
    // template refers to...
    var ex = new RegExp("/?assets/stylesheets/version-select.css$");
    var sheet = document.querySelector('link[href$="version-select.css"]');

    var ABS_BASE_URL = sheet.href.replace(ex, "");
    var CURRENT_VERSION = ABS_BASE_URL.split("/").pop();

    function makeSelect(options, selected) {
        var select = document.createElement("select");
        select.classList.add("form-control");

        options.forEach(function(i) {
            var option = new Option(i.text, i.value, undefined,
                i.value === selected);
            select.add(option);
        });

        return select;
    }

    function insertAfter(referenceNode, newNode) {
        referenceNode.parentNode.insertBefore(newNode, referenceNode.nextSibling);
    }

    var xhr = new XMLHttpRequest();
    xhr.open("GET", ABS_BASE_URL + "/../versions.json");
    xhr.onload = function() {
        var versions = JSON.parse(this.responseText);

        var realVersion = versions.find(function(i) {
            return i.version === CURRENT_VERSION ||
                i.aliases.includes(CURRENT_VERSION);
        }).version;

        var select = makeSelect(versions.map(function(i) {
            return {
                text: i.title,
                value: i.version
            };
        }), realVersion);
        select.addEventListener("change", function(event) {
            window.location.href = ABS_BASE_URL + "/../" + this.value;
        });

        var container = document.createElement("div");
        container.id = "version-selector";
        container.appendChild(select);

        var header = document.querySelector(".md-header-nav__source");
        //header.parentNode.insertBefore(container, header);
        insertAfter(header, container);
    };
    xhr.send();
});
