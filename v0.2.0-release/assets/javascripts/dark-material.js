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

(() => {

  const preferToggle = function(e) {
    if (localStorage.getItem("data-md-prefers-color-scheme") === "true") {
      document.querySelector("body").setAttribute("data-md-color-scheme", (e.matches) ? "slate" : "default");
    }
  };

  const setupTheme = function(body) {
    const preferSupported = window.matchMedia("(prefers-color-scheme)").media !== "not all";
    let scheme = localStorage.getItem("data-md-color-scheme");
    let prefers = localStorage.getItem("data-md-prefers-color-scheme");

    if (!scheme) {
      scheme = "slate";
    }
    if (!prefers) {
      prefers = "false";
    }

    if (prefers === "true" && preferSupported) {
      scheme = (window.matchMedia("(prefers-color-scheme: dark)").matches) ? "slate" : "default";
    } else {
      prefers = "false";
    }

    body.setAttribute("data-md-prefers-color-scheme", prefers);
    body.setAttribute("data-md-color-scheme", scheme);

    if (preferSupported) {
      const matchListener = window.matchMedia("(prefers-color-scheme: dark)");
      matchListener.addListener(preferToggle);
    }
  };

  const observer = new MutationObserver(mutations => {
    mutations.forEach(mutation => {
      if (mutation.type === "childList") {
        if (mutation.addedNodes.length) {
          for (let i = 0; i < mutation.addedNodes.length; i++) {
            const el = mutation.addedNodes[i];

            if (el.nodeType === 1 && el.tagName.toLowerCase() === "body") {
              setupTheme(el);
              break;
            }
          }
        }
      }
    });
  });

  observer.observe(document.querySelector("html"), {childList: true});
})();

window.toggleScheme = function() {
  const body = document.querySelector("body");
  const preferSupported = window.matchMedia("(prefers-color-scheme)").media !== "not all";
  let scheme = body.getAttribute("data-md-color-scheme");
  let prefer = body.getAttribute("data-md-prefers-color-scheme");

  if (preferSupported && scheme === "default" && prefer !== "true") {
    prefer = "true";
    scheme = (window.matchMedia("(prefers-color-scheme: dark)").matches) ? "slate" : "default";
  } else if (preferSupported && prefer === "true") {
    prefer = "false";
    scheme = "slate";
  } else if (scheme === "slate") {
    prefer = "false";
    scheme = "default";
  } else {
    prefer = "false";
    scheme = "slate";
  }
  localStorage.setItem("data-md-prefers-color-scheme", prefer);
  localStorage.setItem("data-md-color-scheme", scheme);
  body.setAttribute("data-md-prefers-color-scheme", prefer);
  body.setAttribute("data-md-color-scheme", scheme);
};
