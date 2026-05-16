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

import logging as log

from .helper import import_dependency_safe, logger_handler

# define logger
logger = log.getLogger("YT_backend")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

yt_dlp = import_dependency_safe("yt_dlp", error="silent")
if yt_dlp is not None:
    from yt_dlp import YoutubeDL

    class YT_backend:
        """
        Internal YT-DLP Backend Class for extracting metadata from Streaming URLs.

        Shared across CamGear and FFGear APIs.

        Parameters:
            source_url (string): defines the URL of source stream
            logging (bool): enables/disables logging.
            options (dict): provides ability to alter yt-dlp backend params.
        """

        def __init__(
            self, source_url: str, logging: bool = False, **stream_params: dict
        ):
            # initialize global params
            self.__logging = logging
            self.is_livestream = False
            self.streams_metadata = {}
            self.streams = {}

            # define supported resolution values
            self.supported_resolutions = {
                "256x144": "144p",
                "426x240": "240p",
                "640x360": "360p",
                "854x480": "480p",
                "1280x720": "720p",
                "1920x1080": "1080p",
                "2560x1440": "1440p",
                "3840x2160": "2160p",
                "7680x4320": "4320p",
            }

            # assign source_url
            self.source_url = source_url

            # define default options for yt-dlp backend
            self.ydl_opts = {
                "format": "best*[vcodec!=none]",
                "quiet": True,
                "prefer_insecure": False,
                "no_warnings": not logging,
                "dump_single_json": True,
                "extract_flat": True,
                "skip_download": True,
            }
            # remove any attribute from user dict
            # that can cause API to fail
            stream_params.pop("format", None)
            stream_params.pop("dump_single_json", None)
            stream_params.pop("extract_flat", None)

            # extract exclusive params
            std_hdrs = stream_params.pop("std_headers", None)
            if std_hdrs is not None and isinstance(std_hdrs, dict):
                yt_dlp.utils.std_headers.update(std_hdrs)

            # update with user defined options
            self.ydl_opts.update(stream_params)

            # extract metadata
            self.meta_data = self.__extract_meta()

            # check if source url is supported
            if (
                self.meta_data is not None  # meta-data is valid
                and "entries" not in self.meta_data  # playlists are not supported
                and len(self.meta_data.get("formats", {}))
                > 0  # video formats must exist
            ):
                self.is_livestream = self.meta_data.get("is_live", False)
                self.streams_metadata = self.meta_data.get("formats", {})
                self.streams = self.__extract_streams()
                if self.streams:
                    logger.info(
                        "[Backend] :: Streaming URL is fully supported. Available Streams are: [{}]".format(
                            ", ".join(list(self.streams.keys()))
                        )
                    )
                else:
                    raise ValueError(
                        "[Backend] :: Streaming URL isn't supported. No usable video streams found!"
                    )
            else:
                # otherwise notify user
                raise ValueError(
                    "[Backend] :: Streaming URL isn't valid{}".format(
                        ". Playlists aren't supported yet!"
                        if self.meta_data is not None and "entries" in self.meta_data
                        else "!"
                    )
                )

        def __extract_meta(self):
            extracted_data = None
            # run parser
            with YoutubeDL(self.ydl_opts) as ydl:
                try:
                    # parse data
                    extracted_data = ydl.extract_info(self.source_url, download=False)
                except yt_dlp.utils.DownloadError as e:
                    # raise errors
                    raise RuntimeError(" [Backend] : " + str(e))
            # return data
            return extracted_data

        def __extract_streams(self):
            # extract streams
            streams = {}
            streams_copy = {}
            for stream in self.streams_metadata:
                # extract useable metadata
                stream_dim = stream.get("resolution", "")
                stream_url = stream.get("url", "")
                stream_protocol = stream.get("protocol", "")
                stream_with_video = stream.get("vcodec", "none") != "none"
                stream_with_audio = stream.get("acodec", "none") != "none"
                # streams must contain video
                if (
                    stream_with_video
                    and stream_dim
                    and stream_url
                    and stream_protocol != "http_dash_segments"
                ):
                    # check if stream resolution is supported
                    if stream_dim in self.supported_resolutions:
                        stream_res = self.supported_resolutions[stream_dim]
                        if (
                            not stream_with_audio  # prefer audioless
                            or stream_protocol in ["https", "http"]  # prefer http/https
                            or stream_res not in streams  # check if already not in dict
                        ):
                            streams[stream_res] = stream_url
                    # otherwise make a copy
                    if (
                        not stream_with_audio  # prefer audioless
                        or stream_protocol in ["https", "http"]  # prefer http/https
                        or stream_dim
                        not in streams_copy  # check if already not in dict
                    ):
                        streams_copy[stream_dim] = stream_url
            # use copy to decide best or worst
            if not streams_copy:
                return {}
            streams["best"] = streams_copy[list(streams_copy.keys())[-1]]
            streams["worst"] = streams_copy[next(iter(streams_copy.keys()))]
            return streams

else:
    YT_backend = None
