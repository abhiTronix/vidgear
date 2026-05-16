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

if __name__ == "__main__":
    # import necessary libs
    import argparse
    import sys

    import yaml

    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "[VidGear:ERROR] :: Failed to detect correct uvicorn executables, install it with `pip3 install uvicorn` command."
        )

    from ..helper import Backend

    # define argument parser and parse command line arguments
    usage = """python -m vidgear.gears.asyncio [-h] [-m MODE] [-s SOURCE] [-a API] [-S] [-b BACKEND]
                [-cs COLORSPACE] [-cn CAMERA_NUM] [-r RESOLUTION] [-f FRAMERATE]
                [-yt] [-sd SOURCE_DEMUXER] [-ff FRAME_FORMAT] [-cf CUSTOM_FFMPEG]
                [-td TIME_DELAY] [-ip IPADDRESS] [-pt PORT] [-l] [-op OPTIONS]"""

    ap = argparse.ArgumentParser(
        usage=usage,
        description="Runs WebGear/WebGear_RTC Video Server through terminal.",
    )
    ap.add_argument(
        "-m",
        "--mode",
        type=str,
        default="mjpeg",
        choices=["mjpeg", "webrtc"],
        help='Whether to use "MJPEG" or "WebRTC" mode for streaming.',
    )
    # VideoGear API backend selection
    ap.add_argument(
        "-a",
        "--api",
        type=str,
        default="camgear",
        choices=[b.value for b in Backend],
        help="Selects the capture backend for VideoGear. Choices: %(choices)s. Default: camgear.",
    )
    # deprecated --enablePiCamera flag (kept for backward compatibility)
    _ep_kwargs = {
        "action": "store_true",
        "default": False,
        "help": "[DEPRECATED] Use `--api pigear` instead. Sets the flag to access PiGear API.",
    }
    if sys.version_info >= (3, 13):
        _ep_kwargs["deprecated"] = True
    ap.add_argument("-ep", "--enablePiCamera", **_ep_kwargs)
    ap.add_argument(
        "-S",
        "--stabilize",
        action="store_true",
        default=False,
        help="Enables real-time video stabilization.",
    )
    # CamGear/FFGear parameters
    ap.add_argument(
        "-s",
        "--source",
        default=0,
        type=str,
        help="Path to input source (device index, filepath, URL, or glob pattern).",
    )
    ap.add_argument(
        "-yt",
        "--stream_mode",
        action="store_true",
        default=False,
        help="Enables YouTube/yt_dlp Stream Mode in CamGear/FFGear API.",
    )
    ap.add_argument(
        "-b",
        "--backend",
        default=0,
        type=int,
        help="Sets the backend of the video source in CamGear API (e.g. cv2.CAP_DSHOW).",
    )
    # FFGear parameters
    ap.add_argument(
        "-sd",
        "--source_demuxer",
        type=str,
        default=None,
        help='[FFGear only] FFmpeg demuxer for the source (e.g. "v4l2", "dshow"). Default: auto-detect.',
    )
    ap.add_argument(
        "-ff",
        "--frame_format",
        type=str,
        default="bgr24",
        help='[FFGear only] Pixel format for decoded frames (e.g. "bgr24", "gray"). Default: bgr24.',
    )
    ap.add_argument(
        "-cf",
        "--custom_ffmpeg",
        type=str,
        default="",
        help="[FFGear only] Path to a custom FFmpeg executable. Default: use PATH.",
    )
    # PiGear parameters
    ap.add_argument(
        "-cn",
        "--camera_num",
        default=0,
        type=int,
        help="[PiGear only] Sets the camera module index.",
    )
    ap.add_argument(
        "-r",
        "--resolution",
        default=(640, 480),
        help="[PiGear only] Sets the resolution (width,height) for the camera module.",
    )
    ap.add_argument(
        "-f",
        "--framerate",
        default=30,
        type=int,
        help="[PiGear only] Sets the framerate for the camera module.",
    )
    # common parameters
    ap.add_argument(
        "-cs",
        "--colorspace",
        type=str,
        default=None,
        help="Sets the colorspace of the output video stream.",
    )
    ap.add_argument(
        "-td",
        "--time_delay",
        default=0,
        type=int,
        help="Sets the time delay (in seconds) before start reading the frames.",
    )
    # WebGear/WebGear_RTC server params
    ap.add_argument(
        "-ip",
        "--ipaddress",
        type=str,
        default="0.0.0.0",
        help="Uvicorn binds the socket to this ipaddress.",
    )
    ap.add_argument(
        "-pt",
        "--port",
        type=int,
        default=8000,
        help="Uvicorn binds the socket to this port.",
    )
    ap.add_argument(
        "-l",
        "--logging",
        action="store_true",
        default=False,
        help="Enables/disables error logging, essential for debugging.",
    )
    ap.add_argument(
        "-op",
        "--options",
        type=str,
        default=None,
        help="Sets the parameters supported by APIs (whichever being accessed) to the input videostream. "
        "Wrap your dict value in single or double quotes.",
    )
    args = vars(ap.parse_args())

    options = {}
    # handle `options` params
    if args["options"] is not None:
        parsed = yaml.safe_load(args["options"])
        if isinstance(parsed, dict):
            options = parsed

    # resolve Backend enum from CLI --api value
    # (enablePiCamera, if set, is handled downstream by VideoGear)
    api = Backend(args["api"])

    if args["mode"] == "mjpeg":
        from .webgear import WebGear

        # initialize WebGear object
        web = WebGear(
            enablePiCamera=args["enablePiCamera"] or None,
            api=api,
            stabilize=args["stabilize"],
            source=args["source"],
            camera_num=args["camera_num"],
            stream_mode=args["stream_mode"],
            backend=args["backend"],
            source_demuxer=args["source_demuxer"],
            frame_format=args["frame_format"],
            custom_ffmpeg=args["custom_ffmpeg"],
            colorspace=args["colorspace"],
            resolution=args["resolution"],
            framerate=args["framerate"],
            logging=args["logging"],
            time_delay=args["time_delay"],
            **options,
        )
    else:
        from .webgear_rtc import WebGear_RTC

        # initialize WebGear_RTC object
        web = WebGear_RTC(
            enablePiCamera=args["enablePiCamera"] or None,
            api=api,
            stabilize=args["stabilize"],
            source=args["source"],
            camera_num=args["camera_num"],
            stream_mode=args["stream_mode"],
            backend=args["backend"],
            source_demuxer=args["source_demuxer"],
            frame_format=args["frame_format"],
            custom_ffmpeg=args["custom_ffmpeg"],
            colorspace=args["colorspace"],
            resolution=args["resolution"],
            framerate=args["framerate"],
            logging=args["logging"],
            time_delay=args["time_delay"],
            **options,
        )
    # run this object on Uvicorn server
    uvicorn.run(web(), host=args["ipaddress"], port=args["port"])

    # close app safely
    web.shutdown()
