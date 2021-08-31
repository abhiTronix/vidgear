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
    # import neccessary libs
    import yaml
    import argparse

    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "[VidGear:ERROR] :: Failed to detect correct uvicorn executables, install it with `pip3 install uvicorn` command."
        )

    # define argument parser and parse command line arguments
    usage = """python -m vidgear.gears.asyncio [-h] [-m MODE] [-s SOURCE] [-ep ENABLEPICAMERA] [-S STABILIZE]
                [-cn CAMERA_NUM] [-yt stream_mode] [-b BACKEND] [-cs COLORSPACE]
                [-r RESOLUTION] [-f FRAMERATE] [-td TIME_DELAY]
                [-ip IPADDRESS] [-pt PORT] [-l LOGGING] [-op OPTIONS]"""

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
    # VideoGear API specific params
    ap.add_argument(
        "-s",
        "--source",
        default=0,
        type=str,
        help="Path to input source for CamGear API.",
    )
    ap.add_argument(
        "-ep",
        "--enablePiCamera",
        type=bool,
        default=False,
        help="Sets the flag to access PiGear(if True) or otherwise CamGear API respectively.",
    )
    ap.add_argument(
        "-S",
        "--stabilize",
        type=bool,
        default=False,
        help="Enables/disables real-time video stabilization.",
    )
    ap.add_argument(
        "-cn",
        "--camera_num",
        default=0,
        help="Sets the camera module index that will be used by PiGear API.",
    )
    ap.add_argument(
        "-yt",
        "--stream_mode",
        default=False,
        type=bool,
        help="Enables YouTube Mode in CamGear API.",
    )
    ap.add_argument(
        "-b",
        "--backend",
        default=0,
        type=int,
        help="Sets the backend of the video source in CamGear API.",
    )
    ap.add_argument(
        "-cs",
        "--colorspace",
        type=str,
        help="Sets the colorspace of the output video stream.",
    )
    ap.add_argument(
        "-r",
        "--resolution",
        default=(640, 480),
        help="Sets the resolution (width,height) for camera module in PiGear API.",
    )
    ap.add_argument(
        "-f",
        "--framerate",
        default=30,
        type=int,
        help="Sets the framerate for camera module in PiGear API.",
    )
    ap.add_argument(
        "-td",
        "--time_delay",
        default=0,
        help="Sets the time delay(in seconds) before start reading the frames.",
    )
    # define WebGear exclusive params
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
    # define common params
    ap.add_argument(
        "-l",
        "--logging",
        type=bool,
        default=False,
        help="Enables/disables error logging, essential for debugging.",
    )
    ap.add_argument(
        "-op",
        "--options",
        type=str,
        help="Sets the parameters supported by APIs(whichever being accessed) to the input videostream, \
                    But make sure to wrap your dict value in single or double quotes.",
    )
    args = vars(ap.parse_args())

    options = {}
    # handle `options` params
    if not (args["options"] is None):
        options = yaml.safe_load(args["options"])

    if args["mode"] == "mjpeg":
        from .webgear import WebGear

        # initialize WebGear object
        web = WebGear(
            enablePiCamera=args["enablePiCamera"],
            stabilize=args["stabilize"],
            source=args["source"],
            camera_num=args["camera_num"],
            stream_mode=args["stream_mode"],
            backend=args["backend"],
            colorspace=args["colorspace"],
            resolution=args["resolution"],
            framerate=args["framerate"],
            logging=args["logging"],
            time_delay=args["time_delay"],
            **options
        )
    else:
        from .webgear_rtc import WebGear_RTC

        # initialize WebGear object
        web = WebGear_RTC(
            enablePiCamera=args["enablePiCamera"],
            stabilize=args["stabilize"],
            source=args["source"],
            camera_num=args["camera_num"],
            stream_mode=args["stream_mode"],
            backend=args["backend"],
            colorspace=args["colorspace"],
            resolution=args["resolution"],
            framerate=args["framerate"],
            logging=args["logging"],
            time_delay=args["time_delay"],
            **options
        )
    # run this object on Uvicorn server
    uvicorn.run(web(), host=args["ipaddress"], port=args["port"])

    # close app safely
    web.shutdown()
