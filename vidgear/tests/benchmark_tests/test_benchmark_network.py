# import libraries
from vidgear.gears.asyncio import NetGear_Async
from vidgear.gears import VideoGear
from vidgear.gears import NetGear
import numpy as np
import logging as log
from .fps import FPS
import pytest, sys, asyncio, os, tempfile
from vidgear.gears.asyncio.helper import logger_handler

logger = log.getLogger("Benchmark NetworkGears")
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


def return_testvideo_path():
    """
    returns Test Video path
    """
    path = "{}/Downloads/Test_videos/BigBuckBunny_4sec.mp4".format(
        tempfile.gettempdir()
    )
    return os.path.abspath(path)


async def client_iterator(client):
    # loop over Client's Asynchronous Frame Generator
    fps_async = FPS().start()
    async for frame in client.recv_generator():
        # check if NoneType
        if frame is None:
            break
        fps_async.update()
        # await before continuing
        await asyncio.sleep(0.000001)
    logger.info("NetGear_Async approx. FPS: {:.2f}".format(fps_async.average_fps()))


pytestmark = pytest.mark.asyncio


@pytest.mark.skipif(
    sys.version_info >= (3, 8),
    reason="python3.8 is not supported yet by pytest-asyncio",
)
async def test_benchmark_Netgear_Async():
    """
    Benchmark NetGear Async in FPS
    """
    try:
        # launch server with valid source
        server = NetGear_Async(source=return_testvideo_path(), pattern=1).launch()
        # launch client
        client = NetGear_Async(receive_mode=True, pattern=1).launch()
        # gather and run tasks
        input_coroutines = [server.task, client_iterator(client)]
        res = await asyncio.gather(*input_coroutines, return_exceptions=True)
    except Exception as e:
        pytest.fail(str(e))
    finally:
        # close
        server.close(skip_loop=True)
        client.close(skip_loop=True)


@pytest.mark.skipif(
    sys.version_info >= (3, 8),
    reason="python3.8 is not supported yet by pytest-asyncio",
)
async def test_benchmark_NetGear():
    """
    Benchmark NetGear original in FPS
    """
    try:
        # open stream with valid source
        stream = VideoGear(source=return_testvideo_path()).start()
        # open server and client
        client = NetGear(receive_mode=True, pattern=1)
        server = NetGear(pattern=1)
        # start FPS handler
        fps = FPS().start()
        # playback
        while True:
            frame_server = stream.read()
            if frame_server is None:
                break
            fps.update()  # update
            server.send(frame_server)  # send
            frame_client = client.recv()  # recv
        stream.stop()
    except Exception as e:
        pytest.fail(str(e))
    finally:
        # close
        server.close()
        client.close()
        logger.info("NetGear approx. FPS: {:.2f}".format(fps.average_fps()))
