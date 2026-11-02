import asyncio


def pytest_asyncio_loop_factories(config, item):
    # SelectorEventLoop: default on Unix, required on Windows (no Proactor)
    return {"selector": asyncio.SelectorEventLoop}
