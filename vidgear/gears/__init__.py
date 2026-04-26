# import the necessary packages
import importlib
import sys
import types

from packaging.version import parse


def get_module_version(module=None):
    """
    ## get_module_version

    Retrieves version of specified module

    Parameters:
        name (ModuleType): module of datatype `ModuleType`.

    **Returns:** version of specified module as string
    """
    # check if module type is valid
    assert module is not None and isinstance(
        module, types.ModuleType
    ), "[VidGear CORE:ERROR] :: Invalid module!"

    # get version from attribute
    version = getattr(module, "__version__", None)
    # retry if failed
    if version is None:
        # some modules uses a capitalized attribute name
        version = getattr(module, "__VERSION__", None)
    # raise if still failed
    if version is None:
        raise ImportError(
            "[VidGear CORE:ERROR] ::  Can't determine version for module: `{}`!".format(
                module.__name__
            )
        )
    return str(version)


def import_core_dependency(
    name, pkg_name=None, custom_message=None, version=None, mode="gte"
):
    """
    ## import_core_dependency

    Imports specified core dependency. By default(`error = raise`), if a dependency is missing,
    an ImportError with a meaningful message will be raised. Also, If a dependency is present,
    but version is different than specified, an error is raised.

    Parameters:
        name (string): name of dependency to be imported.
        pkg_name (string): (Optional) package name of dependency(if different `pip` name). Otherwise `name` will be used.
        custom_message (string): (Optional) custom Import error message to be raised.
        version (string): (Optional) required minimum/maximum version of the dependency to be imported.
        mode (boolean): (Optional) Possible values "gte"(greater then equal), "lte"(less then equal), "exact"(exact). Default is "gte".

    **Returns:** `None`
    """
    # check specified parameters
    assert name and isinstance(
        name, str
    ), "[VidGear CORE:ERROR] :: Kindly provide name of the dependency."

    # extract name in case of relative import
    sub_class = ""
    name = name.strip()
    if name.startswith("from"):
        name = name.split(" ")
        name, sub_class = (name[1].strip(), name[-1].strip())

    # check mode of operation
    assert mode in ["gte", "lte", "exact"], "[VidGear CORE:ERROR] :: Invalid mode!"

    # specify package name of dependency(if defined). Otherwise use name
    install_name = pkg_name if pkg_name is not None else name

    # create message
    msg = (
        custom_message
        if custom_message is not None
        else "Failed to find its core dependency '{}'. Install it with  `pip install {}` command.".format(
            name, install_name
        )
    )
    # try importing dependency
    try:
        module = importlib.import_module(name)
        module = getattr(module, sub_class) if sub_class else module
    except Exception as e:
        if isinstance(e, ModuleNotFoundError):
            # raise message
            raise ModuleNotFoundError(msg) from None
        else:
            # raise error+message
            raise ImportError(msg) from e

    # check if minimum required version
    if (version) is not None:
        # Handle submodules
        parent_module = name.split(".")[0]
        if parent_module != name:
            # grab parent module
            module_to_get = sys.modules[parent_module]
        else:
            module_to_get = module

        # extract version
        module_version = get_module_version(module_to_get)
        # verify
        if mode == "exact":
            if parse(module_version) != parse(version):
                # create message
                msg = "Unsupported version '{}' found. Vidgear requires '{}' dependency with exact version '{}' installed!".format(
                    module_version, parent_module, version
                )
                # raise
                raise ImportError(msg)
        elif mode == "lte":
            if parse(module_version) > parse(version):
                # create message
                msg = "Unsupported version '{}' found. Vidgear requires '{}' dependency installed with older version '{}' or smaller!".format(
                    module_version, parent_module, version
                )
                # raise
                raise ImportError(msg)
        else:
            if parse(module_version) < parse(version):
                # create message
                msg = "Unsupported version '{}' found. Vidgear requires '{}' dependency installed with newer version '{}' or greater!".format(
                    module_version, parent_module, version
                )
                # raise
                raise ImportError(msg)
    return module


# import core dependencies
import_core_dependency(
    "cv2",
    pkg_name="opencv-python",
    version="3",
)
import_core_dependency("numpy")
import_core_dependency("colorlog")
import_core_dependency("requests")
import_core_dependency("from tqdm import tqdm", pkg_name="tqdm")

# import all APIs
from .camgear import CamGear
from .netgear import NetGear
from .pigear import PiGear
from .screengear import ScreenGear
from .streamgear import StreamGear
from .videogear import VideoGear
from .writegear import WriteGear

__all__ = [
    "CamGear",
    "NetGear",
    "PiGear",
    "ScreenGear",
    "StreamGear",
    "VideoGear",
    "WriteGear",
]

__author__ = "Abhishek Thakur (@abhiTronix) <abhi.una12@gmail.com>"
