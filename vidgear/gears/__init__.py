# import the necessary packages
import sys
import types
import importlib
from pkg_resources import parse_version


def get_module_version(module=None):
    """
    ## get_module_version

    Retrieves version of specified module

    Parameters:
        name (ModuleType): module of datatype `ModuleType`.

    **Returns:** version of specified module as string
    """
    # check if module type is valid
    assert not (module is None) and isinstance(
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
    install_name = pkg_name if not (pkg_name is None) else name

    # create message
    msg = (
        custom_message
        if not (custom_message is None)
        else "Failed to find its core dependency '{}'. Install it with  `pip install {}` command.".format(
            name, install_name
        )
    )
    # try importing dependency
    try:
        module = importlib.import_module(name)
        if sub_class:
            module = getattr(module, sub_class)
    except ImportError:
        # raise
        raise ImportError(msg) from None

    # check if minimum required version
    if not (version) is None:
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
            if parse_version(module_version) != parse_version(version):
                # create message
                msg = "Unsupported version '{}' found. Vidgear requires '{}' dependency with exact version '{}' installed!".format(
                    module_version, parent_module, version
                )
                # raise
                raise ImportError(msg)
        elif mode == "lte":
            if parse_version(module_version) > parse_version(version):
                # create message
                msg = "Unsupported version '{}' found. Vidgear requires '{}' dependency installed with older version '{}' or smaller!".format(
                    module_version, parent_module, version
                )
                # raise
                raise ImportError(msg)
        else:
            if parse_version(module_version) < parse_version(version):
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
    custom_message="Failed to find core dependency '{}'. Install it with  `pip install opencv-python` command.",
)
import_core_dependency(
    "numpy",
    mode="lte",
)
import_core_dependency(
    "colorlog",
)
import_core_dependency("requests")
import_core_dependency("from tqdm import tqdm", pkg_name="tqdm")

# import all APIs
from .camgear import CamGear
from .pigear import PiGear
from .videogear import VideoGear
from .netgear import NetGear
from .writegear import WriteGear
from .screengear import ScreenGear
from .streamgear import StreamGear

__all__ = [
    "PiGear",
    "CamGear",
    "VideoGear",
    "ScreenGear",
    "WriteGear",
    "NetGear",
    "StreamGear",
]

__author__ = "Abhishek Thakur (@abhiTronix) <abhi.una12@gmail.com>"
