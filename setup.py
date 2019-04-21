import os
import setuptools
from pkg_resources import parse_version
from setuptools import setup

def test_opencv():
    """
    This function is workaround to 
    test if correct OpenCV Library version has already been installed
    on the machine or not. Returns True if installed.
    """

    try:
        # import OpenCV Binaries
        import cv2
        
        if parse_version(cv2.__version__) >= parse_version('3'):
            # check whether OpenCV Binaries are 3.x+
            pass
        else:
            raise ImportError('Incompatible (< 3.0) OpenCV version-{} Installation found on this machine!'.format(parse_version(cv2.__version__)))

    except ImportError:
        return True

    return False


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='vidgear',
    packages=['vidgear','vidgear.gears'],
    version='0.1.4',
    description='A Lightweight Multi-Threaded OpenCV Video Processing Python Wrapper Library',
    license='MIT License',
    author='abhiTronix',
    install_requires = ["pafy"] 
    + (["opencv-python"] if test_opencv() else []) 
    + (["opencv-contrib-python"] if test_opencv() else []) 
    + (["picamera"] if ("arm" in os.uname()[4][:3]) else []),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email='abhi.una12@gmail.com',
    url='https://github.com/abhiTronix/vidgear',
    download_url='https://github.com/abhiTronix/vidgear/tarball/0.1.4',
    keywords=['computer vision', 'multi-thread', 'opencv', 'opencv2', 'opencv4', 'picamera'],
    classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7'],
    python_requires='>=2.7',
    scripts=[],
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/abhiTronix/vidgear/issues',
        'Funding': 'https://paypal.me/AbhiTronix?locale.x=en_GB',
        'Source': 'https://github.com/abhiTronix/vidgear',},
)
