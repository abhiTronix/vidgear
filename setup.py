"""
============================================
vidgear library code is placed under the MIT license
Copyright (c) 2019 Abhishek Thakur

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
===============================================
"""

import platform
import setuptools
from pkg_resources import parse_version
from setuptools import setup

def test_opencv():
    """
    This function is workaround to 
    test if correct OpenCV Library version has already been installed
    on the machine or not. Returns True if previously not installed.
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
    long_description = long_description.replace(" [x]", "") #Readme Hack

setup(
    name='vidgear',
    packages=['vidgear','vidgear.gears'],
    version='0.1.5-dev',
    description='Powerful multi-threaded OpenCV and FFmpeg build Turbo Video Processing Python library powerpacked with unique trailblazing features.',
    license='MIT License',
    author='abhiTronix',
    install_requires = ["pafy", "mss", "youtube-dl", "requests","numpy"] 
    + (["opencv-contrib-python"] if test_opencv() else []) 
    + (["picamera"] if ("arm" in platform.uname()[4][:3]) else []),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email='abhi.una12@gmail.com',
    url='https://github.com/abhiTronix/vidgear',
    download_url='https://github.com/abhiTronix/vidgear/tarball/0.1.5-dev',
    keywords=['computer vision', 'multi-thread', 'python', 'opencv', 'cv2', 'opencv4', 'Video Processing'],
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
