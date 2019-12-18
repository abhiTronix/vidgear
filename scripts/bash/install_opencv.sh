
#!/bin/sh

# Copyright (c) 2019 Abhishek Thakur(@abhiTronix) <abhi.una12@gmail.com>

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

######################################
#  Installing OpenCV Binaries        #
######################################

OPENCV_VERSION='4.2.0-pre'

#determining system specific temp directory
TMPFOLDER=$(python -c 'import tempfile; print(tempfile.gettempdir())')

#determining system Python suffix and  version
PYTHONSUFFIX=$(python -c 'import platform; a = platform.python_version(); print(".".join(a.split(".")[:2]))')
PYTHONVERSION=$(python -c 'import platform; print(platform.python_version())')
PYTHONVERSIONMIN=$(python3 -c 'import platform; print(platform.python_version()[:3])')

echo "Installing OpenCV..."

echo "Installing OpenCV Dependencies..."

sudo apt-get install -y --allow-unauthenticated build-essential cmake pkg-config gfortran libavutil-dev ffmpeg

sudo apt-get install -y --allow-unauthenticated yasm libv4l-dev libgtk-3-dev libtbb-dev libavresample-dev

sudo apt-get install -y --allow-unauthenticated libavcodec-dev libavformat-dev libswscale-dev libopenexr-dev

sudo apt-get install -y --allow-unauthenticated libxvidcore-dev libx264-dev libatlas-base-dev libtiff5-dev python3-dev liblapacke-dev

sudo apt-get install -y --allow-unauthenticated zlib1g-dev libjpeg-dev checkinstall libwebp-dev libpng-dev libopenblas-dev libopenblas-base

sudo apt-get install -y --allow-unauthenticated libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools

echo "Installing OpenCV Library"

cd $TMPFOLDER

export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH

curl -s https://api.github.com/repos/abhiTronix/OpenCV-Travis-Builds/releases/latest \
| grep "OpenCV-$OPENCV_VERSION-$PYTHONVERSIONMIN.*.deb" \
| cut -d : -f 2,3 \
| tr -d \" \
| wget -qi -

sudo dpkg -i OpenCV-$OPENCV_VERSION-$PYTHONVERSIONMIN.*.deb

sudo ln -s /usr/local/lib/python$PYTHONSUFFIX/site-packages/*.so $HOME/virtualenv/python$PYTHONVERSION/lib/python$PYTHONSUFFIX/site-packages

sudo ldconfig

echo "OpenCV working version is $(python -c 'import cv2; print(cv2.__version__)')"

echo "Done Installing OpenCV...!!!"