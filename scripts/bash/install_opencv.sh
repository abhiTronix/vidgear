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

########################################
# Install OpenCV Custom Binaries for   #
#        CI Linux Environments         #
########################################

#determining system specific temp directory
TMPFOLDER=$(python -c 'import tempfile; print(tempfile.gettempdir())')

#determining system Python suffix and  version
PYTHONSUFFIX=$(python -c 'import platform; a = platform.python_version(); print(".".join(a.split(".")[:2]))')
PYTHONVERSION=$(python -c 'import platform; print(platform.python_version())')

echo "$PYTHONSUFFIX"
echo "$PYTHONVERSION"

echo "Installing OpenCV Dependencies..."

# Update package list
sudo apt-get update -qq

# Install build tools
echo "Installing build essentials..."
sudo apt-get install -y -qq --allow-unauthenticated build-essential gfortran cmake python3-dev
sudo apt-get install -y -qq --allow-unauthenticated pkg-config cmake-data

# Install video/codec dependencies
echo "Installing video and codec dependencies..."
sudo apt-get install -y -qq --allow-unauthenticated \
    libavutil-dev ffmpeg yasm libv4l-dev \
    libxvidcore-dev libx264-dev \
    libavcodec-dev libavformat-dev libswscale-dev libswresample-dev

# Install image format dependencies
echo "Installing image format dependencies..."
sudo apt-get install -y -qq --allow-unauthenticated \
    libtiff5-dev libjpeg-dev libpng-dev libwebp-dev libopenexr-dev

# Install math libraries
echo "Installing math libraries..."
sudo apt-get install -y -qq --allow-unauthenticated \
    libatlas-base-dev liblapacke-dev libopenblas-dev libopenblas-base

# Install GUI and parallel processing dependencies
echo "Installing GUI and parallel processing dependencies..."
sudo apt-get install -y -qq --allow-unauthenticated \
    libgtk-3-dev libtbb-dev

# Install other required dependencies
echo "Installing other dependencies..."
sudo apt-get install -y -qq --allow-unauthenticated \
    zlib1g-dev checkinstall

# Install GStreamer dependencies
echo "Installing GStreamer dependencies..."
sudo apt-get install -y -qq --allow-unauthenticated \
    libunwind-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 gstreamer1.0-pulseaudio


cd "$TMPFOLDER || exit"

export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH

echo "Installing v4l2loopback Kernel Module"

git clone https://github.com/v4l2loopback/v4l2loopback.git || exit
cd v4l2loopback || exit
make && sudo make install 
sudo depmod -a

cd "$TMPFOLDER" || exit

echo "Installing OpenCV Library"

RETRY=3
while [ "$RETRY" -gt 0 ]; do
  LATEST_VERSION=$(curl -sL https://api.github.com/repos/abhiTronix/OpenCV-CI-Releases/releases/latest |
    grep "OpenCV-.*.*-*-$PYTHONSUFFIX.*.deb" |
    grep -Eo "(http|https)://[a-zA-Z0-9./?=_%:-]*")
  echo "Found version: $LATEST_VERSION. Downloading..."
  curl -LO $LATEST_VERSION
  #opencv version to install
  OPENCV_FILENAME=$(basename "$LATEST_VERSION")
  echo "Installing OpenCV File: $OPENCV_FILENAME"
  if [ -n "$LATEST_VERSION" ] && [ -f $(find . -name "$OPENCV_FILENAME") ]; then
    echo "Downloaded OpenCV binary: $OPENCV_FILENAME successfully at $LATEST_VERSION"
    break
  else
    echo "Retrying: $RETRY!!!"
    RETRY=$((RETRY+1))
    sleep 3
  fi
done

if [ -z "$LATEST_VERSION" ]; then
  echo "Something is wrong!"
  exit 1
fi

echo "Installing OpenCV file: $OPENCV_FILENAME"

sudo dpkg -i "$OPENCV_FILENAME"

sudo ln -s /usr/local/lib/python$PYTHONSUFFIX/site-packages/*.so /opt/hostedtoolcache/Python/$PYTHONVERSION/x64/lib/python$PYTHONSUFFIX/site-packages

sudo ldconfig

echo "Python working version is $(which python)"

echo "OpenCV working version is $(python -c 'import cv2; print(cv2.__version__)')"

echo "Pip working version is $(python -m pip show pip) - $(python -m pip --version) - $(which pip)"

echo "Done Installing OpenCV...!!!"
