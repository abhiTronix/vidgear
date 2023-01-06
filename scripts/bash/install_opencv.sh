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

sudo apt-get install -y -qq --allow-unauthenticated build-essential cmake pkg-config gfortran libavutil-dev ffmpeg

sudo apt-get install -y -qq --allow-unauthenticated yasm libv4l-dev libgtk-3-dev libtbb-dev libswresample-dev

sudo apt-get install -y -qq --allow-unauthenticated libavcodec-dev libavformat-dev libswscale-dev libopenexr-dev

sudo apt-get install -y -qq --allow-unauthenticated libxvidcore-dev libx264-dev libatlas-base-dev libtiff5-dev python3-dev liblapacke-dev

sudo apt-get install -y -qq --allow-unauthenticated zlib1g-dev libjpeg-dev checkinstall libwebp-dev libpng-dev libopenblas-dev libopenblas-base

sudo apt-get install -y -qq --allow-unauthenticated libgstreamer1.0-0 libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev

sudo apt-get install -y -qq --allow-unauthenticated gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio

echo "Installing OpenCV Library"

cd "$TMPFOLDER || exit"

export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH

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

echo "OpenCV working version is $(python -c 'import cv2; print(cv2.__version__)')"

echo "Done Installing OpenCV...!!!"
