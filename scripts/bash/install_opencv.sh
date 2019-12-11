
#!/bin/sh

#Copyright (c) 2019 Abhishek Thakur

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.


######################################
#  Installing OpenCV 4.1.0 Binaries  #
######################################

OPENCV_VERSION='4.1.2'

#determining system specific temp directory
TMPFOLDER=$(python -c 'import tempfile; print(tempfile.gettempdir())')

#determining system Python suffix and  version
PYTHONSUFFIX=$(python -c 'import platform; a = platform.python_version(); print(".".join(a.split(".")[:2]))')
PYTHONVERSION=$(python -c 'import platform; print(platform.python_version())')

echo "Installing OpenCV..."

echo "Installing OpenCV Dependencies..."

sudo apt-get install -y --allow-unauthenticated build-essential cmake pkg-config gfortran

sudo apt-get install -y --allow-unauthenticated yasm libv4l-dev libgtk-3-dev libtbb-dev

sudo apt-get install -y --allow-unauthenticated libavcodec-dev libavformat-dev libswscale-dev libjasper-dev libopenexr-dev

sudo apt-get install -y --allow-unauthenticated libxvidcore-dev libx264-dev libatlas-base-dev libtiff5-dev python3-dev

sudo apt-get install -y --allow-unauthenticated zlib1g-dev libjpeg-dev checkinstall libwebp-dev libpng-dev libopenblas-base ffmpeg

sudo apt-get install -y --allow-unauthenticated libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools

echo "Installing OpenCV Library"

cd $TMPFOLDER

wget https://github.com/abhiTronix/OpenCV-Travis-Builds/releases/download/opencv-$OPENCV_VERSION/OpenCV-$OPENCV_VERSION-$PYTHONVERSION.deb

sudo dpkg -i OpenCV-$OPENCV_VERSION-$(python -c 'import platform; print(platform.python_version())').deb

sudo ln -s /usr/local/lib/python$PYTHONSUFFIX/site-packages/*.so $HOME/virtualenv/python$PYTHONVERSION/lib/python$PYTHONSUFFIX/site-packages

sudo ldconfig

echo "OpenCV working version is $(python -c 'import cv2; print(cv2.__version__)')"

echo "Done Installing OpenCV...!!!"