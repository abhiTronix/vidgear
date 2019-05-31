######################################
#  Installing OpenCV  Binaries       #
######################################


OPENCV_VERSION='4.1.0'

PYTHONSUFFIX=$(python -c 'import platform; a = platform.python_version(); print(".".join(a.split(".")[:2]))')
PYTHONVERSION=$(python -c 'import platform; print(platform.python_version())')

echo "Installing OpenCV..."

echo "Installing OpenCV Dependencies..."

sudo apt-get install -y build-essential cmake pkg-config gfortran

sudo apt-get install -y yasm libv4l-dev libgtk-3-dev libtbb-dev

sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libjasper-dev libopenexr-dev

sudo apt-get install -y libxvidcore-dev libx264-dev libatlas-base-dev libtiff5-dev python3-dev

sudo apt-get install -y zlib1g-dev libjpeg-dev checkinstall libwebp-dev libpng-dev libopenblas-base

sudo apt-get install -y libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools

echo "Installing OpenCV Library"

cd $HOME

wget https://github.com/abhiTronix/OpenCV-Travis-Builds/releases/download/latest/OpenCV-$OPENCV_VERSION-$PYTHONVERSION.deb

sudo dpkg -i OpenCV-$OPENCV_VERSION-$(python -c 'import platform; print(platform.python_version())').deb

sudo ldconfig

sudo ln -s /usr/local/lib/python$PYTHONSUFFIX/site-packages/*.so $.+93-HOME/virtualenv/python$PYTHONVERSION/lib/python$PYTHONSUFFIX/site-packages

sudo ldconfig

echo "OpenCV working version is $(python -c 'import cv2; print(cv2.__version__)')"

echo "Done Installing OpenCV...!!!"

