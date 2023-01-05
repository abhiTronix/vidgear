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

#determining system specific temp directory
TMPFOLDER=$(python -c 'import tempfile; print(tempfile.gettempdir())')

# Creating necessary directories
mkdir -p "$TMPFOLDER"/temp_mpd    # MPD assets temp path
mkdir -p "$TMPFOLDER"/temp_m3u8   # M3U8 assets temp path
mkdir -p "$TMPFOLDER"/temp_write  # For testing WriteGear Assets.
mkdir -p "$TMPFOLDER"/temp_ffmpeg # For downloading FFmpeg Static Binary Assets.
mkdir -p "$TMPFOLDER"/Downloads
mkdir -p "$TMPFOLDER"/Downloads/{FFmpeg_static,Test_videos}

# Acknowledging machine architecture
MACHINE_BIT=$(uname -m)

#Defining alternate ffmpeg static binaries date/version
ALTBINARIES_DATE="12-07-2022"

# Acknowledging machine OS type
case $(uname | tr '[:upper:]' '[:lower:]') in
linux*)
  OS_NAME=linux
  ;;
darwin*)
  OS_NAME=osx
  ;;
msys*)
  OS_NAME=windows
  ;;
*)
  OS_NAME=notset
  ;;
esac

#Download and Configure FFmpeg Static
cd "$TMPFOLDER"/Downloads/FFmpeg_static

if [ $OS_NAME = "linux" ]; then

  echo "Downloading Linux64 Static FFmpeg Binaries..."
  curl -LO https://github.com/abhiTronix/ffmpeg-static-builds/raw/master/$ALTBINARIES_DATE/linux/ffmpeg-git-amd64-static.tar.xz
  tar -xJf ffmpeg-git-amd64-static.tar.xz
  rm *.tar.*
  mv ffmpeg* ffmpeg

elif [ $OS_NAME = "windows" ]; then

  echo "Downloading Win64 Static FFmpeg Binaries..."
  curl -LO https://github.com/abhiTronix/ffmpeg-static-builds/raw/master/$ALTBINARIES_DATE/windows/ffmpeg-latest-win64-static.zip
  unzip -qq ffmpeg-latest-win64-static.zip
  rm ffmpeg-latest-win64-static.zip
  mv ffmpeg-latest-win64-static ffmpeg

else

  echo "Downloading MacOS64 Static FFmpeg Binary..."
  curl -LO https://github.com/abhiTronix/ffmpeg-static-builds/raw/master/$ALTBINARIES_DATE/macOS/ffmpeg-latest-macos64-static.zip
  unzip -qq ffmpeg-latest-macos64-static.zip
  rm ffmpeg-latest-macos64-static.zip
  mv ffmpeg-latest-macos64-static ffmpeg

fi

# Downloading Test Data
cd "$TMPFOLDER"/Downloads/Test_videos || exit

echo "Downloading Test-Data..."
curl https://gitlab.com/abhiTronix/Imbakup/-/raw/master/Images/big_buck_bunny_720p_1mb.mp4 -o BigBuckBunny_4sec.mp4
curl https://gitlab.com/abhiTronix/Imbakup/-/raw/master/Images/big_buck_bunny_720p_1mb_vo.mp4 -o BigBuckBunny_4sec_VO.mp4
curl https://gitlab.com/abhiTronix/Imbakup/-/raw/master/Images/big_buck_bunny_720p_1mb_ao.aac -o BigBuckBunny_4sec_AO.aac
curl -L https://github.com/abhiTronix/Imbakup/releases/download/vid-001/BigBuckBunny.mp4 -o BigBuckBunny.mp4
curl -L https://github.com/abhiTronix/Imbakup/releases/download/vid-001/jellyfish-50-mbps-hd-h264.mkv -o 50_mbps_hd_h264.mkv
curl -L https://github.com/abhiTronix/Imbakup/releases/download/vid-001/jellyfish-90-mbps-hd-hevc-10bit.mkv -o 90_mbps_hd_hevc_10bit.mkv
curl -L https://github.com/abhiTronix/Imbakup/releases/download/vid-001/jellyfish-120-mbps-4k-uhd-h264.mkv -o 120_mbps_4k_uhd_h264.mkv
echo "Done Downloading Test-Data!"
