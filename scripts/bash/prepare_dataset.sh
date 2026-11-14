#!/bin/bash

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

set -e # abort on first failing command

# Determine system temp dir (python first: tests use tempfile.gettempdir(), which differs from /tmp on Windows)
TMPFOLDER=$(python3 -c 'import tempfile; print(tempfile.gettempdir())' 2>/dev/null || python -c 'import tempfile; print(tempfile.gettempdir())' 2>/dev/null || true)
[ -z "$TMPFOLDER" ] && TMPFOLDER="${TMPDIR:-/tmp}"
TMPFOLDER="${TMPFOLDER%/}/testing_dir"

# Print the resolved path
echo "The resolved testing folder path is: $TMPFOLDER"

# Creating necessary directories
mkdir -p "$TMPFOLDER/temp_mpd" \
  "$TMPFOLDER/temp_m3u8" \
  "$TMPFOLDER/temp_write" \
  "$TMPFOLDER/temp_ffmpeg" \
  "$TMPFOLDER/Downloads/FFmpeg_static" \
  "$TMPFOLDER/Downloads/Test_videos"

# Defining alternate ffmpeg static binaries date/version (overridable via first argument)
ALTBINARIES_DATE="${1:-"08-11-2026"}"

# Acknowledging machine OS type
case $(uname -s | tr '[:upper:]' '[:lower:]') in
linux*)
  OS_NAME=linux
  ;;
darwin*)
  OS_NAME=osx
  ;;
msys* | cygwin* | mingw*)
  OS_NAME=windows
  ;;
*)
  echo "Error: unsupported OS: $(uname -s)" >&2
  exit 1
  ;;
esac

# fail on HTTP errors, follow redirects, retry flaky networks
download() { curl -fL --retry 3 "$@"; }

# Download and Configure FFmpeg Static
cd "$TMPFOLDER/Downloads/FFmpeg_static"

# Clean residual artifacts from previous runs
rm -rf ffmpeg ffmpeg-*

BASE_URL="https://github.com/abhiTronix/ffmpeg-static-builds/raw/master/$ALTBINARIES_DATE"

if [ "$OS_NAME" = "linux" ]; then
  echo "Downloading Linux Static FFmpeg Binaries..."
  download -O "$BASE_URL/linux/ffmpeg-git-amd64-static.tar.xz"
elif [ "$OS_NAME" = "windows" ]; then
  echo "Downloading Windows Static FFmpeg Binaries..."
  download -O "$BASE_URL/windows/ffmpeg-latest-win64-static.7z"
else
  echo "Downloading MacOS64 Static FFmpeg Binary..."
  download -O "$BASE_URL/macOS/ffmpeg-latest-macos64-static.zip"
fi

# Extract by archive type (zip, 7z, tar.*); every archive unpacks to ffmpeg/bin/
for ARCHIVE in ffmpeg-*; do
  case "$ARCHIVE" in
  *.zip) unzip -qq "$ARCHIVE" ;;
  *.7z) "$(command -v 7z || command -v 7zz || command -v 7za)" x -y "$ARCHIVE" >/dev/null ;; # 7z name varies by platform
  *.tar.*) tar -xf "$ARCHIVE" ;;
  *)
    echo "Error: unsupported archive type: $ARCHIVE" >&2
    exit 1
    ;;
  esac
  rm "$ARCHIVE"
done

# Downloading Test Data
cd "$TMPFOLDER/Downloads/Test_videos"

echo "Downloading Test-Data..."
download https://gitlab.com/abhiTronix/Imbakup/-/raw/master/Images/big_buck_bunny_720p_1mb.mp4 -o BigBuckBunny_4sec.mp4
download https://gitlab.com/abhiTronix/Imbakup/-/raw/master/Images/big_buck_bunny_720p_1mb_vo.mp4 -o BigBuckBunny_4sec_VO.mp4
download https://gitlab.com/abhiTronix/Imbakup/-/raw/master/Images/big_buck_bunny_720p_1mb_ao.aac -o BigBuckBunny_4sec_AO.aac
echo "Done Downloading Test-Data!"

if [ "$OS_NAME" = "linux" ]; then
  echo "Setting up ffmpeg with v4l2loopback"
  sudo modprobe v4l2loopback devices=1 video_nr=0 exclusive_caps=1 card_label='VCamera'
  sudo usermod -aG video "$USER" && sudo chmod 666 /dev/video0
  nohup sudo ffmpeg -hide_banner -loglevel error -re -stream_loop -1 -i "$TMPFOLDER"/Downloads/Test_videos/BigBuckBunny_4sec_VO.mp4 -f v4l2 -pix_fmt yuv420p /dev/video0 &
  echo "$USER ALL=NOPASSWD:$(which v4l2-ctl)" | (sudo su -c 'EDITOR="tee" visudo -f /etc/sudoers.d/v4l2ctl')
  v4l2-ctl --list-devices
  echo "Done"
fi

# Set permissions for the testing directories (Necessary for sudo commands)
if [ "$OS_NAME" != "windows" ]; then
  sudo chmod -R a+rw "$TMPFOLDER/temp_mpd" \
    "$TMPFOLDER/temp_m3u8" \
    "$TMPFOLDER/temp_write" \
    "$TMPFOLDER/temp_ffmpeg" \
    "$TMPFOLDER/Downloads/FFmpeg_static"
fi
