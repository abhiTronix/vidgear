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

#determining system specific temp directory
TMPFOLDER=$(python -c 'import tempfile; print(tempfile.gettempdir())')

# Creating necessary directories 
mkdir -p "$TMPFOLDER"/Downloads
mkdir -p "$TMPFOLDER"/Downloads/{FFmpeg_static,Test_videos}

# Acknowledging machine architecture
MACHINE_BIT=$(uname -m)

#Defining alternate ffmpeg static binaries date/version
ALTBINARIES_DATE=02-12-19

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

	echo "Downloading Linux Static FFmpeg Binaries..."
	if [ "$MACHINE_BIT" = "x86_64" ]; then
	  curl https://github.com/abhiTronix/ffmpeg-static-builds/raw/master/$ALTBINARIES_DATE/ffmpeg-release-amd64-static.tar.xz -o ffmpeg-release-amd64-static.tar.xz
	  tar -xJf ffmpeg-release-amd64-static.tar.xz
	  rm *.tar.*
	  mv ffmpeg* ffmpeg
	else
	  curl https://github.com/abhiTronix/ffmpeg-static-builds/raw/master/$ALTBINARIES_DATE/ffmpeg-release-i686-static.tar.xz -o ffmpeg-release-i686-static.tar.xz
	  tar -xJf ffmpeg-release-i686-static.tar.xz
	  rm *.tar.*
	  mv ffmpeg* ffmpeg
	fi

elif [ $OS_NAME = "windows" ]; then

	echo "Downloading Windows Static FFmpeg Binaries..."
	if [ "$MACHINE_BIT" = "x86_64" ]; then
	  curl https://github.com/abhiTronix/ffmpeg-static-builds/raw/master/$ALTBINARIES_DATE/ffmpeg-latest-win64-static.zip -o ffmpeg-latest-win64-static.zip
	  unzip -qq ffmpeg-latest-win64-static.zip
	  rm ffmpeg-latest-win64-static.zip
	  mv ffmpeg-latest-win64-static ffmpeg
	else
	  curl https://github.com/abhiTronix/ffmpeg-static-builds/raw/master/$ALTBINARIES_DATE/ffmpeg-latest-win32-static.zip -o ffmpeg-latest-win32-static.zip
	  unzip -qq ffmpeg-latest-win32-static.zip
	  rm ffmpeg-latest-win32-static.zip
	  mv ffmpeg-latest-win32-static ffmpeg
	fi

else

	echo "Downloading MacOS64 Static FFmpeg Binary..."
	curl -LO https://github.com/abhiTronix/ffmpeg-static-builds/raw/master/$ALTBINARIES_DATE/ffmpeg-latest-macos64-static.zip
	unzip -qq ffmpeg-latest-macos64-static.zip
	rm ffmpeg-latest-macos64-static.zip
	mv ffmpeg-latest-macos64-static ffmpeg

fi

# Downloading Test Data
cd "$TMPFOLDER"/Downloads/Test_videos

echo "Downloading Test-Data..."
curl http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4 -o BigBuckBunny.mp4
curl https://raw.githubusercontent.com/abhiTronix/Imbakup/master/Images/big_buck_bunny_720p_1mb.mp4 -o BigBuckBunny_4sec.mp4
curl http://jell.yfish.us/media/jellyfish-20-mbps-hd-hevc-10bit.mkv -o 20_mbps_hd_hevc_10bit.mkv
curl http://jell.yfish.us/media/jellyfish-50-mbps-hd-h264.mkv -o 50_mbps_hd_h264.mkv
curl http://jell.yfish.us/media/jellyfish-90-mbps-hd-hevc-10bit.mkv -o 90_mbps_hd_hevc_10bit.mkv
curl http://jell.yfish.us/media/jellyfish-120-mbps-4k-uhd-h264.mkv -o 120_mbps_4k_uhd_h264.mkv
echo "Done Downloading Test-Data!"