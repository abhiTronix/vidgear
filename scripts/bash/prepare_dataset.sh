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

mkdir -p $HOME/Downloads
mkdir -p $HOME/Downloads/{FFmpeg_static,Test_videos}

cd $HOME/Downloads/FFmpeg_static

OS_TYPE=$(uname)
MACHINE_BIT=$(uname -m)

#Download and Configure FFmpeg Static
if [ $OS_TYPE = "Linux" ]; then
	
	if [ $MACHINE_BIT = "x86_64" ]; then
	  curl https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o ffmpeg-release-amd64-static.tar.xz
	  tar -xJf ffmpeg-release-amd64-static.tar.xz
	  rm *.tar.*
	  mv ffmpeg* ffmpeg
	else
	  curl https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-i686-static.tar.xz -o ffmpeg-release-i686-static.tar.xz
	  tar -xJf ffmpeg-release-i686-static.tar.xz
	  rm *.tar.*
	  mv ffmpeg* ffmpeg
	fi

elif [ $OS_TYPE = "Windows" ]; then

	if [ $MACHINE_BIT = "x86_64" ]; then
	  curl https://ffmpeg.zeranoe.com/builds/win64/static/ffmpeg-latest-win64-static.zip -o ffmpeg-latest-win64-static.zip
	  unzip -qq ffmpeg-latest-win64-static.zip
	  rm ffmpeg-latest-win64-static.zip
	  mv ffmpeg-latest-win64-static ffmpeg
	else
	  curl https://ffmpeg.zeranoe.com/builds/win32/static/ffmpeg-latest-win32-static.zip -o ffmpeg-latest-win32-static.zip
	  unzip -qq ffmpeg-latest-win32-static.zip
	  rm ffmpeg-latest-win32-static.zip
	  mv ffmpeg-latest-win32-static ffmpeg
	fi

else

	curl -LO https://ffmpeg.zeranoe.com/builds/macos64/static/ffmpeg-latest-macos64-static.zip
	unzip -qq ffmpeg-latest-macos64-static.zip
	rm ffmpeg-latest-macos64-static.zip
	mv ffmpeg-latest-macos64-static ffmpeg
	ls
	
fi

cd $HOME/Downloads/Test_videos
# Download Test-Data
curl http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4 -o BigBuckBunny.mp4
curl https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4 -o BigBuckBunny_4sec.mp4
curl http://jell.yfish.us/media/jellyfish-20-mbps-hd-hevc-10bit.mkv -o 20_mbps_hd_hevc_10bit.mkv
curl http://jell.yfish.us/media/jellyfish-50-mbps-hd-h264.mkv -o 50_mbps_hd_h264.mkv
curl http://jell.yfish.us/media/jellyfish-90-mbps-hd-hevc-10bit.mkv -o 90_mbps_hd_hevc_10bit.mkv
curl http://jell.yfish.us/media/jellyfish-120-mbps-4k-uhd-h264.mkv -o 120_mbps_4k_uhd_h264.mkv

ls