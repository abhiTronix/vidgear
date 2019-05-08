mkdir $HOME/download || echo "Already exists.";
mkdir $HOME/download/FFmpeg_static || echo "Already exists.";
mkdir $HOME/download/Test_videos || echo "Already exists.";
mkdir $HOME/download/Test_images || echo "Already exists.";

cd $HOME/download/FFmpeg_static

MACHINE_TYPE=`uname -m`

if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then 

	if [ ${MACHINE_TYPE} == 'x86_64' ]; then
	  curl https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o ffmpeg-release-amd64-static.tar.xz
	  tar -xJf ffmpeg-release-amd64-static.tar.xz
	  rm ffmpeg-release-amd64-static.tar.xz
	  mv ffmpeg-4.1.3-amd64-static ffmpeg
	else
	  curl https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-i686-static.tar.xz -o ffmpeg-release-i686-static.tar.xz
	  tar -xJf ffmpeg-release-i686-static.tar.xz
	  rm ffmpeg-release-i686-static.tar.xz
	  mv ffmpeg-4.1.3-i686-static ffmpeg
	fi

else

    if [ ${MACHINE_TYPE} == 'x86_64' ]; then
	  curl https://ffmpeg.zeranoe.com/builds/win64/static/ffmpeg-latest-win64-static.zip -o ffmpeg-latest-win64-static.zip
	  unzip ffmpeg-latest-win64-static.zip
	  rm ffmpeg-latest-win64-static.zip
	  mv ffmpeg-latest-win64-static ffmpeg
	else
	  curl https://ffmpeg.zeranoe.com/builds/win32/static/ffmpeg-latest-win32-static.zip -o ffmpeg-latest-win32-static.zip
	  unzip ffmpeg-latest-win32-static.zip
	  rm ffmpeg-latest-win32-static.zip
	  mv ffmpeg-latest-win32-static ffmpeg
	fi
fi

cd $HOME/download/Test_videos

curl http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4 -o BigBuckBunny.mp4
curl https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4 -o BigBuckBunny_4sec.mp4
curl http://jell.yfish.us/media/jellyfish-20-mbps-hd-hevc-10bit.mkv -o 20_mbps_hd_hevc_10bit.mkv
curl http://jell.yfish.us/media/jellyfish-50-mbps-hd-h264.mkv -o 50_mbps_hd_h264.mkv
curl http://jell.yfish.us/media/jellyfish-90-mbps-hd-hevc-10bit.mkv -o 90_mbps_hd_hevc_10bit.mkv
curl http://jell.yfish.us/media/jellyfish-120-mbps-4k-uhd-h264.mkv -o 120_mbps_4k_uhd_h264.mkv

