mkdir $HOME/download || echo "Already exists.";
mkdir $HOME/download/FFmpeg_static || echo "Already exists.";
mkdir $HOME/download/Test_videos || echo "Already exists.";
mkdir $HOME/download/Test_images || echo "Already exists.";

cd $HOME/download/FFmpeg_static

MACHINE_TYPE=`uname -m`

if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then 

	if [ ${MACHINE_TYPE} == 'x86_64' ]; then
	  wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -O ffmpeg-release-amd64-static.tar.xz
	  tar -xJf ffmpeg-release-amd64-static.tar.xz
	  rm ffmpeg-release-amd64-static.tar.xz
	  mv ffmpeg-4.1.3-amd64-static ffmpeg
	else
	  wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-i686-static.tar.xz -O ffmpeg-release-i686-static.tar.xz
	  tar -xJf ffmpeg-release-i686-static.tar.xz
	  rm ffmpeg-release-i686-static.tar.xz
	  mv ffmpeg-4.1.3-i686-static ffmpeg
	fi

else

    if [ ${MACHINE_TYPE} == 'x86_64' ]; then
	  wget https://ffmpeg.zeranoe.com/builds/win64/static/ffmpeg-latest-win64-static.zip -O ffmpeg-latest-win64-static.zip
	  unzip ffmpeg-latest-win64-static.zip
	  rm ffmpeg-latest-win64-static.zip
	  mv ffmpeg-latest-win64-static ffmpeg
	else
	  wget https://ffmpeg.zeranoe.com/builds/win32/static/ffmpeg-latest-win32-static.zip -O ffmpeg-latest-win32-static.zip
	  unzip ffmpeg-latest-win32-static.zip
	  rm ffmpeg-latest-win32-static.zip
	  mv ffmpeg-latest-win32-static ffmpeg
	fi
fi

cd $HOME/download/Test_videos

wget http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4 -O BigBuckBunny.mp4
wget https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4 -O BigBuckBunny_4sec.mp4
wget http://jell.yfish.us/media/jellyfish-20-mbps-hd-hevc-10bit.mkv -O 20_mbps_hd_hevc_10bit.mkv
wget http://jell.yfish.us/media/jellyfish-50-mbps-hd-h264.mkv -O 50_mbps_hd_h264.mkv
wget http://jell.yfish.us/media/jellyfish-90-mbps-hd-hevc-10bit.mkv -O 90_mbps_hd_hevc_10bit.mkv
wget http://jell.yfish.us/media/jellyfish-120-mbps-4k-uhd-h264.mkv -O 120_mbps_4k_uhd_h264.mkv

