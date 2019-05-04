import os, pytest, platform, tempfile, shutil, sys
from vidgear.gears import WriteGear

def getBitmode(_os_windows=True):
	"""
	Function to return bitmode
	"""
	if _os_windows:
		return 'win64' if platform.machine().endswith('64') else 'win32'
	else:
		return 'amd64' if platform.machine().endswith('64') else 'i686'

"""
Auxliary test for FFmpeg Static binaries installation on linux: 
"""

def test_download_ffmpeg_linux(path = os.path.join(tempfile.gettempdir(),'ffmpeg')):
	if os.name != 'nt':
		try:
			#inialize varibles
			file_url = 'https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-{}-static.tar.xz'.format(getBitmode(False))
			file_name = os.path.join(os.path.abspath(path),'ffmpeg-release-{}-static.tar.xz'.format(getBitmode(False)))
			file_path = os.path.join(os.path.abspath(path), 'ffmpeg-4.1.3-{}-static/ffmpeg'.format(getBitmode(False)))
			base_path, _ = os.path.split(file_path) #extract file base path

			#check if file already exists
			if os.path.isfile(file_path):
				pass #skip download if does 
			else:
				import requests
				
				if check_python_version() == 2:

					from backports import lzma

				import tarfile

				#check if given pth has write access
				assert os.access(path, os.W_OK), "Permission Denied: Cannot write ffmpeg binaries to directory = " + path
				#remove leftovers
				if os.path.isfile(file_name):
					os.remove(file_name)
				#download and write file to the given path

				with open(file_name, "wb") as f:
					response  = requests.get(file_url, stream=True)
					total_length = response.headers.get('content-length')
					if total_length is None: # no content length header
						f.write(response.content)
					else:
						dl = 0
						total_length = int(total_length)
						for data in response.iter_content(chunk_size=4096):
							dl += len(data)
							f.write(data)
							done = int(50 * dl / total_length)
							sys.stdout.write("\r[{}{}]{}{}".format('=' * done, ' ' * (50-done), done * 2, '%') )    
							sys.stdout.flush()
				print("\nExtracting executables, Please Wait...")
				if check_python_version()==2:
					with lzma.LZMAFile(file_name, "r") as f:
						with tarfile.open(fileobj=f) as tar:
							tar.extractall(base_path)			
				else:
					with tarfile.open(file_name, 'r:xz') as tar:
						tar.extractall(base_path)
						tar.close()
				print("\nChecking Files...")
				if os.path.isfile(file_path):
					pass
				else:
					folder = os.path.join(os.path.abspath(os.path.join(file_path ,"..")), 'ffmpeg-4.1.3-{}-static'.format(getBitmode(False)))
					folder_dest = os.path.abspath(os.path.join(file_path ,".."))
					files = os.listdir(folder)
					for file in files:
						shutil.move(os.path.join(folder, file), folder_dest)
					os.rmdir(folder)
					assert os.path.isfile(file_path), "Failed to extract Linux Static binary files!"
				#perform cleaning
				os.remove(file_name)
				print("FFmpeg binaries for Linux Configured Successfully at {}!".format(file_path))
		except:
			pytest.fail(str(e))

"""
Testing WriteGear Class Initialization:
"""
test_data_class = [
	('','','', {}, False),
	('Output.mp4',True,'', {}, True),
	('Output.mp4',False,'', {}, True),
	('Output.mp4',True,'', {}, True),
	('Output.mp4',True,'', {"-vcodec":"libx264", "-crf": 0, "-preset": "fast"}, True),
	('Output.mp4',True,'', {" -vcodec  ":" libx264", "   -crf": 0, "-preset    ": " fast "}, True),
	('Output.mp4',False,'', {"   -fourcc   ":"  MJPG ", "   -fps  ": 30}, True),
	('Output.mp4',False,'', {"-fourcc":"MJPG", "-fps": 30}, True),
	('Output.mp4',True,os.path.join(tempfile.gettempdir(), 'ffmpeg-latest-{}-static/bin/ffmpeg.exe'.format(getBitmode()) if os.name == 'nt' else 'ffmpeg-4.1.3-{}-static'.format(getBitmode(False))), {"-vcodec":"libx264", "-crf": 0, "-preset": "fast"}, True),
	('Output.mp4',True,'wrong_test_path', {" -vcodec  ":" libx264", "   -crf": 0, "-preset    ": " fast "}, False)]
@pytest.mark.parametrize('f_name, compression, c_ffmpeg, output_params, result', test_data_class)
def test_writegear_class(f_name, compression, c_ffmpeg, output_params, result):
	try:
		WriteGear(output_filename = f_name, compression_mode = compression , custom_ffmpeg = c_ffmpeg, logging = True, **output_params)
	except Exception as e:
		if result:
			pytest.fail(str(e))
			
