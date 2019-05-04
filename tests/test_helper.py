import os, pytest, platform, tempfile, shutil, sys
from vidgear.gears.helper import download_ffmpeg_binaries
from vidgear.gears.helper import validate_ffmpeg
from vidgear.gears.helper import get_valid_ffmpeg_path
from vidgear.gears.helper import check_python_version

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
def test_download_ffmpeg_linux(path = tempfile.gettempdir()):
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
		except Exception as e:
			pytest.fail(str(e))


"""
Testing FFmpeg Static binaries installation on Windows:
Parametrized Values => userdefined_path, wrong_test_path, temporary_path 
"""
@pytest.mark.parametrize('paths', ['..', 'wrong_test_path', tempfile.gettempdir()])
def test_ffmpeg_binaries_download(paths):
	_windows  = True if os.name == 'nt' else False
	file_path = ''
	try: 
		file_path = download_ffmpeg_binaries(path = paths, os_windows = _windows)
		if file_path:
			assert os.path.isfile(file_path), "FFmpeg download failed!"
			if paths != tempfile.gettempdir():
				shutil.rmtree(os.path.abspath(os.path.join(file_path ,"../..")))
	except Exception as e:
		if paths == 'wrong_test_path' and "Permission Denied:" in str(e):
			pass
		else:
			pytest.fail(str(e))


"""
Testing FFmpeg Static binaries validation on Windows:
Parametrized Values => wrong_test_path, true_temporary_path 
"""
@pytest.mark.parametrize('paths', ['wrong_test_path', tempfile.gettempdir()])
def test_validate_ffmpeg(paths):
	FFmpeg_path = ''
	if os.name == 'nt':
		FFmpeg_path += os.path.join(paths, 'ffmpeg-latest-{}-static/bin/ffmpeg.exe'.format(getBitmode()))
	else:
		FFmpeg_path += os.path.join(paths, 'ffmpeg-4.1.3-{}-static/ffmpeg'.format(getBitmode(False)))
	try:
		output = validate_ffmpeg(FFmpeg_path, logging = True)
		if paths != 'wrong_test_path':
			assert bool(output), "Validation Test failed at path: {}".format(FFmpeg_path)
	except Exception as e:
		if paths == 'wrong_test_path':
			pass
		else:
			pytest.fail(str(e))


"""
Testing FFmpeg excutables validation and correction:
"""
test_data = [('','', True), 
('wrong_test_path', '', False), 
('', 'wrong_test_path', False), 
(os.path.join(tempfile.gettempdir(), 'ffmpeg-latest-{}-static/bin/ffmpeg.exe'.format(getBitmode()) if os.name == 'nt' else 'ffmpeg-4.1.3-{}-static'.format(getBitmode(False))), '', True),
(os.path.join(tempfile.gettempdir(), '{}/ffmpeg{}'.format('ffmpeg-latest-{}-static/bin'.format(getBitmode()) if os.name == 'nt' else 'ffmpeg-4.1.3-{}-static'.format(getBitmode(False)),'.exe'if os.name == 'nt' else '')), '', True),
(os.path.join(tempfile.gettempdir(), '{}'.format('ffmpeg-latest-{}-static/bin'.format(getBitmode()) if os.name == 'nt' else 'ffmpeg-4.1.3-{}-static'.format(getBitmode(False)))), '', True), 
('', tempfile.gettempdir(), True)]

@pytest.mark.parametrize('paths, ffmpeg_download_paths, results', test_data)
def test_get_valid_ffmpeg_path(paths, ffmpeg_download_paths, results):
	_windows  = True if os.name == 'nt' else False
	try:
		output = get_valid_ffmpeg_path(custom_ffmpeg = paths, is_windows = _windows, ffmpeg_download_path = ffmpeg_download_paths, logging = True)
		if not (paths == 'wrong_test_path' or ffmpeg_download_paths == 'wrong_test_path'):
			assert bool(output) == results, "FFmpeg excutables validation and correction Test failed at path: {} and FFmpeg ffmpeg_download_paths: {}".format(paths, ffmpeg_download_paths)
	except Exception as e:
		if paths == 'wrong_test_path' or ffmpeg_download_paths == 'wrong_test_path':
			pass
		else:
			pytest.fail(str(e))