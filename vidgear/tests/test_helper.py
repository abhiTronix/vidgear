import os, pytest, platform, tempfile, shutil, sys
from vidgear.gears.helper import download_ffmpeg_binaries
from vidgear.gears.helper import validate_ffmpeg
from vidgear.gears.helper import get_valid_ffmpeg_path
from vidgear.gears.helper import check_python_version


def return_static_ffmpeg():
	path = ''
	if os.name == 'nt':
		path += os.path.join(os.environ['USERPROFILE'],'Downloads/FFmpeg_static/ffmpeg/bin/ffmpeg.exe')
	else:
		path += os.path.join(os.environ['HOME'],'Downloads/FFmpeg_static/ffmpeg/ffmpeg')
	return os.path.abspath(path)


def test_ffmpeg_static_installation():
	startpath = os.path.abspath(os.path.join( os.environ['USERPROFILE'] if os.name == 'nt' else os.environ['HOME'],'Downloads/FFmpeg_static'))
	for root, dirs, files in os.walk(startpath):
		level = root.replace(startpath, '').count(os.sep)
		indent = ' ' * 4 * (level)
		print('{}{}/'.format(indent, os.path.basename(root)))
		subindent = ' ' * 4 * (level + 1)
		for f in files:
			print('{}{}'.format(subindent, f))

"""
Testing FFmpeg Static binaries installation on Windows:
Parametrized Values => userdefined_path_empty, userdefined_path_exist, wrong_test_path, temporary_path 
"""
@pytest.mark.parametrize('paths', ['..','wrong_test_path', tempfile.gettempdir()])
def test_ffmpeg_binaries_download(paths):
	_windows  = True if os.name == 'nt' else False
	file_path = ''
	try: 
		file_path = download_ffmpeg_binaries(path = paths, os_windows = _windows)
		if file_path:
			assert os.path.isfile(file_path), "FFmpeg download failed!"
			if paths != return_static_ffmpeg():
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
@pytest.mark.parametrize('paths', ['wrong_test_path', return_static_ffmpeg()])
def test_validate_ffmpeg(paths):
	try:
		output = validate_ffmpeg(paths, logging = True)
		if paths != 'wrong_test_path':
			assert bool(output), "Validation Test failed at path: {}".format(paths)
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
('', tempfile.gettempdir(), True), 
(return_static_ffmpeg(), '', True),
(os.path.dirname(return_static_ffmpeg()), '', True)]

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