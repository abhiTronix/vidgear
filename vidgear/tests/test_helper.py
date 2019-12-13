"""
============================================
vidgear library code is placed under the MIT license
Copyright (c) 2019 Abhishek Thakur

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
===============================================
"""

import os, pytest, tempfile, shutil, platform
from os.path import expanduser

from vidgear.gears.helper import download_ffmpeg_binaries
from vidgear.gears.helper import validate_ffmpeg
from vidgear.gears.helper import get_valid_ffmpeg_path
from vidgear.gears.helper import generate_auth_certificates



def return_static_ffmpeg():
	"""
	returns system specific FFmpeg static path
	"""
	path = ''
	if platform.system() == 'Windows':
		path += os.path.join(tempfile.gettempdir(),'Downloads/FFmpeg_static/ffmpeg/bin/ffmpeg.exe')
	elif platform.system() == 'Darwin':
		path += os.path.join(tempfile.gettempdir(),'Downloads/FFmpeg_static/ffmpeg/bin/ffmpeg')
	else:
		path += os.path.join(tempfile.gettempdir(),'Downloads/FFmpeg_static/ffmpeg/ffmpeg')
	return os.path.abspath(path)



def test_ffmpeg_static_installation():
	"""
	Test to ensure successful FFmpeg static Installation on Windows
	"""
	startpath = os.path.abspath(os.path.join(tempfile.gettempdir(),'Downloads/FFmpeg_static'))
	for root, dirs, files in os.walk(startpath):
		level = root.replace(startpath, '').count(os.sep)
		indent = ' ' * 4 * (level)
		print('[LOG]: {}{}/'.format(indent, os.path.basename(root)))
		subindent = ' ' * 4 * (level + 1)
		for f in files:
			print('[LOG]: {}{}'.format(subindent, f))



@pytest.mark.parametrize('paths', ['..','wrong_test_path', tempfile.gettempdir()])
def test_ffmpeg_binaries_download(paths):
	"""
	Testing Static FFmpeg auto-download on Windows OS
	"""
	_windows  = True if os.name == 'nt' else False
	if os.name == 'posix' and path == tempfile.gettempdir():
		#for incrementing codecov only
		_windows  = True
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



@pytest.mark.parametrize('paths', ['wrong_test_path', return_static_ffmpeg()])
def test_validate_ffmpeg(paths):
	"""
	Testing downloaded FFmpeg Static binaries validation on Windows OS
	"""
	try:
		output = validate_ffmpeg(paths, logging = True)
		if paths != 'wrong_test_path':
			assert bool(output), "Validation Test failed at path: {}".format(paths)
	except Exception as e:
		if paths == 'wrong_test_path':
			pass
		else:
			pytest.fail(str(e))



test_data = [('','', True), 
('wrong_test_path', '', False), 
('', 'wrong_test_path', False),
('', tempfile.gettempdir(), True), 
(return_static_ffmpeg(), '', True),
(os.path.dirname(return_static_ffmpeg()), '', True)]

@pytest.mark.parametrize('paths, ffmpeg_download_paths, results', test_data)
def test_get_valid_ffmpeg_path(paths, ffmpeg_download_paths, results):
	"""
	Testing FFmpeg excutables validation and correction:
	"""
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



test_data = [(os.path.join(expanduser("~"),".vidgear"), False, True), 
(os.path.join(expanduser("~"),".vidgear"), True, True),
('test_folder', False, True), 
(tempfile.gettempdir(), False, True), 
(tempfile.gettempdir(), True, True)]

@pytest.mark.parametrize('paths, overwrite_cert, results', test_data)
def test_generate_auth_certificates(paths, overwrite_cert, results):
	"""
	Testing auto-Generation and auto-validation of CURVE ZMQ keys/certificates 
	"""
	try:
		if overwrite_cert: print('[WARNING]: Overwriting ZMQ Authentication certificates over previous ones!')
		output = generate_auth_certificates(paths, overwrite = overwrite_cert)
		if paths != 'wrong_test_path':
			assert bool(output) == results
	except Exception as e:
		if paths == 'wrong_test_path':
			pass
		else:
			pytest.fail(str(e))