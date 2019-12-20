"""
===============================================
vidgear library source-code is deployed under the Apache 2.0 License:

Copyright (c) 2019 Abhishek Thakur(@abhiTronix) <abhi.una12@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
===============================================
"""

import os, pytest, tempfile, shutil, platform
from os.path import expanduser
import logging as log

from vidgear.gears.helper import download_ffmpeg_binaries
from vidgear.gears.helper import validate_ffmpeg
from vidgear.gears.helper import get_valid_ffmpeg_path
from vidgear.gears.helper import generate_auth_certificates

logger = log.getLogger('Test_helper')


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
		logger.debug('{}{}/'.format(indent, os.path.basename(root)))
		subindent = ' ' * 4 * (level + 1)
		for f in files:
			logger.debug('{}{}'.format(subindent, f))



@pytest.mark.parametrize('paths', ['..','wrong_test_path', tempfile.gettempdir()])
def test_ffmpeg_binaries_download(paths):
	"""
	Testing Static FFmpeg auto-download on Windows OS
	"""
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
		if overwrite_cert: logger.warning('Overwriting ZMQ Authentication certificates over previous ones!')
		output = generate_auth_certificates(paths, overwrite = overwrite_cert)
		if paths != 'wrong_test_path':
			assert bool(output) == results
	except Exception as e:
		if paths == 'wrong_test_path':
			pass
		else:
			pytest.fail(str(e))