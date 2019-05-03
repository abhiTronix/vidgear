import os, pytest, platform, tempfile, shutil
from vidgear.gears.helper import download_ffmpeg_binaries

@pytest.mark.parametrize('paths', [tempfile.gettempdir(), '..', 'wrong_test_path'])
def test_ffmpeg_binaries_download(paths):
	_windows  = True if os.name == 'nt' else False
	file_path = ''
	try: 
		file_path = download_ffmpeg_binaries(path = paths, os_windows = _windows)

		if file_path:
			assert os.path.isfile(file_path), "FFmpeg download failed!"
			shutil.rmtree(os.path.abspath(os.path.join(file_path ,"../..")))
	except Exception as e:
		print(e)
		if paths == 'wrong_test_path' and "Permission Denied:" in str(e):
			pass
		else:
			pytest.fail(str(e))
