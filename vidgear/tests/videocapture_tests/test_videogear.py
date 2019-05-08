import pytest
from vidgear.gears import VideoGear


def test_PiGear_import():
	with pytest.raises(ImportError):
		stream = VideoGear(enablePiCamera = True, logging = True).start() # define various attributes and start the stream