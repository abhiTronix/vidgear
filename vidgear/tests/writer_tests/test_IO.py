"""
Copyright (c) 2019 Abhishek Thakur

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""


from vidgear.gears import WriteGear
import sys
import numpy as np
import os
from numpy.testing import assert_equal
import pytest

def test_assertfailedwrite():
	"""
	IO Test - made to fail with Wrong Output file path
	"""
	np.random.seed(0)
	# generate random data for 10 frames
	random_data = np.random.random(size=(10, 1080, 1920, 3)) * 255
	input_data = random_data.astype(np.uint8)

	with pytest.raises(AssertionError):
		# wrong folder path does not exist
		writer = WriteGear("wrong_path/output.mp4")
		writer.write(input_data)
		writer.close()



def test_failedextension():
	"""
	IO Test - made to fail with filename with wrong extention
	"""
	np.random.seed(0)
	# generate random data for 10 frames
	random_data = np.random.random(size=(10, 1080, 1920, 3)) * 255
	input_data = random_data.astype(np.uint8)
	
	# 'garbage' extension does not exist
	with pytest.raises(ValueError):
		writer = WriteGear("garbage.garbage")
		writer.write(input_data)
		writer.close()
