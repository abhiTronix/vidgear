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

from vidgear.gears import WriteGear
import numpy as np
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