from vidgear.gears import WriteGear
import sys
import numpy as np
import os
from numpy.testing import assert_equal
import pytest

def test_assertfailedwrite():

    np.random.seed(0)

    # generate random data for 10 frames
    random_data = np.random.random(size=(10, 1080, 1920, 3)) * 255
    input_data = random_data.astype(np.uint8)

    with pytest.raises(AssertionError):
        # 'garbage' folder does not exist
        writer = WriteGear("wrong_path/output.mp4")
        writer.write(input_data)
        writer.close()

def test_failedextension():

    np.random.seed(0)

    # generate random data for 10 frames
    random_data = np.random.random(size=(10, 1080, 1920, 3)) * 255
    input_data = random_data.astype(np.uint8)
    
    # 'garbage' extension does not exist
    with pytest.raises(ValueError):
        writer = WriteGear("garbage.garbage")
        writer.write(input_data)
        writer.close()
