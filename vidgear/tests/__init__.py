# Faking
import sys
from .utils import fake_picamera

sys.modules["picamera"] = fake_picamera.picamera
sys.modules["picamera.array"] = fake_picamera.picamera.array

__author__ = "Abhishek Thakur (@abhiTronix) <abhi.una12@gmail.com>"