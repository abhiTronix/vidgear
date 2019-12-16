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
from vidgear.gears import NetGear
from vidgear.gears import VideoGear

import pytest
import cv2
import random
import tempfile
import os
import numpy as np
import traceback
from zmq.error import ZMQError



def return_testvideo_path():
	"""
	returns Test Video path
	"""
	path = '{}/Downloads/Test_videos/BigBuckBunny_4sec.mp4'.format(tempfile.gettempdir())
	return os.path.abspath(path)



def test_playback():
	"""
	Tests NetGear Bare-minimum network playback capabilities
	"""
	try:
		#open stream
		stream = VideoGear(source=return_testvideo_path()).start()
		#open server and client with default params
		client = NetGear(receive_mode = True)
		server = NetGear()
		#playback
		while True:
			frame_server = stream.read()
			if frame_server is None: break
			server.send(frame_server) #send
			frame_client = client.recv() #recv
		#clean resources
		stream.stop()
		server.close()
		client.close()
	except Exception as e:
		if isinstance(e, (ZMQError, ValueError)):
			print(traceback.print_tb(e.__traceback__))
		else:
			pytest.fail(str(e))



@pytest.mark.parametrize('pattern', [1, 2, 3]) #0:(zmq.PAIR,zmq.PAIR), 1:(zmq.REQ,zmq.REP), 2:(zmq.PUB,zmq.SUB) (#3 is incorrect value)
def test_patterns(pattern):
	"""
	Testing NetGear different messaging patterns
	"""
	#open stream
	try:
		stream = VideoGear(source=return_testvideo_path()).start()
		
		#define parameters
		options = {'flag' : 0, 'copy' : True, 'track' : False}
		client = NetGear(pattern = pattern, receive_mode = True, logging = True, **options)
		server = NetGear(pattern = pattern, logging = True, **options)
		#initialize 
		frame_server = None
		#select random input frame from stream
		i = 0
		while (i < random.randint(10, 100)):
			frame_server = stream.read()
			i+=1
		#check if input frame is valid
		assert not(frame_server is None)
		#send frame over network
		server.send(frame_server)
		frame_client = client.recv()
		#clean resources
		stream.stop()
		server.close()
		client.close()
		#check if recieved frame exactly matches input frame
		assert np.array_equal(frame_server, frame_client)
	except Exception as e:
		if isinstance(e, (ZMQError, ValueError)):
			print(traceback.print_tb(e.__traceback__))
		else:
			pytest.fail(str(e))


def test_compression():
	"""
	Testing NetGear's real-time frame compression capabilities
	"""
	try:
		#open streams
		stream = VideoGear(source=return_testvideo_path()).start()
		#define client parameters
		options = {'compression_param':cv2.IMREAD_COLOR} #read color image 
		client = NetGear(pattern = 1, receive_mode = True, logging = True, **options)
		#define server parameters
		options = {'compression_format': '.jpg', 'compression_param':[cv2.IMWRITE_JPEG_OPTIMIZE, 20]} #JPEG compression
		server = NetGear(pattern = 1, logging = True, **options)
		#send over network
		while True:
			frame_server = stream.read()
			if frame_server is None: break
			server.send(frame_server)
			frame_client = client.recv()
		#clean resources
		stream.stop()
		server.close()
		client.close()
	except Exception as e:
		if isinstance(e, (ZMQError, ValueError)):
			print(traceback.print_tb(e.__traceback__))
		else:
			pytest.fail(str(e))
 


test_data_class = [
	(0,1, tempfile.gettempdir(), True),
	(1,1, tempfile.gettempdir(), False)]
@pytest.mark.parametrize('pattern, security_mech, custom_cert_location, overwrite_cert', test_data_class)
def test_secure_mode(pattern, security_mech, custom_cert_location, overwrite_cert):
	"""
	Testing NetGear's Secure Mode
	"""
	try:
		#open stream
		stream = VideoGear(source=return_testvideo_path()).start()

		#define security mechanism
		options = {'secure_mode': security_mech, 'custom_cert_location': custom_cert_location, 'overwrite_cert': overwrite_cert}
			
		#define params
		server = NetGear(pattern = pattern, logging = True, **options)
		client = NetGear(pattern = pattern, receive_mode = True, logging = True, **options)
		#initialize
		frame_server = None
		#select random input frame from stream
		i = 0
		while (i < random.randint(10, 100)):
			frame_server = stream.read()
			i+=1
		#check input frame is valid
		assert not(frame_server is None)
		#send and recv input frame
		server.send(frame_server)
		frame_client = client.recv()
		#clean resources
		stream.stop()
		server.close()
		client.close()
		#check if recieved frame exactly matches input frame
		assert np.array_equal(frame_server, frame_client)
	except Exception as e:
		if isinstance(e, (ZMQError, ValueError)):
			print(traceback.print_tb(e.__traceback__))
		else:
			pytest.fail(str(e))




@pytest.mark.parametrize('target_data', [[1,'string',['list']], {1:'apple', 2: 'cat'}])
def test_bidirectional_mode(target_data):
	"""
	Testing NetGear's Bidirectional Mode with different datatypes
	"""
	try:
		print('[LOG] Given Input Data: {}'.format(target_data))

		#open strem
		stream = VideoGear(source=return_testvideo_path()).start()
		#activate bidirectional_mode
		options = {'bidirectional_mode': True}
		#define params
		client = NetGear(receive_mode = True, logging = True, **options)
		server = NetGear(logging = True, **options)
		#get frame from stream
		frame_server = stream.read()
		assert not(frame_server is None)

		#sent frame and data from server to client
		server.send(frame_server, message = target_data)
		#client recieves the data and frame and send its data
		server_data, frame_client = client.recv(return_data = target_data)
		#server recieves the data and cycle continues
		client_data = server.send(frame_server, message = target_data)

		#clean resources
		stream.stop()
		server.close()
		client.close()

		#print data recieved at client-end and server-end
		print('[LOG] Data recieved at Server-end: {}'.format(server_data))
		print('[LOG] Data recieved at Client-end: {}'.format(client_data))
		
		#check if recieved frame exactly matches input frame
		assert np.array_equal(frame_server, frame_client)
		#check if client-end data exactly matches server-end data
		assert client_data == server_data
	except Exception as e:
		if isinstance(e, (ZMQError, ValueError)):
			print(traceback.print_tb(e.__traceback__))
		else:
			pytest.fail(str(e))




def test_multiserver_mode():
	"""
	Testing NetGear's Multi-Server Mode with three unique servers
	"""
	try:
		#open network stream
		stream = VideoGear(source=return_testvideo_path()).start()

		#define and activate Multi-Server Mode
		options = {'multiserver_mode': True}

		#define a single client
		client = NetGear(port = ['5556', '5557', '5558'], pattern = 1, receive_mode = True, logging = True, **options)
		#define client-end dict to save frames inaccordance with unique port 
		client_frame_dict = {}

		#define three unique server
		server_1 = NetGear(pattern = 1, port = '5556', logging = True, **options) #at port `5556`
		server_2 = NetGear(pattern = 1, port = '5557', logging = True, **options) #at port `5557`
		server_3 = NetGear(pattern = 1, port = '5558', logging = True, **options) #at port `5558`

		#generate a random input frame
		frame_server = None
		i = 0
		while (i < random.randint(10, 100)):
			frame_server = stream.read()
			i+=1
		#check if input frame is valid
		assert not(frame_server is None)

		#send frame from Server-1 to client and save it in dict
		server_1.send(frame_server)
		unique_address, frame = client.recv()
		client_frame_dict[unique_address] = frame
		#send frame from Server-2 to client and save it in dict
		server_2.send(frame_server)
		unique_address, frame = client.recv()
		client_frame_dict[unique_address] = frame
		#send frame from Server-3 to client and save it in dict
		server_3.send(frame_server)
		unique_address, frame = client.recv()
		client_frame_dict[unique_address] = frame
		
		#clean all resources
		stream.stop()
		server_1.close()
		server_2.close()
		server_3.close()
		client.close()

		#check if recieved frames from each unique server exactly matches input frame
		for key in client_frame_dict.keys():
			assert np.array_equal(frame_server, client_frame_dict[key])

	except Exception as e:
		if isinstance(e, (ZMQError, ValueError)):
			print(traceback.print_tb(e.__traceback__))
		else:
			pytest.fail(str(e)) 