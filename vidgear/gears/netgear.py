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

# import the necessary packages
from threading import Thread
from pkg_resources import parse_version
import numpy as np
import time
import random


try:
	# import OpenCV Binaries
	import cv2
	# check whether OpenCV Binaries are 3.x+
	if parse_version(cv2.__version__) >= parse_version('3'):
		pass
	else:
		raise ImportError('OpenCV library version >= 3.0 is only supported by this library')
except ImportError as error:
	raise ImportError('Failed to detect OpenCV executables, install it with `pip install opencv-contrib-python` command.')



class NetGear:

	"""
	NetGear is exclusively designed to transfer video frames synchronously between interconnecting systems over the network in real-time. 
	This is achieved by implementing a high-level wrapper around PyZmQ python library that contains python bindings for ZeroMQ - a 
	high-performance asynchronous distributed messaging library that aim to be used in distributed or concurrent applications. 
	It provides a message queue, but unlike message-oriented middleware, a ZeroMQ system can run without a dedicated message broker. 
	Furthermore, NetGear currently supports two ZeroMQ messaging patterns: i.e zmq.PAIR and zmq.REQ and zmq.REP and the supported 
	protocol are: 'tcp', 'upd', 'pgm', 'inproc', 'ipc'.

	Threaded Queue Mode => Sequentially adds and releases frames to/from deque and handles overflow of this queue. It utilizes 
	deques that support thread-safe, memory efficient appends and pops from either side of the deque with approximately the 
	same O(1) performance in either direction.  

	:param address(string): sets the valid network address of the server/client. Network addresses unique identifiers across the 
							network. Its default value of this parameter is based on mode it is working, 'localhost' for Send Mode
							and `*` for Receive Mode.

	:param port(string): sets the valid network port of the server/client. A network port is a number that identifies one side 
							of a connection between two devices on network. It is used determine to which process or application 
							a message should be delivered. Its default value is `5555` .

	:param protocol(string): sets the valid messaging protocol between server and client. A network protocol is a set of established rules
							 that dictates how to format, transmit and receive data so computer network devices - from servers and 
							 routers to endpoints - can communicate regardless of the differences in their underlying infrastructures, 
							 designs or standards. Supported protocol are: 'tcp', 'upd', 'pgm', 'inproc', 'ipc'. Its default value is `tcp`.

	:param pattern(int): sets the supported messaging pattern(flow of communication) between server and client. Messaging patterns are network 
							oriented architectural pattern that describes the flow of communication between interconnecting systems.
							vidgear provides access to ZeroMQ's pre-optimized sockets which enables you to take advantage of these patterns.
							The supported patterns are:
								0: zmq.PAIR -> In this, the communication is bidirectional. There is no specific state stored within the socket. 
												There can only be one connected peer.The server listens on a certain port and a client connects to it.
								1. zmq.REQ/zmq.REP -> In this, ZMQ REQ sockets can connect to many servers. The requests will be
														interleaved or distributed to both the servers. socket zmq.REQ will block 
														on send unless it has successfully received a reply back and socket zmq.REP 
														will block on recv unless it has received a request.
							Its default value is `0`(i.e zmq.PAIR).

	:param (boolean) receive_mode: set this flag to select the Netgear's Mode of operation. This basically activates `Receive Mode`(if True) and `Send Mode`(if False). 
									Furthermore `recv()` function will only works when this flag is enabled(i.e. `Receive Mode`) and `send()` function will only works 
									when this flag is disabled(i.e.`Send Mode`). Checkout VidGear docs for usage details.
									Its default value is False(i.e. Send Mode is activated by default).

	:param **options(dict): can be used to pass parameters to NetGear Class. 
							/This attribute provides the flexibility to manipulate ZeroMQ input parameters 
							/directly. Checkout vidgear docs for usage details.

	:param (boolean) logging: set this flag to enable/disable error logging essential for debugging. Its default value is False.

	"""
	
	def __init__(self, address = None, port = None, protocol = None,  pattern = 0, receive_mode = False, logging = False, **options):

		try:
			#import PyZMQ library
			import zmq
			#import ZMQError
			from zmq.error import ZMQError
			#assign values to global variable
			self.zmq = zmq
			self.ZMQError = ZMQError
		except ImportError as error:
			#raise error
			raise ImportError('pyzmq python library not installed. Kindly install it with `pip install pyzmq` command.')

		#log and enable threaded queue mode
		if logging:
			print('[LOG]:  Threaded Mode is enabled by default for NetGear.')
		#import deque
		from collections import deque
		#define deque and assign it to global var
		self.queue = deque(maxlen=96) #max len 96 to check overflow

		#define valid messaging pattern `0`: zmq.PAIR, `1`:(zmq.REQ,zmq.REP), and `1`:(zmq.SUB,zmq.PUB)
		valid_messaging_patterns = {0:(zmq.PAIR,zmq.PAIR), 1:(zmq.REQ,zmq.REP), 2:(zmq.PUB,zmq.SUB)}

		# initialize messaging pattern
		msg_pattern = None
		# check whether user-defined messaging pattern is valid
		if isinstance(pattern, int) and pattern in valid_messaging_patterns:
			#assign value 
			msg_pattern = valid_messaging_patterns[pattern]
			self.pattern = pattern #add it to global variable for further use
		else:
			#otherwise default to 0:`zmq.PAIR`
			self.pattern = 0
			msg_pattern = valid_messaging_patterns[self.pattern]
			if logging:
				#log it
				print('[LOG]:  Wrong pattern, Defaulting to `zmq.PAIR`! Kindly refer Docs for more Information.')
		
		#check  whether user-defined messaging protocol is valid
		if protocol in ['tcp', 'upd', 'pgm', 'inproc', 'ipc']:
			pass
		else:
			# else default to `tcp` protocol
			protocol = 'tcp'
			if logging:
				#log it
				print('[LOG]:  Protocol is not valid or provided. Defaulting to `tcp` protocol! Kindly refer Docs for more Information.')

		#generate random device id
		self.id = ''.join(random.choice('0123456789ABCDEF') for i in range(5))

		self.msg_flag = 0 #handles connection flags
		self.msg_copy = False #handles whether to copy data
		self.msg_track = False #handles whether to track packets

		self.multiserver_mode = False
		recv_filter = ''
		
		try: 
			#reformat dict
			options = {k.lower().strip(): v for k,v in options.items()}
			# apply attributes to source if specified and valid
			for key, value in options.items():
				if key == 'multiserver_mode' and isinstance(value, bool) and self.pattern == 2:
					self.multiserver_mode = value
				elif key == 'filter' and isinstance(value, str):
					recv_filter = value
				elif key == 'flag' and isinstance(value, str):
					self.msg_flag = getattr(zmq, value)
				elif key == 'copy' and isinstance(value, bool):
					self.msg_copy = value
				elif key == 'track' and isinstance(value, bool):
					self.msg_track = value
				else:
					pass
		except Exception as e:
			# Catch if any error occurred
			if logging:
				print('[Exception]: '+ e)
			
		# enable logging if specified
		self.logging = logging

		# initialize termination flag
		self.terminate = False

		# initialize exit_loop flag
		self.exit_loop = False

		# initialize and assign receive mode to global variable
		self.receive_mode = receive_mode

		# define messaging context
		self.msg_context = zmq.Context()

		#check whether `receive_mode` is enabled by user
		if receive_mode:

			# if does than define connection address and port
			if address is None: #define address
				address = 'localhost' if self.multiserver_mode else '*' 
			
			if self.multiserver_mode:
				if port is None or not isinstance(port, (tuple, list)):
					raise ValueError('Incorrect port value! Kindly provide a list/tuple of ports at Receiver-end while Multi-Server mode is enabled. For more information refer VidGear docs.')
				else:
					print('[LOG]: Enabling Multi-Server Mode at PORTS: {}!'.format(port))
				self.port_buffer = []
			else:
				port = '5555'

			try:
				# initialize and define thread-safe messaging socket
				self.msg_socket = self.msg_context.socket(msg_pattern[1])

				if self.multiserver_mode:
					for pt in port:
						# connect socket to given protocol, address and port
						self.msg_socket.connect(protocol+'://' + str(address) + ':' + str(pt))
						self.msg_socket.setsockopt(zmq.LINGER, 0)
					# define socket options
					self.msg_socket.setsockopt_string(zmq.SUBSCRIBE, recv_filter)
				else:
					# bind socket to given protocol, address and port
					self.msg_socket.bind(protocol+'://' + str(address) + ':' + str(port))
					# define socket options
					if self.pattern == 2:
						self.msg_socket.setsockopt_string(zmq.SUBSCRIBE,'')
					else:
						self.msg_socket.setsockopt(zmq.LINGER, 0)					

			except Exception as e:
				# otherwise raise value error
				raise ValueError('Failed to bind address: {} and pattern: {}! Kindly recheck all parameters.'.format((protocol+'://' + str(address) + ':' + str(port)), pattern))
			
			# initialize and start threading instance
			self.thread = Thread(target=self.update, args=())
			self.thread.daemon = True
			self.thread.start()

			if logging:
				#log it
				print('[LOG]:  Successfully Binded to address: {}.'.format(protocol+'://' + str(address) + ':' + str(port)))
				print('[LOG]:  Multi-threaded Receive Mode is enabled Successfully!')
				print('[LOG]:  This device Unique ID is {}.'.format(self.id))
				print('[LOG]:  Receive Mode is activated successfully!')
		else:

			#otherwise default to `Send Mode

			if address is None: #define address
				address = '*' if self.multiserver_mode else 'localhost'

			if self.multiserver_mode:
				if port is None:
					raise ValueError('Incorrect port value! Kindly provide a unique & valid port value at Server-end while Multi-Server mode is enabled. For more information refer VidGear docs.')
				else:
					print('[LOG]: Enabling Multi-Server Mode at PORT: {} on this device!'.format(port))
					self.port = port
			else:
				port = 5555  #define port
				
			try:
				# initialize and define thread-safe messaging socket
				self.msg_socket = self.msg_context.socket(msg_pattern[0])

				if self.multiserver_mode:
					# connect socket to given protocol, address and port
					self.msg_socket.bind(protocol+'://' + str(address) + ':' + str(port))
				else:
					if self.pattern == 1:
						# if pattern is 1, define additional flags
						self.msg_socket.REQ_RELAXED = True
						self.msg_socket.REQ_CORRELATE = True

					# connect socket to given protocol, address and port
					self.msg_socket.connect(protocol+'://' + str(address) + ':' + str(port))

					# define socket options
					self.msg_socket.setsockopt(zmq.LINGER, 0)

			except Exception as e:
				# otherwise raise value error
				raise ValueError('Failed to connect address: {} and pattern: {}! Kindly recheck all parameters.'.format((protocol+'://' + str(address) + ':' + str(port)), pattern))

			if logging:
				#log it
				print('[LOG]:  Successfully connected to address: {}.'.format(protocol+'://' + str(address) + ':' + str(port)))
				print('[LOG]:  This device Unique ID is {}.'.format(self.id))
				print('[LOG]:  Send Mode is successfully activated and ready to send data!')


	def update(self):
		"""
		Updates recovered frames from messaging network to the queue
		"""
		# initialize frame variable
		frame = None
		# keep looping infinitely until the thread is terminated
		while not self.exit_loop:

			# check if global termination_flag is enabled  
			if self.terminate:
				# check whether there is still frames in queue before breaking out
				if len(self.queue)>0:
					continue
				else:
					break

			#check queue buffer for overflow
			if len(self.queue) < 96:
				pass
			else:
				#stop iterating if overflowing occurs
				time.sleep(0.000001)
				continue
			# extract json data out of socket
			msg_json = self.msg_socket.recv_json(flags=self.msg_flag)

			# check if terminate_flag` is enabled in json
			if msg_json['terminate_flag']:
				if self.multiserver_mode:
					self.port_buffer.remove(msg_json['port'])
					if not self.port_buffer:
						print('Termination signal received from all Servers!!!')
						self.terminate = True
					continue
				else:
					if self.pattern == 1:
						# if pattern is 1, then send back server the info about termination
						self.msg_socket.send_string('Termination signal received from server!')
					#assign values to global termination flag
					self.terminate = msg_json['terminate_flag']
					continue

			try:
				assert int(msg_json['pattern']) == self.pattern
			except (AssertionError) as e:
				raise ValueError("Messaging pattern on the Server-end & Client-end must a valid pairs! Kindly refer VidGear docs.")
				self.terminate = True
				continue


			# extract array from socket
			msg_data = self.msg_socket.recv(flags=self.msg_flag, copy=self.msg_copy, track=self.msg_track)

			if self.pattern != 2:
				# send confirmation message to server for debugging
				self.msg_socket.send_string('Data received on device: {} !'.format(self.id))

			# recover frame from array buffer
			frame_buffer = np.frombuffer(msg_data, dtype=msg_json['dtype'])
			# reshape frame
			frame = frame_buffer.reshape(msg_json['shape'])
			#extract if any message from server and display it
			if msg_json['message']:
				print(msg_json['message'])

			if self.multiserver_mode: 
				if not msg_json['port'] in self.port_buffer:
					self.port_buffer.append(msg_json['port'])
				# append recovered unique port and frame to queue
				self.queue.append((msg_json['port'],frame))
			else:
				# append recovered frame to queue
				self.queue.append(frame)
		# finally properly close the socket
		self.msg_socket.close()



	def recv(self):
		"""
		return the recovered frame
		"""
		# check whether `receive mode` is activated
		if self.receive_mode:
			pass
		else:
			#otherwise raise value error and exit
			raise ValueError('recv() function cannot be used while receive_mode is disabled. Kindly refer vidgear docs!')
			self.terminate = True
		# check whether or not termination flag is enabled
		while not self.terminate:
			# check if queue is empty
			if len(self.queue)>0:
				return self.queue.popleft()
			else:
				continue
		# otherwise return NoneType
		return None



	def send(self, frame, message = None):
		"""
		send the frames over the messaging network

		:param frame(ndarray): frame array to send
		:param message(string/int): additional message for the client(s)  
		"""
		# check whether `receive_mode` is disabled
		if not self.receive_mode:
			pass
		else:
			#otherwise raise value error and exit
			raise ValueError('send() function cannot be used while receive_mode is enabled. Kindly refer vidgear docs!')
			self.terminate = True

		# define exit_flag and assign value
		exit_flag = True if (frame is None or self.terminate) else False
		#check whether exit_flag is False
		if not exit_flag:
			#check whether the incoming frame is contiguous
			if frame.flags['C_CONTIGUOUS']:
				pass
			else:
				#otherwise make it contiguous
				frame = np.ascontiguousarray(frame, dtype=frame.dtype)

		if self.multiserver_mode:
			# prepare the json dict and assign values with unique port
			msg_dict = dict(terminate_flag = exit_flag,
							port = self.port,
							pattern = str(self.pattern),
							message = message if not(message is None) else '',
							dtype = str(frame.dtype),
							shape = frame.shape)
		else:
			# prepare the json dict and assign values
			msg_dict = dict(terminate_flag = exit_flag,
							message = message if not(message is None) else '',
							pattern = self.pattern,
							dtype = str(frame.dtype),
							shape = frame.shape)

		# send the json dict
		self.msg_socket.send_json(msg_dict, self.msg_flag|self.zmq.SNDMORE)
		# send the frame array with correct flags
		self.msg_socket.send(frame, flags = self.msg_flag, copy=self.msg_copy, track=self.msg_track)
		# wait for confirmation

		if self.pattern != 2:
			if self.logging:
				# log confirmation 
				print(self.msg_socket.recv())
			else:
				# otherwise be quiet
				self.msg_socket.recv()



	def close(self):
		"""
		Terminates the NetGear processes safely
		"""
		if self.logging:
			#log it
			print(' \n[LOG]: Terminating various {} Processes \n'.format('Receive Mode' if self.receive_mode else 'Send Mode'))
		#  whether `receive_mode` is enabled or not
		if self.receive_mode:
			# indicate that process should be terminated
			self.terminate = True
			# check whether queue mode is empty
			if not(self.queue is None):
				self.queue.clear()
			# call immediate termination
			self.exit_loop = True
			# wait until stream resources are released (producer thread might be still grabbing frame)
			if self.thread is not None: 
				self.thread.join()
				self.thread = None
				#properly handle thread exit
		else:
			# otherwise indicate that the thread should be terminated
			self.terminate = True
			if self.multiserver_mode:
				# send termination flag to client
				term_dict = dict(terminate_flag = True, port = self.port)
			else:
				# send termination flag to client
				term_dict = dict(terminate_flag = True)
			self.msg_socket.send_json(term_dict)
			# properly close the socket
			self.msg_socket.close()