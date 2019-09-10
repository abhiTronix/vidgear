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
from .helper import check_python_version
from .helper import generate_auth_certificates
import numpy as np
import time
import os
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
	Furthermore, NetGear currently supports three ZeroMQ messaging patterns: i.e zmq.PAIR, zmq.REQ and zmq.REP,and zmq.PUB,zmq.SUB whereas
	supported protocol are: 'tcp', 'upd', 'pgm', 'inproc', 'ipc'.

	Multi-Server Mode:  This mode in NetGear API can robustly handle multiple servers at once through exclusive Publish/Subscribe (zmq.PUB/zmq.SUB) 
						messaging pattern for seamless Live Streaming across various device at the same time. Each device new server on network is 
						identied using its unique port address. Also, when all the connected servers on the network get disconnected, the client 
						itself automatically exits too. This mode can be activated through`multiserver_mode` option boolean attribute during 
						netgear initialization easily.

	:param address(string): sets the valid network address of the server/client. Network addresses unique identifiers across the 
							network. Its default value of this parameter is based on mode it is working, 'localhost' for Send Mode
							and `*` for Receive Mode.

	:param port(string/dict/list): sets the valid network port of the server/client. A network port is a number that identifies one side 
							of a connection between two devices on network. It is used determine to which process or application 
							a message should be delivered. In Multi-Server Mode a unique port number must required at each server, and a
							list/tuple of port addresses of each connected server is required at clients end.

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
								2. zmq.PUB,zmq.SUB -> Publish/Subscribe is another classic pattern where senders of messages, called publishers, 
														do not program the messages to be sent directly to specific receivers, called subscribers. 
														Messages are published without the knowledge of what or if any subscriber of that knowledge exists.
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
			print('[LOG]: Threaded Queue Mode is enabled by default for NetGear.')
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
				print('[LOG]: Wrong pattern value, Defaulting to `zmq.PAIR`! Kindly refer Docs for more Information.')
		
		#check  whether user-defined messaging protocol is valid
		if protocol in ['tcp', 'upd', 'pgm', 'inproc', 'ipc']:
			pass
		else:
			# else default to `tcp` protocol
			protocol = 'tcp'
			if logging:
				#log it
				print('[LOG]: protocol is not valid or provided. Defaulting to `tcp` protocol!')

		#generate random device id
		self.id = ''.join(random.choice('0123456789ABCDEF') for i in range(5))

		self.msg_flag = 0 #handles connection flags
		self.msg_copy = False #handles whether to copy data
		self.msg_track = False #handles whether to track packets

		self.multiserver_mode = False #handles multiserver_mode state
		recv_filter = '' #user-defined filter to allow specific port/servers only in multiserver_mode

		valid_security_mech = {0:'Grasslands', 1:'StoneHouse', 2:'IronHouse'}
		self.secure_mode = 0
		auth_cert_dir = ''
		self.auth_publickeys_dir = ''
		self.auth_secretkeys_dir = ''
		overwrite_cert = False
		custom_cert_location = ''
		
		try: 
			#reformat dict
			options = {k.lower().strip(): v for k,v in options.items()}
			# assign values to global variables if specified and valid
			for key, value in options.items():

				if key == 'multiserver_mode' and isinstance(value, bool) and self.pattern == 2:
					self.multiserver_mode = value
				elif key == 'filter' and isinstance(value, str):
					recv_filter = value

				elif key == 'secure_mode' and isinstance(value,int) and (value in valid_security_mech):
					try:
						assert check_python_version() >= 3,"[ERROR]: ZMQ Security feature is not available with python version < 3.0."
						assert zmq.zmq_version_info() >= (4,0), "[ERROR]: ZMQ Security feature is not supported in libzmq version < 4.0."
						self.secure_mode = value
					except AssertionError as e:
						print(e)
				elif key == 'custom_cert_location' and isinstance(value,str):
					try:
						assert os.access(value, os.W_OK), "[ERROR]: Permission Denied!, cannot write ZMQ authentication certificates to '{}' directory!".format(value)
						assert not(os.path.isfile(value)), "[ERROR]: `custom_cert_location` value must be the path to a directory and not to a file!"
						custom_cert_location = os.path.abspath(value)
					except AssertionError as e:
						print(e)
				elif key == 'overwrite_cert' and isinstance(value,bool):
					overwrite_cert = value

				elif key == 'flag' and isinstance(value, int):
					self.msg_flag = value
				elif key == 'copy' and isinstance(value, bool):
					self.msg_copy = value
				elif key == 'track' and isinstance(value, bool):
					self.msg_track = value
				else:
					pass
		except Exception as e:
			# Catch if any error occurred
			if logging:
				print(e)


		if self.secure_mode:

			#import libraries
			import zmq.auth
			from zmq.auth.thread import ThreadAuthenticator

			if logging and overwrite_cert: print('[WARNING]: Overwriting ZMQ Authentication certificates over previous ones!')

			try:
				if custom_cert_location:
					if os.path.isdir(custom_cert_location):
						(auth_cert_dir, self.auth_secretkeys_dir, self.auth_publickeys_dir) = generate_auth_certificates(custom_cert_location, overwrite = overwrite_cert)
					else:
						raise ValueError("[ERROR]: Invalid `custom_cert_location` value!")
				else:
					from os.path import expanduser
					(auth_cert_dir, self.auth_secretkeys_dir, self.auth_publickeys_dir) = generate_auth_certificates(os.path.join(expanduser("~"),".vidgear"), overwrite = overwrite_cert)

				if logging:
					print('[LOG]: `{}` is the default location for storing ZMQ authentication certificates/keys.'.format(auth_cert_dir))

			except Exception as e:
				# Catch if any error occurred
				print(e)
				self.secure_mode = 0
				print('[WARNING]: ZMQ Security Mechanism is disabled for this connection!')
		else:
			if logging: print('[LOG]: ZMQ Security Mechanism is disabled for this connection!')

			
		# enable logging if specified
		self.logging = logging

		# initialize termination flag
		self.terminate = False

		# initialize exit_loop flag
		self.exit_loop = False

		# initialize and assign receive mode to global variable
		self.receive_mode = receive_mode

		# define messaging context instance
		self.msg_context = zmq.Context.instance() # TODO changed


		#check whether `receive_mode` is enabled by user
		if receive_mode:

			# if does than define connection address
			if address is None: #define address
				address = 'localhost' if self.multiserver_mode else '*' 
			
			#check if multiserver_mode is enabled
			if self.multiserver_mode:
				# check if unique server port address list/tuple is assigned or not in multiserver_mode
				if port is None or not isinstance(port, (tuple, list)):
					# raise error if not
					raise ValueError('Incorrect port value! Kindly provide a list/tuple of ports at Receiver-end while Multi-Server mode is enabled. For more information refer VidGear docs.')
				else:
					#otherwise log it
					print('[LOG]: Enabling Multi-Server Mode at PORTS: {}!'.format(port))
				#create port address buffer for keeping track of incoming server's port
				self.port_buffer = []
			else:
				# otherwise assign local port address
				port = '5555'

			try:
				if self.secure_mode > 0 and not(self.multiserver_mode):
					# Start an authenticator for this context.
					auth = ThreadAuthenticator(self.msg_context)
					auth.start()
					auth.allow(str(address))
					# Tell authenticator to use the certificate in a directory
					if self.secure_mode == 2:
						auth.configure_curve(domain='*', location=self.auth_publickeys_dir)
					else:
						# Tell the authenticator how to handle CURVE requests
						auth.configure_curve(domain='*', location=zmq.auth.CURVE_ALLOW_ANY)

				# initialize and define thread-safe messaging socket
				self.msg_socket = self.msg_context.socket(msg_pattern[1])

				if self.multiserver_mode:
					#if multiserver_mode is enabled assign port addresses to zmq socket
					for pt in port:
						if self.secure_mode > 0: 
							client_secret_file = os.path.join(self.auth_secretkeys_dir, "client.key_secret")
							client_public, client_secret = zmq.auth.load_certificate(client_secret_file)
							self.msg_socket.curve_secretkey = client_secret
							self.msg_socket.curve_publickey = client_public

							server_public_file = os.path.join(self.auth_publickeys_dir, "server.key")
							server_public, _ = zmq.auth.load_certificate(server_public_file)
							# The client must know the server's public key to make a CURVE connection.
							self.msg_socket.curve_serverkey = server_public
						# connect socket to given server protocol, address and ports
						self.msg_socket.connect(protocol+'://' + str(address) + ':' + str(pt))
						self.msg_socket.setsockopt(zmq.LINGER, 0)
					# define socket options
					self.msg_socket.setsockopt_string(zmq.SUBSCRIBE, recv_filter)
				else:
					# otherwise bind socket to given protocol, address and port normally

					if self.secure_mode > 0:
						server_secret_file = os.path.join(self.auth_secretkeys_dir, "server.key_secret")
						server_public, server_secret = zmq.auth.load_certificate(server_secret_file)
						self.msg_socket.curve_secretkey = server_secret
						self.msg_socket.curve_publickey = server_public
						self.msg_socket.curve_server = True  # must come before bind

					self.msg_socket.bind(protocol+'://' + str(address) + ':' + str(port))
					# define exclusive socket options for patterns
					if self.pattern == 2:
						self.msg_socket.setsockopt_string(zmq.SUBSCRIBE,'')
					else:
						self.msg_socket.setsockopt(zmq.LINGER, 0)					

			except Exception as e:
				# otherwise raise value error if errored
				if self.multiserver_mode:
					raise ValueError('[ERROR]: Multi-Server Mode, failed to connect to ports: {} with pattern: {}! Kindly recheck all parameters.'.format( str(port), pattern))
				else:
					raise ValueError('[ERROR]: Failed to bind address: {} and pattern: {}! Kindly recheck all parameters.'.format((protocol+'://' + str(address) + ':' + str(port)), pattern))
			
			# initialize and start threading instance
			self.thread = Thread(target=self.update, args=())
			self.thread.daemon = True
			self.thread.start()

			if logging:
				#log progress
				print('[LOG]: Successfully Binded to address: {}.'.format(protocol+'://' + str(address) + ':' + str(port)))
				print('[LOG]: Multi-threaded Receive Mode is enabled Successfully!')
				print('[LOG]: This device Unique ID is {}.'.format(self.id))
				print('[LOG]: Receive Mode is activated successfully!')
				if self.secure_mode: print('[LOG]: Successfully enabled ZMQ Security Mechanism: `{}` for this connection.'.format(valid_security_mech[self.secure_mode]))
		else:

			#otherwise default to `Send Mode`

			if address is None: #define address
				address = '*' if self.multiserver_mode else 'localhost'

			#check if multiserver_mode is enabled
			if self.multiserver_mode:
				# check if unique server port address is assigned or not in multiserver_mode
				if port is None:
					#raise error is not
					raise ValueError('Incorrect port value! Kindly provide a unique & valid port value at Server-end while Multi-Server mode is enabled. For more information refer VidGear docs.')
				else:
					#otherwise log it
					print('[LOG]: Enabling Multi-Server Mode at PORT: {} on this device!'.format(port))
					#assign value to global variable
					self.port = port
			else:
				port = 5555  #define port normally
				
			try:
				if self.secure_mode > 0 and self.multiserver_mode:
					# Start an authenticator for this context.
					auth = ThreadAuthenticator(self.msg_context)
					auth.start()
					auth.allow(str(address))
					# Tell authenticator to use the certificate in a directory
					if self.secure_mode == 2:
						auth.configure_curve(domain='*', location=self.auth_publickeys_dir)
					else:
						# Tell the authenticator how to handle CURVE requests
						auth.configure_curve(domain='*', location=zmq.auth.CURVE_ALLOW_ANY)

				# initialize and define thread-safe messaging socket
				self.msg_socket = self.msg_context.socket(msg_pattern[0])

				if self.multiserver_mode:
					if self.secure_mode > 0: 
						server_secret_file = os.path.join(self.auth_secretkeys_dir, "server.key_secret")
						server_public, server_secret = zmq.auth.load_certificate(server_secret_file)
						self.msg_socket.curve_secretkey = server_secret
						self.msg_socket.curve_publickey = server_public
						self.msg_socket.curve_server = True  # must come before bind
					# connect socket to protocol, address and a unique port if multiserver_mode is activated
					self.msg_socket.bind(protocol+'://' + str(address) + ':' + str(port))
				else:
					if self.pattern == 1:
						# if pattern is 1, define additional flags
						self.msg_socket.REQ_RELAXED = True
						self.msg_socket.REQ_CORRELATE = True

					if self.secure_mode > 0: 
						client_secret_file = os.path.join(self.auth_secretkeys_dir, "client.key_secret")
						client_public, client_secret = zmq.auth.load_certificate(client_secret_file)
						self.msg_socket.curve_secretkey = client_secret
						self.msg_socket.curve_publickey = client_public

						server_public_file = os.path.join(self.auth_publickeys_dir, "server.key")
						server_public, _ = zmq.auth.load_certificate(server_public_file)

						# The client must know the server's public key to make a CURVE connection.
						self.msg_socket.curve_serverkey = server_public

					# connect socket to given protocol, address and port
					self.msg_socket.connect(protocol+'://' + str(address) + ':' + str(port))

					# define socket options
					self.msg_socket.setsockopt(zmq.LINGER, 0)

			except Exception as e:
				# otherwise raise value error
				raise ValueError('Failed to connect address: {} and pattern: {}! Kindly recheck all parameters.'.format((protocol+'://' + str(address) + ':' + str(port)), pattern))

			if logging:
				#log progress
				print('[LOG]: Successfully connected to address: {}.'.format(protocol+'://' + str(address) + ':' + str(port)))
				print('[LOG]: This device Unique ID is {}.'.format(self.id))
				print('[LOG]: Send Mode is successfully activated and ready to send data!')
				if self.secure_mode: print('[LOG]: Successfully enabled ZMQ Security Mechanism: `{}` for this connection.'.format(valid_security_mech[self.secure_mode]))

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

			# check if terminate_flag` received
			if msg_json['terminate_flag']:
				#if multiserver_mode is enabled 
				if self.multiserver_mode:
					# check from which ports signal is received
					self.port_buffer.remove(msg_json['port'])
					#if termination signal received from all servers then exit client.
					if not self.port_buffer:
						print('Termination signal received from all Servers!!!')
						self.terminate = True #termination
					continue
				else:
					if self.pattern == 1:
						# if pattern is 1, then send back server the info about termination
						self.msg_socket.send_string('Termination signal received from server!')
					#termination
					self.terminate = msg_json['terminate_flag']
					continue

			try:
				#check if pattern is same at both server's and client's end.
				assert int(msg_json['pattern']) == self.pattern
			except AssertionError as e:
				#otherwise raise error and exit 
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

			if self.multiserver_mode:
				# check if multiserver_mode

				#save the unique port addresses
				if not msg_json['port'] in self.port_buffer:
					self.port_buffer.append(msg_json['port'])
			
				#extract if any message from server and display it
				if msg_json['message']:
					self.queue.append((msg_json['port'], msg_json['message'], frame))
				else:
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

		#check if multiserver_mode is activated
		if self.multiserver_mode:
			# prepare the exclusive json dict and assign values with unique port
			msg_dict = dict(terminate_flag = exit_flag,
							port = self.port,
							pattern = str(self.pattern),
							message = message if not(message is None) else '',
							dtype = str(frame.dtype),
							shape = frame.shape)
		else:
			# otherwise prepare normal json dict and assign values
			msg_dict = dict(terminate_flag = exit_flag,
							message = message if not(message is None) else '',
							pattern = str(self.pattern),
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
			# otherwise in `send_mode`, inform client(s) that the termination is reached
			self.terminate = True
			#check if multiserver_mode
			if self.multiserver_mode:
				# send termination flag to client with its unique port
				term_dict = dict(terminate_flag = True, port = self.port)
			else:
				# otherwise send termination flag to client
				term_dict = dict(terminate_flag = True)
			self.msg_socket.send_json(term_dict)
			# properly close the socket
			self.msg_socket.close()