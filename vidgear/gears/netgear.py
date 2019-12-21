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

# import the necessary packages
from threading import Thread
from pkg_resources import parse_version
from .helper import generate_auth_certificates
from collections import deque
import numpy as np
import time
import os
import random
import logging as log


try:
	# import OpenCV Binaries
	import cv2
	# check whether OpenCV Binaries are 3.x+
	if parse_version(cv2.__version__) < parse_version('3'):
		raise ImportError('[NetGear:ERROR] :: OpenCV API version >= 3.0 is only supported by this library.')
except ImportError as error:
	raise ImportError('[NetGear:ERROR] :: Failed to detect correct OpenCV executables, install it with `pip3 install opencv-python` command.')



class NetGear:

	"""
	NetGear is exclusively designed to transfer video frames synchronously between interconnecting systems over the network in real-time. 
	This is achieved by implementing a high-level wrapper around PyZmQ python library that contains python bindings for ZeroMQ - a 
	high-performance asynchronous distributed messaging library that aim to be used in distributed or concurrent applications. 
	It provides a message queue, but unlike message-oriented middleware, a ZeroMQ system can run without a dedicated message broker. 
	Furthermore, NetGear currently supports three ZeroMQ messaging patterns: i.e zmq.PAIR, zmq.REQ and zmq.REP,and zmq.PUB,zmq.SUB whereas
	supported protocol are: 'tcp', 'upd', 'pgm', 'inproc', 'ipc'.

	Multi-Server Mode:  This mode enables NetGear API to robustly handle multiple servers at once through exclusive Publish/Subscribe (zmq.PUB/zmq.SUB) 
						messaging pattern for seamless Live Streaming across various device at the same time. Each device new server on network is 
						identified using its unique port address. Also, when all the connected servers on the network get disconnected, the client 
						itself automatically exits too. This mode can be activated through`multiserver_mode` option boolean attribute during 
						NetGear API initialization easily.

	Secure Mode: This mode provides secure ZeroMQ's Security Layers for NetGear API that enables strong encryption on data, and (as far as we know) unbreakable 
				 authentication between the Server and the Client with the help of custom auth certificates/keys. It's default value is `Grassland`:0 => which means no
				 security at all. 

				 This mode supports the two most powerful ZMQ security mechanisms:

				 * `StoneHouse`: 1 => which switches to the "CURVE" security mechanism, giving us strong encryption on data, and unbreakable authentication. 
									  Stonehouse is the minimum you would use over public networks and assures clients that they are speaking to an authentic server while allowing any client 
									  to connect. It is less secure but at the same time faster than IronHouse security mechanism.

				 * `IronHouse`: 2 => which extends Stonehouse with client public key authentication. This is the strongest security model ZMQ have today, 
										protecting against every attack we know about, except end-point attacks (where an attacker plants spyware on a machine 
										to capture data before it's encrypted, or after it's decrypted). IronHouse enhanced security comes at a price of additional latency.


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

	:param **options(dict): can be used to pass flexible parameters to NetGear API. 
							This attribute provides the flexibility to manipulate ZeroMQ internal parameters 
							directly. Checkout vidgear docs for usage details.

	:param (boolean) logging: set this flag to enable/disable error logging essential for debugging. Its default value is False.

	"""
	
	def __init__(self, address = None, port = None, protocol = None,  pattern = 0, receive_mode = False, logging = False, **options):

		try:
			#import PyZMQ library
			import zmq
			#import ZMQError
			from zmq.error import ZMQError
			#assign values to global variable for further use
			self.__zmq = zmq
			self.__ZMQError = ZMQError
		except ImportError as error:
			#raise error
			raise ImportError('[NetGear:ERROR] :: pyzmq python library not installed. Kindly install it with `pip install pyzmq` command.')

		# enable logging if specified
		self.__logging = False
		self.__logger = log.getLogger('NetGear')
		if logging: self.__logging = logging

		#define valid messaging patterns => `0`: zmq.PAIR, `1`:(zmq.REQ,zmq.REP), and `1`:(zmq.SUB,zmq.PUB)
		valid_messaging_patterns = {0:(zmq.PAIR,zmq.PAIR), 1:(zmq.REQ,zmq.REP), 2:(zmq.PUB,zmq.SUB)}

		# initialize messaging pattern
		msg_pattern = None
		# check whether user-defined messaging pattern is valid
		if isinstance(pattern, int) and pattern in valid_messaging_patterns:
			#assign value 
			msg_pattern = valid_messaging_patterns[pattern]
			self.__pattern = pattern #add it to global variable for further use
		else:
			#otherwise default to 0:`zmq.PAIR`
			self.__pattern = 0
			msg_pattern = valid_messaging_patterns[self.__pattern]
			#log it
			if self.__logging: self.__logger.warning('Wrong pattern value, Defaulting to `zmq.PAIR`! Kindly refer Docs for more Information.')
		
		#check  whether user-defined messaging protocol is valid
		if not(protocol in ['tcp', 'udp',  'pgm', 'epgm', 'inproc', 'ipc']):
			# else default to `tcp` protocol
			protocol = 'tcp'
			#log it
			if self.__logging: self.__logger.warning('protocol is not valid or provided. Defaulting to `tcp` protocol!')

		#generate random device id
		self.__id = ''.join(random.choice('0123456789ABCDEF') for i in range(5))

		self.__msg_flag = 0 #handles connection flags
		self.__msg_copy = True #handles whether to copy data
		self.__msg_track = False #handles whether to track packets

		self.__multiserver_mode = False #handles multiserver_mode state
		recv_filter = '' #user-defined filter to allow specific port/servers only in multiserver_mode

		#define bi-directional data transmission mode
		self.__bi_mode = False #handles bi_mode state
		self.__bi_data = None #handles return data

		#define valid  ZMQ security mechanisms => `0`: Grasslands, `1`:StoneHouse, and `1`:IronHouse
		valid_security_mech = {0:'Grasslands', 1:'StoneHouse', 2:'IronHouse'}
		self.__secure_mode = 0 #handles ZMQ security layer status
		auth_cert_dir = '' #handles valid ZMQ certificates dir 
		self.__auth_publickeys_dir = '' #handles valid ZMQ public certificates dir
		self.__auth_secretkeys_dir = '' #handles valid ZMQ private certificates dir
		overwrite_cert = False #checks if certificates overwriting allowed
		custom_cert_location = '' #handles custom ZMQ certificates path

		#handle force socket termination if there's latency in network
		self.__force_close = False

		#define stream compression handlers
		self.__compression = '' #disabled by default
		self.__compression_params = None
		
		#reformat dict
		options = {k.lower().strip(): v for k,v in options.items()}

		# assign values to global variables if specified and valid
		for key, value in options.items():
			if key == 'multiserver_mode' and isinstance(value, bool):
				if pattern > 0:
					# multi-server mode
					self.__multiserver_mode = value
				else:
					self.__multiserver_mode = False
					self.__logger.critical('Multi-Server is disabled!')
					raise ValueError('[NetGear:ERROR] :: `{}` pattern is not valid when Multi-Server Mode is enabled. Kindly refer Docs for more Information.'.format(pattern))
			elif key == 'filter' and isinstance(value, str):
				#custom filter in multi-server mode
				recv_filter = value

			elif key == 'secure_mode' and isinstance(value,int) and (value in valid_security_mech):
				#secure mode 
				try:
					assert zmq.zmq_version_info() >= (4,0), "[NetGear:ERROR] :: ZMQ Security feature is not supported in libzmq version < 4.0."
					self.__secure_mode = value
				except Exception as e:
					self.__logger.exception(str(e))
			elif key == 'custom_cert_location' and isinstance(value,str):
				# custom auth certificates path
				try:
					assert os.access(value, os.W_OK), "[NetGear:ERROR] :: Permission Denied!, cannot write ZMQ authentication certificates to '{}' directory!".format(value)
					assert not(os.path.isfile(value)), "[NetGear:ERROR] :: `custom_cert_location` value must be the path to a directory and not to a file!"
					custom_cert_location = os.path.abspath(value)
				except Exception as e:
					self.__logger.exception(str(e))
			elif key == 'overwrite_cert' and isinstance(value,bool):
				# enable/disable auth certificate overwriting
				overwrite_cert = value

			# handle encoding and decoding if specified
			elif key == 'compression_format' and isinstance(value,str) and value.lower().strip() in ['.jpg', '.jpeg', '.bmp', '.png']: #few are supported
				# enable encoding
				if not(receive_mode): self.__compression = value.lower().strip()
			elif key == 'compression_param':
				# specify encoding/decoding params
				if receive_mode and isinstance(value, int):
					self.__compression_params = value
					if self.__logging: self.__logger.debug("Decoding flag: {}.".format(value))
				elif not(receive_mode) and isinstance(value, (list,tuple)):
					if self.__logging: self.__logger.debug("Encoding parameters: {}.".format(value))
					self.__compression_params = list(value)
				else:	
					if self.__logging: self.__logger.warning("Invalid compression parameters: {} skipped!".format(value))
					self.__compression_params = cv2.IMREAD_COLOR if receive_mode else [] # skip to defaults

			# enable bi-directional data transmission if specified
			elif key == 'bidirectional_mode' and isinstance(value, bool):
				# check if pattern is valid
				if pattern < 2:
					self.__bi_mode = True
				else:
					self.__bi_mode = False
					self.__logger.critical('Bi-Directional data transmission is disabled!')
					raise ValueError('[NetGear:ERROR] :: `{}` pattern is not valid when Bi-Directional Mode is enabled. Kindly refer Docs for more Information.'.format(pattern))

			# enable force socket closing if specified
			elif key == 'force_terminate' and isinstance(value, bool):
				# check if pattern is valid
				if address is None and not(receive_mode):
					self.__force_close = False
					self.__logger.critical('Force termination is disabled for local servers!')
				else:
					self.__force_close = True
					if self.__logging: self.__logger.warning("Force termination is enabled for this connection!")

			# various ZMQ flags 
			elif key == 'flag' and isinstance(value, int):
				self.__msg_flag = value
			elif key == 'copy' and isinstance(value, bool):
				self.__msg_copy = value
			elif key == 'track' and isinstance(value, bool):
				self.__msg_track = value
			else:
				pass

		#handle secure mode 
		if self.__secure_mode:
			#import important libs
			import zmq.auth
			from zmq.auth.thread import ThreadAuthenticator

			# log if overwriting is enabled
			if overwrite_cert: 
				if not receive_mode:
					if self.__logging: self.__logger.warning('Overwriting ZMQ Authentication certificates over previous ones!')
				else:
					overwrite_cert = False
					if self.__logging: self.__logger.critical('Overwriting ZMQ Authentication certificates is disabled for Client-end!')

			#generate and validate certificates path
			try:
				#check if custom certificates path is specified
				if custom_cert_location:
					if os.path.isdir(custom_cert_location): #custom certificate location must be a directory
						(auth_cert_dir, self.__auth_secretkeys_dir, self.__auth_publickeys_dir) = generate_auth_certificates(custom_cert_location, overwrite = overwrite_cert)
					else:
						raise ValueError("[NetGear:ERROR] :: Invalid `custom_cert_location` value!")
				else:
					# otherwise auto-generate suitable path
					from os.path import expanduser
					(auth_cert_dir, self.__auth_secretkeys_dir, self.__auth_publickeys_dir) = generate_auth_certificates(os.path.join(expanduser("~"),".vidgear"), overwrite = overwrite_cert)
				
				#log it
				if self.__logging: self.__logger.debug('`{}` is the default location for storing ZMQ authentication certificates/keys.'.format(auth_cert_dir))
			except Exception as e:
				# catch if any error occurred
				self.__logger.exception(str(e))
				# also disable secure mode
				self.__secure_mode = 0
				self.__logger.warning('ZMQ Security Mechanism is disabled for this connection!')
		else:
			#log if disabled
			if self.__logging: self.__logger.warning('ZMQ Security Mechanism is disabled for this connection!')

		#handle bi_mode
		if self.__bi_mode:
			#disable bi_mode if multi-server is enabled
			if self.__multiserver_mode:
				self.__bi_mode = False
				self.__logger.critical('Bi-Directional Data Transmission is disabled when Multi-Server Mode is Enabled due to incompatibility!')
			else:
				#enable force termination by default
				self.__force_close = True
				if self.__logging: 
					self.__logger.warning("Force termination is enabled for this connection by default!")
					self.__logger.debug('Bi-Directional Data Transmission is enabled for this connection!')

		# initialize termination flag
		self.__terminate = False

		# initialize exit_loop flag
		self.__exit_loop = False

		# initialize and assign receive mode to global variable
		self.__receive_mode = receive_mode

		# define messaging context instance
		self.__msg_context = zmq.Context.instance()

		#check whether `receive_mode` is enabled by user
		if receive_mode:

			# if does than define connection address
			if address is None: address = '*' #define address
			
			#check if multiserver_mode is enabled
			if self.__multiserver_mode:
				# check if unique server port address list/tuple is assigned or not in multiserver_mode
				if port is None or not isinstance(port, (tuple, list)):
					# raise error if not
					raise ValueError('[NetGear:ERROR] :: Incorrect port value! Kindly provide a list/tuple of ports while Multi-Server mode is enabled. For more information refer VidGear docs.')
				else:
					#otherwise log it
					self.__logger.debug('Enabling Multi-Server Mode at PORTS: {}!'.format(port))
				#create port address buffer for keeping track of incoming server's port
				self.__port_buffer = []
			else:
				# otherwise assign local port address if None
				if port is None: port = '5555'

			try:
				# initiate and handle secure mode
				if self.__secure_mode > 0:
					# start an authenticator for this context
					auth = ThreadAuthenticator(self.__msg_context)
					auth.start()
					auth.allow(str(address)) #allow current address

					#check if `IronHouse` is activated
					if self.__secure_mode == 2:
						# tell authenticator to use the certificate from given valid dir
						auth.configure_curve(domain='*', location=self.__auth_publickeys_dir)
					else:
						#otherwise tell the authenticator how to handle the CURVE requests, if `StoneHouse` is activated
						auth.configure_curve(domain='*', location=zmq.auth.CURVE_ALLOW_ANY)

				# initialize and define thread-safe messaging socket
				self.__msg_socket = self.__msg_context.socket(msg_pattern[1])
				if self.__pattern == 2 and not(self.__secure_mode): self.__msg_socket.set_hwm(1)

				if self.__multiserver_mode:
					# if multiserver_mode is enabled, then assign port addresses to zmq socket
					for pt in port:
						# enable specified secure mode for the zmq socket
						if self.__secure_mode > 0:
							# load server key
							server_secret_file = os.path.join(self.__auth_secretkeys_dir, "server.key_secret")
							server_public, server_secret = zmq.auth.load_certificate(server_secret_file)
							# load  all CURVE keys
							self.__msg_socket.curve_secretkey = server_secret
							self.__msg_socket.curve_publickey = server_public
							# enable CURVE connection for this socket
							self.__msg_socket.curve_server = True

						# define socket options
						if self.__pattern == 2: self.__msg_socket.setsockopt_string(zmq.SUBSCRIBE, recv_filter)

						# bind socket to given server protocol, address and ports
						self.__msg_socket.bind(protocol+'://' + str(address) + ':' + str(pt))

					# define socket optimizer
					self.__msg_socket.setsockopt(zmq.LINGER, 0)

				else:
					# enable specified secure mode for the zmq socket
					if self.__secure_mode > 0:
						# load server key
						server_secret_file = os.path.join(self.__auth_secretkeys_dir, "server.key_secret")
						server_public, server_secret = zmq.auth.load_certificate(server_secret_file)
						# load  all CURVE keys
						self.__msg_socket.curve_secretkey = server_secret
						self.__msg_socket.curve_publickey = server_public
						# enable CURVE connection for this socket
						self.__msg_socket.curve_server = True

					# define exclusive socket options for patterns
					if self.__pattern == 2: self.__msg_socket.setsockopt_string(zmq.SUBSCRIBE,'')

					# bind socket to given protocol, address and port normally
					self.__msg_socket.bind(protocol+'://' + str(address) + ':' + str(port))

					# define socket optimizer
					self.__msg_socket.setsockopt(zmq.LINGER, 0)
											
			except Exception as e:
				self.__logger.exception(str(e))
				# otherwise raise value error if errored
				if self.__secure_mode: self.__logger.warning('Failed to activate ZMQ Security Mechanism: `{}` for this address!'.format(valid_security_mech[self.__secure_mode]))
				if self.__multiserver_mode:
					raise ValueError('[NetGear:ERROR] :: Multi-Server Mode, failed to connect to ports: {} with pattern: {}! Kindly recheck all parameters.'.format( str(port), pattern))
				else:
					raise ValueError('[NetGear:ERROR] :: Failed to bind address: {} and pattern: {}! Kindly recheck all parameters.'.format((protocol+'://' + str(address) + ':' + str(port)), pattern))
			
			#log and enable threaded queue mode
			if self.__logging: self.__logger.debug('Threaded Queue Mode is enabled by default for NetGear.')
			#define deque and assign it to global var
			self.__queue = deque(maxlen=96) #max len 96 to check overflow

			# initialize and start threading instance
			self.__thread = Thread(target=self.__update, name='NetGear', args=())
			self.__thread.daemon = True
			self.__thread.start()

			if self.__logging:
				#finally log progress
				self.__logger.debug('Successfully Binded to address: {} with pattern: {}.'.format((protocol+'://' + str(address) + ':' + str(port)), pattern))
				if self.__secure_mode: self.__logger.debug('Enabled ZMQ Security Mechanism: `{}` for this address, Successfully!'.format(valid_security_mech[self.__secure_mode]))
				self.__logger.debug('Multi-threaded Receive Mode is enabled Successfully!')
				self.__logger.debug('Device Unique ID is {}.'.format(self.__id))
				self.__logger.debug('Receive Mode is activated successfully!')
		else:
			#otherwise default to `Send Mode`
			if address is None: address = 'localhost'#define address

			#check if multiserver_mode is enabled
			if self.__multiserver_mode:
				# check if unique server port address is assigned or not in multiserver_mode
				if port is None:
					#raise error is not
					raise ValueError('[NetGear:ERROR] :: Kindly provide a unique & valid port value at Server-end. For more information refer VidGear docs.')
				else:
					#otherwise log it
					self.__logger.debug('Enabling Multi-Server Mode at PORT: {} on this device!'.format(port))
					#assign value to global variable
					self.port = port
			else:
				# otherwise assign local port address if None
				if port is None: port = '5555'
				
			try:
				# initiate and handle secure mode
				if self.__secure_mode > 0:
					# start an authenticator for this context
					auth = ThreadAuthenticator(self.__msg_context)
					auth.start()
					auth.allow(str(address)) #allow current address

					#check if `IronHouse` is activated
					if self.__secure_mode == 2:
						# tell authenticator to use the certificate from given valid dir
						auth.configure_curve(domain='*', location=self.__auth_publickeys_dir)
					else:
						#otherwise tell the authenticator how to handle the CURVE requests, if `StoneHouse` is activated
						auth.configure_curve(domain='*', location=zmq.auth.CURVE_ALLOW_ANY)

				# initialize and define thread-safe messaging socket
				self.__msg_socket = self.__msg_context.socket(msg_pattern[0])

				if self.__pattern == 1:
					# if pattern is 1, define additional flags
					self.__msg_socket.REQ_RELAXED = True
					self.__msg_socket.REQ_CORRELATE = True
				if self.__pattern == 2 and not(self.__secure_mode): self.__msg_socket.set_hwm(1) # if pattern is 2, define additional optimizer

				# enable specified secure mode for the zmq socket
				if self.__secure_mode > 0:
					# load client key
					client_secret_file = os.path.join(self.__auth_secretkeys_dir, "client.key_secret")
					client_public, client_secret = zmq.auth.load_certificate(client_secret_file) 
					# load  all CURVE keys
					self.__msg_socket.curve_secretkey = client_secret
					self.__msg_socket.curve_publickey = client_public
					# load server key
					server_public_file = os.path.join(self.__auth_publickeys_dir, "server.key")
					server_public, _ = zmq.auth.load_certificate(server_public_file)
					# inject public key to make a CURVE connection.
					self.__msg_socket.curve_serverkey = server_public

				# connect socket to given protocol, address and port
				self.__msg_socket.connect(protocol+'://' + str(address) + ':' + str(port))

				# define socket options
				self.__msg_socket.setsockopt(zmq.LINGER, 0)

			except Exception as e:
				self.__logger.exception(str(e))
				#log if errored
				if self.__secure_mode: self.__logger.warning('Failed to activate ZMQ Security Mechanism: `{}` for this address!'.format(valid_security_mech[self.__secure_mode]))
				# raise value error
				raise ValueError('[NetGear:ERROR] :: Failed to connect address: {} and pattern: {}! Kindly recheck all parameters.'.format((protocol+'://' + str(address) + ':' + str(port)), pattern))

			if self.__logging:
				#finally log progress
				self.__logger.debug('Successfully connected to address: {} with pattern: {}.'.format((protocol+'://' + str(address) + ':' + str(port)), pattern))
				if self.__secure_mode: self.__logger.debug('Enabled ZMQ Security Mechanism: `{}` for this address, Successfully!'.format(valid_security_mech[self.__secure_mode]))
				self.__logger.debug('This device Unique ID is {}.'.format(self.__id))
				self.__logger.debug('Send Mode is successfully activated and ready to send data!')



	def __update(self):
		"""
		Updates recovered frames from messaging network to the queue
		"""
		# initialize frame variable
		frame = None

		# keep looping infinitely until the thread is terminated
		while not self.__exit_loop:

			# check if global termination_flag is enabled  
			if self.__terminate:
				# check whether there is still frames in queue before breaking out
				if len(self.__queue)>0:
					continue
				else:
					break

			#check queue buffer for overflow
			if len(self.__queue) >= 96:
				#stop iterating if overflowing occurs
				time.sleep(0.000001)
				continue

			# extract json data out of socket
			msg_json = self.__msg_socket.recv_json(flags=self.__msg_flag)

			# check if terminate_flag` received
			if msg_json['terminate_flag']:
				#if multiserver_mode is enabled 
				if self.__multiserver_mode:
					# check and remove from which ports signal is received
					if msg_json['port'] in self.__port_buffer:
						# if pattern is 1, then send back server the info about termination
						if self.__pattern == 1: self.__msg_socket.send_string('Termination signal received at client!')
						self.__port_buffer.remove(msg_json['port'])
						if self.__logging: self.__logger.warning('Termination signal received from server at port: {}!'.format(msg_json['port']))
					#if termination signal received from all servers then exit client.
					if not self.__port_buffer:
						self.__logger.warning('Termination signal received from all Servers!!!')
						self.__terminate = True #termination
					continue
				else:
					# if pattern is 1, then send back server the info about termination
					if self.__pattern == 1: self.__msg_socket.send_string('Termination signal received at client!')
					#termination
					self.__terminate = True
					#notify client
					if self.__logging: self.__logger.warning('Termination signal received from server!')
					continue

			#check if pattern is same at both server's and client's end.
			if int(msg_json['pattern']) != self.__pattern:
				raise ValueError("[NetGear:ERROR] :: Messaging patterns on both Server-end & Client-end must a valid pairs! Kindly refer VidGear docs.")
				self.__terminate = True
				continue

			# extract array from socket
			msg_data = self.__msg_socket.recv(flags=self.__msg_flag, copy=self.__msg_copy, track=self.__msg_track)

			if self.__pattern != 2:
				# check if bi-directional mode is enabled
				if self.__bi_mode:
					# handle return data
					bi_dict = dict(data = self.__bi_data)
					self.__msg_socket.send_json(bi_dict, self.__msg_flag)
				else:
					# send confirmation message to server
					self.__msg_socket.send_string('Data received on device: {} !'.format(self.__id))

			# recover frame from array buffer
			frame_buffer = np.frombuffer(msg_data, dtype=msg_json['dtype'])
			# reshape frame
			frame = frame_buffer.reshape(msg_json['shape'])

			#check if encoding was enabled
			if msg_json['compression']: 
				frame = cv2.imdecode(frame, self.__compression_params)
				#check if valid frame returned
				if frame is None:
					#otherwise raise error and exit
					raise ValueError("[NetGear:ERROR] :: `{}` Frame Decoding failed with Parameter: {}".format(msg_json['compression'], self.__compression_params))
					self.__terminate = True
					continue

			if self.__multiserver_mode:
				# check if multiserver_mode

				#save the unique port addresses
				if not msg_json['port'] in self.__port_buffer:
					self.__port_buffer.append(msg_json['port'])
			
				#extract if any message from server and display it
				if msg_json['message']:
					self.__queue.append((msg_json['port'], msg_json['message'], frame))
				else:
					# append recovered unique port and frame to queue
					self.__queue.append((msg_json['port'],frame))
			else:
				#extract if any message from server if Bi-Directional Mode is enabled
				if self.__bi_mode and msg_json['message']:
					# append grouped frame and data to queue
					self.__queue.append((msg_json['message'], frame))
				else:
					# otherwise append recovered frame to queue
					self.__queue.append(frame)

		# finally properly close the socket
		self.__msg_socket.close()



	def recv(self, return_data = None):
		"""
		return the recovered frame

		:param return_data: handles return data for bi-directional mode 
		"""
		# check whether `receive mode` is activated
		if not(self.__receive_mode):
			#raise value error and exit
			raise ValueError('[NetGear:ERROR] :: `recv()` function cannot be used while receive_mode is disabled. Kindly refer vidgear docs!')
			self.__terminate = True
		
		#handle bi-directional return data
		if (self.__bi_mode and not(return_data is None)): self.__bi_data = return_data

		# check whether or not termination flag is enabled
		while not self.__terminate:
			# check if queue is empty
			if len(self.__queue)>0:
				return self.__queue.popleft()
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
		if self.__receive_mode:
			#raise value error and exit
			raise ValueError('[NetGear:ERROR] :: `send()` function cannot be used while receive_mode is enabled. Kindly refer vidgear docs!')
			self.__terminate = True

		# define exit_flag and assign value
		exit_flag = True if (frame is None or self.__terminate) else False

		#check whether exit_flag is False
		if not(exit_flag) and not(frame.flags['C_CONTIGUOUS']):
			#check whether the incoming frame is contiguous
			frame = np.ascontiguousarray(frame, dtype=frame.dtype)

		#handle encoding
		if self.__compression:
			retval, frame = cv2.imencode(self.__compression, frame, self.__compression_params)
			#check if it works
			if not(retval): 
				#otherwise raise error and exit
				raise ValueError("[NetGear:ERROR] :: Frame Encoding failed with format: {} and Parameters: {}".format(self.__compression, self.__compression_params))
				self.__terminate = True


		#check if multiserver_mode is activated
		if self.__multiserver_mode:
			# prepare the exclusive json dict and assign values with unique port
			msg_dict = dict(terminate_flag = exit_flag,
							compression=str(self.__compression),
							port = self.port,
							pattern = str(self.__pattern),
							message = message,
							dtype = str(frame.dtype),
							shape = frame.shape)
		else:
			# otherwise prepare normal json dict and assign values
			msg_dict = dict(terminate_flag = exit_flag,
							compression=str(self.__compression),
							message = message,
							pattern = str(self.__pattern),
							dtype = str(frame.dtype),
							shape = frame.shape)

		# send the json dict
		self.__msg_socket.send_json(msg_dict, self.__msg_flag|self.__zmq.SNDMORE)
		# send the frame array with correct flags
		self.__msg_socket.send(frame, flags = self.__msg_flag, copy=self.__msg_copy, track=self.__msg_track)
		# wait for confirmation

		if self.__pattern != 2:
			#check if bi-directional data transmission is enabled
			if self.__bi_mode:
				#handle return data
				return_dict = self.__msg_socket.recv_json(flags=self.__msg_flag)
				return return_dict['data'] if return_dict else None
			else:
				#otherwise log normally
				recv_confirmation = self.__msg_socket.recv()
				# log confirmation 
				if self.__logging : self.__logger.debug(recv_confirmation)
			



	def close(self):
		"""
		Terminates the NetGear processes safely
		"""
		if self.__logging:
			#log it
			self.__logger.debug('Terminating various {} Processes.'.format('Receive Mode' if self.__receive_mode else 'Send Mode'))
		#  whether `receive_mode` is enabled or not
		if self.__receive_mode:
			# indicate that process should be terminated
			self.__terminate = True
			# check whether queue mode is empty
			if not(self.__queue is None):
				self.__queue.clear()
			# call immediate termination
			self.__exit_loop = True
			# wait until stream resources are released (producer thread might be still grabbing frame)
			if self.__thread is not None: 
				self.__thread.join()
				self.__thread = None
				#properly handle thread exit

			# properly close the socket
			self.__msg_socket.close(linger=0)
		else:
			# indicate that process should be terminated
			self.__terminate = True
			if self.__multiserver_mode:
				#check if multiserver_mode
				# send termination flag to client with its unique port
				term_dict = dict(terminate_flag = True, port = self.port)
			else:
				# otherwise send termination flag to client
				term_dict = dict(terminate_flag = True)
			# otherwise inform client(s) that the termination has been reached
			if self.__force_close:
				#overflow socket with termination signal
				for _ in range(500): self.__msg_socket.send_json(term_dict)
			else:
				self.__msg_socket.send_json(term_dict)
				#check for confirmation if available
				if self.__pattern < 2: 
					if self.__pattern == 1: self.__msg_socket.recv()
			# properly close the socket
			self.__msg_socket.close(linger=0)