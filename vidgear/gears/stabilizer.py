# This is a modified algorithm/code based on findings of `Simple video stabilization using OpenCV` 
# published on February 20, 2014 by nghiaho12 (http://nghiaho.com/?p=2093)

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
from collections import deque
from .helper import check_CV_version
import numpy as np
import cv2

us

class Stabilizer:
	"""
	This class enables real-time video stabilization for vidgear which can provide powerful video stabilization with latency and at expense of 
	little to no additional computational power requirement. It tracks the salient feature array of frames and uses this as an anchor point to cancel out all
	 perturbations relative to it.

	:param smoothing_radius` (int) : to alter averaging window size. It handles the quality of stabilization at expense of latency and sudden panning. 
									/ Larger its value, less will be panning, more will be latency and vice-versa. It's default value is 25.
	:param border_size (int) : to alter output border cropping. It's will crops the border to reduce the black borders from stabilization being too noticeable. 
								/ Larger its value, more will be cropping. It's default value is 0 (i.e. no cropping).			
	:param border_type (string) : to change the border mode. Valid border types are 'black', 'reflect', 'reflect_101', 'replicate' and 'wrap'. It's default value is 'black'

	:param (boolean) logging: set this flag to enable/disable error logging essential for debugging. Its default value is False.

	"""
	
	def __init__(self, smoothing_radius = 25, border_type = 'black', border_size = 0 , logging = False):

		# initialize deques for handling input frames and its indexes
		self.frame_queue = deque(maxlen=smoothing_radius)
		self.frame_queue_indexes = deque(maxlen=smoothing_radius)

		# define and create Adaptive histogram equalization (AHE) object for optimizations
		self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

		# initialize global vars
		self.smoothing_radius = smoothing_radius #averaging window, handles the quality of stabilization at expense of latency and sudden panning
		self.smoothed_path = None #handles the smoothed path with box filter
		self.path = None #handles path i.e cumulative sum of pevious_2_current transformations along a axis
		self.transforms = [] #handles pevious_2_current transformations [dx,dy,da]
		self.frame_transforms_smoothed = None #handles smoothed array of pevious_2_current transformations w.r.t to frames
		self.previous_gray = None #handles previous gray frame
		self.previous_keypoints = None #handles previous detect_GFTTed keypoints w.r.t previous gray frame
		self.frame_height, self.frame_width = (0, 0) #handles width and height of input frames

		# define valid border modes
		border_modes = {'black': cv2.BORDER_CONSTANT,'reflect': cv2.BORDER_REFLECT, 'reflect_101': cv2.BORDER_REFLECT_101, 'replicate': cv2.BORDER_REPLICATE, 'wrap': cv2.BORDER_WRAP}
		# choose valid border_mode from border_type
		if border_type in ['black', 'reflect', 'reflect_101', 'replicate', 'wrap']:
			#initialize global border mode variable 
			self.border_mode = border_modes[border_type]
		else:
			#otherwise log if not
			if logging:
				print('Invalid input border type!')
			self.border_mode = border_modes['black'] #reset to default mode
			
		# define normalized box filter
		self.box_filter = np.ones(smoothing_radius)/smoothing_radius

		# define cropping factor, Crops the border to reduce the black borders from stabilization being too noticeable.
		self.border_crop_size = border_size

		#decide whether to log
		self.logging = logging



	def stabilize(self, frame):
		"""
		Return stabilized video frame

		:param frame(numpy.ndarray): input video frame
		"""
		#check if frame is None
		if frame is None:
			#return if it does
			return
		
		# initiate transformations capturing
		if not self.frame_queue:
			#for first frame
			previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to gray
			previous_gray = self.clahe.apply(previous_gray) #optimize gray frame
			self.previous_keypoints = cv2.goodFeaturesToTrack(previous_gray, maxCorners=200, qualityLevel=0.05, minDistance=30.0, blockSize=3 , mask=None, useHarrisDetector=False, k=0.04) #track features using GFTT
			self.frame_height, self.frame_width = frame.shape[:2] #save input frame height and width
			self.frame_queue.append(frame) #save frame to deque
			self.frame_queue_indexes.append(0) #save frame index to deque
			self.previous_gray = previous_gray[:] #save gray frame clone for further processing

		elif self.frame_queue_indexes[-1] <= self.smoothing_radius - 1:
			#for rest of frames 
			self.frame_queue.append(frame) #save frame to deque
			self.frame_queue_indexes.append(self.frame_queue_indexes[-1]+1) #save frame index
			self.generate_transformations() #generate transformations
			if self.frame_queue_indexes[-1] == self.smoothing_radius - 1:
				#calculate smooth path once transformation capturing is completed
				for i in range(3):
					#apply normalized box filter to the path
				   self.smoothed_path[:,i] = self.box_filter_convolve((self.path[:,i]), window_size=self.smoothing_radius)
				#calculate deviation of path from smoothed path
				deviation = self.smoothed_path - self.path
				#save smoothed transformation
				self.frame_transforms_smoothed = self.frame_transform + deviation
		else:
			#start applying transformations
			self.frame_queue.append(frame) #save frame to deque
			self.frame_queue_indexes.append(self.frame_queue_indexes[-1]+1) #save frame index
			self.generate_transformations() #generate transformations
			#calculate smooth path once transformation capturing is completed
			for i in range(3):
				#apply normalized box filter to the path
			   self.smoothed_path[:,i] = self.box_filter_convolve((self.path[:,i]), window_size=self.smoothing_radius)
			#calculate deviation of path from smoothed path
			deviation = self.smoothed_path - self.path
			#save smoothed transformation
			self.frame_transforms_smoothed = self.frame_transform + deviation
			#return transformation applied stabilized frame
			return self.apply_transformations()



	def generate_transformations(self):
		"""
		Generate pevious_2_current transformations [dx,dy,da]
		"""
		frame_gray = cv2.cvtColor(self.frame_queue[-1], cv2.COLOR_BGR2GRAY) #retrieve current frame and convert to gray
		frame_gray = self.clahe.apply(frame_gray) #optimize it

		#calculate optical flow using Lucas–Kanade differential method
		curr_kps, status, error = cv2.calcOpticalFlowPyrLK(self.previous_gray, frame_gray, self.previous_keypoints, None)

		#select only valid keypoints
		valid_curr_kps = curr_kps[status==1] #current
		valid_previous_keypoints = self.previous_keypoints[status==1] #previous
		
		#calculate optimal affine transformation between pevious_2_current keypoints
		if check_CV_version() == 3:
			#backward compatibility with OpenCV3 
			transformation = cv2.estimateRigidTransform(valid_previous_keypoints,valid_curr_kps, False)
		else:
			transformation = cv2.estimateAffinePartial2D(valid_previous_keypoints,valid_curr_kps)[0]
		
		#check if transformation is not None
		if not(transformation is None):
			# pevious_2_current translation in x direction
			dx = transformation[0, 2]
			# pevious_2_current translation in y direction
			dy = transformation[1, 2]
			# pevious_2_current rotation in angle
			da = np.arctan2(transformation[1, 0], transformation[0, 0])
		else:
			#otherwise zero it
			dx = dy = da = 0

		#save this transformation
		self.transforms.append([dx, dy, da])

		#calculate path from cumulative transformations sum
		self.frame_transform = np.array(self.transforms, dtype='float32')
		self.path = np.cumsum(self.frame_transform, axis=0)
		#create smoothed path from a copy of path
		self.smoothed_path=np.copy(self.path)

		#re-calculate and save GFTT keypoints for current gray frame
		self.previous_keypoints = cv2.goodFeaturesToTrack(frame_gray, maxCorners=200, qualityLevel=0.05, minDistance=30.0, blockSize=3 , mask=None, useHarrisDetector=False, k=0.04)
		#save this gray frame for further processing
		self.previous_gray = frame_gray[:]



	def box_filter_convolve(self, path, window_size):
		"""
		applies normalized linear box filter to path w.r.t to averaging window
		
		:param: path(nd.array)(cumulative sum of transformations)
		:param: averaging window size(int)
		"""
		#pad path to size of averaging window
		path_padded = np.pad(path, (window_size, window_size), 'median')
		#apply linear box filter to path
		path_smoothed = np.convolve(path_padded, self.box_filter, mode='same')
		#crop the smoothed path to original path 
		path_smoothed = path_smoothed[window_size:-window_size]
		#assert if cropping is completed
		assert path.shape == path_smoothed.shape
		#return smoothed path
		return path_smoothed



	def apply_transformations(self):
		"""
		Applies affine transformation to the given frame 
		build from previously calculated transformations
		"""

		#extract frame and its index from deque
		queue_frame = self.frame_queue.popleft()
		queue_frame_index = self.frame_queue_indexes.popleft()

		#create border around extracted frame w.r.t border_size
		bordered_frame = cv2.copyMakeBorder(queue_frame, top=self.border_crop_size, bottom=self.border_crop_size, left=self.border_crop_size, right=self.border_crop_size, borderType=self.border_mode, value=[0, 0, 0])
		alpha_bordered_frame = cv2.cvtColor(bordered_frame, cv2.COLOR_BGR2BGRA) #create alpha channel
		#extract alpha channel
		alpha_bordered_frame[:, :, 3] = 0 
		alpha_bordered_frame[self.border_crop_size:self.border_crop_size + self.frame_height, self.border_crop_size:self.border_crop_size + self.frame_width, 3] = 255

		#extracting Transformations w.r.t frame index
		dx = self.frame_transforms_smoothed[queue_frame_index,0] #x-axis
		dy = self.frame_transforms_smoothed[queue_frame_index,1] #y-axis
		da = self.frame_transforms_smoothed[queue_frame_index,2] #angle

		#building 2x3 transformation matrix from extracted transformations
		queue_frame_transform = np.zeros((2,3), np.float32)
		queue_frame_transform[0,0] = np.cos(da)
		queue_frame_transform[0,1] = -np.sin(da)
		queue_frame_transform[1,0] = np.sin(da)
		queue_frame_transform[1,1] = np.cos(da)
		queue_frame_transform[0,2] = dx
		queue_frame_transform[1,2] = dy

		#Applying an affine transformation to the frame
		frame_wrapped = cv2.warpAffine(alpha_bordered_frame, queue_frame_transform, alpha_bordered_frame.shape[:2][::-1], borderMode=self.border_mode)

		#drop alpha channel
		frame_stabilized = frame_wrapped[:, :, :3]

		#finally return stabilized frame
		return frame_stabilized


	def clean(self):
		"""
		clean resources(deque)
		"""
		# check if deque present
		if self.frame_queue:
			#clear frame deque
			self.frame_queue.clear()
			#clear frame indexes deque
			self.frame_queue_indexes.clear()





	