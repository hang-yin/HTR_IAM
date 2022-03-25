from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import re

import argparse
import os
import random
import numpy as np
import cv2
import math
import copy
from os import listdir
from os.path import isfile, join
import pickle
import sys
import pickle
# import matplotlib.pyplot as plt

# paths to data files
# within data, there's "ascii" and "lineStrokes"
# "ascii" folder contains the labels
# "lineStrokes" folder contains the strokes
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(FILE_PATH, "data/")
LABEL_DATA_PATH = os.path.join(DATA_PATH, "ascii/")
STROKES_DATA_PATH = os.path.join(DATA_PATH, "lineStrokes/")

ESCAPE_CHAR = '~!@#$%^&*()_+{}:"<>?`-=[];\',./|\n'


# function for finding label of one whole textline
def find_textline_by_id(filename):
	"""
	Inputs:
		filename: string, textline prefix, eg: 'a01-020w-01'
	Return:
		label: string, label of one whole textline, eg: 'No secret talks # - Macleod.'
	"""
	dir_name_L1 = filename[:3]  # eg: 'a01'
	dir_name_L2 = filename[:7]  # eg: 'a01-020'
	file_name = filename[:-3] + ".txt"  # eg: 'a01-020w.txt' or 'a01-020.txt'
	line_id = int(filename[-2:])  # eg: 1
	filepath = os.path.join(
		LABEL_DATA_PATH, dir_name_L1, dir_name_L2, file_name)
	line_counter = -2  # because line start after 2 new lines from "CSR:\n"
	label = []
	flag = False
	for line in open(filepath, 'r'):
		if line.startswith('CSR'):
			flag = True
		if flag:
			line_counter += 1
		if line_counter == line_id:
			for char in ESCAPE_CHAR:
				line = line.replace(char, '')
			label = line
			break
	return label

# function for showing image
def imshow( grayMatrix, visiable=True, scale=2.0 ):
	image = ( grayMatrix + 1.0 ) / 2.0 * 255
	image[image == 127.5 ] = 127.5 

	newWidth  = int(image.shape[0] * scale)
	newHeight = int(image.shape[1] * scale )

	image = np.array( image, np.uint8 )
	# print("line 17", image.shape)
	image = cv2.resize( image, (newHeight, newWidth) )
	image = np.flip( image, 0 )
	if visiable is True:
		cv2.imshow( 'Image', image )
		cv2.waitKey(1)
		cv2.destroyAllWindows()
	return image

# calculate Euclidean distance given point 1, point 2
def dist( point_1, point_2 ):
	return math.sqrt( ( point_1.x - point_2.x ) ** 2 + ( point_1.y - point_2.y ) ** 2 )

# calculate angle given point 1, point 2, point_3
def angle( point_1, point_2, point_3 ):
	p_2_p_1_x 	= point_1.x - point_2.x
	p_2_p_3_x 	= point_3.x - point_2.x
	p_2_p_1_y 	= point_1.y - point_2.y 
	p_2_p_3_y 	= point_3.y - point_2.y

	cos_value 	= ( p_2_p_1_x * p_2_p_3_x + p_2_p_1_y * p_2_p_3_y ) / math.sqrt( p_2_p_1_x ** 2 + p_2_p_1_y ** 2 ) / math.sqrt( p_2_p_3_x ** 2 + p_2_p_3_y ** 2 )
	
	# do projection 
	if cos_value > 1.0:
		cos_value 	= 1.0
	elif cos_value < -1.0:
		cos_value 	= -1.0
		
	return 	math.acos( cos_value )

# define point class: x, y, time stamp, re time stamp, id, features
class Point(object):
	"""docstring for Point"""
	def __init__(self, x=-1, y=-1, timeStamp=-1, reTimeStamp=-1):
		super(Point, self).__init__()
		self.x = x
		self.y = y
		self.timeStamp = timeStamp
		self.reTimeStamp = reTimeStamp
		self.id = None
		self.features = np.array([0.0] * 7)

# define stroke class: id, raw points, sampled points, scaled points
class Stroke(object):
	"""docstring for Stroke"""
	def __init__(self, id):
		super(Stroke, self).__init__()
		self.id = id
		self.rawPoints = list()
		self.sampledPoints = None
		self.scaledPoints = None

	def addRawPoint(self, point):
		self.rawPoints.append(point)

# define signature class
class Signature(object):
	"""docstring for Signature"""
	def __init__(self, stroke=None):
		super(Signature, self).__init__()
		self.id = -1
		self.targetHeight = 124
		self.padding = 2
		self.scaledPoints = list()
		self.scaledRawPoints = list()
		self.scaledSampledPoints = list()
		if stroke == None:
			self.rawPoints = list()
		else:
			self.rawPoints = stroke
		self.timestamps = list()
		self.penStatuses = list()
		self.label = -1
		self.userId = -1
		self.localId = -1 # id for its user
		self.userName = ""
		self.strokes = list()
		self.startTimeStamp = -1

	def showStrokes(self, strokes, type=0, visible=True, scalar=None, label=1):
		# determine the size of image
		max_x = -float( 'inf' )
		min_x = float( 'inf' )
		max_y = -float( 'inf' )
		min_y = float( 'inf' )		

		for stroke in strokes:

			if type == 0:
				max_x = max( max_x, max( [ point.x for point in stroke.rawPoints ] ) )
				min_x = min( min_x, min( [ point.x for point in stroke.rawPoints ] ) )
				max_y = max( max_y, max( [ point.y for point in stroke.rawPoints ] ) )
				min_y = min( min_y, min( [ point.y for point in stroke.rawPoints ] ) )				
			elif type == 1:
				max_x = max( max_x, max( [ point.x for point in stroke.sampledPoints ] ) )
				min_x = min( min_x, min( [ point.x for point in stroke.sampledPoints ] ) )
				max_y = max( max_y, max( [ point.y for point in stroke.sampledPoints ] ) )
				min_y = min( min_y, min( [ point.y for point in stroke.sampledPoints ] ) )		
			elif type == 2:
				max_x = max( max_x, max( [ point.x for point in stroke.scaledPoints ] ) )
				min_x = min( min_x, min( [ point.x for point in stroke.scaledPoints ] ) )
				max_y = max( max_y, max( [ point.y for point in stroke.scaledPoints ] ) )
				min_y = min( min_y, min( [ point.y for point in stroke.scaledPoints ] ) )		

		margin = 5
		if scalar is None:
			scalar = 400.0 / float( max_y - min_y )
		else:
			scalar = 1.0

		width = int( ( max_x - min_x ) * scalar + 2 * margin )
		height = int( ( max_y - min_y ) * scalar + 2 * margin )

		print( max_y - min_y, width, height)
		image = 255 * np.ones( ( height, width, 3 ), np.uint8 )

		color = (0,255,0)
		if label == 0:
			color = (255, 0, 0)

		for stroke in strokes:
			if type == 0:
				for point in stroke.rawPoints:
					cv2.circle( image, ( int((point.x-min_x)*scalar) + margin, int((point.y-min_y)*scalar) + margin ), 1, (0,0,0), 2 )

				for ( i, point ) in enumerate( stroke.rawPoints ):
					if i >= 1:
						cv2.line(image, ( int(scalar*(stroke.rawPoints[i-1].x-min_x))+margin, int(scalar*(stroke.rawPoints[i-1].y-min_y))+margin ), \
							( int(scalar*(stroke.rawPoints[i].x-min_x))+margin, int(scalar*(stroke.rawPoints[i].y-min_y))+margin ) , color, 1)
			elif type == 1:
				for point in stroke.sampledPoints:
					cv2.circle( image, ( int((point.x-min_x)*scalar)+margin, int((point.y-min_y)*scalar)+margin ), 1, (0,0,0), 2 )

				for ( i, point ) in enumerate( stroke.sampledPoints ):
					if i >= 1:
						cv2.line(image, ( int(scalar*(stroke.sampledPoints[i-1].x-min_x))+margin, int(scalar*(stroke.sampledPoints[i-1].y-min_y))+margin ), \
							( int(scalar*(stroke.sampledPoints[i].x-min_x))+margin, int(scalar*(stroke.sampledPoints[i].y-min_y))+margin ) , color, 1)
			elif type == 2:
				for point in stroke.scaledPoints:
					cv2.circle( image, ( int((point.x-min_x)*scalar)+margin, int((point.y-min_y)*scalar)+margin ), 1, (0,0,0), 2 )

				for ( i, point ) in enumerate( stroke.scaledPoints ):
					if i >= 1:
						cv2.line(image, ( int(scalar*(stroke.scaledPoints[i-1].x-min_x))+margin, int(scalar*(stroke.scaledPoints[i-1].y-min_y))+margin ), \
							( int(scalar*(stroke.scaledPoints[i].x-min_x))+margin, int(scalar*(stroke.scaledPoints[i].y-min_y))+margin ) , color, 1)
		image = np.flip( image, 0 )

		if visible is True:
			cv2.imshow( 'Signature', image )
			cv2.waitKey(0)
		return image

	def show(self, points, visible=True, scalar=None, label=1):
		# determine the size of image
		max_x = -float( 'inf' )
		min_x = float( 'inf' )
		max_y = -float( 'inf' )
		min_y = float( 'inf' )		
		
		max_x = max( max_x, max( [ point.x for point in points ] ) )
		min_x = min( min_x, min( [ point.x for point in points ] ) )
		max_y = max( max_y, max( [ point.y for point in points ] ) )
		min_y = min( min_y, min( [ point.y for point in points ] ) )

		margin = 5
		if scalar is None:
			scalar = 400.0 / float( max_y - min_y )
		else:
			scalar = 1.0

		width = int( ( max_x - min_x ) * scalar + 2 * margin )
		height = int( ( max_y - min_y ) * scalar + 2 * margin )

		# print( width, height )
		image = 255 * np.ones( ( height, width, 3 ), np.uint8 )

		color = (0,255,0)
		if label == 0:
			color = (255, 0, 0)

		for point in points:
			cv2.circle( image, ( int((point.x-min_x)*scalar), int((point.y-min_y)*scalar) ), 1, (0,0,0), 2 )

		for ( i, point ) in enumerate( points ):
			if i >= 1:
				cv2.line(image, ( int(scalar*(points[i-1].x-min_x)), int(scalar*(points[i-1].y-min_y)) ), \
					( int(scalar*(points[i].x-min_x)), int(scalar*(points[i].y-min_y)) ) , color, 1)

		image = np.flip( image, 0 )

		if visible is True:
			cv2.imshow( 'Signature', image )
			cv2.waitKey(0)
		return image

	def saveShow(self, points, path, visible=False, scalar=None):
		image = self.show(points, visible, scalar)
		cv2.imwrite( path, image)

	def scaleByHeight(self, strokes, targetHeight, type=1):

		# determine the size of image
		max_x = -float( 'inf' )
		min_x = float( 'inf' )
		max_y = -float( 'inf' )
		min_y = float( 'inf' )		

		for stroke in strokes:

			if type == 0:
				max_x = max( max_x, max( [ point.x for point in stroke.rawPoints ] ) )
				min_x = min( min_x, min( [ point.x for point in stroke.rawPoints ] ) )
				max_y = max( max_y, max( [ point.y for point in stroke.rawPoints ] ) )
				min_y = min( min_y, min( [ point.y for point in stroke.rawPoints ] ) )				
			elif type == 1:
				max_x = max( max_x, max( [ point.x for point in stroke.sampledPoints ] ) )
				min_x = min( min_x, min( [ point.x for point in stroke.sampledPoints ] ) )
				max_y = max( max_y, max( [ point.y for point in stroke.sampledPoints ] ) )
				min_y = min( min_y, min( [ point.y for point in stroke.sampledPoints ] ) )			

		scale = float(targetHeight) / max( (max_y - min_y ), 1e-8 )

		for stroke in strokes:
			if type == 0:
				stroke.scaledRawPoints = list()
				for point in stroke.rawPoints:
					stroke.scaledRawPoints.append( Point( float(point.x-min_x) * scale, float(point.y-min_y) * scale, point.timeStamp, point.reTimeStamp ) )
			elif type == 1:
				stroke.scaledSampledPoints = list()
				for point in stroke.sampledPoints:
					stroke.scaledSampledPoints.append( Point( float(point.x-min_x) * scale, float(point.y-min_y) * scale, point.timeStamp, point.reTimeStamp ) )						
		
	def scaleByHeightPoints(self, points, targetHeight):
		# determine the size of image
		max_x = -float( 'inf' )
		min_x = float( 'inf' )
		max_y = -float( 'inf' )
		min_y = float( 'inf' )		
		
		max_x = max( max_x, max( [ point.x for point in points ] ) )
		min_x = min( min_x, min( [ point.x for point in points ] ) )
		max_y = max( max_y, max( [ point.y for point in points ] ) )
		min_y = min( min_y, min( [ point.y for point in points ] ) )

		scale = float(targetHeight) / max( (max_y - min_y ), 1e-8 )

		scaledPoints = list()
		for point in points:
			scaledPoints.append( Point( float(point.x-min_x) * scale, float(point.y-min_y) * scale, point.timeStamp, point.reTimeStamp ) )

		return scaledPoints

	def calculatePSFeatures(self, strokes):
		# determine the size of image
		max_x = -float( 'inf' )
		min_x = float( 'inf' )

		for stroke in strokes:
			max_x = max( max_x, max( [ point.x for point in stroke.scaledRawPoints ] ) )
			min_x = min( min_x, min( [ point.x for point in stroke.scaledRawPoints ] ) )
			max_x = max( max_x, max( [ point.x for point in stroke.scaledSampledPoints ] ) )
			min_x = min( min_x, min( [ point.x for point in stroke.scaledSampledPoints ] ) )					
		self.length = math.floor( max_x - min_x )
	
		width = self.length+2*self.padding
		height = self.targetHeight+2*self.padding
		channel = 7
		shape = ( int(height), int(width), int(channel) )
		self.PSFeatures = np.zeros( shape ) 

		for stroke in strokes:

			pointFeatures = np.zeros((len(stroke.scaledSampledPoints), int(channel)))
			# calculate products (discrete)
			for (i, point) in enumerate(stroke.scaledSampledPoints):
				# calculate order as 0
				point.features[0] = 1.0
				# calculate order as 1 and 2
				if i != len(stroke.scaledSampledPoints) - 1:
					point.features[1] = stroke.scaledSampledPoints[i+1].x - stroke.scaledSampledPoints[i].x
					point.features[2] = stroke.scaledSampledPoints[i+1].y - stroke.scaledSampledPoints[i].y
					point.features[3] = 0.5 * (point.features[1]**2)
					point.features[4] = 0.5 * (point.features[1]*point.features[2])
					point.features[5] = 0.5 * (point.features[2]*point.features[1])
					point.features[6] = 0.5 * (point.features[2]**2)			
				pointFeatures[i] = np.array(point.features)				

			# Normalized to [-1,1] 
			for c in range(channel):
				maxValue = np.amax( pointFeatures[:,c] )
				minValue = np.amin( pointFeatures[:,c] )

				if maxValue != minValue:
					pointFeatures[:,c] = 2.0*(pointFeatures[:,c]-minValue)/(maxValue-minValue)-1.0
				elif maxValue == minValue and maxValue != 0:
					pointFeatures[:,c] /= maxValue	

			# reset point's features
			for i in range(pointFeatures.shape[0]):
				stroke.scaledSampledPoints[i].features = pointFeatures[i,:]

			# set tensor:
			for (i, point) in enumerate(stroke.scaledSampledPoints):
				self.PSFeatures[int(point.y+self.padding)][int(point.x+self.padding)][:] = point.features

	def uniformSampling(self, strokes):

		for stroke in strokes:
			stroke.sampledPoints = self.uniformSamplingPoints(stroke.rawPoints)
	
	def uniformSamplingPoints(self, inputPoints):		

		points = copy.deepcopy(inputPoints)

		# calculate resample spacing
		S = self.resampleSpacing(points)
		D = 0

		resamplePoints = list()
		prePoint = None

		i = 0
		while i < len(points):
			curPoint = points[i]
			if i == 0:
				prePoint = copy.deepcopy(curPoint)
				resamplePoints.append(copy.deepcopy(curPoint))
			else:
				d = dist(prePoint, curPoint)

				# if D + d >= S, then we can add a new point 
				if D + d >= S:
					# construct a new point
					newPoint = Point()
					
					if curPoint.x - prePoint.x == 0 and curPoint.y - prePoint.y == 0:
						prePoint = copy.deepcopy(curPoint)
						i += 1
						continue

					if curPoint.x - prePoint.x != 0:
						newPoint.x = prePoint.x + ( ( S - D ) / d ) * ( curPoint.x - prePoint.x ) 
					else:
						newPoint.x = prePoint.x

					if curPoint.y - prePoint.y == 0:
						newPoint.y = prePoint.y
					else:
						newPoint.y = prePoint.y + ( ( S - D ) / d ) * ( curPoint.y - prePoint.y )

					timeInterval = float(curPoint.timeStamp) - float(prePoint.timeStamp)
					deltaTime = S / d * timeInterval
					newPoint.timeStamp = prePoint.timeStamp + deltaTime

					retimeInterval = float(curPoint.reTimeStamp) - float(prePoint.reTimeStamp)
					redeltaTime = S / d * retimeInterval
					newPoint.reTimeStamp = prePoint.reTimeStamp + redeltaTime
					resamplePoints.append(copy.deepcopy(newPoint))
					points.insert(i, copy.deepcopy(newPoint))
					D = 0
					prePoint = copy.deepcopy(newPoint)
				else:
					D += d	
					prePoint = copy.deepcopy(curPoint)

			i += 1

		return resamplePoints

	def resampleSpacing(self, points, const=200.0):

		topLeftX = float( 'inf' )
		topLeftY = float( 'inf' )
		bottomRightX = -float( 'inf' )
		bottomRightY = -float( 'inf' )

		for point in points:
			if point.x <= topLeftX:
				topLeftX = point.x
			if point.x >= bottomRightX:
				bottomRightX = point.x
			if point.y <= topLeftY:
				topLeftY = point.y
			if point.y >= bottomRightY:
				bottomRightY = point.y

		diagonalDist = math.sqrt( (bottomRightX-topLeftX)**2\
					+ (bottomRightY-topLeftY)**2 )

		S = diagonalDist / const

		return S

def extraction( path, format ):

	#onlyfiles = [f for f in sorted(listdir(path)) if isfile(join(path, f))]

	signatures = list()
	labels = list()
	
	if format == "IAM":
		count = 0
		with open(path, "r") as trainFile:
			for line in trainFile.readlines():
				count += 1
				print(str(count) + ' / 775 finished')
				line = line.strip('\n')
				line = line[1:]
				for root, directories, files in sorted(os.walk(STROKES_DATA_PATH)):
					files = sorted(files)
					for file_name in files:
						if line in file_name:
							############# label data #############
							# split our .xml (eg: a01-020w-01.xml -> a01-020w-01)
							text_line_id = file_name[:-4]
							label_text_line = find_textline_by_id(text_line_id)
							# print(label_text_line)
							if len(label_text_line) == 0: 
								break
							#print(label_text_line)
							labels.append(label_text_line)
							############# trajectory data #############
							text_line_path = os.path.join(root, file_name)
							#print(file_name)
							#parser = ET.XMLParser(encoding = 'iso-8859-1')
							e_tree = ET.parse(text_line_path)
							e_tree = e_tree.getroot()
							
							signature = Signature()
							#signature.id = text_line_id

							curStrokeId = -1

							#  first for loop: find start and end time stamps
							for stroke in e_tree.findall('StrokeSet/Stroke'):
								signature.strokes.append(Stroke(curStrokeId))
								curStrokeId += 1
								if curStrokeId == 0:
									# used float instead of int here
									signature.startTimeStamp = float(stroke.get('start_time'))
								signature.endTimeStamp = float(stroke.get('end_time'))

							mx = int(e_tree.findall('WhiteboardDescription/DiagonallyOppositeCoords')[0].get('x'))
							my = int(e_tree.findall('WhiteboardDescription/DiagonallyOppositeCoords')[0].get('y'))

							#  second for loop: record stroke and point info
							curStrokeId = -1
							for stroke in e_tree.findall('StrokeSet/Stroke'):
								curStrokeId += 1
								for point in stroke.findall('Point'):
									if curStrokeId >= 0:
										signature.strokes[curStrokeId].addRawPoint(
											Point(int(point.get('x')),
												my-int(point.get('y')),
												float(point.get('time')) - signature.startTimeStamp,
												signature.endTimeStamp - float(point.get('time'))))


							signature.uniformSampling(signature.strokes)
							signature.scaleByHeight(signature.strokes, signature.targetHeight,0)
							signature.scaleByHeight(signature.strokes, signature.targetHeight,1)
							# signature.showStrokes(signature.strokes, 0, True, None, signature.label)
							# signature.showStrokes(signature.strokes, 1, True, None, signature.label)
							# signature.showStrokes(signature.strokes, 2, True, None, signature.label)
							signature.calculatePSFeatures(signature.strokes)
							signature.features = signature.PSFeatures
							signature.TEPSFeatures = None
							signature.TimeFeatures = None
							signatures.append(signature.features)
							

							# print(np.shape(signature.features))
							'''
							for c in range(7):
								image = imshow(signature.features[:,:,c], True)
								store_path = os.path.join(os.path.expanduser('~'),'Desktop','HTR_IAM','figures',text_line_id+'_PSF_'+str(c)+'.png')
								if not cv2.imwrite(store_path,image):
									raise Exception("Could not write image")
							'''
							print("Finished a file ", file_name)
		return [signatures, labels]
		'''
		with open('train_signatures.pickle', 'wb') as pickleOutput:
			pickle.dump(signatures, pickleOutput, protocol=pickle.HIGHEST_PROTOCOL)
		with open('train_labels.pickle', 'wb') as pickleOutput1:
			pickle.dump(labels, pickleOutput1, protocol=pickle.HIGHEST_PROTOCOL)
		'''
									
'''
		# parse STROKES (.xml)
		for root, directories, files in sorted(os.walk(STROKES_DATA_PATH)):
			directories = sorted(directories)
			files = sorted(files)
			for file_name in files:  # TextLine files
				if file_name != ".DS_Store":
					############# label data #############
					# split our .xml (eg: a01-020w-01.xml -> a01-020w-01)
					text_line_id = file_name[:-4]
					label_text_line = find_textline_by_id(text_line_id)
					if len(label_text_line) == 0: 
						break
					labels.append(label_text_line)
					############# trajectory data #############
					text_line_path = os.path.join(root, file_name)
					#print(file_name)
					#parser = ET.XMLParser(encoding = 'iso-8859-1')
					e_tree = ET.parse(text_line_path)
					e_tree = e_tree.getroot()
					
					signature = Signature()
					#signature.id = text_line_id

					curStrokeId = -1

					#  first for loop: find start and end time stamps
					for stroke in e_tree.findall('StrokeSet/Stroke'):
						signature.strokes.append(Stroke(curStrokeId))
						curStrokeId += 1
						if curStrokeId == 0:
							# used float instead of int here
							signature.startTimeStamp = float(stroke.get('start_time'))
						signature.endTimeStamp = float(stroke.get('end_time'))

					mx = int(e_tree.findall('WhiteboardDescription/DiagonallyOppositeCoords')[0].get('x'))
					my = int(e_tree.findall('WhiteboardDescription/DiagonallyOppositeCoords')[0].get('y'))

					#  second for loop: record stroke and point info
					curStrokeId = -1
					for stroke in e_tree.findall('StrokeSet/Stroke'):
						curStrokeId += 1
						for point in stroke.findall('Point'):
							if curStrokeId >= 0:
								signature.strokes[curStrokeId].addRawPoint(
									Point(int(point.get('x')),
										my-int(point.get('y')),
										float(point.get('time')) - signature.startTimeStamp,
										signature.endTimeStamp - float(point.get('time'))))


					signature.uniformSampling(signature.strokes)
					signature.scaleByHeight(signature.strokes, signature.targetHeight,0)
					signature.scaleByHeight(signature.strokes, signature.targetHeight,1)
					# signature.showStrokes(signature.strokes, 0, True, None, signature.label)
					# signature.showStrokes(signature.strokes, 1, True, None, signature.label)
					# signature.showStrokes(signature.strokes, 2, True, None, signature.label)
					signature.calculatePSFeatures(signature.strokes)
					signature.features = signature.PSFeatures
					signature.TEPSFeatures = None
					signature.TimeFeatures = None
					signatures.append(signature)
					
					for c in range(7):
						image = imshow(signature.features[:,:,c], True)
						store_path = os.path.join(os.path.expanduser('~'),'Desktop','HTR_IAM','figures',text_line_id+'_PSF_'+str(c)+'.png')
						if not cv2.imwrite(store_path,image):
							raise Exception("Could not write image")
					
			print("Finished a file ", files)
'''

		


def main():
	parser = argparse.ArgumentParser(description="Signature folder to be extracted")
	#parser.add_argument("--path", type=str, default="data/SVC2004/Task1", help="path directory of signature data")
	parser.add_argument("--path", type=str, default="data/IAM/lineStrokes", help="path directory of signature data")
	#parser.add_argument("--type", type=str, default="SVC", help="format of signature data")
	parser.add_argument("--type", type=str, default="IAM", help="format of signature data")
	parser.add_argument("--output", type=str, default="data/extracted", help="output file of generated data")
	opts = parser.parse_args()
	
	extraction(opts.path, opts.type)


if __name__ == '__main__':
	main()


