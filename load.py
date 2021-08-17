import os
import torch
from torchvision.io import read_image
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
import pickle
from PSFforIAM import *

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


with open('train_signatures.pickle', 'rb') as f:
    train_signatures = pickle.load(f)

with open('train_labels.pickle', 'rb') as f:
    train_labels = pickle.load(f)



#train_signatures = np.array(train_signatures)
#train_labels = np.array(train_labels)


print(len(train_signatures))
#print(np.array(train_signatures[1]).shape)
print(train_labels[1])
#print(train_signatures[0].shape)

