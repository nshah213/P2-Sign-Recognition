import pickle
import os

training_file = './Data_Pickle_Files/train.p'
#validation_file= './Data_Pickle_Files/valid.p'

with open(training_file, mode='rb') as f:
    data = pickle.load(f)
    
X_train, y_train = data['features'], data['labels']

import argparse
import cv2
import numpy as np
cv2.namedWindow("ROI",cv2.WINDOW_NORMAL)

training_shape_file = './Data_Pickle_Files/shapeTypeTrain.p'

"""
### Run once to collect the shape of all the unique signs in the training dataset
#	Saves datafile './Data_Pickle_Files/shapeTypeTrain.p'with information of the shape 
#	of the unique signs in the dataset and the label of the sign for further use

ts_shape = []
ts_shape_label = []
(unique,idx) = np.unique(y_train, return_index=True)
print(idx)
j=0
for k in idx:
	target_value = unique[j]
	j+=1
	index = np.where(y_train == target_value)
	image_blue = np.reshape(X_train[k][:,:,0],(32,32,1))
	image_red = np.reshape(X_train[k][:,:,2],(32,32,1))
	image_green = np.reshape(X_train[k][:,:,1],(32,32,1))
	image = np.concatenate((image_red,image_green,image_blue),2)
	print(np.shape(image))
	cv2.imshow('ROI',image)
	
	# if the 'r' key is pressed, reset the cropping region
	ts_shape_label.append(target_value)	
	outerBreak = True	
	while(outerBreak):
		key = cv2.waitKey(1) & 0xFF		
		if (key == ord("t")):
			ts_shape.append(0)
			outerBreak = False
		if (key ==ord("c")):
			ts_shape.append(1)
			outerBreak = False
		if (key ==ord("q")):
			ts_shape.append(2)
			outerBreak = False

p_file = training_shape_file	#if not os.path.isfile(p_file):
print('Saving data to pickle file...')
try:
	with open(training_shape_file, 'wb') as pfile:
			pickle.dump(
       		{
				'ts_shape': ts_shape,
				'ts_shape_label': ts_shape_label
       		},
			pfile, pickle.HIGHEST_PROTOCOL)
except Exception as e:
	print('Unable to save data to', pickle_file, ':', e)
	raise
"""

#cv2.namedWindow("Blue",cv2.WINDOW_NORMAL)
#cv2.namedWindow("Red",cv2.WINDOW_NORMAL)
#cv2.namedWindow("Green",cv2.WINDOW_NORMAL)
#cv2.namedWindow("Edge",cv2.WINDOW_NORMAL)

### Load the data from pickel file containing the shape of all the unique signs in the training dataset
#	loads datafile './Data_Pickle_Files/shapeTypeTrain.p'with information of the shape 
#	of the unique signs in the dataset and the label of the sign

with open(training_shape_file, mode='rb') as f:
    data = pickle.load(f)
    
ts_shape, ts_shape_label = data['ts_shape'], data['ts_shape_label']
(unique,idx) = np.unique(y_train, return_index=True)

subset_crop = []
subset_index = []
scale_factor = 2
cropped_images_train =[]
for j in range (len(X_train)):#(10000,11000):	
	image = X_train[j]
	#image_blue = np.reshape(X_train[j][:,:,0],(32,32,1))
	#print(np.shape(image_blue))
	#image_red = np.reshape(X_train[j][:,:,2],(32,32,1))
	#image_green = np.reshape(X_train[j][:,:,1],(32,32,1))	
	#image = np.concatenate((image_red,image_green,image_blue),2)
	roi = []
	for m in range (len(idx)):
		if (y_train[j]==ts_shape_label[m]):
			break
	if ts_shape[m] == 1:	
		#print(np.shape(np.zeros((32,32,1),dtype = np.uint8)))
		#print(np.shape(image_red))
		tmp = np.zeros((32,32,1),dtype = np.uint8)
		#cv2.imshow('Blue',np.concatenate((image_blue,tmp,tmp),2))
		#cv2.imshow('Red',np.concatenate((tmp,tmp,image_red),2))
		#cv2.imshow('Green',np.concatenate((tmp,image_green,tmp),2))
		#mask_red = [image_red>(image_)]	
		#blur = cv2.bilateralFilter(image,5,200,200)
		RED_MIN = 0
		RED_MAX = 30 # 45/360*180 for opencv hsv
		BLUE_MIN = 110 # 220 on 360 scale
		BLUE_MAX = 132 # 265 on 360 scale	

		#zoom = cv2.resize(image,(32*scale_factor,32*scale_factor),interpolation = cv2.INTER_LINEAR)
		grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	
		#hsvColor = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		#print()
		#print(grayScale[:,0])
		#print()
		lower_red = np.array([RED_MIN, 0, 0])
		upper_red = np.array([RED_MAX, 255, 255])
		lower_blue = np.array([BLUE_MIN, 20, 20])
		upper_blue = np.array([BLUE_MAX, 255, 255])
		#create a color selection based mask for the image
		#maskRed = cv2.inRange(hsvColor,lower_red,upper_red)
		#maskBlue = cv2.inRange(hsvColor,lower_blue,upper_blue)
		#image_hue = np.reshape(maskRed[:,:,0],(32,32,1))
		#print(maskRed[0])	
		#print(np.shape(maskRed))	
		#print(grayScale[0])
		#tmp2 = grayScale.copy()
		#red_region = cv2.bitwise_and(grayScale, maskRed)
		#print(red_region[0])
		#grayScaleZoom = cv2.cvtColor(zoom, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(grayScale,(3,3),0)
		#print(zoom[0])	
		#zoom = cv2.resize(image,(32*scale_factor,32*scale_factor),interpolation = cv2.INTER_CUBIC)
		#blurZoom = cv2.GaussianBlur(grayScaleZoom, (7,7),0.1)
		#atZoom = blurZoom.copy()	
		#atZoom = cv2.adaptiveThreshold(blurZoom,1.,1, 0, blockSize = 5, C = 0)
		#clahe = cv2.createCLAHE(tileGridSize = (32,32))
		#image_gray = clahe.apply(grayScale)
		image_gray = cv2.equalizeHist(grayScale)
		image_edge = cv2.Canny(image_gray,30,60)
		#tmp2 = cv2.equalizeHist(red_region)	
#		#image_edges = cv2.Canny(image_gray,2,4,apertureSize = 3, L2gradient = True)
		#image_edges_zoom = cv2.GaussianBlur(cv2.Canny(blurZoom, 40,80),(61,61),0.1)
		#cv2.imshow('Edge',image_edges)
		circles = cv2.HoughCircles(image_gray,cv2.HOUGH_GRADIENT,3,30,param1=50,param2=1,minRadius=8,maxRadius=16)
		image_circle_draw = image.copy()
		image_circle = image.copy()
		#image_contour = image.copy()	
		if (isinstance(circles,type(None))==False):
			circles = np.uint16(np.around(circles))
			for i in circles[0,:]:
				# draw the outer circle
				cv2.circle(image_circle_draw,(i[0],i[1]),i[2],(0,255,0),1)
				# draw the center of the circle
				cv2.circle(image_circle_draw,(i[0],i[1]),2,(0,0,255),1)
			#ROI_Mask = selectCircularROI(circle[0], 32, 32)
			#image_circle = cv2.bitwise_and(image, )
			circle = circles[0]		
			#print(circle)
			x_Lef, y_Top = circle[0,0].astype(int)-circle[0,2].astype(int)-2, circle[0,1].astype(int)-circle[0,2].astype(int)-2
			x_Rig, y_Bot = circle[0,0].astype(int)+circle[0,2].astype(int)+2, circle[0,1].astype(int)+circle[0,2].astype(int)+2
			if x_Lef < 0:
				x_Lef = 0
			if y_Top < 0:
				y_Top = 0
			if x_Rig > 32:
				x_Rig = 32
			if y_Bot > 32:
				y_Bot = 32
			assert(y_Top < y_Bot)
			assert(x_Lef < x_Rig)
	
			new_scale_horz = 32
			new_scale_vert = 32
			#print(Pt1_sel)
			# if there are two reference points, then crop the region of interest
			# from teh image and display it
			Neg_bias = -1
			Pos_bias = 1
			#y_Top_bias = -4
			#y_Bot_bias = 4
			x_Lef += Neg_bias
			y_Top += Neg_bias
			x_Rig += Pos_bias
			y_Bot += Pos_bias
			if x_Lef < 0:
				x_Lef = 0
			if y_Top < 0:
				y_Top = 0
			if x_Rig > 32:
				x_Rig = 32
			if y_Bot > 32:
				y_Bot = 32
			roi = cv2.resize(image[x_Lef:x_Rig, y_Top:y_Bot],(new_scale_horz,new_scale_vert),interpolation = cv2.INTER_CUBIC)
		if (isinstance(circles,type(None))==True):
			roi = image.copy()
	else:
		roi = image.copy()
	cv2.imshow("ROI", roi)	
	cropped_images_train.append(roi)	
	debug_on = False	
	while(debug_on):	
		#cv2.imshow("ROI", image_contour)		
		key = cv2.waitKey(1) & 0xFF
		if (key == ord("c")):
			debug_on = False 
### Function - fillCircle
#		Used for selecting circular ROI to fit circle detected to represent the traffic sign
#	Input - circle to select as output of HoughCircles, x_len of image, y_len of image
def selectCircularROI(circle, x_len = 32, y_len = 32):
	ROI = np.zeros((32,32,1),dtype = np.uint8)
	for i in range(x_len):
		for j in range(y_len):
			if ((i-circle[0])**2 + (j-circle[1])**2 - circle[2]**2<= 0):
				ROI[i,j,0] = 255
	return ROI

if (True):
	pickle_file = './Data_Pickle_Files/croppedUnprocessedTrain_circle2.p'
	print('Saving data to pickle file...')
	try:
		with open('./Data_Pickle_Files/croppedUnprocessedTrain_circle2.p', 'wb') as pfile:
			pickle.dump(
                   {
	                   'train_dataset_crop': cropped_images_train
                   },
                   pfile, pickle.HIGHEST_PROTOCOL)
	except Exception as e:
		print('Unable to save data to', pickle_file, ':', e)
		raise

print('Data cached in pickle file.')
# close all open windows
cv2.destroyAllWindows()
"""	
	#tmp1 = cv2.Canny(image_edges,1,2)
	#cv2.imshow('Edge',image_edges_zoom)	
	cv2.imshow('Edge',image_edges)
	lines = cv2.HoughLinesP(image_edges,rho = 3,theta = np.pi/360, threshold = 3, minLineLength = 5, maxLineGap = 0)	
	#img, contours, hierachy = cv2.findContours(image_edge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	print(np.shape(lines))
	if (isinstance(lines,type(None))==False):	
		for l in range(len(lines)):	
			x0,y0,x1,y1 = lines[l,0,:]
			cv2.line(image_contour, (x0,y0),(x1,y1),(0,0,255),1)
		
			#approx = cv2.approxPolyDP(cnt,0.02*cv2.arcLength(cnt,True),True)	
		debug_on = False			
		while(debug_on):	
			cv2.imshow("ROI", image_contour)		
			key = cv2.waitKey(1) & 0xFF
			if (key == ord("c")):
				debug_on = False
	#image_contours = cv2.findContours
	#image_triangles = cv2.minEnclosingTriangle(image_gray)
	# print(image_triangle)
	#idx = idx + 1
"""

