import os
import glob
import argparse

# Kerasa / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

from keras.models import load_model
from layers import BilinearUpSampling2D
from loss import depth_loss_function
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt
import numpy as np 
import cv2
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import skimage
from skimage.transform import resize
import cv2
import math 


cap = cv2.VideoCapture(0)
plasma = plt.get_cmap('plasma')

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=True)

print('\nModel loaded ({0}).'.format(args.model))




# Input Videos
while(True):
	ret, inputs = cap.read()
	inputs = cv2.resize(inputs, (800,800) )
	#IN = inputs
	binary_img = cv2.cvtColor(inputs, cv2.COLOR_BGR2GRAY)
	binary_img = cv2.GaussianBlur(binary_img, (7,7),0)  
	ret,binary_img = cv2.threshold(binary_img,100,255,cv2.THRESH_BINARY)
	binary_img = cv2.resize(binary_img,(960,960))
	IN = cv2.resize(inputs,(960,960))

	inputs = np.clip(np.asarray(inputs, dtype=float) / 255, 0, 1)
	outputs = predict(model, inputs)
	rescaled = outputs[0][:,:,0]	 
	rescaled = rescaled - np.min(rescaled)
	rescaled = rescaled / np.max(rescaled)
	outputs = plasma(rescaled)[:,:,0]
	#outputs = outputs[0 : 240 , 0 : 240] 
	outputs = cv2.resize(outputs, (800,800) )




	img = outputs
	 
	delta_y = 200
	delta = 200
	delta_unit = 40
	shift = 10
	center = (img.shape[0] / 2 )
	frame_ref = img[ center - (delta_y/2)  : center + (delta_y/2) , 0 : 800 ]
	center_r  = center
	center_l  = center  
	center_x  = center 
	rotation  = 0
	flag = 0 
	threshold = 0.85
	font = cv2.FONT_HERSHEY_SIMPLEX	
	R = 2

	check = 0
	


	while(center_l != (delta/2) and center_r != (img.shape[0] - (delta/2))):

	        frame = frame_ref[ 0 : delta_y , center_x - (delta/2)  : center_x + (delta/2) ] 	
	        count = 0
	        
	        while(count != 9):
	                frame_quanta = frame[  0 : delta_y , delta_unit * (count) : delta_unit * (count + 1) ]
	                
	                
	                average = ( np.average(frame_quanta) )	                 
	                average = (average * math.exp(3*(average-1)))
	                average = 1 - average
	                print average
	                
	                if(average > threshold and (rotation % 2 == 0)):
	                        rotation = rotation + 1  
	                        center_l = center_l - shift
	                        center_x = center_l 
	                        break 

	                elif(average > threshold and (rotation % 2 != 0)):
	                        rotation = rotation + 1     
	                        center_r = center_r + shift                                   
	                        center_x = center_r
	                        break 

	                else:
	                        count = count + 1

	        if (count == 9):

	            if (rotation % 2 == 0 and center_l != center and center_r != center):
	                R = -1

	            elif (rotation % 2 != 0):
	                R = 1

	            elif (center_r == center and center_l == center):
	                R = 2

	            flag = 1
	            break 


 	
	if (rotation % 2 == 0 and center_l != center and center_r != center):
	        R = 1

	elif (rotation % 2 != 0):
	        R = -1


	x1 = center_x - (delta/2)
	y1 = center   - (delta_y/2)
	x2 = center_x + (delta/2)
	y2 = center   + (delta_y/2)

	

	if (flag != 0):
	    image_ocv = cv2.rectangle(inputs, (x1, y1), (x2, y2), (255,255,255), 1)

	elif (flag == 0 or math.isnan(distance) or (distance < 0.5)):                    
	    cv2.putText(inputs,'NO PATH',(350,400), font, 1,(255,255,255),1,cv2.LINE_AA)       


	if   (R == 1):
	    cv2.putText(inputs,'Rotate Right',(350,400), font, 1,(0,0,0),1,cv2.LINE_AA)


	elif (R == -1):
	    cv2.putText(inputs,'Rotate Left',(350,400), font, 1,(0,0,0),1,cv2.LINE_AA)





	cv2.imshow("Disparity Map" , img)
	cv2.imshow("Frame" , inputs)

	if (cv2.waitKey(1) & 0xFF == ord('q')):
	    break