import cv2
import cv2.cv as cv
import Image
import numpy as np
import os
from os import listdir
from os.path import isfile, join

'''
Requires that three folders be present where the script is run:
tomatoImages = raw, unprocessed images
croppedImages = after finding circles
resizedImages = resized to 50 x 50

The functions named crop don't actually crop, just find circles,  I never got
the circles to reliably work so I didn't figure out a way to crop stuff out
'''

def resize_images(image_list):
	size = 50, 50
	resized_images = {}
	
	for image in image_list:
		title = 'resized_' + image.split('_')[1].split('.')[0]
		image = Image.open(image)
		image.thumbnail(size, Image.ANTIALIAS)
		resized_images[image] = title

	os.chdir('../resizedImages')
	
	for image in resized_images:
		image.save(resized_images[image], 'PNG')

def get_images():
	return [ f for f in listdir('./tomatoImages') if isfile(join('./tomatoImages',f)) ]

#from a CV tutorial about finding circles
def testCV(image_list):
	os.chdir('./tomatoImages')
	cropped_images = {}

	for image in image_list:
		img = cv2.imread(image,0)
		img = cv2.medianBlur(img,5)
		cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

		'''
		May want to experiment with different min/max radii, if the min is too
		small, it will find circles in everything, also high resolution images
		take forever, may want to scale them down first
		'''
		circles = cv2.HoughCircles(img,cv.CV_HOUGH_GRADIENT,1,20,
		                            param1=50,param2=30,minRadius=0,maxRadius=0)

		circles = np.uint16(np.around(circles))
		for i in circles[0,:]:
			if i[2] > 100:
			    # draw the outer circle
			    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
			    # draw the center of the circle
			    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

		cropped_images['cropped_' + image] = cimg

	os.chdir('../croppedImages')

	for image in cropped_images:
		cv2.imwrite(image, cimg)

	return cropped_images.keys()

def crop_and_resize(image_list):
	cropped_images = testCV(image_list)
	resize_images(cropped_images) #current dir = croppedImages 


def main():
	crop_and_resize(get_images())

main()