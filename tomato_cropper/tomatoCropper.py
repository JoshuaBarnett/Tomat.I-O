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

a complementary program that might help to develop would be one to filter through a directory, show images and have user assign as tomato or not, then sort accordingly.
With heuristically mined data, it would allow us to maintain 100% accuracy with greatly increased throughput time
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
        img = cv2.imread(image,1) # read in image
        img2 = cv2.medianBlur(img,5) #duplicate and blur
        img2 = cv2.Canny(img2,.9,.6,3) #canny edge detect on blurred
        #cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR) #not sure
        circles = cv2.HoughCircles(img2,cv.CV_HOUGH_GRADIENT,1,20,
		                            param1=100,param2=50,minRadius=10,maxRadius=75) #find circcles in detected edge img (play with params to improve performance)
        circles = np.uint16(np.around(circles)) #make nparray of [[xcent,ycent,radius]]
        '''
         May want to experiment with different min/max radii, if the min is too
		small, it will find circles in everything, also high resolution images
		take forever, may want to scale them down first param1 and 2 seem to have sig. effect on results
        '''
    cropped_images['cropped_' + image] = img2 #?
    ''' #maybe split function here?'''
    os.chdir('../croppedImages') #switch to cropped directory
    croppedImgPadSize = 30 #HOW MUCH PIXEL PADDING AROUND CIRCLE WHEN CROPPED
    xx = 0
    for i in circles[0,:]:#for each circle found
        xx += 1
        print str(i[0]) + " " + str(i[1]) + " " + str(i[2]) #print circle info
        cv2.imwrite(str(xx)+".png",img[i[0]-i[2]-croppedImgPadSize:i[2]+i[0]+croppedImgPadSize,i[1]-i[2]-croppedImgPadSize:i[1]+i[2]+croppedImgPadSize]) #write img of circle contents w/ pad
        if i[2] > 100:#?
# draw the outer circle
            cv2.circle(img2,(i[0],i[1]),i[2],(0,255,0),2)
# draw the center of the circle
            cv2.circle(img2,(i[0],i[1]),2,(0,0,255),3)


    #for image in cropped_images:
    #    cv2.imwrite(image, img2)

    return cropped_images.keys() #?

def crop_and_resize(image_list):
	cropped_images = testCV(image_list)
	resize_images(cropped_images) #current dir = croppedImages 
'''#maybe resize before saving file? will savve hella memory+time'''

def main():
	testCV(get_images())

main()
