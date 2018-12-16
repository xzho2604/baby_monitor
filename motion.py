#https://software.intel.com/en-us/node/754940

from __future__ import print_function
import numpy as np
import cv2
import argparse

sdThresh = 10
font = cv2.FONT_HERSHEY_SIMPLEX
#TODO: Face Detection 1

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument( "--m")
args = vars(ap.parse_args())
flag = 1 if args['m'] =="night" else 0

#gamma transformation for night mode
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

#calculating distance between different frames
def distMap(frame1, frame2):
    """outputs pythagorean distance between two frames"""
    frame1_32 = np.float32(frame1)
    frame2_32 = np.float32(frame2)
    diff32 = frame1_32 - frame2_32
    norm32 = np.sqrt(diff32[:,:,0]**2 + diff32[:,:,1]**2 + diff32[:,:,2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
    dist = np.uint8(norm32*255)
    return dist

cv2.namedWindow('frame')
cv2.namedWindow('dist')

#capture video stream from camera source. 0 refers to first camera, 1 referes to 2nd and so on.
cap = cv2.VideoCapture('http://192.168.1.12:8080/video')

_, frame1 = cap.read()
_, frame2 = cap.read()


facecount = 0
while(True):
    _, frame3 = cap.read()
    if flag :frame3 = adjust_gamma(frame3,gamma =3.0) 

    rows, cols, _ = np.shape(frame3)
    cv2.imshow('dist', frame3)
    dist = distMap(frame1, frame3)

    frame1 = frame2
    frame2 = frame3

    # apply Gaussian smoothing
    mod = cv2.GaussianBlur(dist, (9,9), 0)

    # apply thresholding
    _, thresh = cv2.threshold(mod, 100, 255, 0)

    # calculate st dev test
    _, stDev = cv2.meanStdDev(mod)

    cv2.imshow('dist', mod)
    cv2.putText(frame2, "Standard Deviation - {}".format(round(stDev[0][0],0)), (70, 70), font, 1, (255, 0, 255), 1, cv2.LINE_AA)
    if stDev > sdThresh:
            print("Motion detected.. Do something!!!");
            print('\a');
            #TODO: Face Detection 2

    cv2.imshow('frame', frame2)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

