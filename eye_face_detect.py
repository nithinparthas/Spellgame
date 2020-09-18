# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 22:57:11 2020

@author: Nithin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 17:22:05 2020

@author: Nithin
"""
import cv2
import dlib
import numpy as np
import time

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def eye_on_mask(mask, side, shape):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
    except:
        pass
    
def nothing(x):
    pass


# loop over frames from the video stream

def dummy_timer(scan_duration):
     start_time = time.time()
     curr_time  = time.time()
     while (curr_time - start_time < scan_duration ):
        curr_time = time.time() 
     cv2.waitKey(1)
     
def capture_image(Show_eyes, Show_image, scan_duration, cap, detector, predictor, left, right, kernel, THRESHOLD):

  mid_prev = 0
  mid_diff = 0
  max_mid_diff = 0
  min_mid_diff = 0
  first_time = 1
  frm_cnt = 0
  start_time = time.time()
  while( (time.time() - start_time) < scan_duration ):
    frm_cnt = frm_cnt + 1  
#    print("time=%1.2f start=%1.2f frm=%d" %(time.time(), start_time, frm_cnt))
    ret, img = cap.read()
    #print("currr=%f st=%f REACHED\n" % time.time(), start_time)
    # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Commenting out zoom in function
    # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # img = cv2.resize(img, (w*3,h*3))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = gray
    rects = detector(gray, 1)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left, shape)
        mask = eye_on_mask(mask, right, shape)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        if not first_time:
            mid_diff = mid - mid_prev   
            if mid_diff > max_mid_diff:
                max_mid_diff = mid_diff
            if mid_diff < min_mid_diff:
                min_mid_diff = mid_diff
        else:
            first_time = 0
        mid_prev = mid
 
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
 #       threshold = cv2.getTrackbarPos('threshold', 'image')
        threshold = THRESHOLD
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2) #1
        thresh = cv2.dilate(thresh, None, iterations=4) #2
        thresh = cv2.medianBlur(thresh, 3) #3
        thresh = cv2.bitwise_not(thresh)
        contouring(thresh[:, 0:mid], mid, img)
        contouring(thresh[:, mid:], mid, img, True)
        # for (x, y) in shape[36:48]:
        #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
    # show te image with the face detections + facial landmarks
    if Show_eyes:
       imgS = cv2.resize(img, (320, 240))
       cv2.moveWindow('eyes', 350,150)
       cv2.imshow('eyes', imgS)
    if Show_image:
       threshS = cv2.resize(thresh, (320, 240))
       cv2.moveWindow('gaze', 150,300)
       cv2.imshow('gaze', threshS)
      
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#  print("MAX is %d\n" %max_mid_diff)

  return(max_mid_diff)

