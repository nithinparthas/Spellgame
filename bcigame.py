# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 09:40:32 2020

@author: Nithin
"""
import cv2
import dlib
import numpy as np
import pickle
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import matplotlib.pyplot as plt
import math
from draw_flashboard import *
from eye_face_detect import *
from response_models import *
from bcigame_trie import *

 #Primary Simulation Control
TextFileName = 'spell1.txt'  # Name of file which contains the text 
simulation_mode = 2          # 0: Ideal simulation (no subject), auto-generated response 
                             # 1: Gaussian data generation  (no subject), generated from PDF
                             # 2: Subject webcam capture (normal operation)
                             
Show_eyes = 1                # 1: Show the processed webcam eye contours on the screen
scan_scheme = 0              # 0: Deterministic, 1:  Random
DFLT_SCAN_DURATION = 1.5     # Flashboard scan wait time in seconds (keep it very small for simulation_mode 0 and 1 as its not required)

#Secondary Simulation Control 

THRESHOLD = 250            # This is for the eye movement detect, leave it at 250 
DIST_THRESHOLD = 0.95      # Similar to Matlab, this variable determines a succesful character decode
bspace_prob = 0.05         # Probability of backspace
MIN_SCAN_DURATION = 1.0    # Min scan wait time in the case where dynamic speedup is used (flag_speedup has to be set to 1) in steps of INCR_SCAN_DURATION
INCR_SCAN_DURATION = 0.1   # Increment steps when dynamic speedup is used (flag_speedup has to be set to 1)
flag_em_fit = 1            # 1: Compute EM fit, 0: Do not
flag_fix_distraction = 1   # 1: Fix distraction, 0 : Do not
flag_distraction = 1       # 1: Check if there is user distraction 0: Do not
flag_speedup = 1           # 1: Speedup if there is no distraction to challenge focussed subjects 0: Do nothing
flag_em_stats = 1          # 1: Compute EM Algorithm statistics, 0: Do not
flag_mean_stats = 1        # 1: Compute statistics such as mean user movement etc. 0: Do not
flag_use_wordmodel = 1     # 1: Use word model to precompute probabilities, 0: Do not
flag_display_widget = 1    # 1: Display the guide information widget, 0: Do not
Show_image = 0             # 1: Will display processed webcam image of the eye contour in black and white 
debug_em_flag = 0          # 1: Debug dump/analysis of EM algorithm variables
debug_distraction = 0      # 1: Debug dump/analysis dump out of distraction computation variables
debug_idx_flag = 0         # 1: Debug dump out of internal sim control index variables
debug_wmodel = 0           # 1: Debug dump from word model transition probability computation
debug_speedup = 0          # 1: Debug dump from speedup (mode is valid with flag_speedup)
debug_dist_updates = 0     # 1: Debug updates to the dist variable
WidgetDisplayTime = 1000   # Instruction Widget display time in msec

scan_duration = DFLT_SCAN_DURATION
    # Keeping main routine clean, get definitions through function call
font, fontsize, rows, cols, charlist, left, right = get_variables()
bimage, gimage, grimage, rimage, textbgnd_img, display_bgndimg, eye  = create_tile(charlist) # Create the basic flashboard

with open(TextFileName, "r") as f:  # Read the text to be spelled now
    text=f.read()
#text = 'this_game_is_a_test_to_check_speed_of_subject_response_to_the_game'
#text = 'zyxwvrt'
 
cv2.destroyAllWindows()        # Remove any old windows created by the program (unclosed ones)
winname = "FLASHBOARD"         # Window name of the displayed flashboard
wintarget = "FOCUS"            # Window name of a target to indicate how much user show move the head

cv2.namedWindow(winname)
cv2.namedWindow(wintarget)
cv2.moveWindow(winname, 750,0) # Move windows to proper locations on laptop screen
cv2.moveWindow(wintarget, 0,100)
cv2.imshow(wintarget, eye)     # Display the window
cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)
textsrcimg = display_srctext(text, textbgnd_img)
displayimg = display_infoimg('STARTING ...', display_bgndimg)

img = cv2_get_full_image(bimage, gimage, rimage, grimage, textbgnd_img, displayimg, textsrcimg, 12, 0, [0,0], 0)
cv2.imshow(winname,img)
fp = open("trie.pkl","rb")
trie = pickle.load(fp)
fp.close()
totcharcnt = get_total_charcnt(charlist, trie)
cv2.waitKey(10000)

# Initialize Variables
declist, rowcol_scanorder = [], []
detector = dlib.get_frontal_face_detector()  # Use frontal face
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') 
#face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
   
cap = cv2.VideoCapture(0) # 0: Integrated laptop Webcam, 1: External Webcam
ret, img = cap.read()
thresh = img.copy()
kernel = np.ones((9, 9), np.uint8)

# Initialize internal variables
MAXCOUNT = len(text)
simstarttime = time.time()   
attpdf, nonattpdf, em_attprob_list, em_nonattprob_list, distraction_list, em_allprob_list = [], [], [], [], [0, 0], [0,0]
sum_att, sum_nonatt, attcount, nonattcount, errcnt, bspacecnt, bspace_char_flag, tot_scan, no_of_char, speedup_char_no = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
confirm_distraction = 0
modelatt = GaussianMixture(n_components=1, init_params='kmeans')
modelnonatt = GaussianMixture(n_components=1, init_params='kmeans') 

count = 0
no_of_char = 0
while count < MAXCOUNT:
   idx = 0   
   no_of_char = no_of_char + 1

   if flag_use_wordmodel: # If word_model is to be used for computing probabilities
      dist = initialize_dist(trie, charlist, declist, bspace_prob, totcharcnt, debug_wmodel)
      if debug_idx_flag:
          print("Initial dist is")
          my_print_float(dist, 1)
   else:  # If not, use uniform probabilities
      dist  = [1/36]*36 # Initialize dist variable, placeholder till language model is incorporated
      
   textimg = display_text(declist, textbgnd_img, 1)
   movement_att_list, movement_nonatt_list = [], []

   while max(dist) < DIST_THRESHOLD:
      tot_scan = tot_scan + 1 # Update statistics
      rowcol_idx, rowcol_scanorder = get_rowcol_idx(idx, scan_scheme, rowcol_scanorder) # Get rowcol index and store computed scan roder for random scan
    
      img = cv2_get_full_image(bimage, gimage, rimage, grimage, textimg, displayimg, textsrcimg, rowcol_idx, flag_fix_distraction, distraction_list, bspace_char_flag)
      cv2.imshow(winname,img)
                                                                        # Capture the user movement
      movement = capture_image(Show_eyes, Show_image, scan_duration, cap, detector, predictor, left, right, kernel, THRESHOLD)
      vld_att = check_if_attended(rowcol_idx, rows, cols, count, charlist, text, bspace_char_flag) # Check to see if attended character is flashed
            
      if debug_idx_flag:
          print("Attended variable is %d %d" %(vld_att, rowcol_idx))
      if simulation_mode == 0: # Ideal simulation
        movement_att, movement_nonatt, prob  = ideal_prob_for_debug(vld_att)
      elif simulation_mode == 1: # Data drawn from Gaussian distribution
        movement_att, movement_nonatt, prob, attpdf, nonattpdf = generate_mixture_prob(vld_att, attpdf, nonattpdf)
   #   print("movement = %1.2f prob=%1.2f\n" %(movement, prob))
      elif simulation_mode == 2: # Webcam and user movement is used
        movement_att, movement_nonatt, prob = get_prob(movement, vld_att)

      movement_att_list, movement_nonatt_list = update_movement_list(movement_att, movement_nonatt, movement_att_list, movement_nonatt_list)
      if flag_mean_stats: # This is to compute the running mean and variance as the simulation progresses
         sum_att, sum_nonatt, attcount, nonattcount = compute_means(vld_att, movement_att, movement_nonatt, sum_att, sum_nonatt, attcount, nonattcount)
    # get EM model prob for movement
      if (count > 2): # Ignore the first two characters so that sufficient statistics are collected for fit
         if flag_em_stats: # Compute em_statistics 
            em_prob, em_attprob_list, em_nonattprob_list, em_allprob_list = get_fit_prob(vld_att, movement_att, movement_nonatt, modelatt, modelnonatt, em_attprob_list, em_nonattprob_list, em_allprob_list)
         if flag_distraction: # This is to compute is subject is distracted
            confirm_distraction, distraction_list = check_for_distraction(vld_att, movement_att, movement_nonatt, em_attprob_list, em_nonattprob_list, em_allprob_list, sum_att, sum_nonatt, attcount, nonattcount, distraction_list, debug_distraction)
                  
             # If flag_speedup is set, compute the Scan duration speedup when user is accurate and focused
         scan_duration, speedup_char_no = update_scan_duration(DFLT_SCAN_DURATION, INCR_SCAN_DURATION, flag_speedup, debug_speedup, em_allprob_list, vld_att, sum_att, attcount,  sum_nonatt, nonattcount, movement_att, movement_nonatt, scan_duration, MIN_SCAN_DURATION, count, speedup_char_no)
  
     
      if flag_em_fit: # If EM fit is to be computed, update for every user response
         modelatt, modelnonatt = fit_em(movement_att_list, movement_nonatt_list, modelatt, modelnonatt, debug_em_flag)

          # Update_dist function updates the probability for the flashboard
      dist = update_dist(prob, rowcol_idx, rows, cols, dist, debug_dist_updates)
      if debug_idx_flag:
         print("curridx=%d idx_max=%d max(dist)=%1.2f prob=%1.2f newdist=%1.2f" %(idx, dist.index(max(dist)), max(dist), prob, max(dist)))
#    print("Idx=%d Prob is %f" %(idx,prob))
      if debug_idx_flag:
         print("Idx=%d"%idx)
      idx = increment_index(idx) # Next scan
   if debug_idx_flag:
      print("Idx is %d" %idx)
      my_print_float(dist,1)
   dec = charlist[dist.index(max(dist))] # Similar to matlab code take the max over all the 36 entries 
   count, declist, errcnt, bspacecnt, bspace_char_flag = update_decoded_list_auto(dec, declist, count, text, errcnt, bspacecnt, bspace_char_flag, debug_idx_flag)
   
   if bspace_char_flag:
      displayimg = display_infoimg('ERROR: ' + dec + ' NEXT: <', display_bgndimg)
      if flag_display_widget:
        display_message("ERROR " + dec + " NEW SCAN BEGINS, NEXT = <", WidgetDisplayTime)
   elif count < len(text):
      displayimg = display_infoimg('LAST: ' + dec + ' NEXT: ' + text[count],  display_bgndimg)
      if flag_display_widget:
        display_message("NEW SCAN BEGINS, NEXT = "+ text[count], WidgetDisplayTime)
   textimg = display_text(declist, textbgnd_img, 1)  # Decoded text information to output on screen
   img = cv2_get_full_image(bimage, gimage, rimage, grimage, textimg, displayimg, textsrcimg, 12, flag_fix_distraction, distraction_list, 0)
   cv2.imshow(winname,img)
   cv2.waitKey(1500) # Short timer prior to beginning new scan

simendtime = time.time()
displayimg = display_infoimg('GAME DONE', display_bgndimg)
img = cv2_get_full_image(bimage, gimage, rimage, grimage, textimg, displayimg, textsrcimg, 12, flag_fix_distraction, distraction_list, 0)
cv2.imshow(winname,img)
cv2.waitKey(1000) # Short timer before ending simulation
finalerr = compute_decoded_errors(text, declist)
print("Input text: %s" %text)
print("Decoded text: %s" %(''.join(declist)))
print("Errors including corrections=%d error_rate=%f BackspaceCnt=%d" %(errcnt, errcnt/len(text), bspacecnt))
print("Total Flashboard Simulation Time %1.2f sec, Avg Time/Char = %1.2f sec, finalerr=%d, final_err_rate=%1.2f" %((simendtime - simstarttime),(simendtime - simstarttime)/MAXCOUNT, finalerr, finalerr/len(text)))
print("Average flashes/char = %1.2f Average_wo_redo=%1.2f" %(tot_scan/len(text), (tot_scan/no_of_char)))
cv2.destroyAllWindows()  # Cleanup by closing all generated windows
                           

