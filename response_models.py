# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:18:06 2020

@author: Nithin
"""

import numpy as np
from numpy.random import normal
from numpy import hstack
import math
from sklearn.mixture import GaussianMixture
import random
from PIL import ImageFont
import time
from gtts import gTTS 
import os
import playsound

def get_words(text):
    
    textwords = [word+'_' for word in text.split('_')]  
    print(textwords)
    textwords[len(textwords)-1] = textwords[len(textwords)-1][:-1]  # Remove _ from final word in the document
    return(textwords)

def initialize_audio():
    if os.path.exists('./decoded.mp3'):
       os.remove("decoded.mp3")
       
def create_audio(dec):
    
   # Passing the text and language to the engine, here we have marked slow=False. Which tells  
   # the module that the converted audio should have a high speed, language = English
       
   if dec == '_':
      dec = 'space'
   elif dec == '<':
      dec = 'backspace'
   elif dec == '.':
      dec = 'period'
   myobj = gTTS(text=dec, lang='en', slow=False)  
   # Saving the converted audio in a mp3 file named welcome  
   myobj.save('decoded.mp3') 
  
   # Playing the converted file 
   playsound.playsound('decoded.mp3')
   os.remove("decoded.mp3")
   
def my_timer(delay):
    simstarttime = time.time() 
    currtime = time.time()
    while currtime < (simstarttime + delay):
        currtime = time.time()
        
        
def get_variables():
    
    
    charlist = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
            'x', 'y', 'z', '1', '2', '3', '4', '5', '6', '7', '.', '<', '_']
    rows = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5]
    cols = [6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11]
    fontsize = 52
    font = ImageFont.truetype("arial.ttf", fontsize)
    left = [36, 37, 38, 39, 40, 41]
    right = [42, 43, 44, 45, 46, 47]
    dist = [1.0/36.0]*36
    dist_sort_loc = list(range(36))
    return(font, fontsize, rows, cols, charlist, left, right, dist, dist_sort_loc)

def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


def update_dist(prob, idx, rows, cols, order_diag,  dist, debug_dist_updates, flashboard_type, charlist_diag):
    
    probs = [1]*36 # Initialize list
    if flashboard_type == 0:
      for r in range (0,36):
        if rows[r] == idx:
           probs[r] = prob 

        if cols[r] == idx:
           probs[r] = prob 
    elif flashboard_type == 1:
      for r in range (0,36):
        if rows[r] == idx:
           probs[order_diag[r]] = prob
        if cols[r] == idx:
           probs[order_diag[r]] = prob
           
    avg_probs = [x/sum(probs) for x in probs]

    for r in range (0,36):
        dist[r] = dist[r]*avg_probs[r]
                      
    sum_dist = sum(dist)
                                         # Normalize without 34 and then renormalize with 34
    if sum_dist > 0:
        dist = [x/sum_dist for x in dist]
 #   my_print_float(dist, 1)
    if debug_dist_updates:
      print("DIST is")
      my_print_float(dist, 1)  
 #   for x in range(len(dist)):
 #       print("%1.2f" %dist[x])
    return(dist)   

def increment_index(idx): # Increment index and wrap around
    idx = idx + 1
    if idx == 12:
       idx = 0
    return(idx)


def check_if_attended(flag_use_wordmodel, currword, words, idx, rows, cols, count, charlist_reg, charlist_diag, text, bspace_char_flag, flashboard_type, loc):
    
    vld_att = 0
    if flashboard_type == 0:
        charlist = charlist_reg
    elif flashboard_type == 1:
        charlist = charlist_diag
        
    if bspace_char_flag > 0: # Next char is a backspace because of previous error
        currchar = charlist.index('<')   # '<' is backspace
    else:
        currchar = charlist.index(text[count])
                     
    if idx < 6:  # Find attended character based on row and col index
                 # First look to see if the word is an attended word, then move on to characters
        if flag_use_wordmodel and len(words) > 0 and currword in words:           
          if idx == rows[charlist.index('7')] and len(words[0]) > 0 and words[0]==currword:
              vld_att = 1
          elif idx == rows[charlist.index('6')] and len(words[1]) > 0 and words[1]==currword:
              vld_att = 1              
          elif idx == rows[charlist.index('5')] and len(words[2]) > 0 and words[2]==currword:
              vld_att = 1
          elif idx == rows[charlist.index('4')] and len(words[3]) > 0 and words[3]==currword:
              vld_att = 1   
          elif idx == rows[charlist.index('3')] and len(words[4]) > 0 and words[4]==currword:
              vld_att = 1
        elif idx == rows[currchar]: # Attended case
           vld_att = 1
    else:
        if flag_use_wordmodel and len(words) > 0 and currword in words:    
          if idx == cols[charlist.index('7')] and len(words[0]) > 0 and words[0]==currword:
              vld_att = 1
          elif idx == cols[charlist.index('6')] and len(words[1]) > 0 and words[1]==currword:
              vld_att = 1              
          elif idx == cols[charlist.index('5')] and len(words[2]) > 0 and words[2]==currword:
              vld_att = 1
          elif idx == cols[charlist.index('4')] and len(words[3]) > 0 and words[3]==currword:
              vld_att = 1   
          elif idx == cols[charlist.index('3')] and len(words[4]) > 0 and words[4]==currword:
              vld_att = 1
        elif idx == cols[currchar]: # Attended case
           vld_att = 1
      
      
#    print(idx, bspace_char_flag, currword, currchar, count, text[count], vld_att)  # DELETE
    return(vld_att)


def ideal_prob_for_debug(vld_att):
    
    if vld_att:
       prob = normpdf(10, 10, 1)/normpdf(10, 2, 1)
       movement_att = 10
       movement_nonatt = 100
    else:
       prob = normpdf(2, 10, 1)/normpdf(2, 2, 1) 
       movement_att = 100
       movement_nonatt = 2
    return(movement_att, movement_nonatt, prob)    

def generate_mixture_prob(vld_att, attpdf, nonattpdf):
    
    mix_prob_att = np.random.uniform(0, 1)
    mix_prob_nonatt = np.random.uniform(0, 1)
    movement_att = 100 # Note that interpretion of 100 is that data is not valid
    movement_nonatt = 100
    if vld_att:
       if  mix_prob_att >= 0.3:
          movement_att  = normal(10, 1, 1) # Formal is (mean, variance, no_samples)
          prob = normpdf(movement_att, 10, 1)/normpdf(movement_att, 2, 1)
          attpdf.append(1)
       else:
          movement_att = normal(4, 1, 1)
          prob = normpdf(movement_att, 10, 1)/normpdf(movement_att, 2, 1)
          attpdf.append(0)
    else:
       if  mix_prob_nonatt >= 0.5:
                                    # Non-attended case
          movement_nonatt = normal(2, 1, 1)     
          prob = normpdf(movement_nonatt, 10, 1)/normpdf(movement_nonatt, 2, 1)
          nonattpdf.append(1)
       else:
          movement_nonatt = normal(1, 1, 1)     
          prob = normpdf(movement_nonatt, 10, 1)/normpdf(movement_nonatt, 2, 1)
          nonattpdf.append(0)
              
 #   print("text is %s, idx is %d, maxd_att=%1.2f maxd_nonatt=%1.2f prob=%1.2f\n" %(text[count], idx, movement_att, movement_nonatt, prob))
 #   print("dist_att=%d dist_nonatt=%d" %(dist_att, dist_nonatt))

    return(movement_att, movement_nonatt, prob, attpdf, nonattpdf)

def get_prob(movement, valid_att):    
    num = normpdf(movement, 10, 1)
    denom = normpdf(movement, 2, 1)
    prob = 0
    if denom != 0:
      prob = num/denom   
    movement_att, movement_nonatt = 100, 100
    if valid_att:
        movement_att = movement
    else:
        movement_nonatt = movement
        
    return(movement_att, movement_nonatt, prob)

def fit_em(movement_att_list, movement_nonatt_list, modelatt, modelnonatt, debug_em_flag):
   
         
    updt_att_model, updt_nonatt_model = 0, 0
    if len(movement_att_list) > 0:
      movement_att_list = hstack(movement_att_list)
      movement_att_arr = movement_att_list.reshape((len(movement_att_list), 1))
      if len(movement_att_arr) > 1:
   #      print("MAXATT" %movement_att_arr)
         updt_att_model = 1
         modelatt.fit(movement_att_arr)
         attmodelout = modelatt.predict(movement_att_arr) # predict lateattnt values

    if len(movement_nonatt_list) > 0:
      movement_nonatt_list = hstack(movement_nonatt_list)
      movement_nonatt_arr = movement_nonatt_list.reshape((len(movement_nonatt_list), 1))
      if len(movement_nonatt_arr) > 1:
    
         modelnonatt.fit(movement_nonatt_arr)
         nonattmodelout = modelnonatt.predict(movement_nonatt_arr) # predict lateattnt values
 #        print("NONATT")
 #        print(movement_nonatt_arr)
         updt_nonatt_model = 1
         
    if debug_em_flag:
      if updt_att_model:
         print("Att parameters: %s " %(modelatt.converged_))
         for i in range(len(modelatt.weights_)):
           print("Weight=%1.2f Mean=%1.2f Var=%1.2f " %(modelatt.weights_[i], modelatt.means_[i], modelatt.covariances_[i]))
      if updt_nonatt_model:
         print("Nonatt parameters: %s " %(modelnonatt.converged_))
         for i in range(len(modelnonatt.weights_)):
           print("Weight=%1.2f Mean=%1.2f Var=%1.2f " %(modelnonatt.weights_[i], modelnonatt.means_[i], modelnonatt.covariances_[i]))
    return(modelatt, modelnonatt) 

def compute_decoded_errors(text, declist):
   diff = 0  
   inptext = []
   inptext[:] = text
   declist = ''.join(declist)
   for i in range(len(inptext)):
     if inptext[i] != declist[i]:
        diff  = diff + 1
   return diff


def get_rowcol_idx(idx, scan_scheme, rowcol_scanorder, row_w, col_w):
    
   if idx == 0: 
     scanorder_row = [0, 1, 2, 3, 4, 5]
     scanorder_col = [6, 7, 8, 9, 10, 11]  
     if scan_scheme == 1:
         random.shuffle(scanorder_row)
         random.shuffle(scanorder_col) 
     elif scan_scheme == 2:
         scanorder_row = row_w
         scanorder_col = col_w
     rowcol_scanorder = scanorder_row + scanorder_col # Merge the row and col scan orders
     
   rowcol_idx = rowcol_scanorder[idx]
   return(rowcol_idx, rowcol_scanorder)
       
def get_fit_prob(vld_att, movement_att, movement_nonatt, modelatt, modelnonatt, em_attprob_list, em_nonattprob_list, em_allprob_list):

  
   if vld_att: #attended variable
      em_prob = normpdf(movement_att, modelatt.means_[0], math.sqrt(modelatt.covariances_[0]))
      em_attprob_list.append(em_prob)
      em_allprob_list.append(em_prob)
 #     print("Attended EM model prob=%1.2f mdiff=%1.2f mean=%1.2f var=%1.2f " %(em_prob, movement_att, modelatt.means_[0], modelatt.covariances_[0]))
   else:
      em_prob = normpdf(movement_nonatt, modelnonatt.means_[0], math.sqrt(modelnonatt.covariances_[0]))
      em_nonattprob_list.append(em_prob)
      em_allprob_list.append(em_prob)
#      print("Non-attended EM model prob=%1.2f mdiff=%1.2f mean=%1.2f var=%1.2f " %(em_prob, movement_nonatt, modelnonatt.means_[0], modelnonatt.covariances_[0]))
 #  print(em_allprob_list)       
   return(em_prob, em_attprob_list, em_nonattprob_list, em_allprob_list)

def compute_means(vld_att, movement_att, movement_nonatt, sum_att, sum_nonatt, attcount, nonattcount):
    
   if vld_att: #attended variable running sum of movement for attended and non-attended variable
      sum_att += movement_att
      attcount += 1
   else:
      sum_nonatt += movement_nonatt
      nonattcount += 1    
   return(sum_att, sum_nonatt, attcount, nonattcount)    
     
def update_decoded_list_auto( dec, declist, flag_use_wordmodel, currword, words, partword, count, text, errcnt, bspacecnt, bspace_char_flag, debug_flag2, word_no):  
   
   word_decoded = 0 
   if bspace_char_flag >= 1:
       currtext = '<'
   else:
       if flag_use_wordmodel and currword in words: # If word is present in the flashboard and wordmodel is used
         currtext = currword
       else:
         currtext = text[count]
         
       if flag_use_wordmodel and len(words) > 0:
           if dec == '7' and len(words[0]) > 0:
               dec = words[0]
               word_decoded = 1
           elif dec == '6' and len(words[1]) > 0:
               dec = words[1]
               word_decoded = 1
           elif dec == '5' and len(words[2]) > 0:
               dec = words[2]
               word_decoded = 1
           elif dec == '4' and len(words[3]) > 0:
               dec = words[3]  
               word_decoded = 1
           elif dec == '3' and len(words[4]) > 0:
               dec = words[4]
               word_decoded = 1
    
   if debug_flag2:
      print("nflag=%d partword=%s dec=%s currword=%s currtext=%s word_decoded=%d, count=%d" %(bspace_char_flag, partword, dec, currword, currtext, word_decoded, count))
#   bspace_char_flag = 0
#   if len(declist) > 1:
#      print("DBG dec_last=%s count=%d currtext=%s dec=%s" %(declist[-1], count, currtext, dec))
   if currtext == '<' and dec != '<':
     if word_decoded:
 #        count = count + len(dec[len(partword)])
         declist.append(dec[len(partword):] )
         dec_fordisplay = dec
     else:
 #        count = count + 1  # Character decoded, so count goes up only by 1
         declist.append(dec)
         dec_fordisplay = dec
         
     dec_foraudio = dec    
     bspacecnt = bspacecnt + 1
     errcnt = errcnt + 1
     bspace_char_flag = bspace_char_flag + 1
   elif currtext == '<' and dec == '<': 
 #    count = count - len(declist[-1])
     declist = declist[:-1]    # Delete element as '<' is really Backspace
     bspacecnt = bspacecnt + 1
     bspace_char_flag = bspace_char_flag - 1
     dec_fordisplay = '<'
     dec_foraudio = '<'

   elif currtext != '<' and dec == '<':
     
     if len(declist) > 0 and ('_' in declist[-1]): # Go to previous word if first decoding in new word was < (corner cases)
        if word_no > 0:
          word_no = word_no - 1
     if count > 0:
       count = count - len(declist[-1])
       
     declist = declist[:-1]    # Delete element as '<' is really Backspace
#     bspace_char_flag = bspace_char_flag - 1
     dec_fordisplay = '<'
     dec_foraudio = '<'
     
   else:

     if currtext != dec:
        errcnt = errcnt + 1
        bspace_char_flag = bspace_char_flag + 1
     else:
        if word_decoded or dec == '_':
            word_no = word_no + 1
            
        if word_decoded:
           count = count + len(dec[len(partword):])
        else:
           count = count + 1
    
     if word_decoded:
         declist.append(dec[len(partword):] ) 
         dec_foraudio = dec[:-1]  # Remove the '_' from the string for audio purposes
         dec_fordisplay = dec
     else:
         declist.append(dec)
         dec_foraudio = dec
         dec_fordisplay = dec
                      
   if bspace_char_flag < 0:
      bspace_char_flag = 0  
   if count < 0:
      count = 0

   if debug_flag2:
      print("Decoded: %s  %s %d %d %d %s" %(dec, currtext, bspace_char_flag, count, word_no, declist))
   
   return(count, declist, errcnt, bspacecnt, bspace_char_flag, word_no, dec_foraudio, dec_fordisplay)

def update_scan_duration(DFLT_SCAN_DURATION, INCR_SCAN_DURATION, flag_speedup, debug_speedup, em_allprob_list, vld_att, sum_att, attcount,  sum_nonatt, nonattcount, movement_att, movement_nonatt, scan_duration, MIN_SCAN_DURATION, count, speedup_char_no)   :
                                                          # Update speedup once per character flash
  confirm_speedup = 0
  if flag_speedup and len(em_allprob_list) >= 3:  # Need at least 2 entries to start
    if (em_allprob_list[-1] > 0.5 ) and (em_allprob_list[-2] > 0.5) : # If last 2 responses have a probability 
      if vld_att:
          mean = sum_att/attcount   # Compute the mean from the running sum
          if movement_att > 0.9*mean : # Make sure current movement is close to the mean
              confirm_speedup = 1
      else:
          mean = sum_nonatt/nonattcount # Compute the mean from the running sum 
          if movement_nonatt > 0.9*mean:
              confirm_speedup = 1
  if confirm_speedup and (speedup_char_no < count):
     if scan_duration > MIN_SCAN_DURATION: # Do not go any faster than min duration
       scan_duration = scan_duration - INCR_SCAN_DURATION
       speedup_char_no = count
  else:
      scan_duration = scan_duration   # Hold value
  if scan_duration <= 0:
      print("WARNING: UNDERFLOW OCCURRING IN scan_duration UPDATE")
  if debug_speedup:
      print("Confirm Speedup = %d" %confirm_speedup)
      print("Default capt duration=%f, actual cap duration=%f, speedup_count=%d charcount=%d" %(DFLT_SCAN_DURATION, scan_duration, speedup_char_no, count))
  return(scan_duration, speedup_char_no)


def check_for_distraction(vld_att, movement_att, movement_nonatt, em_attprob_list, em_nonattprob_list, em_allprob_list, sum_att, sum_nonatt, attcount, nonattcount, dist_list, debug_distraction):
   
   confirm_distraction = 0
   mean = 0.0
   if (em_allprob_list[-1] < 10**-1 ) and (em_allprob_list[-2] < 10**-1 ): 
      if vld_att:
          mean = sum_att/attcount   # Compute the mean from the running sum
          if abs(mean - movement_att) > 0.25*mean :
              confirm_distraction = 1
      else:
          mean = sum_nonatt/nonattcount
          if abs(mean - movement_nonatt) > 0.5*mean:
              confirm_distraction = 1
#   print(movement_att, movement_nonatt, attcount, nonattcount)
   if debug_distraction: 
     print("Distraction is %d, mean is %1.2f, maxd_att=%1.2f maxd_non=%1.2f vldatt=%d" %(confirm_distraction, mean, movement_att, movement_nonatt, vld_att) )      
   dist_list.append(confirm_distraction)
   return(confirm_distraction, dist_list)

def update_movement_list(movement_att, movement_nonatt, movement_att_list, movement_nonatt_list):
    
      if movement_att != 100: # 100 means that the value is not valid
         movement_att_list.append(movement_att)
      if movement_nonatt != 100:
         movement_nonatt_list.append(movement_nonatt)     
      return(movement_att_list, movement_nonatt_list)
  
def my_print_float(var, flag=0):
    if flag:
        print(', '.join('{:0.2e}'.format(i) for i in var))
        
def send_to_audio(flag_output_audio, text4audio, indx):
    
    if flag_output_audio:
      if indx == 0:
         create_audio('Welcome. Now starting game')
      elif indx == 1:
         create_audio(text4audio)
      elif indx == 2:
         create_audio('thank you and goodbye')
         
def get_dec(charlist, charlist_diag, flashboard_type, dist):
    
   dec = charlist[dist.index(max(dist))] # Similar to matlab code take the max over all the 36 entries 
     
   return(dec)