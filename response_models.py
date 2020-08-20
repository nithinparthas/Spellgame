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

def my_timer(delay):
    simstarttime = time.time() 
    currtime = time.time()
    while currtime < (simstarttime + delay):
        currtime = time.time()
        
        
def get_variables():
    
    
    charlist = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
            'x', 'y', 'z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_']
    rows = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5]
    cols = [6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11]
    fontsize = 52
    font = ImageFont.truetype("arial.ttf", fontsize)
    left = [36, 37, 38, 39, 40, 41]
    right = [42, 43, 44, 45, 46, 47]
    return(font, fontsize, rows, cols, charlist, left, right)

def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


def update_dist(prob, idx, rows, cols, dist):
    
    probs = [1]*36 # Initialize list
    for r in range (0,36):
      if rows[r] == idx:
         probs[r] = prob 

      if cols[r] == idx:
         probs[r] = prob 
    avg_probs = [x/sum(probs) for x in probs]

    for r in range (0,36):
        dist[r] = dist[r]*avg_probs[r]
        
        
    sum_dist = sum(dist)
    if sum_dist > 0:
        dist = [x/sum_dist for x in dist]
        
 #   for x in range(len(dist)):
 #       print("%1.2f" %dist[x])
    return(dist)   

def increment_index(idx): # Increment index and wrap around
    idx = idx + 1
    if idx == 12:
       idx = 0
    return(idx)


def check_if_attended(idx, rows, cols, count, charlist, text, bspace_char_flag):
    
    vld_att = 0
    if bspace_char_flag > 0: # Next char is a backspace because of previous error
        curr = charlist.index('<')   # '<' is backspace
    else:
        curr = charlist.index(text[count])
    if idx < 6:
        if idx == rows[curr]: # Attended case
           vld_att = 1
    else:
        if idx == cols[curr]: # Attended case
           vld_att = 1
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
       if  mix_prob_att >= 0.2:
          movement_att  = normal(10, 1, 1) # Formal is (mean, variance, no_samples)
          prob = normpdf(movement_att, 10, 1)/normpdf(movement_att, 2, 1)
          attpdf.append(1)
       else:
          movement_att = normal(4, 1, 1)
          prob = normpdf(movement_att, 10, 1)/normpdf(movement_att, 2, 1)
          attpdf.append(0)
    else:
       if  mix_prob_nonatt >= 0.2:
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
   for i in range(len(inptext)):
     if inptext[i] != declist[i]:
        diff  = diff + 1
   return diff


def get_rowcol_idx(idx, scan_scheme, rowcol_scanorder):
    
   if idx == 0: 
     scanorder_row = [0, 1, 2, 3, 4, 5]
     scanorder_col = [6, 7, 8, 9, 10, 11]  
     if scan_scheme == 1:
         random.shuffle(scanorder_row)
         random.shuffle(scanorder_col)
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

def compute_means(vld_att, movement_att, movement_nonatt, mean_att, mean_nonatt, attcount, nonattcount):
    
   if vld_att: #attended variable
      mean_att += movement_att
      attcount += 1
   else:
      mean_nonatt += movement_nonatt
      nonattcount += 1    
   return(mean_att, mean_nonatt, attcount, nonattcount)    
     
def update_decoded_list_auto( dec, declist, count, text, errcnt, bspacecnt, bspace_char_flag, debug_flag2):  
   
   if bspace_char_flag >= 1:
       currtext = '<'
   else:
       currtext = text[count]
   if debug_flag2:
      print("nflag=%d dec=%s currtext=%s" %(bspace_char_flag, dec, currtext))
#   bspace_char_flag = 0
   if currtext == '<' and dec != '<':
     count = count + 1
     bspacecnt = bspacecnt + 1
     errcnt = errcnt + 1
     bspace_char_flag = bspace_char_flag + 1
     declist.append(dec) 
   elif currtext == '<' and dec == '<':
     declist = declist[:-1]    # Delete element as '<' is really Backspace
     count = count - 1
     bspacecnt = bspacecnt + 1
     bspace_char_flag = bspace_char_flag - 1

   elif currtext != '<' and dec == '<':
     declist = declist[:-1]    # Delete element as '<' is really Backspace
     count = count - 1
     bspace_char_flag = bspace_char_flag - 1
     
   else:
     if currtext != dec:
        errcnt = errcnt + 1
        bspace_char_flag = bspace_char_flag + 1
      
     declist.append(dec)     
     count = count + 1
     
     
   if bspace_char_flag < 0:
      bspace_char_flag = 0  
   if count < 0:
      count = 0
   if debug_flag2:
      print("Decoded: %s  %s %d %d %s" %(dec, currtext, bspace_char_flag, count, declist))
   
   return(count, declist, errcnt, bspacecnt, bspace_char_flag)


def check_for_distraction(vld_att, movement_att, movement_nonatt, em_attprob_list, em_nonattprob_list, em_allprob_list, mean_att, mean_nonatt, attcount, nonattcount, dist_list, debug_distraction):
   
   confirm_distraction = 0
   mean = 0.0
   if (em_allprob_list[-1] < 10**-1 ) and (em_allprob_list[-2] < 10**-1 ): 
      if vld_att:
          mean = mean_att/attcount
          if abs(mean - movement_att) > 0.25*mean :
              confirm_distraction = 1
      else:
          mean = mean_nonatt/nonattcount
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
  
    