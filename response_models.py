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
#    print(textwords)
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
    nexttime = time.time()
    distracted_state = 'nondist'
    return(font, fontsize, rows, cols, charlist, left, right, dist, dist_sort_loc, nexttime, distracted_state)

def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

def updated_distracted_state(flag_use_distraction_detect, distraction_list, DIST_BLOCK_PROB, count, nexttime, distracted_state, distraction_time, nondistraction_time, distraction_model_type, DISTMODEL_STATE_PROB, d_count, nd_count, prev_d_count, prev_nd_count, fixed_distraction_count):
    currtime = time.time()

    block_distracted = 0  # This variable determines whether to force exit from distraction state
    if flag_use_distraction_detect == 1:
       cointoss = np.random.uniform(0, 1)
       if  cointoss <= DIST_BLOCK_PROB and distraction_list[-1] == 1 and distracted_state == 'dist':
           block_distracted = 1
        
    if distraction_model_type == 0: # In this model, subject distraction is a two state MC independent of scan and flashboard
       if nexttime > currtime and block_distracted == 1: # Exit distortion state if block_distorion = 1
          distracted_state = 'nondist'
          fixed_distraction_count = fixed_distraction_count + 1
          nd_count = nd_count + 1
 #         print("EXITED DIST STATE DUE TO FEEDBACK")
       while nexttime <= currtime:
          nexttime = nexttime + distraction_time
          cointoss = np.random.uniform(0, 1)
          if cointoss >= DISTMODEL_STATE_PROB:
             distracted_state = 'nondist'
             if count > 2:  # Ignore first 2 characters
               nd_count = nd_count + 1
          else:
             distracted_state = 'dist' 
             if count > 2:  # Ignore first 2 characters
               d_count = d_count + 1    
#               print("DISTRACTION: Incr d_count %d %f %f %f" %(d_count, nexttime, currtime, distraction_time))
    elif distraction_model_type == 1: # In this model, a new distraction variable is drawn at the conclusion of current one beginning 
      if nexttime <= currtime:        # from current time
          nexttime = currtime + distraction_time
          cointoss = np.random.uniform(0, 1)
          if cointoss >= DISTMODEL_STATE_PROB:
             distracted_state = 'nondist'
             if count > 2:  # Ignore first 2 characters
                nd_count = nd_count + 1
          else:
             distracted_state = 'dist' 
             if count > 2:  # Ignore first 2 characters
                d_count = d_count + 1
      else:
          if distracted_state == 'dist' and block_distracted == 1: # Exit distortion state if block_distorion = 1
             distracted_state = 'nondist'
             fixed_distraction_count = fixed_distraction_count + 1
             nd_count = nd_count + 1
#             print("EXITED DIST STATE DUE TO FEEDBACK")
#    print(distracted_state, count)
    return(nexttime, distracted_state, d_count, nd_count, fixed_distraction_count)     

def print_dist_stats(flag_checkfor_distraction, d_count, nd_count, count_of_detect_distr, count_of_visible_nonunique_distr, fixed_distraction_count):
    
    if flag_checkfor_distraction:
        if count_of_visible_nonunique_distr[0] == 0:
          print("No subject distractions noticed")
        if count_of_visible_nonunique_distr[0] > 0: 
          print("Percentage of non-unique visible distraction events = %2.2f" %(count_of_visible_nonunique_distr[0]/d_count))
          print("Percentage of correctly decoded visible distraction events = %2.2f" %(count_of_detect_distr[0]/count_of_visible_nonunique_distr[0]))
          print("Percentage of incorrectly decoded visible distraction events = %2.2f" %(count_of_detect_distr[1]/count_of_visible_nonunique_distr[0]))
        if ((count_of_detect_distr[0] + count_of_detect_distr[1]) > 0):
             print("Percentage of correctly decoded visible distraction (correct+incorrect) events = %2.2f" %(count_of_detect_distr[0]/(count_of_detect_distr[0] + count_of_detect_distr[1])))
             print("Percentage of correctly decoded visible distraction (correct+incorrect) events = %2.2f" %(count_of_detect_distr[1]/(count_of_detect_distr[0] + count_of_detect_distr[1])))
        if (count_of_detect_distr[5] + count_of_detect_distr[6]) > 0:
           print("Percentage of correctly decoded non-att distraction events (corr+non-corr) = %2.2f" %(count_of_detect_distr[5]/(count_of_detect_distr[5] + count_of_detect_distr[6])))
           print("Percentage of correctly decoded non-att distraction events (corr+non-corr) = %2.2f" %(count_of_detect_distr[6]/(count_of_detect_distr[5] + count_of_detect_distr[6])))

        if d_count > 0:
           print("Percentage of correctly detected overall (vis+invis) distraction events  = %2.2f" %(count_of_detect_distr[0]/d_count))
           print("Percentage of missed overall (vis+invis) distraction events = %2.2f" %(count_of_detect_distr[1]/d_count))
           print("Percentage of fixed distraction events with feedback = %2.2f" %(fixed_distraction_count/d_count))
        if (count_of_detect_distr[3] + count_of_detect_distr[4]) > 0:
           print("Percentage of attended events as a fraction of overall = %2.2f" %(count_of_detect_distr[3]/(count_of_detect_distr[3] + count_of_detect_distr[4])))
        

def update_dist(prob, idx, rows, cols, order_diag,  dist, debug_dist_updates, flashboard_type, charlist_diag, huffman_enable, hflash, charlist):
    
 #   print("In Update_dist block prob= %e" %(prob))
    probs = [1]*36 # Initialize list
    if huffman_enable:
      for r in range (0,36):  
        if charlist[r] in hflash:
           probs[r] = prob
    elif flashboard_type == 0:
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
  #  print(avg_probs)
  #  print("In Update_dist block sum(probs)= %e" %(sum(probs)))
    for r in range (0,36):
        dist[r] = dist[r]*avg_probs[r]
  #  my_print_float(dist, 1)                  
    sum_dist = sum(dist)
 #   print("In Update_dist block sum_dist = %f" %(sum_dist))
                                         # Normalize without 34 and then renormalize with 34
    if sum_dist > 0:
        dist = [x/sum_dist for x in dist]
  #  my_print_float(dist, 1)
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

def check_if_huffman_attended(bspace_char_flag, flag_use_wordmodel, currchar, currword, words, hflash):
    
    vld_att = 0
    if bspace_char_flag > 0: # Next char is a backspace because of previous error
        currchar = '<'  # '<' is backspace
        
    if flag_use_wordmodel and len(words) > 0 and currword in words:           
          if '7' in hflash and len(words) > 0 and words[0]==currword:
              vld_att = 1
          elif '6' in hflash and len(words) > 1 and words[1]==currword:
              vld_att = 1              
          elif '5' in hflash and len(words) > 2 and words[2]==currword:
              vld_att = 1
          elif '4' in hflash and len(words) > 3 and words[3]==currword:
              vld_att = 1   
          elif '3' in hflash and len(words) > 4 and words[4]==currword:
              vld_att = 1
    elif currchar in hflash: # Attended case
              vld_att = 1
    return(vld_att)   

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
          if idx == rows[charlist.index('7')] and len(words) > 0 and words[0]==currword:
              vld_att = 1
          elif idx == rows[charlist.index('6')] and len(words) > 1 and words[1]==currword:
              vld_att = 1              
          elif idx == rows[charlist.index('5')] and len(words) > 2 and words[2]==currword:
              vld_att = 1
          elif idx == rows[charlist.index('4')] and len(words) > 3 and words[3]==currword:
              vld_att = 1   
          elif idx == rows[charlist.index('3')] and len(words) > 4 and words[4]==currword:
              vld_att = 1
        elif idx == rows[currchar]: # Attended case
           vld_att = 1
    else:
        if flag_use_wordmodel and len(words) > 0 and currword in words:    
          if idx == cols[charlist.index('7')] and len(words) > 0 and words[0]==currword:
              vld_att = 1
          elif idx == cols[charlist.index('6')] and len(words) > 1 and words[1]==currword:
              vld_att = 1              
          elif idx == cols[charlist.index('5')] and len(words) > 2 and words[2]==currword:
              vld_att = 1
          elif idx == cols[charlist.index('4')] and len(words) > 3 and words[3]==currword:
              vld_att = 1   
          elif idx == cols[charlist.index('3')] and len(words) > 4 and words[4]==currword:
              vld_att = 1
        elif idx == cols[currchar]: # Attended case
           vld_att = 1
      
      
 #   print(idx, bspace_char_flag, currword, currchar, count, text[count], vld_att)  # DELETE

    return(vld_att)


def ideal_prob_for_code_debug_only(vld_att):
    
    if vld_att:
       prob = normpdf(10, 10, 1)/normpdf(10, 2, 1)
       movement_att = 10
       movement_nonatt = 100
    else:
       prob = normpdf(2, 10, 1)/normpdf(2, 2, 1) 
       movement_att = 100
       movement_nonatt = 2
    return(movement_att, movement_nonatt, prob)    

def generate_mixture_prob(vld_att, attpdf, nonattpdf, distracted_state):
    
    mix_prob_att = np.random.uniform(0, 1)
    mix_prob_nonatt = np.random.uniform(0, 1)
    movement_att = 100 # Note that interpretion of 100 is that data is not valid
    movement_nonatt = 100
    mean_att, sd_att = 6, 1
    mean_nonatt, sd_nonatt = 2, 1
 #   print("DISTRACTED STATE IS %s" %distracted_state)
    if distracted_state == 'dist':
       if vld_att:
           movement_att = 0 
           prob = normpdf(movement_att, mean_att, sd_att)/normpdf(movement_att, mean_nonatt, sd_nonatt)
       else:
           movement_nonatt = 0
           prob = normpdf(movement_nonatt, mean_att, sd_att)/normpdf(movement_nonatt, mean_nonatt, sd_nonatt)
#       print("DISTRACTED STATE %2.2f" %prob)
    elif vld_att:
       if  mix_prob_att >= 0.0:
          movement_att  = normal(mean_att, sd_att, 1) # Formal is (mean, variance, no_samples)
          prob = normpdf(movement_att, mean_att, sd_att)/normpdf(movement_att, mean_nonatt, sd_nonatt)
          attpdf.append(1)
       else:
          movement_att = normal(4, 1, 1)
          prob = normpdf(movement_att, mean_att, sd_att)/normpdf(movement_att, mean_nonatt, sd_nonatt)
          attpdf.append(0)
    else:
       if  mix_prob_nonatt >= 0.0:
                                    # Non-attended case
          movement_nonatt = normal(mean_nonatt, 1, 1)     
          prob = normpdf(movement_nonatt, mean_att, sd_att)/normpdf(movement_nonatt, mean_nonatt, sd_nonatt)
          nonattpdf.append(1)
       else:
          movement_nonatt = normal(1, 1, 1)     
          prob = normpdf(movement_nonatt, mean_att, sd_att)/normpdf(movement_nonatt, mean_nonatt, sd_nonatt)
          nonattpdf.append(0)
              
 #   print("maxd_att=%1.2f maxd_nonatt=%1.2f prob=%1.2f\n" %(movement_att, movement_nonatt, prob))
 #   print("dist_att=%d dist_nonatt=%d" %(dist_att, dist_nonatt))

    return(movement_att, movement_nonatt, prob, attpdf, nonattpdf)

def get_prob(movement, valid_att, modelatt, modelnonatt, count):    

#    if count > 1:
#       print("Mean=%1.2f Var=%1.2f " %(modelatt.means_[0], modelatt.covariances_[0]))
    num = normpdf(movement, 10, 3)
    denom = normpdf(movement, 2, 2)
    prob = 0
    if denom != 0:
      prob = num/denom 
    
    if prob > 10**4 or movement > 10: # Prevents overflows later in computation
        prob = 10**4
    movement_att, movement_nonatt = 100, 100
    if valid_att:
        movement_att = movement
    else:
        movement_nonatt = movement
        
    return(movement_att, movement_nonatt, prob)

def fit_em(movement_att_list, movement_nonatt_list, movement_dist_list, modelatt, modelnonatt, modeldist, debug_em_flag):
          
    updt_att_model, updt_nonatt_model, updt_dist_model = 0, 0, 0
    if len(movement_att_list) > 0:
      movement_att_list = hstack(movement_att_list)
      movement_att_arr = movement_att_list.reshape((len(movement_att_list), 1))
      if len(movement_att_arr) > 1:
   #      print("MAXATT" %movement_att_arr)
         updt_att_model = 1
         modelatt.fit(movement_att_arr)
         attmodelout = modelatt.predict(movement_att_arr) # predict lateattnt values
         
    if len(movement_dist_list) > 0:
      movement_dist_list = hstack(movement_dist_list)
      movement_dist_arr = movement_dist_list.reshape((len(movement_dist_list), 1))
      if len(movement_dist_arr) > 1:
   #      print("MAXATT" %movement_att_arr)
         updt_dist_model = 1
         modeldist.fit(movement_dist_arr)
         distmodelout = modeldist.predict(movement_dist_arr) # predict lateattnt values

    if len(movement_nonatt_list) > 0:
      movement_nonatt_list = hstack(movement_nonatt_list)
      movement_nonatt_arr = movement_nonatt_list.reshape((len(movement_nonatt_list), 1))
      if len(movement_nonatt_arr) > 1:
    
         modelnonatt.fit(movement_nonatt_arr)
         nonattmodelout = modelnonatt.predict(movement_nonatt_arr) # predict lateattnt values
 #        print("NONATT")
 #        print(movement_nonatt_arr)
         updt_nonatt_model = 1
    
 #   debug_em_flag = 1
    if debug_em_flag:
      if updt_att_model:
 #        print("ATT parameters: %s " %(modelatt.converged_))
 #        print(movement_att_list)
         for i in range(len(modelatt.weights_)):
           print("Weight=%1.2f Mean=%1.2f Var=%1.2f " %(modelatt.weights_[i], modelatt.means_[i], modelatt.covariances_[i]))
      if updt_dist_model:
         print("DIST parameters: %s " %(modeldist.converged_))
         for i in range(len(modeldist.weights_)):
           print("Weight=%1.2f Mean=%1.2f Var=%1.2f " %(modeldist.weights_[i], modeldist.means_[i], modeldist.covariances_[i]))
      if updt_nonatt_model:
         print("NONATT parameters: %s " %(modelnonatt.converged_))
         for i in range(len(modelnonatt.weights_)):
           print("Weight=%1.2f Mean=%1.2f Var=%1.2f " %(modelnonatt.weights_[i], modelnonatt.means_[i], modelnonatt.covariances_[i]))
    return(modelatt, modelnonatt, modeldist) 

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
       
def get_fit_prob(simulation_mode, vld_att, movement_att, movement_nonatt, modelatt, modelnonatt, modeldist, em_attprob_list, em_nonattprob_list, em_allprob_list, distracted_state):

   if simulation_mode == 1 or simulation_mode == 2:
     att_Mean, non_Mean = modelatt.means_[0], modelnonatt.means_[0]
     att_SD, non_SD = math.sqrt(modelatt.covariances_[0]), math.sqrt(modelnonatt.covariances_[0])
   elif simulation_mode == 3:
     att_Mean, non_Mean = 0.7101, -0.0523
     att_SD, non_SD = 0.1882, 0.1506

#   print("Movement_att is %f" %(movement_att))
   em_prob = 10
   if vld_att: #attended variable
 #    print(movement_att, att_Mean, att_SD, non_Mean, non_SD)
     denom = normpdf(movement_att, non_Mean, non_SD)
     if denom > 0:
       em_prob = normpdf(movement_att, att_Mean, att_SD)/denom
     em_attprob_list.append(em_prob)
     em_allprob_list.append(em_prob)
 #    print("Att  %s prob=%1.2f mdiff=%1.2f" %(distracted_state, np.log10(em_prob), movement_att))
   else:
     denom = normpdf(movement_nonatt, att_Mean, att_SD)
     if denom > 0:
       em_prob = normpdf(movement_nonatt, non_Mean, non_SD)/denom
     em_nonattprob_list.append(em_prob)
     em_allprob_list.append(em_prob)
      
   return(em_prob, em_attprob_list, em_nonattprob_list, em_allprob_list)

def compute_means(movement_att_list, movement_nonatt_list):
    
   sum_att = sum(movement_att_list)
   attcount = len(movement_att_list)
   sum_nonatt = sum(movement_nonatt_list)
   nonattcount = len(movement_nonatt_list)

   return(sum_att, sum_nonatt, attcount, nonattcount)    

def get_currtext(flag_use_wordmodel, bspace_char_flag, currword, words, text, count):
    
   if bspace_char_flag >= 1:
       currtext = '<'
   else:
      if flag_use_wordmodel and currword in words: # If word is present in the flashboard and wordmodel is used
         currtext = currword
      else:
         currtext = text[count]     
   return(currtext)
         
         
def update_decoded_list_auto( dec, declist, flag_use_wordmodel, currword, words, partword, count, text, errcnt, bspacecnt, bspace_char_flag, debug_flag2, word_no, cpuzzle_mode = 0, orig_word = ''):  
   
   word_decoded = 0 
   if bspace_char_flag >= 1:
       currtext = '<'
   else:
       if flag_use_wordmodel and currword in words: # If word is present in the flashboard and wordmodel is used
         currtext = currword
       else:
         currtext = text[count]
         
       if flag_use_wordmodel and len(words) > 0:
           if dec == '7' and len(words) > 0:
               dec = words[0]
               word_decoded = 1
           elif dec == '6' and len(words) > 1:
               dec = words[1]
               word_decoded = 1
           elif dec == '5' and len(words) > 2:
               dec = words[2]
               word_decoded = 1
           elif dec == '4' and len(words) > 3:
               dec = words[3]  
               word_decoded = 1
           elif dec == '3' and len(words) > 4:
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
     if not cpuzzle_mode:
       if len(declist) > 0 and ('_' in declist[-1]): # Go to previous word if first decoding in new word was < (corner cases), does not apply to crossword
          if word_no > 0:
            word_no = word_no - 1
       if count > 0:
         count = count - len(declist[-1])
       
       declist = declist[:-1]    # Delete element as '<' is really Backspace
#     bspace_char_flag = bspace_char_flag - 1
     else:
       if orig_word[len(declist)-1] == '_':
         declist = declist[:-1]     
     dec_fordisplay = '<'
     dec_foraudio = '<'
 #    print("LOCATION 1A %s" %declist)
   else:

     if currtext != dec:
        errcnt = errcnt + 1
        bspace_char_flag = bspace_char_flag + 1
 #       print("LOC 1b ERROR curr=%s dec=%s bflag=%d" %(currtext, dec, bspace_char_flag))
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
   
   return(count, declist, errcnt, bspacecnt, bspace_char_flag, word_no, dec_foraudio, dec_fordisplay, word_decoded)

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

def check_for_distraction_simmode_2(distracted_state, vld_att, movement_att, movement_nonatt, em_attprob_list, em_nonattprob_list, em_allprob_list, 
                          sum_att, sum_nonatt, attcount, nonattcount, dist_list, debug_distraction, 
                          count_of_detect_distr, movement_att_list, movement_dist_list, movement_nonatt_list):
   
   confirm_distraction, mean = 0, 0
   att_threshold_check = -1
   nonatt_threshold_check = 3

#   print(sum_att/attcount, sum_nonatt/nonattcount)

   if len(em_attprob_list) == 0 or em_attprob_list[-1] <= 0:
       attprob_check = 0
   else:
       attprob_check = np.log10(em_attprob_list[-1])

 #  if vld_att:
#       print("VALID prob_check=%2.2f move=%f" %(attprob_check, movement_att_list[-1]))
     
   if vld_att and attprob_check < att_threshold_check :
         print("Dist_stete= %s CONFIRMING DISTRACTION" %distracted_state)
         confirm_distraction = 1
         movement_dist_list.append(movement_att)    # Add to distraction list
         movement_att_list = movement_att_list[:-1] # Remove distracted element from list, it is added to distracted list
         count_of_detect_distr[0] = count_of_detect_distr[0] + 1 # Dist, Dist
   elif vld_att:
         confirm_distraction = 0
         count_of_detect_distr[1] = count_of_detect_distr[1] + 1 # Dist, NoDist
   
   if len(em_nonattprob_list) == 0 or em_nonattprob_list[-1] <= 0:
       nonattprob_check = 0
   else:
       nonattprob_check = np.log10(em_nonattprob_list[-1])
 
   if not vld_att and (nonattprob_check > nonatt_threshold_check) : # Check if non_attended case exceeds threshold to indicate distraction
         confirm_distraction = 1
         movement_dist_list.append(movement_nonatt)    # Add to distraction list
         count_of_detect_distr[5] = count_of_detect_distr[5] + 1 # Count dist
   elif not vld_att:
         confirm_distraction = 0
         count_of_detect_distr[6] = count_of_detect_distr[6] + 1 
       
   # if vld_att:
   #   print("VALID prob_check=%2.2f movement=%2.2f dist_state=%s" %(attprob_check, movement_att, distracted_state))
   # else:
   #   print("NONVALID prob_check=%2.2f movement=%2.2f dist_state=%s" %(nonattprob_check, movement_nonatt, distracted_state))  
     
          
#   print("Counts %d %d %d %d %d" %(vld_att, d_count, prev_d_count, prev_nd_count, nd_count))
#   print(d_count, count_of_detect_distr, count_of_visible_nonunique_distr)
      
   if vld_att: 
       count_of_detect_distr[3] = count_of_detect_distr[3] + 1 # Track count of attended variables
   if not vld_att:
       count_of_detect_distr[4] = count_of_detect_distr[4] + 1 # Track count of non-attended variables
        
   if debug_distraction: 
     print("Distraction is %d, mean is %1.2f, maxd_att=%1.2f maxd_non=%1.2f vldatt=%d" %(confirm_distraction, mean, movement_att, movement_nonatt, vld_att) )      
   dist_list.append(confirm_distraction)
    
   return(confirm_distraction, dist_list, count_of_detect_distr, movement_att_list, movement_dist_list, movement_nonatt_list)


def check_for_distraction(simulation_mode, distracted_state, vld_att, movement_att, movement_nonatt, em_attprob_list, em_nonattprob_list, em_allprob_list, sum_att, sum_nonatt, 
                          attcount, nonattcount, dist_list, debug_distraction, d_count, nd_count, prev_d_count,
                          prev_nd_count,
                          count_of_detect_distr, count_of_visible_nonunique_distr, movement_att_list, movement_dist_list, movement_nonatt_list):
   
   confirm_distraction = 0
   mean = 0.0
   if simulation_mode == 3:
     att_threshold_check = -2
     nonatt_threshold_check = 3
   elif simulation_mode == 1 or simulation_mode == 0:
     att_threshold_check = -1
     nonatt_threshold_check = 3
#   print(sum_att/attcount, sum_nonatt/nonattcount)
   if vld_att and distracted_state == 'dist':
       count_of_visible_nonunique_distr[0] = count_of_visible_nonunique_distr[0] + 1 # THis is the number of visible non-unique distraction events
   elif (not vld_att) and distracted_state == 'dist':
       count_of_visible_nonunique_distr[1] = count_of_visible_nonunique_distr[1] + 1  # This is the nonvisible nonunique

   if len(em_attprob_list) == 0 or em_attprob_list[-1] <= 0:
       attprob_check = 0
   else:
       attprob_check = np.log10(em_attprob_list[-1])

 #  if vld_att:
#       print("VALID prob_check=%2.2f move=%f" %(attprob_check, movement_att_list[-1]))
      
   if vld_att and attprob_check < att_threshold_check :
     #    print("Dist_stete= %s CONFIRMING DISTRACTION" %distracted_state)
         confirm_distraction = 1
         movement_dist_list.append(movement_att)    # Add to distraction list
         movement_att_list = movement_att_list[:-1] # Remove distracted element from list, it is added to distracted list
         if d_count != prev_d_count and distracted_state == 'dist': # Count the number of unique distraction events which were also detectable
            count_of_detect_distr[0] = count_of_detect_distr[0] + 1 # Dist, Dist
   elif vld_att:
         confirm_distraction = 0
         if d_count != prev_d_count and distracted_state == 'dist': # Count the number of unique distraction events which were non-detectable
            count_of_detect_distr[1] = count_of_detect_distr[1] + 1 # Dist, NoDist
   
   if len(em_nonattprob_list) == 0 or em_nonattprob_list[-1] <= 0:
       nonattprob_check = 0
   else:
       nonattprob_check = np.log10(em_nonattprob_list[-1])
       
   if not vld_att and (nonattprob_check > nonatt_threshold_check): # Check if non-attended case exceeds threshold
         confirm_distraction = 1
         movement_dist_list.append(movement_att)    # Add to distraction list
         movement_nonatt_list = movement_nonatt_list[:-1] # Remove distracted element from list, it is added to distracted list
         if d_count != prev_d_count and distracted_state == 'dist':
           count_of_detect_distr[5] = count_of_detect_distr[5] + 1 # Dist, Dist
   elif not vld_att:
         confirm_distraction = 0
         if d_count != prev_d_count and distracted_state == 'dist':
            count_of_detect_distr[6] = count_of_detect_distr[6] + 1 # Dist, NoDist
   # if vld_att:
   #   print("VALID prob_check=%2.2f movement=%2.2f dist_state=%s" %(attprob_check, movement_att, distracted_state))
   # else:
   #   print("NONVALID prob_check=%2.2f movement=%2.2f dist_state=%s" %(nonattprob_check, movement_nonatt, distracted_state))  
     
   if not vld_att and distracted_state == 'dist':
         if nd_count != prev_nd_count: # Count the number of unique nondistraction events which was not observed
            count_of_detect_distr[2] = count_of_detect_distr[2] + 1 # NoDist, Dist
          
#   print("Counts %d %d %d %d %d" %(vld_att, d_count, prev_d_count, prev_nd_count, nd_count))
#   print(d_count, count_of_detect_distr, count_of_visible_nonunique_distr)
      
   if vld_att: 
       prev_d_count = d_count     
       count_of_detect_distr[3] = count_of_detect_distr[3] + 1 #
   if not vld_att:
       prev_nd_count = nd_count
       count_of_detect_distr[4] = count_of_detect_distr[4] + 1 #
        
#   if (em_allprob_list[-1] < 10**-1 ) and (em_allprob_list[-2] < 10**-1 ): 
          # mean = sum_att/attcount   # Compute the mean from the running sum
          # if abs(mean - movement_att) > 0.25*mean :
 #             confirm_distraction = 1
#      else:
          # mean = sum_nonatt/nonattcount
          # if abs(mean - movement_nonatt) > 0.5*mean:
 #             confirm_distraction = 1
#   print(movement_att, movement_nonatt, attcount, nonattcount)
   if debug_distraction: 
     print("Distraction is %d, mean is %1.2f, maxd_att=%1.2f maxd_non=%1.2f vldatt=%d" %(confirm_distraction, mean, movement_att, movement_nonatt, vld_att) )      
   dist_list.append(confirm_distraction)
    
   return(confirm_distraction, dist_list, count_of_detect_distr, count_of_visible_nonunique_distr, prev_d_count, prev_nd_count, movement_att_list, movement_dist_list, movement_nonatt_list)

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

def load_eeg_data():
   import statistics
   
   scores00, scores01, scores10, scores11, scores0, scores1 = [], [], [], [], [], []
   f = open('EEGscores/scores00.txt', 'r')
   for x in f:
       scores00.append(x.strip())
   f.close()
   f = open('EEGscores/scores01.txt', 'r')
   for x in f:
       scores01.append(x.strip())
   f.close()
   f = open('EEGscores/scores10.txt', 'r')
   for x in f:
       scores10.append(x.strip())
   f.close()
   f = open('EEGscores/scores11.txt', 'r')
   for x in f:
       scores11.append(x.strip())
   f.close()
   f = open('EEGscores/scores0.txt', 'r')
   for x in f:
       scores0.append(x.strip())
   f.close()
   f = open('EEGscores/scores1.txt', 'r')
   for x in f:
       scores1.append(x.strip())
   f.close()
   scores00 = [float(n) for n in scores00]
   scores01 = [float(n) for n in scores01]
   scores10 = [float(n) for n in scores10]
   scores11 = [float(n) for n in scores11]
   scores0 = [float(n) for n in scores0]
   scores1 = [float(n) for n in scores1]
   
   non_Mean = statistics.mean(scores0)
   att_Mean = statistics.mean(scores1)
   non_SD = statistics.pstdev(scores0) 
   att_SD = statistics.pstdev(scores1)
   return(scores00, scores01, scores10, scores11, att_Mean, att_SD, non_Mean, non_SD)
    
def get_EEG(vld_att, prev_vld_att, scores00, scores01, scores10, scores11, att_Mean, att_SD, non_Mean, non_SD, distracted_state):    

#    print(vld_att, distracted_state)
    if distracted_state == 'dist': # If distracted flag is set for the subject, then nonatt score is chosen
        score = 0.00
     #   score = random.choice(scores00)  # independent of vld attended variable
    elif vld_att == 0 and prev_vld_att == 0:
        score = random.choice(scores00)
    elif vld_att == 0 and prev_vld_att == 1:
        score = random.choice(scores01)
    elif vld_att == 1 and prev_vld_att == 0:
        score = random.choice(scores10)
    elif vld_att == 1 and prev_vld_att == 1:
        score = random.choice(scores11)
        
    prob=normpdf(score,att_Mean,att_SD)/normpdf(score,non_Mean,non_SD)  # convert score into probability    
    movement_att, movement_nonatt = 100, 100
    if vld_att:
        movement_att = score
    else:
        movement_nonatt = score
        
    return(movement_att, movement_nonatt, prob)

def check_mode_and_set_flags_for_correctness(simulation_mode, flag_mimick_distractions, flag_em_stats):

    if simulation_mode == 0 or simulation_mode == 2:  # Force flags off in modes where distraction is not model computed
       flag_mimick_distractions = 0
       flag_em_stats = 0
       print("Simulation mode is %d, forcing flag_mimick_distractions to 0" %(simulation_mode))
    return(flag_mimick_distractions, flag_em_stats)


        