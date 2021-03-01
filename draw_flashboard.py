# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 18:09:25 2020

@author: Nithin
"""

import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np
import time
import textwrap
import tkinter as tk
from tkinter import Label
import tkinter.font as tkFont

def put_word_into_image(img, text):
    w, h = 80, 80

    fontsize = 8  # starting font size

    # portion of image width you want text width to be
    img_fraction = 0.90

    font = ImageFont.truetype("arial.ttf", fontsize)
    if len(text) < 5:
        fontsize = 28
        font = ImageFont.truetype("arial.ttf", fontsize)
    else:
      while font.getsize(text)[0] < img_fraction*img.size[0]:
        # iterate until the text size is just larger than the criteria
         fontsize += 1
         font = ImageFont.truetype("arial.ttf", fontsize)

#    print("FONTSIZE IS %d textlen=%d text=%s" %(fontsize, len(str(text[1])), text[1]))

    img_new = center_text_word(img.copy(), font, str(text), w, h, (0,0,0))
    return(img_new)

def enter_word_into_flashboard(bspace_char_flag, words, bim_for_word, gim_for_word, rim_for_word, grim_for_word, bim, gim, rim, grim):
    
    if len(words) > 0 and not(bspace_char_flag):
      bim_w_text = [np.array(put_word_into_image(bim_for_word, x)) for x in words] 
      gim_w_text = [np.array(put_word_into_image(gim_for_word, x)) for x in words] 
      rim_w_text = [np.array(put_word_into_image(rim_for_word, x)) for x in words] 
      grim_w_text = [np.array(put_word_into_image(grim_for_word, x)) for x in words] 

    bim_w, gim_w, rim_w, grim_w = bim, gim, rim, grim
#    cv2.imshow('temp', np.array(bim))
    for i in range(5):
        if i < len(words) and not(bspace_char_flag):
           bim_w[32-i], gim_w[32-i], rim_w[32-i], grim_w[32-i] = bim_w_text[i], gim_w_text[i], rim_w_text[i], grim_w_text[i]
        elif i == 0:
           bim_w[32-i], gim_w[32-i], rim_w[32-i], grim_w[32-i] = getcharimg('7')
        elif i == 1:
           bim_w[32-i], gim_w[32-i], rim_w[32-i], grim_w[32-i] = getcharimg('6')
        elif i == 2:
           bim_w[32-i], gim_w[32-i], rim_w[32-i], grim_w[32-i] = getcharimg('5')      
        elif i == 3:
           bim_w[32-i], gim_w[32-i], rim_w[32-i], grim_w[32-i] = getcharimg('4')
        elif i == 4:
           bim_w[32-i], gim_w[32-i], rim_w[32-i], grim_w[32-i] = getcharimg('3')     
 #   cv2.imshow('temp1', np.array(bim_w))  
    return(bim_w, gim_w, rim_w, grim_w)


def display_message(text, WidgetDisplayTime, done=0):  # Widget to display message

    root = tk.Tk()
    root.title("DIALOG BOX")
    root.geometry("325x150+0+0") #Width x Height
    fontStyle = tkFont.Font(family="Arial", size=28)
    label = tk.LabelFrame(root, width=325, height=150)

    #grid manager to set label localization
    label.grid(row=0, column=0)

    #label row and column configure: first argument is col or row id
    label.grid_rowconfigure(0, weight=2)
    label.grid_columnconfigure(0, weight=2)

    #cancel propagation
    label.grid_propagate(False)

    #Create button and set it localization. You can change it font without changing size of button, but if You set too big not whole will be visible
    button = tk.Button(label, text=text, font=('Helvetica', 24))

    #Use sticky to button took up the whole label area
    button.grid(row=0, column=0, sticky='nesw')

   # main window will get destroyed after X Msec
    root.after(WidgetDisplayTime, root.destroy) 
    #root.mainloop() 
    root.update() # Using update() instead of mainloop() as it is non-blocking  
    if done:
       root.destroy()
    
   
def display_infoimg(text, display_bgndimg):
    font = cv2.FONT_HERSHEY_PLAIN
    font_size = 2
    font_thickness = 2
    
    wrapped_srctext = textwrap.wrap(text, width=50)
    y_text = 33
    w = display_bgndimg.shape[1]
    for i,line in enumerate(wrapped_srctext):
        
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        width = textsize[0]
        height = textsize[1]
        if i == 0:
          S = cv2.putText(display_bgndimg.copy(), line, (int((w - width) / 2), y_text), font, font_size, (255,255,255), font_thickness, lineType = cv2.LINE_AA)
        else:
          S = cv2.putText(S, line, (int((w - width) / 2), y_text), font, font_size, (255,255,255), font_thickness, lineType = cv2.LINE_AA)  
    
    return(S)

def display_srctext(srctext, textbgnd_img):
    font = cv2.FONT_HERSHEY_PLAIN
    font_size = 2
    font_thickness = 1
    wrapped_srctext = textwrap.wrap(srctext, width=50)
    y_text = 40
    w = textbgnd_img.shape[1]
    for i,line in enumerate(wrapped_srctext):
        
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        width = textsize[0]
        height = textsize[1]
        if i == 0:
          S = cv2.putText(textbgnd_img.copy(), line, (int((w - width) / 2), y_text), font, font_size, (0,255,255), font_thickness, lineType = cv2.LINE_AA)
        else:
          S = cv2.putText(S, line, (int((w - width) / 2), y_text), font, font_size, (0,255,255), font_thickness, lineType = cv2.LINE_AA)  
        y_text += 2*height
        
    return(S)  
  
def display_text(declist, textbgnd_img, flg):
     
 #   declist = declist.append('-')
    text = ''.join(declist)
    text = text[-20:]
    if flg == 1:
      text = text+'*'


    font = cv2.FONT_HERSHEY_PLAIN
    font_size = 2
    font_thickness = 1
    wrapped_text = textwrap.wrap(text, width=50)
    y_text = 40
    w = textbgnd_img.shape[1]
    for i,line in enumerate(wrapped_text):
        
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        width = textsize[0]
        height = textsize[1]
        if i == 0:
          I = cv2.putText(textbgnd_img.copy(), line, (int((w - width) / 2), y_text), font, font_size, (0,255,255), font_thickness, lineType = cv2.LINE_AA)
        else:
          I = cv2.putText(I, line, (int((w - width) / 2), y_text), font, font_size, (0,255,255), font_thickness, lineType = cv2.LINE_AA)  
        y_text += 2*height
        
    return(I) 

def pzl_display_all_messages(textwords, words, word_no, bspace_char_flag, dec_fordisplay, display_bgndimg, flag_display_widget, count, text, declist, textbgnd_img, WidgetDisplayTime):
#   print(word_no, textwords[word_no], words, dec_fordisplay)
   if len(dec_fordisplay) > 1:
      dec_fordisplay = dec_fordisplay[-1]
   else:
      dec_fordisplay = ''
   if bspace_char_flag:
      displayimg = display_infoimg('ERROR: ' + dec_fordisplay + ' NEXT: <', display_bgndimg)
      if flag_display_widget:
        display_message('ERROR: ' + dec_fordisplay + '\n' + 'NEXT: <' + '\n\n' + 'NEW SCAN STARTS', WidgetDisplayTime)
   elif len(declist) == 0:
      displayimg = display_infoimg('NEXT: ' + text[count],  display_bgndimg)
      if flag_display_widget:
        display_message('NEXT:  '+ text[count], WidgetDisplayTime) 
   elif word_no >= len(textwords): # Applies only to game and means that game is done
      
      displayimg = display_infoimg('LAST: ' + dec_fordisplay + ' GAME OVER',  display_bgndimg)      
      if flag_display_widget:
        display_message('LAST: ' + dec_fordisplay + '\n' + 'GAME OVER' + '\n\n', WidgetDisplayTime)
   elif textwords[word_no] in words:
      if len(dec_fordisplay) > 0:
         displayimg = display_infoimg('LAST: ' + dec_fordisplay + ' NEXT: ' + textwords[word_no],  display_bgndimg)
      else:
         displayimg = display_infoimg('NEXT: ' + textwords[word_no],  display_bgndimg) 
      if flag_display_widget:
         if len(dec_fordisplay) > 0:
            display_message('LAST: ' + dec_fordisplay + '\n' + 'NEXT: '+ textwords[word_no] + '\n\n' + 'NEW SCAN STARTS', WidgetDisplayTime)
         else:
            display_message('NEXT: '+ textwords[word_no] + '\n\n' + 'NEW SCAN STARTS', WidgetDisplayTime)
   elif count < len(text):
      if len(dec_fordisplay) > 0:
        displayimg = display_infoimg('LAST: ' + dec_fordisplay + ' NEXT: ' + text[count]+' ('+textwords[word_no]+')',  display_bgndimg)
      else:
        displayimg = display_infoimg('NEXT: ' + text[count]+' ('+textwords[word_no]+')',  display_bgndimg) 
      if flag_display_widget:
        if len(dec_fordisplay) > 0:
          display_message('LAST: ' + dec_fordisplay + '\n' + 'NEXT: ' + text[count] + ' ('+textwords[word_no]+')'+'\n\n'+'NEW SCAN STARTS', WidgetDisplayTime) 
        else:
          display_message('NEXT: ' + text[count] + ' ('+textwords[word_no]+')'+'\n\n'+'NEW SCAN STARTS', WidgetDisplayTime) 
   elif count == len(text):
      displayimg = display_infoimg('LAST: ' + dec_fordisplay + ' NEXT: ' + text[count-1],  display_bgndimg)
      if flag_display_widget:
        display_message('LAST: ' + dec_fordisplay + '\n' + 'NEXT: ' + text[count-1]+'\n\n', WidgetDisplayTime)   
   textimg = display_text([x.upper() for x in declist], textbgnd_img, 1)  # Decoded text information to output on screen
   return(displayimg, textimg)
   

def display_all_messages(textwords, words, word_no, bspace_char_flag, dec_fordisplay, display_bgndimg, flag_display_widget, count, text, declist, textbgnd_img, WidgetDisplayTime):
#   print(word_no, textwords[word_no], words, dec_fordisplay)
   if bspace_char_flag:
      displayimg = display_infoimg('ERROR: ' + dec_fordisplay + ' NEXT: <', display_bgndimg)
      if flag_display_widget:
        display_message('ERROR: ' + dec_fordisplay + '\n' + 'NEXT: <' + '\n\n' + 'NEW SCAN STARTS', WidgetDisplayTime)
   elif len(declist) == 0:
      displayimg = display_infoimg('NEXT: ' + text[count],  display_bgndimg)
      if flag_display_widget:
        display_message('NEXT:  '+ text[count], WidgetDisplayTime) 
   elif word_no >= len(textwords): # Applies only to game and means that game is done
      displayimg = display_infoimg('LAST: ' + dec_fordisplay + ' GAME OVER',  display_bgndimg)
      if flag_display_widget:
        display_message('LAST: ' + dec_fordisplay + '\n' + 'GAME OVER' + '\n\n', WidgetDisplayTime)
   elif textwords[word_no] in words:
      displayimg = display_infoimg('LAST: ' + dec_fordisplay + ' NEXT: ' + textwords[word_no],  display_bgndimg)
      if flag_display_widget:
        display_message('LAST: ' + dec_fordisplay + '\n' + 'NEXT: '+ textwords[word_no] + '\n\n' + 'NEW SCAN STARTS', WidgetDisplayTime)
   elif count < len(text):
      displayimg = display_infoimg('LAST: ' + dec_fordisplay + ' NEXT: ' + text[count]+' ('+textwords[word_no]+')',  display_bgndimg)
      if flag_display_widget:
        display_message('LAST: ' + dec_fordisplay + '\n' + 'NEXT: ' + text[count] + ' ('+textwords[word_no]+')'+'\n\n'+'NEW SCAN STARTS', WidgetDisplayTime) 
   elif count == len(text):
      displayimg = display_infoimg('LAST: ' + dec_fordisplay + ' NEXT: ' + text[count-1],  display_bgndimg)
      if flag_display_widget:
        display_message('LAST: ' + dec_fordisplay + '\n' + 'NEXT: ' + text[count-1]+'\n\n', WidgetDisplayTime)
        
   textimg = display_text(declist, textbgnd_img, 1)  # Decoded text information to output on screen
   return(displayimg, textimg)


def create_huffman_flashboard(hattlist, charlist, bim, gim, rim, grim, textimg, displayimg, textsrcimg, flag_fix_distraction, distraction_list, bspace_char_flag, flag_word_complete):
      

    if not flag_fix_distraction: # Colorize the flashboard based on feedback and flag
        him = gim
    elif distraction_list[-1]: # Change attended colors based on last distraction indication or character error (backspace)
        him = rim
    else:
        him = gim
        
    if not flag_fix_distraction: # Colorize the attended row/col based on feedback and flag
        bkim = bim
    elif len(distraction_list) >=2 and distraction_list[-2] and distraction_list[-1]: # Change background colors based on last two distraction indication
        bkim = grim
    else:
        bkim = bim
        
    htile = []
    for ch in range(len(charlist)):
        if charlist[ch] in hattlist:
            htile.append(him[ch])
        else:
            htile.append(bkim[ch])
            
    row0, row1, row2, row3, row4, row5 = cv2.hconcat(htile[0:6]), cv2.hconcat(htile[6:12]), cv2.hconcat(htile[12:18]), cv2.hconcat(htile[18:24]), cv2.hconcat(htile[24:30]), cv2.hconcat(htile[30:36])
    board = cv2.vconcat([row0, row1, row2, row3, row4, row5])
    board = cv2.vconcat([textimg, displayimg, board, textsrcimg])
    return(board)
    
def create_full_flashboard(bim, gim, rim, grim, textimg, displayimg, textsrcimg, idx, flag_fix_distraction, distraction_list, bspace_char_flag, flag_word_complete):
    
    if not flag_fix_distraction: # Colorize the flashboard based on feedback and flag
        him = gim
    elif len(distraction_list) > 0: # Change attended colors based on last distraction indication or character error (backspace)
        if distraction_list[-1]:
            him = rim
        else:
            him = gim
    else:
        him = gim
        
    if not flag_fix_distraction: # Colorize the attended row/col based on feedback and flag
        bkim = bim
    elif len(distraction_list) > 1: # Change background colors based on last two distraction indication
        if  distraction_list[-2] and distraction_list[-1]:
          bkim = grim  
        else:
          bkim = bim
    else:
        bkim = bim
        
    if idx == 0:
        board = cv2.vconcat([him[0], bkim[1], bkim[2], bkim[3], bkim[4], bkim[5]])
    elif idx == 1:
        board = cv2.vconcat([bkim[0], him[1], bkim[2], bkim[3], bkim[4], bkim[5]]) 
    elif idx == 2:
        board = cv2.vconcat([bkim[0], bkim[1], him[2], bkim[3], bkim[4], bkim[5]])
    elif idx == 3:
        board = cv2.vconcat([bkim[0], bkim[1], bkim[2], him[3], bkim[4], bkim[5]])
    elif idx == 4:
        board = cv2.vconcat([bkim[0], bkim[1], bkim[2], bkim[3], him[4], bkim[5]])
    elif idx == 5:
        board = cv2.vconcat([bkim[0], bkim[1], bkim[2], bkim[3], bkim[4], him[5]])
    elif idx == 6:
        board = cv2.hconcat([him[6], bkim[7], bkim[8], bkim[9], bkim[10], bkim[11]])
    elif idx == 7:
        board = cv2.hconcat([bkim[6], him[7], bkim[8], bkim[9], bkim[10], bkim[11]]) 
    elif idx == 8:
        board = cv2.hconcat([bkim[6], bkim[7], him[8], bkim[9], bkim[10], bkim[11]])
    elif idx == 9:
        board = cv2.hconcat([bkim[6], bkim[7], bkim[8], him[9], bkim[10], bkim[11]])
    elif idx == 10:
        board = cv2.hconcat([bkim[6], bkim[7], bkim[8], bkim[9], him[10], bkim[11]])
    elif idx == 11:
        board = cv2.hconcat([bkim[6], bkim[7], bkim[8], bkim[9], bkim[10], him[11]])  
    else:
        board = cv2.hconcat([bkim[6], bkim[7], bkim[8], bkim[9], bkim[10], bkim[11]])  

    board = cv2.vconcat([textimg, displayimg, board, textsrcimg])
    return(board)

def center_text_word(img, font, text, strip_width, strip_height, color=(255, 255, 255)):
    draw = ImageDraw.Draw(img)
    text_width, text_height = draw.textsize(text, font)
  #  offset_x, offset_y = font.getoffset(text)
  #  text_width += offset_x
  #  text_height += offset_y
    text_height += 5
    position = ((strip_width-text_width)/2,(strip_height-text_height)/2)
  #  print(strip_width, strip_height, text_width, text_height, position)
    draw.text(position, text, color, font=font, align="left")
    return img

def center_text(img, font, text, strip_width, strip_height, color=(255, 255, 255)):
    draw = ImageDraw.Draw(img)
    text_width, text_height = draw.textsize(text, font)
  #  offset_x, offset_y = font.getoffset(text)
  #  text_width += offset_x
  #  text_height += offset_y
    text_height += 5
    position = ((strip_width-text_width)/2,(strip_height-text_height)/2)
  #  print(strip_width, strip_height, text_width, text_height, position)
    draw.text(position, text, color, font=font, align="left")
    return img

def puzzle_block(text, font):
  w, h = 50, 50
  shape = [(0, 0), (w - 1, h - 1)]  

  if not text:
    img=Image.new(mode="RGB", size=(w,h),color =(0,0,0))
    img1 = ImageDraw.Draw(img)
    img1.rectangle(shape, fill=(0,0,0))
  else:
    img=Image.new(mode="RGB", size=(w,h),color =(255,255,255))
    img1 = ImageDraw.Draw(img)
    if text == '*':
       img1.rectangle(shape, fill=(0,0,255), outline=(0,0,0), width=1)
    else:
       img1.rectangle(shape, fill=(255,255,255), outline=(0,0,0), width=1)
    img = center_text(img, font, text, w, h, (0,0,0))
#  img.save("testblue.png")
  return img


def blue_block(text, font):
  w, h = 80, 80
  shape = [(0, 0), (w - 1, h - 1)]  
  img=Image.new(mode="RGB", size=(w,h),color =(104,157,155))
  img1 = ImageDraw.Draw(img)
  img1.rectangle(shape, fill=(104,157,155), outline=(0,0,0), width=2)
  if len(text) > 0:
    img = center_text(img, font, text, w, h, (0,0,0))
#  img.save("testblue.png")
  return img

def gray_block(text, font):
    
  w, h = 80, 80
  shape = [(0, 0), (w - 1, h - 1)]  
  img=Image.new(mode="RGB", size=(w,h),color =(255,255,255))
  img1 = ImageDraw.Draw(img)
  img1.rectangle(shape, fill=(255,255,255), outline=(0,0,0), width=2)
  if len(text) > 0:
    img = center_text(img, font, text, w, h, (0,0,0))
#  img.save("testgray.png")
  return img    

def green_block(text, font):
    
  w, h = 80, 80
  shape = [(0, 0), (w - 1, h - 1)]  
  img=Image.new(mode="RGB", size=(w,h),color =(0,255,0))
  img1 = ImageDraw.Draw(img)
  img1.rectangle(shape, fill=(0,255,0), outline=(0,0,0), width=2)
  if len(text) > 0:
    img = center_text(img, font, text, w, h, (0,0,0))
#  img.save("testgray.png")
  return img 

def red_block(text, font):
  w, h = 80, 80
  shape = [(0, 0), (w - 1, h - 1)]  
  img=Image.new(mode="RGB", size=(w,h),color =(0,0,255))
  img1 = ImageDraw.Draw(img)
  img1.rectangle(shape, fill=(0,0,255), outline=(0,0,0), width=2)
  if len(text) > 0:
    img = center_text(img, font, text, w, h, (0,0,0))
  #  img.save("testred.png")
  return img   

def getpzl_img(text, width, height):

    fontsize = 26
    font = ImageFont.truetype("arial.ttf", fontsize)
    im = []
    for j in range(height):
      pzlimg = [np.array(puzzle_block(x,font)) for x in text[j]]
      im.append(cv2.hconcat(pzlimg))
    pzlboard = cv2.vconcat(im)
    return(pzlboard)
        
def getimg(charset):
    fontsize = 52
    charset[33] = '.'
    charset[34] = '<'
    charset[35] = '_'
    font = ImageFont.truetype("arial.ttf", fontsize)
    bimg = [np.array(blue_block(x,font)) for x in charset]   
    gimg = [np.array(gray_block(x,font)) for x in charset]
    rimg = [np.array(red_block(x,font)) for x in charset]
    grimg = [np.array(green_block(x,font)) for x in charset]
    return(bimg, gimg, rimg, grimg)

def getcharimg(mychar):
    fontsize = 52

    font = ImageFont.truetype("arial.ttf", fontsize)
    bimg = np.array(blue_block(mychar,font)) 
    gimg = np.array(gray_block(mychar,font)) 
    rimg = np.array(red_block(mychar,font)) 
    grimg = np.array(green_block(mychar,font)) 
    return(bimg, gimg, rimg, grimg)    

def create_tile(img, flashboard_type, loc):
    
    if flashboard_type == 0:
      im0 = cv2.hconcat([img[0], img[1], img[2], img[3], img[4], img[5]])
      im1 = cv2.hconcat([img[6], img[7], img[8], img[9], img[10], img[11]])
      im2 = cv2.hconcat([img[12], img[13], img[14], img[15], img[16], img[17]])
      im3 = cv2.hconcat([img[18], img[19], img[20], img[21], img[22], img[23]])
      im4 = cv2.hconcat([img[24], img[25], img[26], img[27], img[28], img[29]])
      im5 = cv2.hconcat([img[30], img[31], img[32], img[33], img[34], img[35]])
    
      im6 = cv2.vconcat([img[0], img[6], img[12], img[18], img[24], img[30]])
      im7 = cv2.vconcat([img[1], img[7], img[13], img[19], img[25], img[31]])
      im8 = cv2.vconcat([img[2], img[8], img[14], img[20], img[26], img[32]])
      im9 = cv2.vconcat([img[3], img[9], img[15], img[21], img[27], img[33]])
      im10 = cv2.vconcat([img[4], img[10], img[16], img[22], img[28], img[34]])
      im11 = cv2.vconcat([img[5], img[11], img[17], img[23], img[29], img[35]])
    elif flashboard_type == 1: 
              
      im0 = cv2.hconcat([img[loc[0]],  img[loc[6]],  img[loc[16]], img[loc[24]],  img[loc[30]], img[loc[34]]])
      im1 = cv2.hconcat([img[loc[11]], img[loc[1]],  img[loc[7]],  img[loc[17]],  img[loc[25]], img[loc[31]]])
      im2 = cv2.hconcat([img[loc[20]], img[loc[12]], img[loc[2]],  img[loc[8]],   img[loc[18]], img[loc[26]]])
      im3 = cv2.hconcat([img[loc[27]], img[loc[21]], img[loc[13]], img[loc[3]],   img[loc[9]],  img[loc[19]]])
      im4 = cv2.hconcat([img[loc[32]], img[loc[28]], img[loc[22]], img[loc[14]],  img[loc[4]],  img[loc[10]]])
      im5 = cv2.hconcat([img[loc[35]], img[loc[33]], img[loc[29]], img[loc[23]],  img[loc[15]], img[loc[5]]]) 
      
      im6 =  cv2.vconcat([img[loc[0]],  img[loc[11]], img[loc[20]], img[loc[27]],  img[loc[32]], img[loc[35]]])
      im7 =  cv2.vconcat([img[loc[6]],  img[loc[1]],  img[loc[12]], img[loc[21]],  img[loc[28]], img[loc[33]]])
      im8 =  cv2.vconcat([img[loc[16]], img[loc[7]],  img[loc[2]],  img[loc[13]],  img[loc[22]], img[loc[29]]])
      im9 =  cv2.vconcat([img[loc[24]], img[loc[17]], img[loc[8]],  img[loc[3]],   img[loc[14]], img[loc[23]]])
      im10 = cv2.vconcat([img[loc[30]], img[loc[25]], img[loc[18]], img[loc[9]],   img[loc[4]],  img[loc[15]]])
      im11 = cv2.vconcat([img[loc[34]], img[loc[31]], img[loc[26]], img[loc[19]],  img[loc[10]], img[loc[5]]]) 
    return([im0, im1, im2, im3, im4, im5, im6, im7, im8, im9, im10, im11])

def get_weighted_scanorder(loc, dist, flashboard_type, scan_scheme):
    rows_w, cols_w = [], []
    if scan_scheme == 2:
      if flashboard_type == 0: # Get the order for the weighted scan scheme
         rows = [sum(dist[0:6]), sum(dist[6:12]), sum(dist[12:18]), sum(dist[18:24]), sum(dist[24:30]), sum(dist[30:36])]
         cols = [sum(dist[6*i] for i in range(6)), sum(dist[6*i + 1] for i in range(6)), sum(dist[6*i + 2] for i in range(6)),
                   sum(dist[6*i + 3] for i in range(6)), sum(dist[6*i + 4] for i in range(6)), sum(dist[6*i + 5] for i in range(6))]
      elif flashboard_type == 1:
         rows = [dist[loc[0]]+dist[loc[6]]+dist[loc[16]]+dist[loc[24]]+dist[loc[30]]+dist[loc[34]], dist[loc[11]]+dist[loc[1]]+dist[loc[7]]+dist[loc[17]]+dist[loc[25]]+dist[loc[31]] ,
                    dist[loc[20]]+dist[loc[12]]+dist[loc[2]]+dist[loc[8]]+dist[loc[18]]+dist[loc[26]], dist[loc[27]]+dist[loc[21]]+dist[loc[13]]+dist[loc[3]]+dist[loc[9]]+dist[loc[19]] ,
                    dist[loc[32]]+dist[loc[28]]+dist[loc[22]]+dist[loc[14]]+dist[loc[4]]+dist[loc[10]], dist[loc[35]]+dist[loc[33]]+dist[loc[29]]+dist[loc[23]]+dist[loc[15]]+dist[loc[5]]] 
         cols = [dist[loc[0]]+dist[loc[11]]+dist[loc[20]]+dist[loc[27]]+dist[loc[32]]+dist[loc[35]], dist[loc[6]]+dist[loc[1]]+dist[loc[12]]+dist[loc[21]]+dist[loc[28]]+dist[loc[33]] ,
                    dist[loc[16]]+dist[loc[7]]+dist[loc[2]]+dist[loc[13]]+dist[loc[22]]+dist[loc[29]], dist[loc[24]]+dist[loc[17]]+dist[loc[8]]+dist[loc[3]]+dist[loc[14]]+dist[loc[23]] ,
                    dist[loc[30]]+dist[loc[25]]+dist[loc[18]]+dist[loc[9]]+dist[loc[4]]+dist[loc[15]], dist[loc[34]]+dist[loc[31]]+dist[loc[26]]+dist[loc[19]]+dist[loc[10]]+dist[loc[5]]]
      rows_w = [rows.index(i) for i in sorted(rows, reverse=True)]  
      cols_w = [cols.index(i)+6 for i in sorted(cols, reverse=True)] 
      
    return(rows_w, cols_w)

def create_other_tiles():
    
    textbgnd_img = np.zeros((80, 480, 3), dtype="uint8")
    display_bgndimg = np.ones((40, 480, 3), dtype="uint8")
    #eye = cv2.imread('eye.png')
    #eye = cv2.resize(eye, (352,240))
    eyebgnd_img = np.zeros((300,320, 3), dtype="uint8")
    eye = cv2.circle(eyebgnd_img,(150,150), 100, (0,0,255), -1)
    eye = cv2.circle(eye,(150,150), 25, (0,0,0), -1)
 #   crop_image =gim1()
  #  cv2.imshow('debug14', crop_image)
    return(textbgnd_img, display_bgndimg, eye)
    
 #   cv2.imshow("test", board)
 #   board.show()
 
def create_tile_for_word(): # Create blank tiles to use for word prediction display on flashboard
    fontsize = 12
    font = ImageFont.truetype("arial.ttf", fontsize) 
    bim_for_word = blue_block('',font)
    gim_for_word = gray_block('',font)
    rim_for_word = red_block('',font)
    grim_for_word = green_block('',font)
    return(bim_for_word, gim_for_word, rim_for_word, grim_for_word)
 