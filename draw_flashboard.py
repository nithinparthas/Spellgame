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
import textwrap
import tkinter as tk

def display_message(text):  # Widget to display message

   root = tk.Tk()

   T = tk.Text(root, height=20, width = 50)  
   T.pack()
   T.insert(tk.END,text)
   tk.mainloop()
   
def display_infoimg(text, display_bgndimg):
    font = cv2.FONT_HERSHEY_PLAIN
    font_size = 3
    font_thickness = 4
    wrapped_srctext = textwrap.wrap(text, width=50)
    y_text = 40
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
 
def cv2_get_full_image(bim, gim, rim, grim, textimg, displayimg, textsrcimg, idx, flag_fix_distraction, distraction_list, bspace_char_flag):
    
    if not flag_fix_distraction: # Colorize the flashboard based on feedback and flag
        him = gim
    elif distraction_list[-1]: # Change attended colors based on last distraction indication or character error (backspace)
        him = rim
    else:
        him = gim
        
    if not flag_fix_distraction: # Colorize the attended row/col based on feedback and flag
        bkim = bim
    elif distraction_list[-2] and distraction_list[-1]: # Change background colors based on last two distraction indication
        bkim = grim
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

def blue_block(text, font):
  w, h = 80, 80
  shape = [(0, 0), (w - 1, h - 1)]  
  img=Image.new(mode="RGB", size=(w,h),color =(104,157,155))
  img1 = ImageDraw.Draw(img)
  img1.rectangle(shape, fill=(104,157,155), outline=(0,0,0), width=2)
  img = center_text(img, font, text, w, h, (0,0,0))
#  img.save("testblue.png")
  return img

def gray_block(text, font):
    
  w, h = 80, 80
  shape = [(0, 0), (w - 1, h - 1)]  
  img=Image.new(mode="RGB", size=(w,h),color =(255,255,255))
  img1 = ImageDraw.Draw(img)
  img1.rectangle(shape, fill=(255,255,255), outline=(0,0,0), width=2)
  img = center_text(img, font, text, w, h, (0,0,0))
#  img.save("testgray.png")
  return img    

def green_block(text, font):
    
  w, h = 80, 80
  shape = [(0, 0), (w - 1, h - 1)]  
  img=Image.new(mode="RGB", size=(w,h),color =(0,255,0))
  img1 = ImageDraw.Draw(img)
  img1.rectangle(shape, fill=(0,255,0), outline=(0,0,0), width=2)
  img = center_text(img, font, text, w, h, (0,0,0))
#  img.save("testgray.png")
  return img 

def red_block(text, font):
  w, h = 80, 80
  shape = [(0, 0), (w - 1, h - 1)]  
  img=Image.new(mode="RGB", size=(w,h),color =(0,0,255))
  img1 = ImageDraw.Draw(img)
  img1.rectangle(shape, fill=(0,0,255), outline=(0,0,0), width=2)
  img = center_text(img, font, text, w, h, (0,0,0))
  img.save("testred.png")
  return img   
  
def create_tile(charset):
    
    fontsize = 52
    charset[33] = '.'
    charset[34] = '<'
    charset[35] = '_'
    font = ImageFont.truetype("arial.ttf", fontsize)
  #  img1 = np.array(blue_block('a',font))

    img = [np.array(blue_block(x,font)) for x in charset] 
    bim0 = cv2.hconcat([img[0], img[1], img[2], img[3], img[4], img[5]])
    bim1 = cv2.hconcat([img[6], img[7], img[8], img[9], img[10], img[11]])
    bim2 = cv2.hconcat([img[12], img[13], img[14], img[15], img[16], img[17]])
    bim3 = cv2.hconcat([img[18], img[19], img[20], img[21], img[22], img[23]])
    bim4 = cv2.hconcat([img[24], img[25], img[26], img[27], img[28], img[29]])
    bim5 = cv2.hconcat([img[30], img[31], img[32], img[33], img[34], img[35]])
    
    bim6 = cv2.vconcat([img[0], img[6], img[12], img[18], img[24], img[30]])
    bim7 = cv2.vconcat([img[1], img[7], img[13], img[19], img[25], img[31]])
    bim8 = cv2.vconcat([img[2], img[8], img[14], img[20], img[26], img[32]])
    bim9 = cv2.vconcat([img[3], img[9], img[15], img[21], img[27], img[33]])
    bim10 = cv2.vconcat([img[4], img[10], img[16], img[22], img[28], img[34]])
    bim11 = cv2.vconcat([img[5], img[11], img[17], img[23], img[29], img[35]])
    
    img = [np.array(gray_block(x,font)) for x in charset] 
    gim0 = cv2.hconcat([img[0], img[1], img[2], img[3], img[4], img[5]])
    gim1 = cv2.hconcat([img[6], img[7], img[8], img[9], img[10], img[11]])
    gim2 = cv2.hconcat([img[12], img[13], img[14], img[15], img[16], img[17]])
    gim3 = cv2.hconcat([img[18], img[19], img[20], img[21], img[22], img[23]])
    gim4 = cv2.hconcat([img[24], img[25], img[26], img[27], img[28], img[29]])
    gim5 = cv2.hconcat([img[30], img[31], img[32], img[33], img[34], img[35]])
    
    gim6 = cv2.vconcat([img[0], img[6], img[12], img[18], img[24], img[30]])
    gim7 = cv2.vconcat([img[1], img[7], img[13], img[19], img[25], img[31]])
    gim8 = cv2.vconcat([img[2], img[8], img[14], img[20], img[26], img[32]])
    gim9 = cv2.vconcat([img[3], img[9], img[15], img[21], img[27], img[33]])
    gim10 = cv2.vconcat([img[4], img[10], img[16], img[22], img[28], img[34]])
    gim11 = cv2.vconcat([img[5], img[11], img[17], img[23], img[29], img[35]])
    
    img = [np.array(red_block(x,font)) for x in charset] 
    rim0 = cv2.hconcat([img[0], img[1], img[2], img[3], img[4], img[5]])
    rim1 = cv2.hconcat([img[6], img[7], img[8], img[9], img[10], img[11]])
    rim2 = cv2.hconcat([img[12], img[13], img[14], img[15], img[16], img[17]])
    rim3 = cv2.hconcat([img[18], img[19], img[20], img[21], img[22], img[23]])
    rim4 = cv2.hconcat([img[24], img[25], img[26], img[27], img[28], img[29]])
    rim5 = cv2.hconcat([img[30], img[31], img[32], img[33], img[34], img[35]])
    
    rim6 = cv2.vconcat([img[0], img[6], img[12], img[18], img[24], img[30]])
    rim7 = cv2.vconcat([img[1], img[7], img[13], img[19], img[25], img[31]])
    rim8 = cv2.vconcat([img[2], img[8], img[14], img[20], img[26], img[32]])
    rim9 = cv2.vconcat([img[3], img[9], img[15], img[21], img[27], img[33]])
    rim10 = cv2.vconcat([img[4], img[10], img[16], img[22], img[28], img[34]])
    rim11 = cv2.vconcat([img[5], img[11], img[17], img[23], img[29], img[35]])
    
    img = [np.array(green_block(x,font)) for x in charset] 
    grim0 = cv2.hconcat([img[0], img[1], img[2], img[3], img[4], img[5]])
    grim1 = cv2.hconcat([img[6], img[7], img[8], img[9], img[10], img[11]])
    grim2 = cv2.hconcat([img[12], img[13], img[14], img[15], img[16], img[17]])
    grim3 = cv2.hconcat([img[18], img[19], img[20], img[21], img[22], img[23]])
    grim4 = cv2.hconcat([img[24], img[25], img[26], img[27], img[28], img[29]])
    grim5 = cv2.hconcat([img[30], img[31], img[32], img[33], img[34], img[35]])
    
    grim6 = cv2.vconcat([img[0], img[6], img[12], img[18], img[24], img[30]])
    grim7 = cv2.vconcat([img[1], img[7], img[13], img[19], img[25], img[31]])
    grim8 = cv2.vconcat([img[2], img[8], img[14], img[20], img[26], img[32]])
    grim9 = cv2.vconcat([img[3], img[9], img[15], img[21], img[27], img[33]])
    grim10 = cv2.vconcat([img[4], img[10], img[16], img[22], img[28], img[34]])
    grim11 = cv2.vconcat([img[5], img[11], img[17], img[23], img[29], img[35]])  
    
    textbgnd_img = np.zeros((80, 480, 3), dtype="uint8")
    display_bgndimg = np.ones((40, 480, 3), dtype="uint8")
    #eye = cv2.imread('eye.png')
    #eye = cv2.resize(eye, (352,240))
    eyebgnd_img = np.zeros((300,300, 3), dtype="uint8")
    eye = cv2.circle(eyebgnd_img,(150,150), 100, (0,0,255), -1)
    
 #   crop_image =gim1()
  #  cv2.imshow('debug14', crop_image)
    return([bim0, bim1, bim2, bim3, bim4, bim5, bim6, bim7, bim8, bim9, bim10, bim11],
            [gim0, gim1, gim2, gim3, gim4, gim5, gim6, gim7, gim8, gim9, gim10, gim11], 
            [grim0, grim1, grim2, grim3, grim4, grim5, grim6, grim7, grim8, grim9, grim10, grim11],
            [rim0, rim1, rim2, rim3, rim4, rim5, rim6, rim7, rim8, rim9, rim10, rim11], textbgnd_img, display_bgndimg, eye)
    
 #   cv2.imshow("test", board)
 #   board.show()
    