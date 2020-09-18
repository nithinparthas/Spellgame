# Spellgame
Top level file is bcigame.py

NOTES

1) Currently all the variables to modify are inside bcigame.py. In a future release, I will move them to an external file
2) A few python libraries like cv2, dlib, numpy, PIL, textwrap, matplotlib, sklearn, gtts, playsound and time  have to be installed (standard pip install)
3) Currently supports eye/face tilt or movement. Face movement is from right (as flashboard is on the right of screen) to the left of screen
4) Currently supports a simple text stored in spell1.txt ("This_is_a_test"). Note that underscore represents spaces between words
5) Assuming an integrated webcam on the laptop or desktop. Supports external webcam too but version1 assumes internal webcam
6) Camera webcam video also shows up on screen. This is not required but is helpful for understanding how the program works
