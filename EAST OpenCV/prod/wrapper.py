# -*- coding: utf-8 -*-
"""
@author: yadav

INFO: Wrapper to call main script

3 steps:
Step 1 : Defining input parameters 
Step 2 : Call main script
Step 3 : Draw BB and text on image

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from text_detection_image import *

# ----------------------------------
# Step 1 : Defining input parameters 
# ----------------------------------
# input image to recognize text
image = '.\\data\\images\\car_wash.png'
#image = '.\\data\\images\\lebron_james.jpg'
#image = '.\\data\\images\\sign.jpg'
# EAST mode path
east_model_path = '.\\EAST\\frozen_east_text_detection.pb'
# pytesseract exe path
pyterrsect_exe_path = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# minimum probability required to inspect a regions
min_confidence=0.5
# resized image width (should be multiple of 32)
width=320
# resized image height (should be multiple of 32)
height=320
# BB padding around text
padding = 0.01

# -------------------------
# Step 2 : Call main script
# -------------------------
# running main script
print('[START] running model')
out = text_detection_in_image(image_path_name=image,
                              east_path=east_model_path,
                              pyterrsect_path=pyterrsect_exe_path,
                              min_confidence=min_confidence,
                              width=width, 
                              height=height, 
                              padding=padding  )

# output is BB coordinates and text in below format
# ((startX, startY, endX, endY), text)

print('[INFO] {0} bounding box detected'.format(len(out)))

# ----------------------------------
# Step 3 : Draw BB and text on image
# ----------------------------------
print('[INFO] drawing bounding box on image')

in_image = plt.imread(image)
for ((startX, startY, endX, endY), text) in out:
    print(startX, startY, endX, endY, text)
    print('Text found: {0}'.format(text))
    # draw the bounding box on image
    #bb_clr = (255,0,0) # blue
    bb_clr = (0,255,0) # green
    #bb_clr = (0,0,255) # red
    bb_thikness = 2
    cv2.rectangle(in_image, (startX, startY), (endX, endY), bb_clr, bb_thikness)
    cv2.putText(in_image, text, (startX, startY + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, bb_clr,bb_thikness)

#plt.imshow(in_image)
#plt.show()
cv2.imshow('Text detection', in_image)
cv2.waitKey(0)

# END: 
