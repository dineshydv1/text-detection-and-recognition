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
#%matplotlib inline
#from text_detection_image import *

# ----------------------------------
# Step 1 : Defining input parameters 
# ----------------------------------
# input image to recognize text
image = '.\\static\\images\\car_wash.png'
#image = '.\\data\\images\\lebron_james.jpg'
#image = '.\\data\\images\\sign.jpg'
# EAST mode path
east_model_path = '.\\static\\EAST\\frozen_east_text_detection.pb'
# pytesseract exe path
pyterrsect_exe_path = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# minimum probability required to inspect a regions
min_confidence=0.5
# resized image width (should be multiple of 32)
width=480
# resized image height (should be multiple of 32)
height=480
# BB padding around text
padding=0.01
