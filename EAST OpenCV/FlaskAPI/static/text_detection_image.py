# -*- coding: utf-8 -*-
"""
@author: yadav

INFO: Script to detect and recognize text on image. 

function can be called like below:
text_detection_in_image(image_path_name=<image-path>,
                       east_path=<east-model-path>,
                       pyterrsect_path=<pyterrsect-exe-path>)
Script output:
output is list containing BB coordinates and text in below format
((startX, startY, endX, endY), text)

"""

# --------------------------------------
# Step 1 : Import libraries and set input arguments
# --------------------------------------
# import required libraries
from imutils.object_detection import non_max_suppression
import numpy as np
#import pandas as pd
#import argparse
import time
import cv2
import os
#from matplotlib import pyplot as plt
#%matplotlib inline
import pytesseract


# -------------------------------------
# function to read text probability score and their coordinates. 
# this function will return coordinates with probability greater than min_confidence 
# -------------------------------------
def decode_predictions(scores,   # array of probability scores
                       geometry, # array of coordinates
                       args_min_confidence # minimum confidence score
                       ):
    
    # grab the number of rows and columns from the scores volume,
    # then initialize our set of bounding box rectangles and
    # corresponding confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []        # list to store the bounding box (x,y) coordinates for text regions
    confidences = []  # list to store the porbabilities associated with each bounding box in rects
    
    # loop over the number of rows
    for y in range(numRows):
    #for y in range(1):
        # extract the scores (probabilities), followed by the geometrical 
        # data used to derive potential bounding box coordinates that surround text
        # get once score row (it has numCols values)
        scoresData = scores[0, 0, y]
        # get its bounding box values
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        angleData = geometry[0, 4, y]
        
        # loop thru each numCols values to check if it meets minimum probability criteria
        for x in range(numCols):
            # check if the value has sufficient probability
            #print(x)
            #print(scoresData[x])
            if scoresData[x] < args_min_confidence:
                continue    # go to next value in loop
            
            # compute the offset factor as our resulting feeatures maps
            # will be 4x smaller than the input image
            (offsetX, offsetY) = (x*4.0, y*4.0)
            # IMP: The EAST text detector naturally reduces volume size as the image passes 
            # through the network. Our volume size is actually 4x smaller than our input image 
            # so we multiply by four to bring the coordinates back into respect of our original image.
            
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = angleData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            
            # use the geometry volume to derive the width and height of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            
            # compute both the starting and ending (x,y) coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY + (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            
            # append the bounding box xoordinates and probability score to list
            #print(scoresData[x],startX, startY, endX, endY)
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
        
    # return a tuple of bounding boxes and associated coordinates
    return (rects, confidences)
        

# END decode_predictions------------------------------------



# ---------------------
# main functtion
# ---------------------
    
# Important: The EAST text requires that your input image dimensions be multiples of 32, 
# so if you choose to adjust your --width  and --height variable values, 
# make sure they are multiples of 32!

def text_detection_in_image(image_path_name,      # input image full path and name
                            east_path,            # path to input EAST text detector file
                            pyterrsect_path,      # path to pytesseract exe file
                            min_confidence=0.05,  # minimum probability required to inspect a regions
                            width=640,            # resized image width (should be multiple of 32)
                            height=640,           # resized image height (should be multiple of 32)
                            padding=0.01          # BB padding around text
                            ):
    
    
    args_image = image_path_name
    args_east = east_path
    args_pytesseract = pyterrsect_path
    args_min_confidence = min_confidence
    args_width = width
    args_height = height
    args_padding = padding
    
    print('[INFO] Image file/path passed: ',args_image)
    print('[INFO] EAST passed: ',args_east)
    print('[INFO] pytesseract exe file passed: ',args_pytesseract)
    print('[INFO] min_confidence passed: ', args_min_confidence)
    print('[INFO] width passed: ',args_width)
    print('[INFO] height passed: ',args_height)
    print('[INFO] BB padding: ',args_padding)
    
    # define pytesseract exe file
    pytesseract.pytesseract.tesseract_cmd = args_pytesseract
    
    # --------------------------------------
    # Step 2 : Load OpenCV EAST text detector in memory 
    # --------------------------------------
    
    # load the pre-trained EAST text detector
    print('[INFO] loading EAST text detector...')
    
    # load neuralnet into memory by passing the path of EAST detector
    net = cv2.dnn.readNet(args_east)  
    
    # In order to perform text detection using OpenCV and the EAST deep learning model, 
    # we need to extract the output feature maps of two layers:
    # We define the two output layer names for the EAST detector model that we are interested in:
    # First is the output probabilities. This layer is our output activation 
    # which gives us the probability of a region containing text or not.
    # Second can be used to derive the bounding box coordinates of text. 
    # Second layer is the output feature man that represents the "geometry" of the image. 
    # We will be able to use this geometry to derive the bounding box coordnates of the text in input image 
    
    layerNames = ['feature_fusion/Conv_7/Sigmoid', # probability of region containing text
                  'feature_fusion/concat_3']       # geometry feature map to use for bounding box
    
    
    # --------------------------------------
    # Step 3 : Load and process and resize images
    # --------------------------------------

    # check if input image file exists 
    # if not give error
    if os.path.isfile(args_image) == False:
        print('[ERROR] Image file not found. Pass image full path and name.')
        return 'Failed'
    
    this_image = args_image
    print('[INFO] processing image: ',this_image)
    # load input image and grab its dimension
    image = cv2.imread(this_image) 
    orig = image.copy()
    (H, W) = image.shape[:2]
    
    # set the new width and height and determine the ratio in change. 
    # also called resize factor
    (newW, newH) = (args_width, args_height) 
    rW = W / float(newW)
    rH = H / float(newH)
    
    # resize the image and grab new dimension
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]
    
    
    # --------------------------------------
    # Step 4 : Process image thru EAST model and extract features
    # --------------------------------------
    
    # construct a blob from the image and then perform a forward pass of 
    # the model to obtain the two output layer set
    # convert input image to blob
    # more info at: https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
    blob = cv2.dnn.blobFromImage(image, 
                                 1.0,
                                 (W, H),
                                 (123.68, 116.78, 103.94),
                                 swapRB=True,
                                 crop=False )
    
    start = time.time()
    
    # to preduct text, we can simply set the blob as input and call net.forward
    # by supplying layerNames as parameter to net.forwardm we are instructing openCV to return
    # the two feature maps
    # output geometry used to derive bounding box coordinates of text in input image
    # output scores contains the probability of given region containing text
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()
    
    # showing timing information on text prediction
    print('[INFO] text detection took {:6f} seconds'.format(end - start))
    
    
    # --------------------------------------
    # Step 5 : Loop thru features and keep strong overlapping bounding boxes only
    # --------------------------------------
    
    (rects, confidences) = decode_predictions(scores, geometry, args_min_confidence)
    
    # The final step is to apply non-maxima suppression to our bounding boxes to suppress 
    # weak overlapping bounding boxes and then display the resulting text predictions:
    # apply non-maxima suppresion to suppress weak, overlapping bounding boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    print('[INFO] {0} bounding box detected'.format(boxes.shape[0]))
    
    # --------------------------------------
    # Step 6 : Show final image
    # --------------------------------------
    
    # list to hold final result
    # output is BB corrdinates and text inside BB
    results = []
    
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        #print(startX, startY, endX, endY) 
        # scale the bounding box coordinates based on the respective ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        #print('scaled: ',startX, startY, endX, endY) 
        
        # to get better OCR of text, we can apply a bit of padding surrounding the bounding box
        # this will increase the area of bounding box 
        # and will reduce any probability of cutting thru text
        dX = int((endX - startX) * args_padding)
        dY = int((endY - startY) * args_padding)
        # apply padding to each side of bounding-box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(W, endX + (dX * 2))
        endY = min(H, endY + (dY * 2))
        
        # extract the actual padded ROI
        roi = orig[startY:endY, startX:endX]
        # now that we have extracted ROI (bounding box), 
        # we will pass this part of image tesseract to extract text. 
        # in order to aupply Tesseract to OCR text, we will set few parameter required for it. 
        # -l (language) to english
        # -oem flag of 4 indicating that we wish to use LSTM neural net model for OCR
        # -psm value of 7 which implies that we are treating the ROI as a single line or text
        print('Calling pytesseract')
        config = ("-l eng --oem 1 --psm 7")
        text = pytesseract.image_to_string(roi, config=config)
        #print('test: ',text)
        # in just 2 lines of code, we have used tesseract to recognize a text from an image. 
        # but there is lot happening under the hood. 
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV, then draw the text and a bounding box surrounding the text region
        text = ''.join([c if ord(c) < 128 else '' for c in text]).strip()

        
        # append BB coordinates and OCR text to the return list
        results.append(((startX, startY, endX, endY), text))

    # return all images with bounding boxes
    # return all_image_bb
    # return BB coordinates and text inside them
    return results

# END text_detection_in_image -------------------------------------

