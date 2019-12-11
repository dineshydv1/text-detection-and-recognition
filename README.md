# text-detection-and-recognition
Model to detect and recognize text in images and videos

This repository contains code details of my capstone project undertaken as part of SpringBoard AI & Machine Learning program.

Capstone project I undertook is to build a model to detect and recognize text in images and videos. 

Project Description: Task is to identify text in an image or a video stream. Text can be hand written, printed or in any font. It can include numeric or alphabet characters. Image can be of a landscape, sign board, house number etc. Initially, roject is limited to english alphabets only.

I have used below approaches:

### Approach 1: EAST Algorithm with OpenCV: 
#### PART-1: Text detection:
EAST text detector is a deep learning model which is capable to run near real time and gives very high text-detection accuracy. EAST text detection requires OpenCV 3.4.2 or 4 version. EAST model gives the probability of text and the coordinates of bounding box around text area. 

#### PART-2: Text recognization:
EAST model draws bounding box, called ROI, around the text. These text-ROI are extract from image and pass them into Tessaract LSTM deep learning text recognition algorithm. The output of LSTM will give us actual OCR results.

### Approach 2: YOLO Algorithm:
PENDING. 
