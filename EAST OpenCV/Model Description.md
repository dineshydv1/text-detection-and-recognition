### EAST Algorithm with OpenCV: 
#### PART-1: Text detection:
EAST text detector is a deep learning model which is capable to run near real time and gives very high text-detection accuracy. EAST text detection requires OpenCV 3.4.2 or 4 version. EAST model gives the probability of text and the coordinates of bounding box around text area. 

#### PART-2: Text recognization:
EAST model draws bounding box, called ROI, around the text. These text-ROI are extract from image and pass them into Tessaract LSTM deep learning text recognition algorithm. The output of LSTM will give us actual OCR results.

#### Model Data Flow:

1. INPUT
- IMAGE (picture/photo)
- VIDEO( recorded video or Live feed such as CCTV, Webcam)

2. PRE-PROCESSING
- Resize input image to common format/dimension
- Convert image to blob

3. EAST Model
- Pass image blob to EAST model
- Get output of below layers
 - Probability of given region containing text
 - Bounding-box coordinates of region containing text

4. POST-PROCESSING
- Discard probabilities below minimum confidence level
- Apply non-maxima suppresion to suppress weak, overlapping bounding boxes

5. DRAW BOUNDING BOX
- Scale bounding box coordinates per scaling factor 
- Draw bounding box on image and display image

6. EXTRACT ROI
- Add padding to bounding-box coordinates.
- Extract ROI (bounding-box containing text) from image

7. TEXT RECOGNITION
- Pass ROI to Tesseract model 

8. DISPLAY TEXT ON IMAGE
- Display text along with bounding-box on image


#### Model Description:

1. INPUT
Model input is image containing text which the model should detect and identify. 

2. PRE-PROCESSING
The EAST text requires that your input image dimensions be multiples of 32, so while adjusting width and height values, make sure they are multiples of 32. 
For our model, we are resizing input image to 320 (width) by 320 (height). While resizing, note down the resize ratio/factor. This factor would be required later on to rescale bounding-boxes to draw on image. 

We are using blobFromImage from OpenCV’s new deep neural network ( dnn ) module to prepare image for classification via pre-trained deep-learning models. Function blobFromImage performs mean subtraction and scaling which usually gives better results. 

3. EAST Model
EAST stands for "Efficient and Accurate Scene Text". It is a deep learning model which is capable to run near real time and gives very high text-detection accuracy. From EAST model, we extract output features of 2 layers:
a. First output layer gives the probability/score of a region having text or not. It is output of sigmoid activation. 
b. Second output layers gives the coordinates or the geometry of text area which can be used further to draw bounding box around text area. 


4. POST-PROCESSING
EAST text detector model, reduces volume size of image by a factor of 4, so we need to multiply the geometry coordinates by 4 to bring them back into respect of our original image. 

EAST model gives the probability score of region containing text or not. Probability score would be higher for regions having text and lower for regions not having text. So we ignore the weak text detection areas having probability score less then minimum threshold confidence value. 

We apply non-maxima supression to bounding boxes to supress weak and overlapping bounding boxes. It retains the most likely text regions and eliminates other overlapping regions. 


5. DRAW BOUNDING BOX
Coordinates of bounding boxes are resized as per resizing factor/ratio. This will bring coordinates in respect to the original image size. The resulting bounding boxes are drawn on the image. 


6. EXTRACT ROI
To get better OCR results, we apply padding to bounding box coordinates. This will increase the area around the text and will reduce the probability of text being omitted while extracting the text area. 
The bounding box area of image containing text is the Region-of-Interest (ROI). 


7. TEXT RECOGNITION
We extract ROI from image and pass them into Tessaract LSTM deep learning text recognition algorithm. The output of LSTM will give us actual OCR results. We are using pytesseract library with below parameters:

-l : it controls the language of input text. We will be using eng (engligh) for this example. Tesseract supports many other languages.
--oem : known as OCR engine mode, it controls the type of algorithm used by tesseract. We will pass value of 4 indicating that we wish to use LSTM neural net model for OCR. 
--psm : It controls the automatic page segmentation mode used by tesseract. We will pass value of 7 which implies that we are treating the ROI as a single line or text. 

8. DISPLAY TEXT ON IMAGE
Now that we have identified the text, the final step is to display text along with the bounding box on the image. Strip out any non-ASCII characters from text as OpenCV does not support non-ASCII charaters. 


