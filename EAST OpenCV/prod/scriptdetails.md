# Running thru script:

### Script/function name and input parameters:
Main script is text_detection_image.py
To use script, call function text_detection_in_image insie the script with below arguments. 
- image_path_name     # [Mandatory] input image full path and name
- east_path           # [Mandatory] path to input EAST text detector file
- pyterrsect_path     # [Mandatory] path to pytesseract exe file
- min_confidence=0.5  # [Optional] minimum probability required to inspect a regions
- width=640           # [Optional] resized image width (should be multiple of 32)
- height=640          # [Optional] resized image height (should be multiple of 32)
- padding = 0.05      # [Optional] BB padding around text

### Script output:
Output of script is a list of tuple with bounding box coordinates and text identified inside them. 
Sample output: ((startX, startY, endX, endY), text)


