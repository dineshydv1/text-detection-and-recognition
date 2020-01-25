# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:27:12 2020

https://www.geeksforgeeks.org/deploy-machine-learning-model-using-flask/

"""
import os

from flask import Flask, render_template, url_for, request
#from flask import Flask, render_template, request, jsonify

# import common variable like EAST, tesseract
from static.config import *
# import python functions
from static.text_detection_image import *

# create flask instance taking __name__ of the script
app = Flask(__name__)

# main route
@app.route('/')
def index():
    return render_template('home.html')


# main route
@app.route('/result', methods=['POST'])
def result(): 
    if request.method == "POST":
        #result="THIS IS INSANE"
        selImage=request.form.to_dict()
        getImage=selImage['img']
        outText=""

        try:
            
            out = text_detection_in_image(image_path_name=getImage,
                east_path=east_model_path,
                pyterrsect_path=pyterrsect_exe_path,
                min_confidence=min_confidence,
                width=width, 
                height=height, 
                padding=padding  ) 

            outText="Text in Image: "
            for ((startX, startY, endX, endY), text) in out:
                outText = outText + " " + text
        except Exception as e: 
            print(e)
            outText=e

        return render_template('result.html', result=outText)




# -----------------------
# run script from python directly
if __name__ == '__main__':
	#app.run('0.0.0.0', debug=True)
	app.run(debug=True)







