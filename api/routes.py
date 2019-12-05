from api import api
from flask import request,abort,jsonify
from werkzeug.utils import secure_filename
import os
from flask_cors import cross_origin
from py import api as detectApi
from PIL import Image
import base64

def __init__():
    detectApi.init(flozengraphpath=api.config["PATH_TO_FROZEN_GRAPH"],labelpath=api.config["PATH_TO_LABELS_CSV"])
@api.route('/')
@api.route('/index')
def index():
    return ","+api.config["PATH_TO_FROZEN_GRAPH"]+","+api.config["PATH_TO_LABELS_CSV"]

@api.route('/detect', methods=['POST'])
@cross_origin()
def detect():
    f = request.files['image']
    #print(request.json["image"])
    filename = secure_filename(f.filename)
    filePath = os.path.join(api.config['UPLOAD_FOLDER'], filename)
    f.save(filePath)
    inputImg = Image.open(filePath)
    sub_result = detectApi.predict(inputImg,filenameOutput="OUTPUT_"+filename,threshold=float(api.config["THRESHOLD_CONFIDENCE"]))
    result = dict()
    result["details"]=sub_result["details"]
    result["products"]=sub_result["count"]
    #for key,val in sub_result["count"]:
    #    result["products"].append({key:val})
    with open("OUTPUT_"+filename, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        hh = str(encoded_string)
        #print(hh)
        result["image_base64"] = hh#str(encoded_string)
    
    return jsonify(result)
    
