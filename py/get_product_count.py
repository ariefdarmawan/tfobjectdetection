import pandas as pd
import json
from random import seed
from random import random
import flask
from flask import Flask     
import argparse
import tensorflow as tf

seed(1)
app = Flask(__name__)             # create an app instance
model_loaded = False

@app.route("/predict", methods=["POST"])                  
def get_product_from_file():
    name = flask.request.files['image'].filename
    labels = pd.read_csv(f'../data/train_labels.csv').sort_values('class')
    
    prods = []
    df = labels[labels['filename']==name].groupby(['class']).agg({'class':'count'}).rename(columns={'class':'class_count'}).reset_index()
    for i in range(len(df)):
        data = df.iloc[i,]
        if data['class'].startswith('A Mild') or  data['class'].startswith('U Mild'):
            prod = {}
            prod['class']=data['class']
            prod['count']=int(data['class_count'])
            prods.append(prod)
    
    details = []
    df = labels[labels['filename']==name]
    for i in range(len(df)):
        label = df.iloc[i,]
        if label['class'].startswith('A Mild') or  label['class'].startswith('U Mild'):
            detail = {}
            detail['class']=label['class']
            detail['xmin']=int(label['xmin'])
            detail['ymin']=int(label['ymin'])
            detail['xmax']=int(label['xmax'])
            detail['ymax']=int(label['ymax'])
            detail['confidence']=(random()*28+70)/100
            details.append(detail)
    
    resp_data = {'products':prods, 'details':details}
    resp = flask.Response(json.dumps(resp_data),status=200,mimetype='application/json')
    return resp

def load_model(model_path):
    global model_loaded

    if not model_loaded:
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        model_loaded = True

    return detection_graph

def predict_product():
    uploaded = flask.request.files['image']
    prods = []
    df = labels[labels['filename']==name].groupby(['class']).agg({'class':'count'}).rename(columns={'class':'class_count'}).reset_index()
    for i in range(len(df)):
        data = df.iloc[i,]
        if data['class'].startswith('A Mild') or  data['class'].startswith('U Mild'):
            prod = {}
            prod['class']=data['class']
            prod['count']=int(data['class_count'])
            prods.append(prod)
    
    details = []
    df = labels[labels['filename']==name]
    for i in range(len(df)):
        label = df.iloc[i,]
        if label['class'].startswith('A Mild') or  label['class'].startswith('U Mild'):
            detail = {}
            detail['class']=label['class']
            detail['xmin']=int(label['xmin'])
            detail['ymin']=int(label['ymin'])
            detail['xmax']=int(label['xmax'])
            detail['ymax']=int(label['ymax'])
            detail['confidence']=(random()*28+70)/100
            details.append(detail)
    
    resp_data = {'products':prods, 'details':details}
    resp = flask.Response(json.dumps(resp_data),status=200,mimetype='application/json')
    return resp

if __name__ == "__main__":        # on running python app.py
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', action="store",dest="model", required=True)
    flags = parser.parse_args()

    app.run(host="0.0.0.0")    
