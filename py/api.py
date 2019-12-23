import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image,ImageDraw
from google.protobuf import text_format
from flask import Flask
import pandas as pd

MODEL_NAME = 'inference_graph'
PATH_TO_FROZEN_GRAPH = ""   #'data/graph/frozen_inference_graph.pb'
PATH_TO_LABELS = 'data/label.pbtxt'
PATH_TO_LABELS_CSV = ""     #'data/label_map.csv'

detection_graph = tf.Graph()
def init(flozengraphpath='data/graph/frozen_inference_graph.pb',labelpath='data/label_map.csv'):
    global PATH_TO_FROZEN_GRAPH
    global PATH_TO_LABELS_CSV
    PATH_TO_LABELS_CSV = labelpath
    PATH_TO_FROZEN_GRAPH = flozengraphpath
    print(PATH_TO_LABELS_CSV)
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

basic_color = [(255,0,0),(255,255,0),(0,234,255),(170,0,255),(255,127,0),(191,255,0),(0,149,255),(255,0,170),(255,212,0),(106,255,0),(0,64,255),(237,185,185),(185,215,237),(231,233,185),(220,185,237),(185,237,224),(143,35,35),(35,98,143),(143,106,35),(107,35,143),(79,143,35),(0,0,0),(115,115,115),(204,204,204)]
labels = dict()
def sortVCluster(val):
    return val["vcluster"].minX
class VCluster:
    def __init__(self,minX,maxX):
        self.maxX=maxX
        self.minX=minX
        self.member = []
    '''
        check if another VCluster is within this VCluster
    '''
    def Contains(self,anotherVCluster):
        if (anotherVCluster.minX>=self.minX and anotherVCluster.minX<=self.maxX) or (anotherVCluster.maxX>=self.minX and anotherVCluster.maxX<=self.maxX):
            return True
        return False
    '''
        expand 
    '''
    def Expand(self,anotherVCluster):
        self.minX = min(anotherVCluster.minX,self.minX)
        self.maxX = max(anotherVCluster.maxX,self.maxX)
    def AddMember(self,idx):
        self.member.append(idx)
def predict(image,threshold=0.75,tempdir="./",filenameOutput="hasil.jpeg"):
    global detection_graph
    global sess
    global labels
    #global tensor_dict
    if len(labels)==0:
        data = pd.read_csv(PATH_TO_LABELS_CSV)
        for index, row in data.iterrows():
            #print(d)
            labels[row["tagid"]] = row["label"]
    print(labels)
    image_np = np.array(image)
    with detection_graph.as_default():
        with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                  'num_detections', 'detection_boxes', 'detection_scores',
                  'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = detection_graph.get_tensor_by_name(tensor_name)


                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                
                # Actual detection.
                if 'detection_masks' in tensor_dict:
                    print("detection is here already")
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, 
                        image_np.shape[0], 
                        image_np.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                        feed_dict={image_tensor: np.expand_dims(
                                            image_np, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = np.squeeze(output_dict['detection_scores'][0])
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]

                scores = []
                for i in output_dict['detection_scores']:
                    #if i > 0.5:
                        scores.append(i)
                    #elif i>0.01:
                    #   scores.append(0.6)
                    #else:
                    #    scores.append(0)
                #print(output_dict)
                width, height = image.size
                imDraw = ImageDraw.Draw(image)
                itemCount = len(output_dict['detection_boxes'])
                result = {}
                result["count"] = dict()
                result["details"] = []
                
                tempResultDetail = []
                for i in range(itemCount):
                    # if i==3:
                    #     break
                    if output_dict['detection_scores'][i]<=threshold:
                        continue
                    curLabel = labels[output_dict['detection_classes'][i]]
                    if curLabel in result["count"]:
                        result["count"][curLabel]+=1
                    else:
                        result["count"][curLabel] = 1
                    coord1 = output_dict['detection_boxes'][i]
                    print(coord1)
                    coord1[0]=(coord1[0])*height
                    coord1[1]=(coord1[1])*width
                    coord1[2]=(coord1[2])*height
                    coord1[3]=(coord1[3])*width
                    coord2 = [coord1[1],coord1[0],coord1[3],coord1[2]]
                    print(coord2)
                    color = basic_color[output_dict['detection_classes'][i]-1]
                    coord3 = [str(coord2[0]),str(coord2[1]),str(coord2[2]),str(coord2[3])]
                    tempResultDetail.append({"label":curLabel,"vcluster":VCluster(coord2[1],coord2[3]),"box":coord3,"confidence":str(output_dict['detection_scores'][i])})
                    imDraw.rectangle(coord2,outline=color)
                if len(tempResultDetail)==0:
                    image.save(filenameOutput,"JPEG")
                    return result
                tempResultDetail.sort(key=sortVCluster)
                rowClusters = []
                curVCluster = VCluster(tempResultDetail[0]["vcluster"].minX,tempResultDetail[0]["vcluster"].maxX)
                curRow = 1
                for i in range(len(tempResultDetail)):
                    print(curVCluster.minX,curVCluster.maxX,tempResultDetail[i]["vcluster"].minX,tempResultDetail[i]["vcluster"].maxX)
                    if curVCluster.Contains(tempResultDetail[i]["vcluster"]):
                         curVCluster.Expand(tempResultDetail[i]["vcluster"])
                         del tempResultDetail[i]["vcluster"]
                         print("curRowExpand",curRow)
                    else:
                        rowClusters.append(curVCluster)
                        curRow+=1
                        print("curRowNew",curRow)
                        curVCluster = VCluster(tempResultDetail[i]["vcluster"].minX,tempResultDetail[i]["vcluster"].maxX)
                        del tempResultDetail[i]["vcluster"]
                    
                    tempResultDetail[i]["Row"] = curRow
                result["details"] = tempResultDetail #.append
                image.save(filenameOutput,"JPEG")
                print("=====")
                return result
                # Visualization of the results of a detection.
                #cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
#inputImg = Image.open("Backwall_172.jpeg")
#predict(inputImg)


