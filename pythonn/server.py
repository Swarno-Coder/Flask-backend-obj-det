import cv2
from PIL import Image
import numpy as np
from flask import Flask, request, Response

#####################
app = Flask(__name__)
#####################

classNames= []
classFile = '..\\coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
configPath = '..\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = '..\\frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

@app.route('/req', methods=['POST'])
def send_img():
    
    img = request.files['image']
    img = Image.open(img.stream)
    thres = 0.55 # Threshold to detect object
    
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    print(classIds,bbox)
    
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            
    return request.post()
    
if __name__ == "__main__":
    app.run(debug=True)

