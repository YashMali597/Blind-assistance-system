# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 13:41:16 2022

@author: Dell
"""
from gtts import gTTS  
from playsound import playsound  
import pyttsx3  
import cv2
#img = cv2.imread('Lena.png')
engine = pyttsx3.init()  

cap = cv2.VideoCapture(0)

cap.set(3,640)
cap.set(4,480)
classNames= []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read() .rstrip('\n').split('\n')
print(classNames)

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)


while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold = 0.5)
    print(classIds,bbox)


    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(),bbox):
        cv2.rectangle(img, box,color=(255,0,0),thickness = 3)
        cv2.putText(img,classNames[classId-1],(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
    if (confidence*100)>70:
        print(classNames[classId-1])
        text = "A " + classNames[classId-1] + " is in front of you"
        print(text)
        engine.say(text)  
        engine.runAndWait()  

    cv2.imshow('Output',img)
    cv2.waitKey(1)
