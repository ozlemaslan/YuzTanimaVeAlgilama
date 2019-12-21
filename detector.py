import cv2,os
import sqlite3
import numpy as numpy
import pickle
from PIL import Image


kamera = cv2.VideoCapture(0)
yuz_tanima = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

rec=cv2.face.LBPHFaceRecognizer_create ()
rec.read("trainner/trainner.yml")
path="dataSet"


def getProfile(id):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT*FROM Yuz WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

##profiles={}
while True:

    ret, frame = kamera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    yuzler =yuz_tanima.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in yuzler:
        
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)

        profile=getProfile(id)
        if(profile!=None):
            cv2.putText(frame,str(profile[0]),(x,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.putText(frame, str(profile[1]), (x, y + h+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            cv2.putText(frame, str(profile[2]), (x, y + h+80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            cv2.putText(frame, str(profile[3]), (x, y + h+110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

    cv2.imshow('orjinal', frame)
##    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break
kamera.release()
cv2.destroyAllWindows()
