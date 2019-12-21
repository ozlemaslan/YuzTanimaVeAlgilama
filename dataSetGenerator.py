import cv2
import sqlite3
import numpy as numpy

kamera = cv2.VideoCapture(0)
yuz_tanima = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def insertOrUpdate(Id,Name):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT*FROM Yuz WHERE ID="+str(Id)
    cursor=conn.execute(cmd)
    isRecordExit=0
    for row in cursor:
        isRecordExit=1
    if(isRecordExit==1):
        cmd="UPDATE Yuz SET Ad="+str(Name)+"WHERE ID="+str(Id)
    else:
        cmd="INSERT INTO Yuz(ID,Ad) Values("+str(Id)+","+str(Name)+")"
    conn.execute(cmd)
    conn.commit()
    conn.close()


id=input("id girin:")
name=input("ad girin:")
insertOrUpdate(id,name)
sampleNum=0


while True:
    ret, frame = kamera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    yuzler =yuz_tanima.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in yuzler:

        sampleNum =sampleNum+1
        cv2.imwrite("dataSet/User"+str(id)+"."+str(sampleNum)+".jpg", gray[y:y+h,x:x+w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('face', frame)
    cv2.waitKey(100)
    if sampleNum>=10:
        kamera.release()
        cv2.destroyAllWindows()
        break
