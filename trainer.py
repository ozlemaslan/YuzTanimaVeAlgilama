import os
import cv2
import numpy as np
from PIL import Image

recognizer=cv2.face.LBPHFaceRecognizer_create ()
path="dataSet"

def getImageWithID(path):
    imagePath=[os.path.join(path,f ) for f in os.listdir(path)]
    faces=[]
    Ids=[]

    for imgpath in imagePath:
        faceImg=Image.open(imgpath).convert("L")
        faceNp=np.array(faceImg,"uint8")
        ID =int(os.path.split(imgpath)[-1].split (".")[1])
        faces.append(faceNp)
        Ids.append(ID)
        cv2.imshow("traning",faceNp)
        cv2.waitKey(ID)
    return np.array(Ids),faces

Ids , faces =getImageWithID(path)
recognizer.train(faces,Ids)
recognizer.save("recognizer/traningData.yml")
cv2.destroyAllWindows()

