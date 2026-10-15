
import cv2
from os import listdir
import numpy as np

recog=cv2.face.LBPHFaceRecognizer_create()

root_dir="./dataset"

features=[]
labels=[]

i=0

for subfolder in listdir(root_dir):
    
    # print(subfolder)
    
    folders=f"{root_dir}/{subfolder}"
    
    for file in listdir(folders):
        
        file_path=f"{folders}/{file}"
        
        # print(file_path)
        
        image=cv2.imread(file_path,0)
        
        # cv2.imshow("demo",image)
        # print(image)
        # cv2.waitKey()
        
        features.append(image)
        labels.append(i)
    
    i=i+1
    


print(f"Features are : {features}")

print(f"Labels are  : {labels}")


recog.train( features,np.array( labels))

recog.save("facemodel.yml")

