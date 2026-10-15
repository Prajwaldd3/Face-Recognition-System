import cv2

camera=cv2.VideoCapture(0)
# cameara is object of the calss VideoCatpure

flag,frame=camera.read()

if flag:
    cv2.imshow("result",frame)
    cv2.waitKey()


camera.release()