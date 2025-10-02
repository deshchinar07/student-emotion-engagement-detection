import cv2
import os

def vid():
  vidcap = cv2.VideoCapture("C:/Users/deshc/Desktop/Chinar/HRI/00002.MTS")
  success,image = vidcap.read()
  count = 0
  path = "C:/Users/deshc/Desktop/Chinar/HRI/Lecture 2 Frames"
  while success:
    cv2.imwrite(os.path.join(path, "%d.jpg" % count), image)    
    
    success,image = vidcap.read()
    print('Read a new frame: ' , count, " ", success)
    count += 1 


vid()