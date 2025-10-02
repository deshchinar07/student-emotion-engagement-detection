import tensorflow as tf
from keras.utils import img_to_array
import os
import numpy as np
import cv2

img_height, img_width = 48, 48
batch_size = 32
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
image = cv2.imread(r"Testing Images/happy2.jpg")
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
for (x, y, w, h) in faces:
    cv2.rectangle(img= image, pt1=(x, y), pt2=(x+w, y + h), color=(0, 255, 255), thickness=2)
    roi_gray = img_gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    if np.sum([roi_gray]) != 0:
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims (roi, axis=0)