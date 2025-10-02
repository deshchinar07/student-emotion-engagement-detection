import numpy as np
import tensorflow as tf
from keras.utils import img_to_array

import cv2

#IMAGE TO EMOTION CODE
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

classifier = tf.keras.models.load_model(r"model/emotion_model.h5")

classifier.load_weights(r"model/emotion_model_weights.h5")

try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    print("Error loading cascade classifiers")

cap = cv2.imread(r"Testing Images/happy2.jpg")

img_gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
for (x, y, w, h) in faces:
    cv2.rectangle(img= cap, pt1=(x, y), pt2=(x+w, y + h), color=(0, 255, 255), thickness=2)
    roi_gray = img_gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    if np.sum([roi_gray]) != 0:
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims (roi, axis=0)
        prediction = classifier.predict(roi)[0]
        output = str(emotion_labels[int(np.argmax(prediction))])
        print(output)

    cv2.imshow('output', cap)
    cv2.waitKey(0)
