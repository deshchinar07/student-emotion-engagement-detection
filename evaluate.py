import tensorflow as tf
from keras.utils import img_to_array
import os
import numpy as np
import cv2

# Set your directories
test_dir = r"FER_2013/test"

# Image parameters
img_height, img_width = 48, 48
batch_size = 32
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Define your custom preprocessing function
def custom_preprocess(image):
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
            return roi

# Function to load image with label from path
def load_and_preprocess_image(path, label):
    path = path.numpy().decode('utf-8')
    image = cv2.imread(path)
    if image is None:
        print(f"Failed to load: {path}")
        # Option 1: Skip this sample by returning a valid dummy image, like zeros
        image = np.zeros((48, 48), dtype=np.float32)
    image = custom_preprocess(image)
    return image, label

def tf_wrapper(path, label):
    image, label = tf.py_function(
        load_and_preprocess_image, [path, label], [tf.float32, tf.int32]
    )
    # Set the shapes for TensorFlow (if needed)
    image.set_shape([img_height, img_width, 1])  # For FER2013, grayscale
    label.set_shape([])
    return image, label

class_names = sorted(os.listdir(test_dir))
file_paths = []
labels = []

for label_index, class_name in enumerate(class_names):
    class_dir = os.path.join(test_dir, class_name)
    for file in os.listdir(class_dir):
        file_paths.append(os.path.join(class_dir, file))
        labels.append(label_index)

path_ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

test_ds = path_ds.map(tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Load model and compile
model = tf.keras.models.load_model(r"/Users/deshc/Desktop/Chinar/Research/Engagement Level Detection/Model/model_78.h5")
model.load_weights(r"/Users/deshc/Desktop/Chinar/Research/Engagement Level Detection/Model/model_weights_78.h5")

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Evaluate model
loss, accuracy = model.evaluate(test_ds)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')