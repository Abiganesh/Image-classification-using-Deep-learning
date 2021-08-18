import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
filename='b.jfif'

from tensorflow.keras.preprocessing import image
img=image.load_img(filename,target_size=(224,224))
plt.imshow(img)
#deep learning trained arichitecture
model = tf.keras.applications.MobileNetV2()
arrayimg=image.img_to_array(img)
final=np.expand_dims(arrayimg,axis=0) 
final=tf.keras.applications.mobilenet.preprocess_input(final)
predic=model.predict(final)
from tensorflow.keras.applications import imagenet_utils
result=imagenet_utils.decode_predictions(predic)
print(result[0][0][1])