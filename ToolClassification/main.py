import load_dataset as data

from keras.models import load_model
import cv2
import numpy as np

model = load_model('tool_model_80acc.h5')
x = cv2.imread('dataset/split/test/hammer/0a98bc81e4dba8065f0cc9e6450cabdbc50f96a5.jpg')
x = np.expand_dims(x, axis=0)

y_pred = model.predict(x)
print(y_pred)
