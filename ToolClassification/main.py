import load_dataset as data

from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

# DESIRED SIZE OF REGION
X_SIZE = 64
Y_SIZE = 64

model = load_model('tool_model_80acc.h5')
image = cv2.imread('dataset/test_image.jpg')

y_dim, x_dim, _ = image.shape
STEPSIZE = 50
for x in range(0, x_dim-X_SIZE, STEPSIZE):
    for y in range(0, y_dim-Y_SIZE, STEPSIZE):
        region = image[y:y+64, x:x+64]
        region = np.expand_dims(region, axis=0)
        y_pred = model.predict(region)
        print(y_pred)
        plt.scatter(x, y, color=y_pred)

    print("Rasterizing: {}%".format(x/(x_dim-X_SIZE)*100))
plt.imshow(image)
plt.show()