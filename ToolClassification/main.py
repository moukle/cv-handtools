import load_dataset as data

from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# DESIRED SIZE OF REGION
X_SIZE = 256
Y_SIZE = 512

X_SIZE_TARGET = 64
Y_SIZE_TARGET = 64

model = load_model('tool_model.h5')
image = cv2.imread('dataset/test_image.jpg')


y_dim, x_dim, _ = image.shape
STEPSIZE = 48
for x in range(0, x_dim-X_SIZE, STEPSIZE):
    for y in range(0, y_dim-Y_SIZE, STEPSIZE):
        region = image[y:y+Y_SIZE, x:x+Y_SIZE]
        region = cv2.resize(region, (X_SIZE_TARGET, Y_SIZE_TARGET))
        region = np.expand_dims(region, axis=0)

        # add alpha for plot
        y_pred = model.predict(region)

        if np.max(y_pred) > 0.996:
            print(y_pred)
            y_pred = np.append(y_pred, [1])
            plt.scatter(x, y, s=100, color=y_pred)
            plt.gca().add_patch(Rectangle((x,y),X_SIZE,Y_SIZE,linewidth=2,edgecolor=y_pred,facecolor='none'))

    print("Rasterizing: {}%".format(x/(x_dim-X_SIZE)*100))
plt.imshow(image)
plt.show()