# DEPRECATED


import load_dataset as data

from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# LOAD DATA
# model = load_model('tool_model_80acc.h5')
model = load_model('tool_model.h5')
image = cv2.imread('dataset/test_image5.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

toolPositions = {}
toolLabels = {
    0: "background",
    1: "hammer",
    # 1: "plane",
    2: "wrench"
}

# SELECTED REGION SIZE
X_SIZE = 256
Y_SIZE = 256*2

# SIZE REQUIRED BY THE NETWORK
X_SIZE_TARGET = 224
Y_SIZE_TARGET = 224
X_SIZE_TARGET = 64 *2
Y_SIZE_TARGET = 64 *2

# ACTUAL SHAPE OF THE IMAGE
y_dim, x_dim, _ = image.shape


STEPSIZE = 128
for y in range(0, y_dim-Y_SIZE, STEPSIZE):
    for x in range(0, x_dim-X_SIZE, STEPSIZE):
        region = image[y:y+Y_SIZE, x:x+Y_SIZE]
        region = cv2.resize(region, (X_SIZE_TARGET, Y_SIZE_TARGET))
        region = np.expand_dims(region, axis=0)

        # add alpha for plot [RGBA]
        y_pred = model.predict(region)
        print(y_pred)

        if np.argmax(y_pred) != 0 and np.max(y_pred) > 0.9:
            rect = (y,y+Y_SIZE, x,x+X_SIZE)
            toolPositions[rect] = toolLabels[np.argmax(y_pred)]
            y_pred = np.append(y_pred, [1])
            plt.scatter(x, y, s=100, color=y_pred)
            plt.gca().add_patch(Rectangle((x,y),X_SIZE,Y_SIZE,linewidth=2,edgecolor=y_pred,facecolor='none'))

    print("Rasterizing: {}%".format(y/(y_dim-Y_SIZE)*100))
# print(toolPositions)
plt.imshow(image)
plt.show()