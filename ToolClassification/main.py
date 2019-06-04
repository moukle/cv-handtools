import load_dataset as data

from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# LOAD DATA
model = load_model('tool_model_80acc.h5')
image = cv2.imread('dataset/test_image.jpg')

toolPositions = {}
toolLabels = {
    0: "hammer",
    1: "plane",
    2: "wrench"
}

# SELECTED REGION SIZE
X_SIZE = 256
Y_SIZE = 512

# SIZE REQUIRED BY THE NETWORK
X_SIZE_TARGET = 64 *2
Y_SIZE_TARGET = 64 *2

# ACTUAL SHAPE OF THE IMAGE
y_dim, x_dim, _ = image.shape


STEPSIZE = 128
for x in range(0, x_dim-X_SIZE, STEPSIZE):
    for y in range(0, y_dim-Y_SIZE, STEPSIZE):
        region = image[y:y+Y_SIZE, x:x+Y_SIZE]
        region = cv2.resize(region, (X_SIZE_TARGET, Y_SIZE_TARGET))
        region = np.expand_dims(region, axis=0)

        # add alpha for plot [RGBA]
        y_pred = model.predict(region)

        if np.max(y_pred) > 0.999:
            rect = (y,y+Y_SIZE, x,x+X_SIZE)
            toolPositions[rect] = toolLabels[np.argmax(y_pred)]
            y_pred = np.append(y_pred, [1])
            plt.scatter(x, y, s=100, color=y_pred)
            plt.gca().add_patch(Rectangle((x,y),X_SIZE,Y_SIZE,linewidth=2,edgecolor=y_pred,facecolor='none'))

    print("Rasterizing: {}%".format(x/(x_dim-X_SIZE)*100))
print(toolPositions)
plt.imshow(image)
plt.show()