from keras.models import load_model
import cv2 as cv
import numpy as np

TARGETSIZE = (224,224)

if __name__ == "__main__":
    model = load_model('./models/tool_model.h5')

    TOOLLABELS = { 0: "background", 1: "hammer", 2: "plane", 3: "wrench" }

    orig_img = cv.imread('./data/testing/11.jpg')
    resize = cv.resize(orig_img, TARGETSIZE)
    resize = np.concatenate([resize[np.newaxis]]).astype('float32')

    pred = model.predict(resize)


    print(pred)