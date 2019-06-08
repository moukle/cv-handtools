from keras.models import load_model
import cv2 as cv
import numpy as np


def get_rectangles_from_image(image):
    edges = get_edges_from_image(image)
    contours,_ = cv.findContours(edges, 1, 2)
    rects = get_merged_rectangles_from_contours(contours)
    return rects


def get_edges_from_image(image):
    img = image

    kernel = np.ones((3,3),np.uint8)

    noNoise = img
    noNoise = cv.morphologyEx(noNoise, cv.MORPH_OPEN, kernel)
    noNoise = cv.morphologyEx(noNoise, cv.MORPH_CLOSE, kernel)
    noNoise = cv.GaussianBlur(noNoise, (3,3), 0)

    edges = cv.Canny(noNoise, 0, 255)
    return edges


def get_merged_rectangles_from_contours(contours):
    rects = []
    for cnt in contours:
        x,y,w,h = cv.boundingRect(cnt)
        rects.append([x,y,w,h])
        rects.append([x,y,w,h])

    rects, _ = cv.groupRectangles(rects, 1, 0.2)

    merged = []
    for a in rects:
        bigRect = a
        for b in rects:
            if intersection(bigRect, b):
                bigRect = union(bigRect, b)
        merged.append(bigRect)
    
    return set(merged)


def union(a,b):
  x = min(a[0], b[0])
  y = min(a[1], b[1])
  w = max(a[0]+a[2], b[0]+b[2]) - x
  h = max(a[1]+a[3], b[1]+b[3]) - y
  return (x, y, w, h)


def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return False # or (0,0,0,0) ?
  return (x, y, w, h)

def predict_labels(model, image, rectangles):
    TOOLLABELS = { 0: "background", 1: "hammer", 2: "plane", 3: "wrench" }
    TARGETSIZE = (128, 128)

    labels = []
    for rect in rectangles:
        # GET REGION
        x,y,w,h = rect
        crop = image[y:y+h, x:x+w]
        resize = cv.resize(crop, TARGETSIZE)
        resize = np.concatenate([resize[np.newaxis]]).astype('float32')

        # PREDICT
        y_pred = model.predict(resize)
        print(y_pred)
        labels.append(TOOLLABELS[np.argmax(y_pred)])
    return labels
    

def write_labels_on_image(image, labels, rectangles):
    img = image
    FONT = cv.FONT_HERSHEY_SIMPLEX
    for i, rect in enumerate(rectangles):
        x,y,_,_ = rect
        label = labels[i]
        cv.putText(img, label, (x,y), FONT, 1, (255,255,255), 2)
    return img


def draw_rectangles(image, rectangles):
    img = image
    for rect in rectangles:
        x,y,w,h = rect
        cv.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 2)
    return img


def save_image(img, name):
    cv.imwrite(name+'.jpg', img, [int(cv.IMWRITE_JPEG_QUALITY), 90])


if __name__ == "__main__":
    orig_img = cv.imread('dataset/test_image5.jpg')
    rectangles = get_rectangles_from_image(orig_img)

    model = load_model('tool_model.h5')
    labels = predict_labels(model, orig_img, rectangles)

    labeled_image = write_labels_on_image(orig_img, labels, rectangles)
    labeled_image = draw_rectangles(labeled_image, rectangles)
    save_image(labeled_image, 'labeledImage')
    cv.imshow('labeled_image', labeled_image)
    cv.waitKey(0)