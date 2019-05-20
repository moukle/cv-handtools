import os
import cv2 as cv
import linesDetection as ld

IMGS_PATH = "../imgsRef/"
images = []

for filename in os.listdir(IMGS_PATH):
    img = cv.imread(IMGS_PATH + filename)
    images.append(img)

img = images[5]
lines = ld.lines_from_img(img, True)
longest_line_length, longest_line_index = ld.longest_line(lines)

# Get real size
px_mm_ratio = ld.px_mm_ratio(img, ref_length_in_mm=50)
print("Pixel to mm ratio: 1:{}".format(px_mm_ratio))
real_length = longest_line_length * px_mm_ratio
print("Longest edge is {}mm /  {}px".format(real_length, longest_line_length))


for line in lines:
    ld.draw_line(img, line)
ld.draw_line(img, lines[longest_line_index], (255,0,0))
cv.imwrite('lines.jpg', img, [int(cv.IMWRITE_JPEG_QUALITY), 90])