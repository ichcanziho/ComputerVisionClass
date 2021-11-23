import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("image/1mapa.png")
cv.imshow("mapa", img)


def get_classes(image):
    rows, cols , _ = image.shape
    classes = {}
    current = 1
    pixis = []
    for y in range(rows):
        for x in range(cols):
            color = image[y,x]
            color = list(color)
            pixis.append(color)
    cleans = []
    for p in pixis:
        if p not in cleans:
            cleans.append(p)
    print(cleans)


get_classes(img)

cv.waitKey(0)

