import sys
import cv2 as cv
import numpy as np


def show_image(path):
    img = cv.imread(path, cv.IMREAD_COLOR)
    cv.imshow('image', img)
    cv.waitKey(0)

def main(argv):
    if len(argv) is not 1:
        print("usage")
        sys.exit(1)

    file = argv[0]
    path = "samples/" + file
    show_image(path)


if __name__ == '__main__':
    main(sys.argv[1:])
