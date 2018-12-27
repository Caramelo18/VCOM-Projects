import os
from yolo import YOLO
from PIL import Image

def main():
    yolo = YOLO()
    
    image = Image.open('../data/train/camara/camara-0005.jpg')

    r_image = yolo.detect_image(image)
    r_image.show()

    yolo.close_session()

if __name__ == '__main__':
    main()
