import os
from yolo import YOLO
from PIL import Image
import sys
import re

def main():
    yolo = YOLO()

    if len(sys.argv) < 2:
        print('Missing image name. Usage: \'python detect_image.py arrabida-0034.jpg\'')
    
    image_name = sys.argv[1]
    class_name = re.findall("[a-zA-Z]+", image_name)[0]
    image_path = '../data/validation/{}/{}'.format(class_name, image_name)

    image = Image.open(image_path)
    image = yolo.detect_image(image)
    image.show()
    image.save(image_name)

    yolo.close_session()

if __name__ == '__main__':
    main()