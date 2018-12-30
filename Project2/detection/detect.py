import os
from yolo import YOLO
from PIL import Image

def main():
    print(os.environ.get('DEV_ENV'))
    yolo = YOLO()

    #image = Image.open('../data/images/arrabida/arrabida-0043.jpg')
    #image = Image.open('../data/images/camara/camara-0043.jpg')
    #image = Image.open('../data/images/clerigos/clerigos-0043.jpg')
    image = Image.open('../data/images/musica/musica-0145.jpg')
    #image = Image.open('../data/images/serralves/serralves-0107.jpg')
    #image = Image.open('/home/fabio/Documentos/5o Ano/VCOM/ex2.png')

    r_image = yolo.detect_image(image)
    r_image.show()

    yolo.close_session()

if __name__ == '__main__':
    main()
