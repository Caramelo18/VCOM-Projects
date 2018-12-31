import os
from yolo import YOLO
from PIL import Image

classes = {'arrabida': 0, 'camara': 1, 'clerigos': 2, 'musica': 3, 'serralves': 4}

def main():
    print(os.environ.get('DEV_ENV'))
    open('predicted.txt', 'w').close()

    yolo = YOLO()
    
    for dirname, dirnames, filenames in os.walk('../data/validation'):
        for class_name in dirnames:
            class_path = os.path.join(dirname, class_name)
            class_path, class_dirs, class_files = next(os.walk(class_path))

            for image_name in class_files:
                image_path = os.path.join(class_path, image_name)
                image = Image.open(image_path)
    
                boxes = yolo.detect_image_return_boxes(image)

                line = image_path
                for box in boxes:
                    (left, top, right, bottom, predicted_class, score) = box
                    box_str = '{},{},{},{},{},{}'.format(left, top, right, bottom, classes[predicted_class], score)
                    line += ' ' + box_str
                line += '\n'
                with open('predicted.txt', 'a', encoding='utf-8') as file:
                    file.write(line)

    yolo.close_session()

if __name__ == '__main__':
    main()
