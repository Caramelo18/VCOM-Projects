import os
import xml.etree.ElementTree as ET

classes = {'arrabida': 0, 'camara': 1, 'clerigos': 2, 'musica': 3, 'serralves': 4}


def main():
    for dirname, dirnames, filenames in os.walk('./data/annotations'):
        # print path to all filenames.
        for filename in filenames:
            path = os.path.join(dirname, filename)
            print(path, filename)
            read_file(path)


def read_file(path):
    tree = ET.parse(path)
    root = tree.getroot()

    imgpath = root.find('path').text

    obj = root.find('object')
    obj_name = obj.find('name').text
    obj_id = classes[obj_name]

    bnd_box = obj.find('bndbox')
    x_min = int(float(bnd_box.find('xmin').text))
    y_min = int(float(bnd_box.find('ymin').text))
    x_max = int(float(bnd_box.find('xmax').text))
    y_max = int(float(bnd_box.find('ymax').text))

    str = '{} {},{},{},{},{}'.format(imgpath, x_min, y_min, x_max, y_max, obj_id)

    print(str)

if __name__=='__main__':
    main()
