import os, random, shutil
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

def split_dataset():
    for dirname, dirnames, filenames in os.walk('./images'):
        for class_name in dirnames:
            class_path = os.path.join(dirname, class_name)
            class_path, class_dirs, class_files = next(os.walk(class_path))
            class_count = len(class_files)

            random.shuffle(class_files)
            split_point = int(0.8 * class_count)

            train_files = class_files[:split_point]
            val_files = class_files[split_point:]

            train_dir = './data/train/{}'.format(class_name)
            if not os.path.exists(train_dir):
                os.makedirs(train_dir)
                
            val_dir = './data/validation/{}'.format(class_name)
            if not os.path.exists(val_dir):
                os.makedirs(val_dir)
            
            for train_image in train_files:
                src_path = class_path + '/' + train_image
                dest_path = train_dir + '/' + train_image
                shutil.copyfile(src_path, dest_path)
            
            for val_image in val_files:
                src_path = class_path + '/' + val_image
                dest_path = val_dir + '/' + val_image
                shutil.copyfile(src_path, dest_path)


if __name__=='__main__':
    main()
