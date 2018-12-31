import os, random, shutil
import xml.etree.ElementTree as ET

#classes = {'arrabida': 21, 'camara': 22, 'clerigos': 23, 'musica': 24, 'serralves': 25}
classes = {'arrabida': 0, 'camara': 1, 'clerigos': 2, 'musica': 3, 'serralves': 4}
f =  f = open("annotations.txt", "w")
basepath = '/content/drive/My Drive/VCOM/images/'

def main():
    for dirname, dirnames, filenames in os.walk('./data_annotations/train'):
        # print path to all filenames.
        for filename in filenames:
            path = os.path.join(dirname, filename)
            print(path, filename)
            read_file(path)


def read_file(path):
    tree = ET.parse(path)
    root = tree.getroot()

    img_name = root.find('filename').text

    obj = root.find('object')
    obj_name = obj.find('name').text
    obj_id = classes[obj_name]

    img_path = basepath + obj_name + "/" + img_name

    bnd_box = obj.find('bndbox')
    x_min = int(float(bnd_box.find('xmin').text))
    y_min = int(float(bnd_box.find('ymin').text))
    x_max = int(float(bnd_box.find('xmax').text))
    y_max = int(float(bnd_box.find('ymax').text))

    str = '{} {},{},{},{},{}\n'.format(img_path, x_min, y_min, x_max, y_max, obj_id)
    f.write(str)

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


def split_annotations():
    for image_type in ['train', 'validation']:
        for dirname, dirnames, filenames in os.walk('./data/{}'.format(image_type)):
            for class_name in dirnames:
                class_path = os.path.join(dirname, class_name)
                class_path, class_dirs, class_files = next(os.walk(class_path))

                class_dir = './data_annotations/{}/{}'.format(image_type, class_name)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)

                for image in class_files:
                    image_name = image.split('.')[0]
                    src_annotation_path = './annotations/{}/{}.xml'.format(class_name, image_name)
                    dest_annotation_path = './data_annotations/{}/{}/{}.xml '.format(image_type, class_name, image_name)
                    shutil.copyfile(src_annotation_path, dest_annotation_path)


if __name__=='__main__':
    main()
