Create venv:

python3.6 -m venv venv

Activate venv:

source env/bin/activate

Install packages:

pip install -r requirements.txt


# Classification
## Train Model
Run project.py -tc with the training images on /data/train and validation images on /data/validation. Each of these folders must contain a folder with the images for each landmark

## Predict Image


## Statistics
Run project.py -s and it will print the statistics of the images on /data/validation (with the ground truth annotations in /data_annotations/validation)

# Detection
## Train Model
On /detection folder:
Convert YOLO model to a Keras model and place it in /model_data/ folder.
Place the training and validation annotations files with names training.txt and validation.txt respectively.
Place the file with the classes to be detected on vcom_classes.txt
The images must be placed as mentioned on the classification step
Run train.py

## Detect Image
On /detection folder:
Run detect_image.py with an image name from /data/validation as argument. (ex: python detect_image.py arrabida-0034.jpg)

## Generate predicted.txt file
On /detection folder:
Run detect.py

## Calculate mAP
Instructions on https://github.com/Cartucho/mAP, with the folders mAP and mAP-tiny already including the .txt files for both the YOLO and Tiny TOLO networks.