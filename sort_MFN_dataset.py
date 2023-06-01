import glob
import os
import shutil

main_path = "/home/msypniewski@sap-flex.com/Documents/DATASETS/MaskedFace-Net/CMFD"

annotations_yolo_path = main_path + "/labels_yolo/"
images_yolo_path = main_path + "/images/"
images_to_extract_path = main_path + "/[0-9][0-9][0-9][0-9][0-9]/*.jpg"

images_to_extract = sorted(glob.glob(images_to_extract_path))

for img_extract in images_to_extract:
    file_name = img_extract.split("/")[-1].split(".")[-2]
    img_destination = images_yolo_path + file_name + ".jpg"
    label_extract = annotations_yolo_path + file_name + ".txt"
    if os.path.isfile(label_extract):
        shutil.copy(img_extract, img_destination)

annotations_yolo_path_2 = annotations_yolo_path + "*.txt"
labels_to_check = sorted(glob.glob(annotations_yolo_path_2))

for lab_extract in labels_to_check:
    file_name = lab_extract.split("/")[-1].split(".")[-2]
    img_to_check = images_yolo_path + file_name + ".jpg"
    if not os.path.isfile(img_to_check):
        os.remove(lab_extract)

