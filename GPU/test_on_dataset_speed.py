import cv2
import os
import time 
import numpy as np
#import csv

from Detekcja.models2 import Haar, FaceRecoLib #, Insightface, YoloV5

#detector = Insightface()
#detector = Haar()
detector = FaceRecoLib()
#detector = YoloV5()

#dataset_path = '/home/msypniewski@sap-flex.com/Documents/DATASETS/MaskedFace-Net/IMFD/'
#dataset_path = '/home/msypniewski@sap-flex.com/Documents/DATASETS/MaFa/val/'
#dataset_path = '/home/msypniewski@sap-flex.com/Documents/DATASETS/WiderFace/WIDER_val/'
dataset_path = '/home/piotr/Documents/MS/Face-Detection-Magisterka-main/datasets/MAFA/'
dataset_images_path = dataset_path + 'images'

def main():
    curr_img_num = 1
    times = 0.0
    num_of_images = len(os.listdir(dataset_images_path))

    # For every image in dataset
    for filename in os.listdir(dataset_images_path):
        img = cv2.imread(os.path.join(dataset_images_path, filename))
        t1 = time.time()
        preds = detector.detect_face(img)
        t2 = time.time()
        times = times + (t2-t1)
                
        print(str(curr_img_num) + "/" + str(num_of_images) + "     " + str(t2-t1))
        curr_img_num += 1

    m_times = times/num_of_images
    print("Mean time: ", m_times)

if __name__ == "__main__":
    main()
