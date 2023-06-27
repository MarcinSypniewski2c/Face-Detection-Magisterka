import cv2
import os
import time 
import numpy as np
import csv

from Detekcja.models import Haar, Insightface, FaceRecoLib, YoloV5

#detector = Insightface()
detector = Haar()
#detector = FaceRecoLib()
#detector = YoloV5()

dataset_path = "/path/to/dataset/"
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
                
        print(str(curr_img_num) + "/" + str(num_of_images))
        curr_img_num += 1

    m_times = times/num_of_images
    print("Mean time: ", m_times)

if __name__ == "__main__":
    main()