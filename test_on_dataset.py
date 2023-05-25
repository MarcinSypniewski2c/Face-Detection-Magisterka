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

dataname = "wider_val_haar"
dataset_path = '/home/msypniewski@sap-flex.com/Documents/DATASETS/WiderFace/WIDER_val/'
dataset_labels_path = dataset_path + 'labels_yolo'
dataset_images_path = dataset_path + 'images'

def intersection_and_union(rect1, rect2, retSum = False):
    # Find the coordinates of the overlapping region
    x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]))
    y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]))

    # Calculate the area of the overlapping region
    overlap_area = x_overlap * y_overlap

    # Areas of both rectangles
    area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])

    # Calculate union
    union_area = area1 + area2 - overlap_area

    if retSum:
        return overlap_area, union_area, area1, area2
    else:
        return overlap_area, union_area

def IoU(gt, pr):
    intersectionArea, unionArea = intersection_and_union(gt,pr)
    iou = intersectionArea/unionArea #This should be greater than 0.5 to consider it as a valid detection.
    return iou

def dice_coef(gt, pr):
    i, u, a1, a2 = intersection_and_union(gt, pr, True)
    dice = 2*(i)/(a1 + a2)
    return dice

def main():
    statistics = []
    confusions = []
    curr_img_num = 1
    num_of_images = len(os.listdir(dataset_images_path))

    # For every image in dataset
    for filename in os.listdir(dataset_images_path):
        tp = 0 # True Positive
        fp = 0 # False Positive
        fn = 0 # False Negative
        labelname = filename.split(".")[0] + ".txt"

        img = cv2.imread(os.path.join(dataset_images_path, filename))
        preds = detector.detect_face(img)
        ground_truths = []
        with open(os.path.join(dataset_labels_path, labelname), 'r') as file:
            # For every ground truth box
            for line in file:
                    # Yolo labels xywh to xyxy bounding box
                    vh = img.shape[0]
                    vw = img.shape[1]
                    lsp = line.split(" ")
                    x = float(lsp[1])
                    y = float(lsp[2])
                    w = float(lsp[3])
                    h = float(lsp[4])
                    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2] 
                    xmin_gt = int(xyxy[0] * vw)
                    ymin_gt = int(xyxy[1] * vh)
                    xmax_gt = int(xyxy[2] * vw)
                    ymax_gt = int(xyxy[3] * vh)
                    ground_truth = [xmin_gt, ymin_gt, xmax_gt, ymax_gt]
                    ground_truths.append(ground_truth)

            for pred in preds:
                iou = 0.0
                dco = 0.0
                for ground_truth in ground_truths:
                    curr_iou = IoU(ground_truth, pred[0])
                    if curr_iou > iou:
                        iou = curr_iou
                        dco = dice_coef(ground_truth, pred[0])

                statistics.append([iou, dco])

                if iou >= 0.5:
                    tp = tp + 1
                else:
                    fp = fp + 1

            fn = fn + (len(ground_truths) - tp)
            confusions.append([tp, fp, fn])
                
        print(str(curr_img_num) + "/" + str(num_of_images))
        curr_img_num += 1

    statistics_filename = dataname + "_statistics.csv"
    confusion_filename = dataname + "_confusion.csv"

    with open(statistics_filename, 'w', newline='') as file_s:
        writer = csv.writer(file_s)
        writer.writerows(statistics)
    with open(confusion_filename, 'w', newline='') as file_c:
        writer = csv.writer(file_c)
        writer.writerows(confusions)

if __name__ == "__main__":
    main()