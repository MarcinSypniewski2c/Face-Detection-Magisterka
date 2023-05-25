import os
import cv2
import glob

def pad_string(string, length, char='0'):
    if len(string) >= length:
        return string
    else:
        padding = (length - len(string)) * char
        padded_string = padding + string
        return padded_string

class LFW:
    def __init__(self):
        self.dataset_path = "/home/msypniewski@sap-flex.com/Documents/DATASETS/LFW_people/"
        self.data_images_path = self.dataset_path + "lfw_funneled/*/*.jpg"

    def get_data(self):
        data = []
        names = []
        for path in glob.glob(self.data_images_path, recursive=True):
            #names
            names.append((path.split("/")[-1]).split(".")[0])

            #images
            img = cv2.imread(path)
            data.append(img)

        return data, names