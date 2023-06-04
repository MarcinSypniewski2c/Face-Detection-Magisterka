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

    def get_names(self):
        names = []
        for path in glob.glob(self.data_images_path, recursive=True):
            #names
            names.append((path.split("/")[-1]).split(".")[0])
        return names

    def get_data(self):
        for path in glob.glob(self.data_images_path, recursive=True):

            #images
            #img = cv2.imread(path)
            data = cv2.imread(path)

            yield data
    
class MFN:
    def __init__(self):
        self.dataset_path = "/home/msypniewski@sap-flex.com/Documents/DATASETS/MaskedFace-Net/"
        self.data_images_path = self.dataset_path + "CMFD/images/*.jpg"

    def get_names(self):
        names = []
        for path in glob.glob(self.data_images_path, recursive=True):
            #names
            names.append((path.split("/")[-1]).split(".")[0])
        return names

    def get_data(self):
        for path in glob.glob(self.data_images_path, recursive=True):

            #images
            #img = cv2.imread(path)
            data = cv2.imread(path)

            yield data