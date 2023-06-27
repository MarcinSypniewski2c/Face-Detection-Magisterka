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
        self.dataset_path = "/path/to/dataset/"
        self.data_images_path = self.dataset_path + "images/*/*.jpg"

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
        self.dataset_path = "/path/to/dataset/"
        self.data_images_path_cmfd = self.dataset_path + "CMFD/images/*.jpg"
        self.data_images_path_imfd = self.dataset_path + "IMFD/images/*.jpg"     
        self.data_images_path = [self.data_images_path_cmfd, self.data_images_path_imfd]

    def get_names(self):
        names = []
        for path in glob.glob(self.data_images_path_cmfd, recursive=True):
            #names
            names.append((path.split("/")[-1]).split(".")[0])
        for path in glob.glob(self.data_images_path_imfd, recursive=True):
            #names
            names.append((path.split("/")[-1]).split(".")[0])
        return names

    def get_data(self):
        for data_path in self.data_images_path:
            for path in glob.glob(data_path, recursive=True):
                #images
                #img = cv2.imread(path)
                data = cv2.imread(path)

                yield data