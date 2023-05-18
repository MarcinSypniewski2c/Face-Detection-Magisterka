from time import time
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import csv

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform

from models import InsightFace

reco_filename = "lfw_reco_insight.csv"
thr = 23.0

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=1)
target_names = lfw_people.target_names
print(len(lfw_people.images))
print(lfw_people.target[0])
print(target_names)

reco = InsightFace()

#make database
current_file_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(current_file_dir, "database_lfw")
embeddings_db = []
names_db = []

for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # Adjust the extensions as needed
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        embeds_tmp = reco.get_embeddings(image)
        for emb_t in embeds_tmp:
            embeddings_db.append(emb_t)
            names_db.append(filename)

recognitions = []
i = 0
for img in lfw_people.images:
    image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    embeddings = reco.get_embeddings(image)
    for emb in embeddings:
        dist = []
        for emb_db in embeddings_db:
            dist.append(np.linalg.norm(emb - emb_db))
        min_value, min_index = min((value, index) for index, value in enumerate(dist))
        if min_value < thr:
            name = names_db[min_index]
        else:
            name = "Unknown"
        target_name = target_names[lfw_people.target[i]]
        recognitions.append([target_name, name, min_value])
        i=i+1


with open(reco_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(recognitions)

print(target_names)