import cv2
import itertools
import csv
import numpy as np

from Rozpoznawanie.load_datasets import LFW
from Rozpoznawanie.models import InsightFace
from Detekcja.models import Haar, YoloV5, FaceRecoLib, Insightface

reco_filename = "Rozpoznawanie/results/lfw_reco_insight.csv"

lfw = LFW()
#detector = Haar()
detector = YoloV5()
#detector = FaceRecoLib()
#detector = Insightface()

recognizer = InsightFace()

data, names = lfw.get_data()
data_len = len(data)

i = 1
embeddings = []
embeddings_names = []
recognitions = []

for k, img in enumerate(data):
    print(str(i) + "/" + str(data_len))
    i = i+1
    preds = detector.detect_face(img)
    for pred in preds:
        xmin, ymin, xmax, ymax = pred[0]
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        #cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (100,100,10), 1)
        #cv2.imwrite("zero.jpg",img)
        img = img[ymin:ymax, xmin:xmax]
        embedding = recognizer.get_embeddings(img)
        for emb in embedding:
            embeddings.append(emb)
            embeddings_names.append(names[k])


for ((i, a),(j, b)) in itertools.combinations(enumerate(embeddings), 2):
    dist = np.linalg.norm(a - b)
    recognitions.append([embeddings_names[i], embeddings_names[j], dist])

#  save to file
with open(reco_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(recognitions)
