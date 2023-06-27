import cv2
import itertools
import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from Rozpoznawanie.load_datasets import LFW, MFN
from Rozpoznawanie.models import InsightFace, FaceNet
from Detekcja.models import Haar, YoloV5, FaceRecoLib, Insightface

reco_filename = "Rozpoznawanie/results/mfn_reco_facenet1111.csv"

#dataset = LFW()
dataset = MFN()

#detector = Haar()
detector = YoloV5()
#detector = FaceRecoLib()
#detector = Insightface()

#recognizer = FaceNet()
recognizer = InsightFace()

names = dataset.get_names()
data_len = len(names)

i = 1
embeddings = []
embeddings_names = []
recognitions = []

for k, img in enumerate(dataset.get_data()):
    #print(i)
    print(str(i) + "/" + str(data_len))
    i = i+1
    preds = detector.detect_face(img)
    for pred in preds:
        xmin, ymin, xmax, ymax = pred[0]
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        #cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (100,200,10), 1)
        #cv2.imwrite("zero.jpg",img)
        img = img[ymin:ymax, xmin:xmax]
        if img.size != 0:
            embedding = recognizer.get_embeddings(img)
            for emb in embedding:
                embeddings.append(emb)
                embeddings_names.append(names[k])

embeddings = np.array(embeddings)
#np.reshape(embeddings,(1,-1))
print(embeddings.shape)

curr_j = 0
for j in range(len(embeddings)):
    knc = KNeighborsClassifier(n_neighbors=1)
    curr_embeddings = embeddings
    curr_embeddings = np.delete(curr_embeddings, j, 0)
    curr_embeddings_names = embeddings_names
    curr_embeddings_names = np.delete(curr_embeddings_names, j, 0)
    knc.fit(curr_embeddings, curr_embeddings_names)
    pred = knc.predict([embeddings[j]])

    recognitions.append([str(embeddings_names[j]), pred])

    #print(knc.predict([embeddings[j]]))
    #print(embeddings_names[j])


#  save to file
with open(reco_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(recognitions)
