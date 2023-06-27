import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from Rozpoznawanie.load_datasets import LFW, MFN
from Rozpoznawanie.models import InsightFace, FaceNet
from Detekcja.models import Haar, YoloV5, FaceRecoLib, Insightface

reco_filename = "Rozpoznawanie/results/mfn_reco_insight1.csv"

#dataset = LFW()
dataset = MFN()

#detector = Haar()
detector = YoloV5()
#detector = FaceRecoLib()
#detector = Insightface()

recognizer = FaceNet()
#recognizer = InsightFace()

names = dataset.get_names()
data_len = len(names)

times = 0.0

i = 1
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
            t1 = time.time()
            embedding = recognizer.get_embeddings(img)
            t2 = time.time()
            print(t2 - t1)
            times = times + (t2-t1)

m_times = times/data_len
print("Mean time: ", m_times)


