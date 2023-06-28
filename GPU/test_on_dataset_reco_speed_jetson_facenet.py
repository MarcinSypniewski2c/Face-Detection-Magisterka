import numpy as np
import time
import cv2

import onnxruntime as ort
from Rozpoznawanie.load_datasets import MFN, LFW
from Detekcja.models2 import Haar, FaceRecoLib

print(ort.get_device())

dataset = MFN()
#dataset = LFW()

detector = Haar()

filename = "./Rozpoznawanie/models/facenet.onnx"


sess = ort.InferenceSession(filename)
times=0.0
num_of_images = 0
for k, img in enumerate(dataset.get_data()):
	preds = detector.detect_face(img)
	for pred in preds:
		xmin, ymin, xmax, ymax = pred[0]
		xmin = int(xmin)
		ymin = int(ymin)
		xmax = int(xmax)
		ymax = int(ymax)
		img = img[ymin:ymax, xmin:xmax]
		if img.size != 0:
			img = cv2.resize(img, (160,160))
			img = np.transpose(img, (2, 0, 1))
			print(img.shape)
			t1 = time.time()
			sess.run(None, {"input_1": [img.astype(np.float32)]})[0]
			t2 = time.time()
			print(t2-t1)
			times = times + (t2-t1)
	num_of_images+=1

m_times = times/num_of_images
print("Mean time: ", m_times)
