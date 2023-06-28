import numpy as np
import math
import cv2
import Detekcja.config.config as cfg
import face_recognition
from Detekcja.logger.logger import logger

class Haar:
    """
    Haar cascade classifier for detecting face in image.
    """
    def __init__(self, model='/home/piotr/Documents/MS/Face-Detection-Magisterka-main/Detekcja/models/haarcascade_frontalface_default.xml'):
        self.face_cascade = cv2.CascadeClassifier(model)
        self.SCALE_FACTOR = 1.1
        self.MIN_NEIGHBORS = 7
        self.MIN_SIZE = (10, 10)
        self.MAX_SIZE = None
        self.CSI = cv2.CASCADE_SCALE_IMAGE

        logger.info("Haar detector initialized")
    
    def detect_face(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        bboxes = self.face_cascade.detectMultiScale(gray, 
                                        scaleFactor=self.SCALE_FACTOR, 
                                        minNeighbors=self.MIN_NEIGHBORS, 
                                        minSize=self.MIN_SIZE,
                                        flags=self.CSI)
        return self.postprocess(bboxes)

    def postprocess(self, xywh):
        xyxy = [[x, y, x+w, y+h] for x,y,w,h in xywh]

        masks = [None for _ in range(len(xyxy))]
        return zip(xyxy, masks)

class FaceRecoLib:
    def __init__(self):
        logger.info("Face recognition library detector initialized")

    def detect_face(self, img):
        face_locations = face_recognition.face_locations(img)
        return self.postprocess(face_locations)

    def postprocess(self, trbl):
        xyxy = [[l, t, r, b] for t,r,b,l in trbl]
        masks = [None for _ in range(len(xyxy))]
        return zip(xyxy, masks)
