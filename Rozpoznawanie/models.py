from insightface.app import FaceAnalysis

import tensorflow as tf
import tflite_runtime.interpreter as tflite
import numpy as np
import pickle

import cv2


class InsightFace:
    def __init__(self):
        self.model = FaceAnalysis(name="antelope")
        self.model.prepare(ctx_id=0, det_size=(160,160))

    def recognize(self, img):
        try:
            faces = self.model.get(img)
        except ZeroDivisionError:
            faces = []
        return faces
    
    def get_embeddings(self, img):
        faces = self.recognize(img)
        embeds = []
        for face in faces:
            embeds.append(face[3])
        return embeds

class FaceNet:
    def __init__(self, model_path_face = 'Rozpoznawanie/models/facenet.tflite'):
        self.interpreter_face = tflite.Interpreter(model_path = model_path_face)
        self.interpreter_face.allocate_tensors()

        input_details_face = self.interpreter_face.get_input_details()
        output_details_face = self.interpreter_face.get_output_details()
        self.input_face = input_details_face[0]
        self.output_face = output_details_face[0]
        #self.isUint8_face = self.input_face['dtype'] == np.uint8

    def get_embeddings(self, img):
        img = cv2.resize(img, (160,160))
        #if self.isUint8_face:
        #    scale, zero_point = self.input_face['quantization']
        #    img = (img / scale + zero_point).astype(np.uint8)  # de-scale

        self.interpreter_face.set_tensor(self.input_face['index'], [img.astype(np.float32)])
        self.interpreter_face.invoke()
        embeds = self.interpreter_face.get_tensor(self.output_face['index'])

        #if self.isUint8_face:
        #    scale, zero_point = self.output_face['quantization']
        #    embeds = (embeds.astype(np.float32) - zero_point) * scale  # re-scale
    
        return embeds