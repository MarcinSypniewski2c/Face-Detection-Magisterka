import numpy as np
import math
from face_detection import RetinaFace
import tensorflow as tf
import tflite_runtime.interpreter as tflite
import face_recognition
import cv2
import config as cfg
from logger import logger

def NMS_tflite(boxes, scores):
    '''
    Non Max Suppression of boxes
    return: best boxes and their indices
    '''
    boxes_tensor = tf.convert_to_tensor(boxes)
    scores_tensor = tf.convert_to_tensor(scores)

    selected_indices = tf.image.non_max_suppression(
        boxes_tensor, scores_tensor, max_output_size=10, iou_threshold=0.6)
    selected_boxes = tf.gather(boxes_tensor, selected_indices)

    new_boxes = selected_boxes.numpy()
    new_indices = selected_indices.numpy()

    return new_boxes, new_indices

def classFilter(classdata):
    classes = []  # create a list
    for i in range(classdata.shape[0]):         # loop through all predictions
        classes.append(classdata[i].argmax())   # get the best classification location
    return classes  # return classes (int)

def YOLOdetect(output_data):  # input = interpreter, output is boxes(xyxy), classes, scores
    output_data = output_data[0]                # x(1, 25200, 7) to x(25200, 7)
    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
    classes = classFilter(output_data[..., 5:]) # get classes
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]
    xyxy = np.transpose(xyxy)  # xyxy to [25200, 4]

    return xyxy, classes, scores  # output is boxes(x,y,x,y), classes(int), scores(float) [predictions length]

class Haar:
    """
    Haar cascade classifier for detecting face in image.
    """
    def __init__(self, model=cfg.HAAR_MODEL_PATH):
        self.face_cascade = cv2.CascadeClassifier(model)
        self.SCALE_FACTOR = 1.1
        self.MIN_NEIGHBORS = 3
        self.MIN_SIZE = (100, 100)
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
    
class Retinaface:
    def __init__(self):
        self.retina = RetinaFace()
        logger.info("RetinaFace detector initialized")
    
    def detect_face(self, img):
        resp = self.retina(img)
        return self.postprocess(resp)
    
    def postprocess(self, bls):
        for b,l,s in bls:
            if s > 0.6:
                xyxy = [[b[0], b[1], b[2], b[3]]]
            else:
                xyxy = []

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
    
class YoloV5:
    def __init__(self):
        self.interpreter = tflite.Interpreter(model_path = cfg.YOLOV5_TFLITE_160_MODEL)
        self.interpreter.allocate_tensors()
        self.vw = 0
        self.vh = 0

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.input = input_details[0]
        self.output = output_details[0]
        self.isUint8 = self.input['dtype'] == np.uint8

    def detect_face(self, frame):
        self.vh = frame.shape[0]
        self.vw = frame.shape[1]
        frame = self.preprocess(frame)
        if self.isUint8:
            scale, zero_point = self.input['quantization']
            frame = (frame / scale + zero_point).astype(np.uint8)  # de-scale

        self.interpreter.set_tensor(self.input['index'], frame)
        self.interpreter.invoke()
        pred = self.interpreter.get_tensor(self.output['index'])

        if self.isUint8:
            scale, zero_point = self.output['quantization']
            pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale

        boxes, classes, scores = YOLOdetect(pred)

        new_boxes, new_indices = NMS_tflite(boxes, scores)

        final_boxes = []
        final_scores = []
        final_classes = []

        for ni in range(len(new_indices)):
            if scores[new_indices[ni]] >= 0.6 and scores[new_indices[ni]] <= 1.0:
                final_boxes.append(new_boxes[ni])
                final_scores.append(scores[new_indices[ni]])
                final_classes.append(classes[new_indices[ni]])
        
        final_boxes = np.absolute(final_boxes)

        return self.postprocess(final_boxes)
    
    def postprocess(self, boxes):
        H = self.vh
        W = self.vw
        xyxy = [[int((boxes[ni][0] * W)), int((boxes[ni][1] * H)), int((boxes[ni][2] * W)), int((boxes[ni][3] * H))] for ni in range(len(boxes))]

        #xmin = int((boxes[ni][0] * W))
        #ymin = int((boxes[ni][1] * H))
        #xmax = int((boxes[ni][2] * W))
        #ymax = int((boxes[ni][3] * H))
        masks = [None for _ in range(len(xyxy))]
        return zip(xyxy, masks)

    def preprocess(self, frame):
        frame = cv2.resize(frame, (160,160))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame/255  # normalization to 0-1
        frame = np.expand_dims(frame, axis=0)
        frame = frame.astype(np.float32)

        return frame
