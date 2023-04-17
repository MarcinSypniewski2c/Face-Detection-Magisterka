#Haar
HAAR_MODEL_PATH = 'haarcascades/haarcascade_frontalface_default.xml'

#Mask
MASK_MODEL_PATH = 'models/detector.tflite'
CONFIDENCE=0.9

#YOLO
YOLOV5_TFLITE_160_MODEL = 'models/best-fp16.tflite'
FACE_DETECTION_MODEL = YOLOV5_TFLITE_160_MODEL
MAX_NMS=2
thr=0.2