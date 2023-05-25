#Haar
HAAR_MODEL_PATH = 'Detekcja/haarcascades/haarcascade_frontalface_default.xml'

#Mask
MASK_MODEL_PATH = 'Detekcja/models/detector.tflite'
CONFIDENCE=0.9

#YOLO
YOLOV5_TFLITE_160_MODEL = 'Detekcja/models/best-fp16.tflite'
FACE_DETECTION_MODEL = YOLOV5_TFLITE_160_MODEL
MAX_NMS=2
thr=0.2