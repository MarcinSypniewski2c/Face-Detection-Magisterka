import tflite2onnx

tflite_path = 'Rozpoznawanie/models/facenet.tflite'
onnx_path = 'Rozpoznawanie/models/facenet.onnx'

tflite2onnx.convert(tflite_path, onnx_path)