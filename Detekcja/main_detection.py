import cv2

from models import Haar, Retinaface, FaceRecoLib, YoloV5

import config as cfg
from logger import logger

detector = Retinaface()
#detector = Haar()
#detector = FaceRecoLib()
#detector = YoloV5()

def main():
    logger.info("Starting face detection script")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if ret:
            preds = detector.detect_face(frame)
            for pred in preds:
                xmin, ymin, xmax, ymax = pred[0]
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                mask = pred[1]
                if mask:
                    color = (70,25,255)
                else:
                    color = (70,255,70)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)

            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Closing face recognition script")

if __name__ == "__main__":
    main()