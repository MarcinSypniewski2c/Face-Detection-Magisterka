import cv2

from Detekcja.models import Haar, Insightface, FaceRecoLib, YoloV5

detector = Insightface()
#detector = Haar()
#detector = FaceRecoLib()
#detector = YoloV5()

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if ret:
            preds = detector.detect_face(frame)
            for pred in preds:
                print(pred[0])
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

if __name__ == "__main__":
    main()