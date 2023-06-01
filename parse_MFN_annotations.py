import json
import glob


annotations_path = "/home/msypniewski@sap-flex.com/Documents/DATASETS/MaskedFace-Net/ffhq-features-dataset-master/json/*.json"
annotations_files = sorted(glob.glob(annotations_path))
annotations_yolo_path = "/home/msypniewski@sap-flex.com/Documents/DATASETS/MaskedFace-Net/CMFD/labels_yolo/"
IMG_SIZE = 1024

num_to_parse = 10000
i = 0

for annotations_file in annotations_files:
    print(str(i) + "/" + str(num_to_parse))
    annotations = open(annotations_file)
    annotations_yolo_file = annotations_yolo_path + (annotations_file.split("/"))[-1].split(".")[0] + "_Mask.txt"

    annotations_data = json.load(annotations)
    if annotations_data:
        tlwh = annotations_data[0]["faceRectangle"]

        xmin = tlwh["left"]
        ymin = tlwh["top"]
        xmax = tlwh["left"] + tlwh["width"]
        ymax = tlwh["top"] + tlwh["height"]

        scale = 8/IMG_SIZE

        xmin = scale*xmin
        ymin = scale*ymin
        xmax = scale*xmax
        ymax = scale*ymax

        #print(IMG_SIZE*xmin, IMG_SIZE*ymin, IMG_SIZE*xmax, IMG_SIZE*ymax)

        x = scale*(tlwh["left"] + (tlwh["width"]/2))
        y = scale*(tlwh["top"] + (tlwh["height"]/2))
        w = scale*tlwh["width"]
        h = scale*tlwh["height"]

        xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2] 
        xmin_gt = int(xyxy[0] * IMG_SIZE)
        ymin_gt = int(xyxy[1] * IMG_SIZE)
        xmax_gt = int(xyxy[2] * IMG_SIZE)
        ymax_gt = int(xyxy[3] * IMG_SIZE)

        #print(xmin_gt, ymin_gt, xmax_gt, ymax_gt)

        label = [str("1 " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")]

        #print(label)

        label_to_write = open(annotations_yolo_file,"w")
        label_to_write.writelines(label)
        label_to_write.close()

    if i == num_to_parse:
        break
    else:
        i+=1

