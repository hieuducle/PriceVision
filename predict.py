import numpy as np
from ultralytics import YOLO
import cv2
from model import RetNet_CustomClassifier
import argparse
import torch
from torchvision.transforms import ToTensor
import json
import torch.nn as nn



def get_args():
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--img-path",type=str,default="/home/amin/PycharmProjects/PythonProject/th.jpg")
    parser.add_argument("--checkpoint",type=str,default="trained_models/best_cnn.pt")
    parser.add_argument("--detect-path",type=str,default="yolo_model/yolov8n.pt")
    parser.add_argument("--video-path",type=str,default="test.mp4")
    parser.add_argument("--out-video", type=str, default="output/check_price.mp4")

    args = parser.parse_args()
    return args

def draw_corner_box(frame, bbox, color=(255, 200, 100), thickness=2, corner_length=20):
    xmin, ymin, xmax, ymax = bbox

    cv2.line(frame, (xmin, ymin), (xmin + corner_length, ymin), color, thickness)
    cv2.line(frame, (xmin, ymin), (xmin, ymin + corner_length), color, thickness)

    cv2.line(frame, (xmax, ymin), (xmax - corner_length, ymin), color, thickness)
    cv2.line(frame, (xmax, ymin), (xmax, ymin + corner_length), color, thickness)

    cv2.line(frame, (xmin, ymax), (xmin + corner_length, ymax), color, thickness)
    cv2.line(frame, (xmin, ymax), (xmin, ymax - corner_length), color, thickness)

    cv2.line(frame, (xmax, ymax), (xmax - corner_length, ymax), color, thickness)
    cv2.line(frame, (xmax, ymax), (xmax, ymax - corner_length), color, thickness)


def cls(img,model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    categories = ["Coca", "Mentos", "milk_box", "Nutifood", "TH_water"]
    model.eval()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = ToTensor()(img)
    img = img[None, :, :, :]
    img = img.to(device)
    softmax = nn.Softmax()
    with torch.no_grad():
        output = model(img)
        probs = softmax(output)
    max_idx = torch.argmax(probs)
    predicted_class = categories[max_idx]
    with open("price.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    for key in data.keys():
        if predicted_class == key:
            return predicted_class, data[key]


def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detect_path = args.detect_path
    classifier = RetNet_CustomClassifier(5).to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        classifier.load_state_dict(checkpoint["model"])
    else:
        print("Not found checkpoint")
        exit(0)
    vid = args.video_path
    cap = cv2.VideoCapture(vid)
    detect = YOLO(detect_path)
    total_price = 0
    tracked_objects = {}
    iou_threshold = 0.35

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(args.out_video, fourcc, fps, (1200, 800))

    item_count = {}
    max_box_area = 0.7 * 1200 * 800
    while cap.isOpened():
        flag,frame = cap.read()

        if not flag:
            break

        frame = cv2.resize(frame, (1200, 800))
        # frame = cv2.rotate(frame,cv2.ROTATE_180)
        frame = cv2.flip(frame, -1)


        result = detect.track(frame, persist=True)

        detections = []
        if not result or not hasattr(result[0], "boxes") or result[0].boxes is None:
            continue

        if result[0].boxes.xyxy is None or result[0].boxes.id is None:
            continue
        for obj,track_id in zip(result[0].boxes.xyxy,result[0].boxes.id):
            if track_id is None:
                continue
            track_id = int(track_id)

            xmin,ymin,xmax,ymax = map(int,obj.tolist())
            if (int(xmax-xmin) * int(ymax - ymin) > max_box_area):
                continue

            img = frame[ymin:ymax,xmin:xmax]
            img = np.array(img)

            class_name,price = cls(img,classifier)
            if class_name in tracked_objects:
                is_new_object = True
                for old_track_id, old_bbox in tracked_objects[class_name]:
                    iou = compute_iou((xmin, ymin, xmax, ymax), old_bbox)
                    if iou > iou_threshold:
                        is_new_object = False
                        break

                if is_new_object:
                    total_price += price
                    if class_name not in item_count:
                        item_count[class_name] = 1
                    else:
                        item_count[class_name] += 1
            else:
                total_price += price
                if class_name not in item_count:
                    item_count[class_name] = 1
                else:
                    item_count[class_name] += 1

            if class_name not in tracked_objects:
                tracked_objects[class_name] = []
            tracked_objects[class_name].append((track_id, (xmin, ymin, xmax, ymax)))

            draw_corner_box(frame,[xmin,ymin,xmax,ymax])

            cv2.putText(frame, "ID {} {}".format(track_id,class_name), (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            trig_x = xmin + int((xmax-xmin)/2)
            triangle_pts = np.array([[trig_x, ymin - 30], [trig_x - 80, ymin - 100], [trig_x + 80, ymin - 100]], np.int32)
            cv2.polylines(frame, [triangle_pts], isClosed=True, color=(255, 200, 100), thickness=3)
            cv2.fillPoly(frame, [triangle_pts], color=(255, 200, 100))
            cv2.putText(frame, "{} vnd".format(price), (xmin + 30, ymin-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        y_offset = 200
        for item,count in item_count.items():
            cv2.putText(frame, "{}: {}".format(item,count), (1000, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            y_offset -= 50
        cv2.putText(frame, "Total: {}".format(total_price), (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)
        # cv2.imshow("a",frame)
        #
        # cv2.waitKey(25)
    cap.release()
    out.release()
    cv2.destroyAllWindows()