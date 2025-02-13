import numpy as np
from ultralytics import YOLO
import cv2
from model import RetNet_CustomClassifier
import argparse
import torch
from test import cls
from torchvision.transforms import ToTensor
import json
import torch.nn as nn
from deep_sort_realtime.deepsort_tracker import DeepSort


def draw_text_with_bg(image, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX,
                      font_scale=1, font_thickness=2, text_color=(255, 255, 255),
                      bg_color=(0, 0, 0), alpha=0.5):
    """Vẽ text với nền mờ"""
    overlay = image.copy()
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x, text_y = pos
    box_coords = ((text_x, text_y - text_size[1] - 5), (text_x + text_size[0] + 10, text_y + 5))

    cv2.rectangle(overlay, box_coords[0], box_coords[1], bg_color, -1)  # Vẽ nền đen
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)  # Làm mờ nền
    cv2.putText(image, text, (text_x + 5, text_y), font, font_scale, text_color, font_thickness)

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
def get_args():
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--img-path",type=str,default="/home/amin/PycharmProjects/PythonProject/th.jpg")
    parser.add_argument("--checkpoint",type=str,default="trained_models/best_cnn.pt")
    args = parser.parse_args()
    return args

def cls(img,model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    categories = ["Coca", "Mentos", "milk_box", "Nutifood", "TH_water"]
    model.eval()

    # img = cv2.imread(args.img_path)
    # ori_image = img
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
    """
    Tính chỉ số IOU giữa 2 bounding box.
    box = (xmin, ymin, xmax, ymax)
    """
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
    detect_path = "/home/amin/PycharmProjects/PythonProject/yolov8n.pt"
    classifier = RetNet_CustomClassifier(5).to(device)
    tracker = DeepSort(max_age=30)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        classifier.load_state_dict(checkpoint["model"])
    else:
        print("Not found checkpoint")
        exit(0)
    vid = "test.mp4"
    cap = cv2.VideoCapture(vid)
    detect = YOLO(detect_path)
    frame_count = 0
    skip_frame = 20
    track_classes = {}
    track_price = {}
    skiped_item = set()
    total_price = 0
    count_mentos = 0
    count_nutifood = 0
    tracked_objects = {}  # {class_name: [(track_id, (xmin, ymin, xmax, ymax))]}
    iou_threshold = 0.35
    processed_ids = set()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter("output/check_price.mp4", fourcc, fps, (1200, 800))
    item_count = {}
    max_box_area = 0.7 * 1200 * 800
    while cap.isOpened():
        flag,frame = cap.read()

        if not flag:
            break

        frame = cv2.resize(frame, (1200, 800))
        frame = cv2.rotate(frame,cv2.ROTATE_180)

        result = detect.track(frame, persist=True)

        detections = []
        if not result or not hasattr(result[0], "boxes") or result[0].boxes is None:
            continue

        if result[0].boxes.xyxy is None or result[0].boxes.id is None:
            continue
        for obj,track_id in zip(result[0].boxes.xyxy,result[0].boxes.id):
            if track_id is None:
                continue  # Bỏ qua nếu không có ID
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
            cv2.putText(frame, f"ID {track_id}: {class_name}", (xmin, ymin - 10),
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
        cv2.imshow("a",frame)

        cv2.waitKey(25)
    cap.release()
    out.release()
    cv2.destroyAllWindows()