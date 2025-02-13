from ultralytics import YOLO
import cv2

img = "/home/amin/PycharmProjects/PythonProject/PriceVision/data/train/TH_water/8.jpg"

model = YOLO("/home/amin/PycharmProjects/PythonProject/yolov8n.pt")

result = model(img)
img = cv2.imread(img)
for box in result[0].boxes.xyxy:
    xmin,ymin,xmax,ymax = map(int,box.tolist())
    cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),3)
cv2.imshow("a",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
