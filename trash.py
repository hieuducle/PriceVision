import cv2
import os

# cap = cv2.VideoCapture("/home/amin/PycharmProjects/PythonProject/PriceVision/data/mentos.mp4")
# fps = cap.get(cv2.CAP_PROP_FPS)
# delay = int(1000/fps)
# os.makedirs("output", exist_ok=True)
root = "/home/amin/PycharmProjects/PythonProject/PriceVision/vid"
out_data = "/home/amin/PycharmProjects/PythonProject/PriceVision/data/train"
for vid in os.listdir(root):
    vid_path = os.path.join(root,vid)
    out_data_class = os.path.join(out_data,(os.path.splitext(vid))[0])

    os.makedirs(out_data_class,exist_ok=True)

    cap = cv2.VideoCapture(vid_path)
    idx = 0
    while cap.isOpened():
        flag,frame = cap.read()

        if not flag:
            break
        frame = cv2.resize(frame, (224, 224))
        cv2.imwrite("{}/{}.jpg".format(out_data_class,idx),frame)
        idx = idx + 1


    cap.release()
    cv2.destroyAllWindows()