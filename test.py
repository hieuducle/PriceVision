import json

from torch import no_grad

from model import RetNet_CustomClassifier
import argparse
import torch
import cv2
from torchvision.transforms import ToTensor
import torch.nn as nn
def get_args():
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--img-path",type=str,default="/home/amin/PycharmProjects/PythonProject/th.jpg")
    parser.add_argument("--checkpoint",type=str,default="trained_models/best_cnn.pt")
    args = parser.parse_args()
    return args

def cls(img):
    args = get_args()
    categories = ["coca", "mentos", "milk_box", "nutifood", "TH_water"]
    device = torch.device("cuda" if torch.cuda.is_available() else "gpu")
    model = RetNet_CustomClassifier(5).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model"])
    else:
        print("Not found checkpoint")
        exit(0)
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
    print(data)
    for key in data.keys():
        if predicted_class == key:
            return predicted_class,data[key]
    #         print(data[key])
    # print("The test image is about {} with confident score of {}".format(predicted_class, probs[0, max_idx]))
    # cv2.imshow("{}:{:.2f}%".format(predicted_class, probs[0, max_idx] * 100), ori_image)
    # cv2.waitKey(0)

if __name__ == '__main__':
    args = get_args()

