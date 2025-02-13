from torch.utils.data import Dataset,DataLoader
import os
import cv2
import torch
from torchvision.transforms import Compose, ToTensor


class ItemDataset(Dataset):
    def __init__(self,transform=None):
        self.labels = []
        self.images_path = []
        self.transform = transform
        root = "/home/amin/PycharmProjects/PythonProject/PriceVision/data/train"
        self.categories = ["coca","mentos","milk_box","nutifood","TH_water"]
        for idx,cls_folder in enumerate(self.categories):
            cls_folder_path = os.path.join(root,cls_folder)
            for file_img in os.listdir(cls_folder_path):
                img_path = os.path.join(cls_folder_path,file_img)
                self.images_path.append(img_path)
                self.labels.append(idx)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images_path[idx]
        label = self.labels[idx]
        img = cv2.imread(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # img = torch.from_numpy(img).float()
        img = self.transform(img)
        return img,label


if __name__ == '__main__':
    transform = Compose([
        ToTensor()
    ])
    dataset = ItemDataset(transform=transform)
    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True
    )
    for img,label in train_dataloader:
        print(img.shape)
        # print(label)


