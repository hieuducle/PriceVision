from torch.utils.data import DataLoader
from dataset import ItemDataset
from model import RetNet_CustomClassifier
import torch
import argparse
import torch.nn as nn
import torch.optim
from torchvision.transforms import Compose, ToTensor
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os

def get_args():
    parser = argparse.ArgumentParser(description="item classifier")
    parser.add_argument("--batch-size","-b",type=int,default=4)
    parser.add_argument("--epochs","-e",type=int,default=100)
    parser.add_argument("--trained_models", "-t", type=str, default="trained_models")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    args = parser.parse_args()
    return args


def train(args):
    transform = Compose([
        ToTensor()
    ])
    train_dataset = ItemDataset(transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )
    test_dataset = ItemDataset(transform=transform)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.trained_models,exist_ok=True)
    model = RetNet_CustomClassifier(5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-3,momentum=0.9)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0
        best_acc = 0
    num_iters = len(train_dataloader)
    for epoch in range(start_epoch,args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader,colour="green")
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
    #         forward
            outputs = model(images)
            loss_value = criterion(outputs,labels)
            progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss {:.3f}".format(epoch+1, args.epochs, iter+1, num_iters, loss_value))

    #         backward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        model.eval()
        all_predictions = []
        all_labels = []
        for iter, (images,labels) in enumerate(test_dataloader):
            all_labels.extend(labels)
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                predictions = model(images)
                indices = torch.argmax(predictions.cpu(),dim=1)
                all_predictions.extend(indices)
                loss_value = criterion(predictions,labels)

        all_labels = [label.item() for label in all_labels]
        all_predictions = [predictions.item() for predictions in all_predictions]
        accuracy = accuracy_score(all_labels, all_predictions)
        print("Epoch {}: Accuracy: {}".format(epoch + 1, accuracy))
        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, "{}/last_cnn.pt".format(args.trained_models))
        if accuracy > best_acc:
            checkpoint = {
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, "{}/best_cnn.pt".format(args.trained_models))
            best_acc = accuracy


if __name__ == '__main__':
    args = get_args()
    train(args)


