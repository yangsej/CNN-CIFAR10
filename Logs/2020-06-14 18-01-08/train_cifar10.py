# -*- coding: utf-8 -*-
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import CifarNet as cfn
# import DPN

import os
import sys
import shutil
# import winsound
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true")
parser.add_argument("--show_data", action="store_true")

# global data
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


DATA_PATH = "./data"

NUM_WORKERS = 2
EPOCHS = 20
BATCH_SIZE = 512
WEIGHT_DECAY = 1e-3
LEARNING_RATE = 1e-2

now = datetime.now()
now_str = now.strftime("%F %H-%M-%S")
DIR_PATH = "./%s" % (now_str)

OUT_FILE = None
if __name__ == "__main__":
    if not os.path.exists(DIR_PATH):
        os.mkdir(DIR_PATH)
    shutil.copy2("./CifarNet.py", "%s/CifarNet.py" % DIR_PATH)
    shutil.copy2("./train_cifar10.py", "%s/train_cifar10.py" % DIR_PATH)
    OUT_FILE = open("%s/cifar_net.txt" % DIR_PATH, "w")
MODEL_PATH = "%s/cifar_net.pth" % DIR_PATH 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    args = parser.parse_args()
    if args.test:
        test_dataloader = cifar10_dataloader(
            train=False, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )
        # model = DPN.DPN92().to(device)
        model = cfn._cifarnet(pretrained=args.test, path=MODEL_PATH).to(device)
        test(test_dataloader, model, args.show_data)
        return

    if args.show_data:
        dataloader = cifar10_dataloader(
            train=False, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
        show_data(dataloader)
        return

    train_dataloader = cifar10_dataloader(
        train=True, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    test_dataloader = cifar10_dataloader(
        train=False, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    # model = DPN.DPN92().to(device)
    model = cfn._cifarnet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY
    )

    for epoch in range(EPOCHS):
        print("Epoch %i" % (epoch+1), file=OUT_FILE)

        train(train_dataloader, model, criterion, optimizer, epoch)
        torch.save(model.state_dict(), MODEL_PATH)

        test(train_dataloader, model, args.show_data)
        test(test_dataloader, model, args.show_data)


def cifar10_dataloader(
    root=DATA_PATH,
    train=True,
    transform=None,
    shuffle=False,
    download=True,
    batch_size=4,
    num_workers=0,
):

    if transform is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    dataset = torchvision.datasets.CIFAR10(
        root=root, train=train, transform=transform, download=download
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
    )

    return dataloader


def train(dataloader, model, criterion, optimizer, epoch):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        images, labels = data[0].to(device), data[1].to(device)
        logit = model(images)
        loss = criterion(logit, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(running_loss / 2000.0, file=OUT_FILE)
            running_loss = 0.0
        elif i == len(dataloader)-1:
            print(running_loss / float(len(dataloader)), file=OUT_FILE)


def test(dataloader, model, show_data):
    if show_data:
        dataiter = iter(dataloader)
        images, labels = dataiter.next()
        # show images
        imshow(torchvision.utils.make_grid(images))
        output = model(images.to(device))
        _, predicted = torch.max(output, 1)
        print(
            "GT", " ".join("%6s" % classes[labels[j]] for j in range(4)), file=OUT_FILE
        )
        print(
            "PT",
            " ".join("%6s" % classes[predicted[j]] for j in range(4)),
            file=OUT_FILE,
        )
        print(file=OUT_FILE)

    correct = 0
    total = 0
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print(
        "Accuracy : %d %%"
        % (100 * correct / total),
        file=OUT_FILE,
    )
    print(file=OUT_FILE)
    for i in range(10):
        print(
            "Accuracy of %5s : %2d %%"
            % (classes[i], 100 * class_correct[i] / class_total[i] if class_total[i] else 0),
            file=OUT_FILE,
        )
    print(file=OUT_FILE)


def show_data(dataloader):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))

    # print labels
    print(" ".join("%10s" % classes[labels[j]] for j in range(4)), file=OUT_FILE)


def imshow(img: torch.Tensor):
    import matplotlib.pyplot as plt
    import numpy as np

    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    main()
    # winsound.Beep(6000, 5000)
