import os
import json

import time
import datetime
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from model2 import vgg2


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 100

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    train_dataset = datasets.CIFAR10(root='./cifar10', train=True,
                                     download=False, transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                               shuffle=True)

    validate_dataset = datasets.CIFAR10(root='./cifar10', train=False,
                                        download=False, transform=data_transform["val"])
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=16,
                                                  shuffle=False)

    classes = ('deg75', 'deg105', 'deg135_left', 'deg135_right',
               'deg165_left', 'deg165_right', 'deg195')

    train_num = len(train_dataset)
    val_num = len(validate_dataset)

    net = vgg2(model_name="vgg11", num_classes=7, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    # optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 40
    best_acc = 0.0
    save_path = './{}Net_CIFAR10.pth'
    train_steps = len(train_loader)
    trainL = []
    valL = []
    print('\nstart time', datetime.datetime.now())
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        pretime = datetime.datetime.now()
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            # print(labels)
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        aftertime = datetime.datetime.now()
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        times = (aftertime-pretime).seconds
        print('this epoch costs %.3f seconds' % (times))
        trainL.append(running_loss / train_steps)
        valL.append(val_accurate)
        # print(trainL)
        # print(valL)
        # f = open('vgg_trainloss.txt', 'w')
        # f.write(str(trainL))
        # f.close()
        # f = open('vgg_val.txt', 'w')
        # f.write(str(valL))
        # f.close()

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    f = open('vgg_cifar_trainloss.txt', 'w')
    f.write(str(trainL))
    f.close()
    f = open('vgg——cifar_val.txt', 'w')
    f.write(str(valL))
    f.close()
    print('\nend time', datetime.datetime.now())
    print('Finished Training')


if __name__ == '__main__':
    main()



