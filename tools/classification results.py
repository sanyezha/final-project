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
import matplotlib.pyplot as plt
import numpy as np

from model2 import vgg2


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 25

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    validate_dataset = datasets.CIFAR10(root='./cifar10', train=False,
                                        download=False, transform=data_transform["val"])
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=25,
                                                  shuffle=False)

    model = vgg2(model_name="vgg11", num_classes=7, init_weights=False).to(device)
    # load model weights
    weights_path = "./{}Net_CIFAR10.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()
    with torch.no_grad():
        val_bar = tqdm(validate_loader)
        for val_data in val_bar:
            val_images, val_labels = val_data
            print(val_labels)
            outputs = model(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
    img = val_images / 2 + 0.5
    classes = ('deg75', 'deg95', 'deg115', 'deg135', 'deg155', 'deg175', 'deg195')

    fig, ax = plt.subplots(4, 2, sharex=True, sharey=True, constrained_layout=True)
    num = 0
    for i in range(4):
        for j in range(2):
            plt.xticks([])
            plt.yticks([])
            ima = img[num]
            a = 2 * i + j
            print(a)
            ax[i, j].set_title('label {}, \npredicted {}'.format(classes[val_labels[a]], classes[predict_y[a]]))
            ima = np.transpose(ima, (1, 2, 0))
            ax[i, j].imshow(ima.detach().cpu().numpy())
            num = num + 1
    plt.suptitle('Examples of VGG results on CIFAR10')
    plt.savefig('Examples of VGG results on CIFAR10.png')
    plt.show()
    plt.close()



if __name__ == '__main__':
    main()