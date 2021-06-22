import json
from model import WaveletCnnModel
from downloader import download
from trainer import train_model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import datasets

import numpy as np
import os

from torch import optim
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split


def main():
    with open("config.json", "r") as read_file:
        config = json.load(read_file)
    config = config['settings']
    seed = int(config['seed'])

    print("here")
    torch.manual_seed(seed)
    if config['cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    data_dir = config['data_dir']

    print("before download")

    # download(config["csv_path"], data_dir)

    ######################################################
    # PREPARING data #

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor()
        ]),
        'valid': transforms.Compose([
            transforms.ToTensor()
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'valid']}
    print("here1")
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                                  shuffle=True)
                   for x in ['train', 'valid']}
    print("here2")
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes

    ######################################################
    # BEGINING TO TRAIN MODEL #
    model_ft = WaveletCnnModel(config['n_classes'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        model_ft.cuda()
    # model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-5)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(model_ft,
                           dataloaders,
                           device,
                           dataset_sizes,
                           criterion,
                           optimizer_ft,
                           exp_lr_scheduler,
                           num_epochs=20)
    torch.save(model_ft.state_dict(), config["save_weights_path"])


if __name__ == "__main__":
    main()