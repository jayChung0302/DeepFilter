from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import os
import sys
import copy
import csv
import time
from tqdm import tqdm
from apex import amp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Dataset directory', default='/dataset')
parser.add_argument('--save_dir', type=str, help='save folder', default='models')
parser.add_argument('--save_model', type=str, help='save model name, csv', default='trainedNet')
parser.add_argument('--init_lr', type=float, help='initial learning rate', default=0.01)
parser.add_argument('--batch_size', type=int, help='batch size', default=64)
parser.add_argument('--num_classes', type=int, help='number of classes', default=2)
parser.add_argument('--exp_name', type=str, help='experiment name', default='dryrun')
parser.add_argument('--num_epoch', type=int, help='number of steps', default=2)
parser.add_argument('--use_fp16', action='store_true', help='using FP16')

args = parser.parse_args()


def main():
    # data preparation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(25),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4309, 0.4302, 0.4463], std=[0.1254, 0.1282, 0.1152])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4309, 0.4302, 0.4463], std=[0.1254, 0.1282, 0.1152])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # network
    net = models.mobilenet_v2(pretrained=True)
    head = nn.Linear(in_features=1280, out_features=args.num_classes)
    net.classifier[1] = head
    net = net.to(device)

    # optimizer, utilizer
    if args.use_fp16:
        net, _ = amp.initialize(net, [], opt_level="O2")

    optimizer_ft = optim.SGD(net.parameters(), lr=args.init_lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.05, patience=2, verbose=True,
                                                      threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0)
    epoch_loss = 1000
    since = time.time()
    writer = SummaryWriter(f'logs/{args.exp_name}')

    # save parameters in csv file #
    if not os.path.isdir(args.save_model+'.csv'):
        with open(args.save_model+'.csv', 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['phase', 'epoch', 'accuracy', 'loss'])

    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0

    print('\n===> Training Start')
    print('train data size : ', dataset_sizes['train'])
    print('validation data size : ', dataset_sizes['val'])

    for epoch in range(args.num_epoch):
        print('-' * 20)
        print('\n===> epoch {}/{}'.format(epoch + 1, args.num_epoch))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print('\n{} mode'.format(phase))

            if phase == 'train':
                # scheduler.step()
                scheduler.step(epoch_loss)
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)  # load to cuda
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer_ft.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict())
        print()
        with open(args.save_model+'.csv', 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([phase, epoch + 1, best_acc, epoch_loss])

    # training finished.
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    net.load_state_dict(best_model_wts)
    torch.save(net, os.path.join(args.save_dir, args.save_model + '.pth'))


if __name__ == '__main__':
    main()
