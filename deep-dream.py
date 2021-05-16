import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torch.optim as optim
import numpy as np
import mypy
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import parser

random.seed(0)
parser.add_argument('--exp_name', type=str, help='experiment name', default='dryrun')


def dream_loss():
    pass



if __name__ == '__main__':
    args = parser.parse_args()
    noise = torch.rand((1,3,224,224), requires_grad=True, device='cpu')
    loss_fn = nn.CrossEntropyLoss()
    net = models.mobilenet_v2(pretrained=True)
    optimizer = optim.Adam([noise], lr=0.01, betas=[0.5, 0.9], eps = 1e-8)
    net.eval()
    writer = SummaryWriter(f'logs/{args.exp_name}')
    target = torch.tensor([1])
    for i in tqdm(range(100)):
        outputs = net(noise)
        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()
        if i%10 == 0:
            
    print()
    print()

