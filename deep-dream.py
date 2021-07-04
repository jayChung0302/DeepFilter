import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.utils as vutils

import numpy as np
import mypy
import random
import sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
# import scipy.ndimage as nd
from utils import preprocess, deprocess, clip, FeatureRegHook

random.seed(0)
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, help='experiment name', default='dryrun')
parser.add_argument('--batch_size', type=int, help='batch size', default='32')
parser.add_argument('--lr', type=float, help='learning rate', default='0.1')
parser.add_argument('--correlate_scale', type=float, help='', default='2.5e-5')
parser.add_argument('--bn_reg_scale', type=float, help='', default='1e2')
parser.add_argument('--l2_scale', type=float, help='', default='1e-3')
parser.add_argument('--target_class', type=int, help='', default='-1')
parser.add_argument('--num_epoch', type=int, help='', default='5000')
parser.add_argument('--num_classes', type=int, help='', default='1000')
# parser.add_argument('--use_amp', type=int, help='', default='False')



args = parser.parse_args()

def dream(image, model, iterations, lr):
    """ Updates the image to maximize outputs for n iterations """
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor
    image = Variable(Tensor(image), requires_grad=True)
    for i in range(iterations):
        model.zero_grad()
        out = model(image)
        loss = out.norm()
        loss.backward()
        avg_grad = np.abs(image.grad.data.cpu().numpy()).mean()
        norm_lr = lr / avg_grad
        image.data += norm_lr * image.grad.data
        image.data = clip(image.data)
        image.grad.data.zero_()
    return image.cpu().data.numpy()

def deep_dream(image, model, iterations, lr, octave_scale, num_octaves):
    """ Main deep dream method """
    image = preprocess(image).unsqueeze(0).cpu().data.numpy()

    # Extract image representations for each octave
    octaves = [image]
    for _ in range(num_octaves - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1 / octave_scale, 1 / octave_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(tqdm.tqdm(octaves[::-1], desc="Dreaming")):
        if octave > 0:
            # Upsample detail to new octave dimension
            detail = nd.zoom(detail, np.array(octave_base.shape) / np.array(detail.shape), order=1)
        # Add deep dream detail from previous octave to new base
        input_image = octave_base + detail
        # Get new deep dream image
        dreamed_image = dream(input_image, model, iterations, lr)
        # Extract deep dream details
        detail = dreamed_image - octave_base

    return deprocess(dreamed_image)

def main():
    #TODO
    # 1. Add scheduler
    # 2. Add mods
    
    lim_0, lim_1 = 2, 2
    noise = torch.rand((args.batch_size, 3, 224, 224), requires_grad=True, device='cuda')
    loss_fn = nn.CrossEntropyLoss()
    net = models.resnet50(pretrained=True).cuda()
    optimizer = optim.Adam([noise], lr=args.lr, betas=[0.5, 0.9], eps = 1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, \
            eta_min=0)
    net.eval()
    writer = SummaryWriter(f'logs/{args.exp_name}')
    if args.target_class != -1:
        target = torch.tensor([args.target_class] * args.batch_size, device='cuda')
    else:
        target = torch.tensor([i % args.num_classes for i in range(args.batch_size)], device='cuda')

    loss_r_feature_layers = []
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(FeatureRegHook(module))

    for i in tqdm(range(args.num_epoch)):
        net.zero_grad()
        optimizer.zero_grad()

        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        inputs_jit = torch.roll(noise, shifts=(off1,off2), dims=(2,3))

        outputs = net(inputs_jit)
        loss = loss_fn(outputs, target)
        
        # nearby pixels should be correlated
        diff1 = inputs_jit[:,:,:,:-1] - inputs_jit[:,:,:,1:]
        diff2 = inputs_jit[:,:,:-1,:] - inputs_jit[:,:,1:,:]
        diff3 = inputs_jit[:,:,1:,:-1] - inputs_jit[:,:,:-1,1:]
        diff4 = inputs_jit[:,:,:-1,:-1] - inputs_jit[:,:,1:,1:]

        loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
        loss = loss + args.correlate_scale * loss_var
        loss_distr = sum([mod.r_feature for mod in loss_r_feature_layers])
        loss = loss + args.bn_reg_scale * loss_distr # best for noise before BN
        loss = loss + args.l2_scale * torch.norm(inputs_jit, 2)
        loss.backward()
        optimizer.step()
        inputs_grid = torchvision.utils.make_grid(noise, normalize=True)
        if i%100 == 0:
            writer.add_scalar('training loss', loss, i)
            writer.add_image('optimized_noise', inputs_grid, i)
            vutils.save_image(noise.data.clone(),
                              './logs/{}/output_{}.png'.format(args.exp_name, i),
                              normalize=True, scale_each=True, nrow=10)

if __name__ == '__main__':
    main()
