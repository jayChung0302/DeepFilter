import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils as vutils


import numpy as np
import cv2
import mypy
import random
import sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
import PIL.Image as Image
# import scipy.ndimage as nd
from utils import preprocess, deprocess, clip, BatchHook, GroupHook, InstanceHook

random.seed(0)
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, help='experiment name', default='dryrun')
parser.add_argument('--on_image', help='', default=False, action='store_true')
parser.add_argument('--batch_size', type=int, help='batch size', default='32')
parser.add_argument('--lr', type=float, help='learning rate', default='0.1')
parser.add_argument('--correlate_scale', type=float, help='', default='2.5e-5')
parser.add_argument('--bn_reg_scale', type=float, help='', default='1e2')
parser.add_argument('--l2_scale', type=float, help='', default='1e-3')
parser.add_argument('--target_class', type=int, help='', default='-1')
parser.add_argument('--num_epoch', type=int, help='', default='5000')
parser.add_argument('--num_classes', type=int, help='', default='1000')
parser.add_argument('--use_amp', help='', default=False, action='store_true')


args = parser.parse_args()

def dream(image, model, iterations, lr):
    """ Updates the image to maximize outputs for n iterations """
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

def get_octave(image:torch.tensor, octave_scale:float=1.4, num_octaves:int=10):
    images_in_octave = [image]
    scale_factor = 1
    for _ in range(num_octaves):
        scale_factor = round(scale_factor / octave_scale)
        image = F.interpolate(image, scale_factor=scale_factor, interpolation='nearest', \
            recompute_scale_factor=True)
        images_in_octave.append(image)
    return images_in_octave


def deep_dream(image, model, iterations, lr, octave_scale, num_octaves):
    """ Main deep dream method """
    # Extract image representations for each octave
    octaves = get_octave(image, octave_scale, num_octaves)

    detail = torch.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(tqdm.tqdm(octaves[::-1], desc="Dreaming")):
        if octave > 0:
            # Upsample detail to new octave dimension
            detail = F.interpolate(detail, (octave_base.shape[-2], octave_base.shape[-1]), \
                interpolation='nearest', recompute_scale_factor=True)
        # Add deep dream detail from previous octave to new base
        input_image = octave_base + detail
        # Get new deep dream image
        dreamed_image = dream(input_image, model, iterations, lr)
        # Extract deep dream details
        detail = dreamed_image - octave_base
    return dreamed_image
    # return deprocess(dreamed_image)

def main():
    if args.use_amp:
        from apex import amp

    if args.on_image:
        img = Image.open('./img/Lenna.png')
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)
        ])
        img = transform(img).unsqueeze(0).repeat(args.batch_size, 1, 1, 1)\
            .clone().detach().requires_grad_(True)
        # input_org = torch.tensor(transform(img), device='cuda', requires_grad=True, pin_memory=True): deprecated
        
        input = img.cuda()
    else:
        input = torch.rand((args.batch_size, 3, 224, 224), requires_grad=True, device='cuda')

    lim_0, lim_1 = 2, 2
    loss_fn = nn.CrossEntropyLoss()
    net = models.resnet50(pretrained=True).cuda()
    optimizer = optim.Adam([input], lr=args.lr, betas=[0.5, 0.9], eps = 1e-8)
    net, optimizer = amp.initialize(net, optimizer, opt_level="O1", loss_scale='dynamic')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    net.eval()
    writer = SummaryWriter(f'logs/{args.exp_name}')
    if args.target_class != -1:
        target = torch.tensor([args.target_class] * args.batch_size, device='cuda')
    else:
        target = torch.tensor([i % args.num_classes for i in range(args.batch_size)], device='cuda')

    loss_r_feature_layers = []
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(BatchHook(module))
        if isinstance(module, nn.GroupNorm):
            loss_r_feature_layers.append(GroupHook(module))
        if isinstance(module, nn.InstanceNorm2d):
            loss_r_feature_layers.append(InstanceHook(module))

    for i in tqdm(range(args.num_epoch)):
        net.zero_grad()
        optimizer.zero_grad()

        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        inputs_jit = torch.roll(input, shifts=(off1,off2), dims=(2,3))

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
        loss = loss + args.bn_reg_scale * loss_distr # best for input before BN
        loss = loss + args.l2_scale * torch.norm(inputs_jit, 2)
        if args.use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()
        inputs_grid = torchvision.utils.make_grid(input, normalize=True)
        if i%100 == 0:
            writer.add_scalar('training loss', loss, i)
            writer.add_image('optimized_input', inputs_grid, i)
            vutils.save_image(input.data.clone(),
                              './logs/{}/output_{}.png'.format(args.exp_name, i),
                              normalize=True, scale_each=True, nrow=10)

if __name__ == '__main__':
    main()
