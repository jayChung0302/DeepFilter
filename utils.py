import torch
from torch.nn.modules.normalization import GroupNorm
import torchvision.transforms as transforms
import cv2
import numpy as np

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
preprocess = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize(mean, std),
    ])

# Mosaic
def mosaic(img, ratio=0.1):
    small = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

def mosaic_area(img, x, y, width, height, ratio=0.1):
    dst = img.copy()
    dst[y:y + height, x:x + width] = mosaic(dst[y:y + height, x:x + width], ratio)
    return dst

def deprocess(image_np):
    image_np = image_np.squeeze().transpose(1, 2, 0)
    image_np = image_np * std.reshape((1, 1, 3)) + mean.reshape((1, 1, 3))
    image_np = np.clip(image_np, 0.0, 255.0)
    return image_np


def clip(image_tensor):
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[0, c] = torch.clamp(image_tensor[0, c], -m / s, (1 - m) / s)
    return image_tensor

class FeatureRegHook:
    
    def __init__(self, module: torch.nn):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        raise NotImplementedError

    def close(self):
        self.hook.remove()

class BatchHook(FeatureRegHook):
    '''
    https://arxiv.org/abs/1912.08795
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module: torch.nn):
        super(BatchHook, self).__init__(module)

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        input = input[0].permute(1, 0, 2, 3).contiguous().view(nch, -1)
        mean = input.mean(dim=1)
        var = input.var(dim=1, unbiased=False)

        r_feature = torch.norm(module.running_var - var, 2) + torch.norm(
            module.running_mean - mean, 2)

        self.r_feature = r_feature
        

class InstanceHook(FeatureRegHook):
    def __init__(self, module: torch.nn):
        super(InstanceHook, self).__init__(module)
    
    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        input = input[0].contiguous().view(nch, -1)
        mean = input.mean(dim=1)
        var = input.var(dim=1, unbiased=False)

        r_feature = torch.norm(module.running_var - var, 2) + torch.norm(
            module.running_mean - mean, 2)

        self.r_feature = r_feature

class GroupHook(FeatureRegHook):
    def __init__(self, module: torch.nn):
        super(GroupHook, self).__init__(module)
        self.ngroups = module.num_groups
    
    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        input = input[0].contiguous().view(nch, self.ngroups, -1)
        mean = input.mean(dim=2)
        var = input.var(dim=2, unbiased=False)

        r_feature = torch.norm(module.running_var - var, 2) + torch.norm(
            module.running_mean - mean, 2)

        self.r_feature = r_feature

class CamHook(FeatureRegHook):
    def __init__(self, module: torch.nn):
        super(CamHook, self).__init__(module)
    
    def hook_fn(self, module, input, output):
        r_feature = output[0].contiguous() + input[0].contiguous()
        
        self.r_feature = r_feature
        
if __name__ == '__main__':
    img = cv2.imread('img/Lenna.png')
    dst_area = mosaic_area(img, 220, 230, 150, 150)
    cv2.imwrite('cv-mosaic.jpeg', dst_area)
    