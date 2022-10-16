import torch
import torch.nn.functional as F
import numpy as np
import math
import random
import time


# http://d2l.ai/_modules/d2l/torch.html#Timer
# Defined in file: ./chapter_linear-networks/linear-regression.md
class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()


    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]


    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)


    def sum(self):
        """Return the sum of time."""
        return sum(self.times)


    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


# https://github.com/GuanshuoXu/RSNA-STR-Pulmonary-Embolism-Detection/blob/main/trainval/2nd_level/seresnext50_128.py
# line 59-72
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def apply_window(image, WL, WW):
    upper, lower = WL + WW // 2, WL - WW // 2
    X = np.clip(image.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X * 255.0).astype('uint8')
    return X


# Data Augmentation
# https://github.com/deep-learning-with-pytorch/dlwpt-code/blob/master/p2ch12/dsets.py
# line 160 - 214
def apply_transform_3d(image_tensor, transform_3d=None):
    image_tensor = image_tensor.unsqueeze(0)  # 1 x C x D x H x W
    
    transform_t = torch.eye(4)

    if 'crop' in transform_3d:
        original_size = image_tensor.shape[3]
        cropped_size = transform_3d['crop']
        gap = original_size - cropped_size
        h_gap = random.randint(0, gap)
        w_gap = random.randint(0, gap)
        image_tensor = image_tensor[..., h_gap:(h_gap+cropped_size), w_gap:(w_gap+cropped_size)]

    if 'rotate' in transform_3d:
        random_angle_degree = random.random() * 360
        if random_angle_degree <= transform_3d['rotate']:
            angle_rad = random_angle_degree / 360 * math.pi * 2
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)

            rotation_t = torch.tensor([
                [c, -s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ])

            transform_t @= rotation_t

    affine_t = F.affine_grid(
            transform_t[:3].unsqueeze(0).to(torch.float32),
            image_tensor.size(),
            align_corners=False,
        )

    image_tensor = F.grid_sample(
            image_tensor,
            affine_t,
            padding_mode='border',
            align_corners=False,
        )
    return image_tensor.squeeze(0)  # C x D x H x W
