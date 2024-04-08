"""
this version is modified from https://github.com/dkumazaw/mobilenetv3-pytorch/
"""
import numpy as np
import os
import shutil
import logging

import torch

BEST_MODEL_PATH = 'model_best.pt'


class AveTracker:
    def __init__(self):
        self.average = 0
        self.sum = 0
        self.counter = 0

    def update(self, value, n):
        self.sum += value * n
        self.counter += n
        self.average = self.sum / self.counter


class Cutout:
    """Applies cutout to input image"""

    def __init__(self, length):
        assert length >= 0
        self.length = length

    def __call__(self, img):
        if self.length > 0:
            h, w = img.shape[1], img.shape[2]
            mask = img.new_ones((h, w), dtype=torch.bool)
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0
            mask = mask.expand_as(img)
            img *= mask
        return img


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state_dict, is_best, savedir, epoch):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    filename = os.path.join(savedir, 'checkpoint_ep{}.pt'.format(epoch))
    torch.save(state_dict, filename)

    if is_best:
        best_filename = os.path.join(savedir, BEST_MODEL_PATH)
        shutil.copyfile(filename, best_filename)


def load_best_model_state_dict(savedir):
    """Loads best model's state dict"""
    return torch.load(os.path.join(savedir, BEST_MODEL_PATH))


def create_logger(filename: str, logger_prefix: str):
    logging.basicConfig(
        filename=filename,
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG
    )
    logger = logging.getLogger(logger_prefix)
    consoleHandler = logging.StreamHandler()
    logger.addHandler(consoleHandler)

    return logger

