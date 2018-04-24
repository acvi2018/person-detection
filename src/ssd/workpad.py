import os
import sys
module_path = os.path.abspath(os.path.join('/home/vijin/iith/project/workpad/ssd.pytorch'))
if module_path not in sys.path:
    sys.path.append(module_path)
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import pickle
import numpy as np

from ssd import build_ssd
from data import v2, v1, AnnotationTransform, MiniDroneDataset, detection_collate, OkutamaDataset
from utils.augmentations import SSDAugmentation


net = build_ssd('test', 300, 2)    # initialize SSD
net.load_weights('/home/vijin/iith/project/workpad/ssd.pytorch/weights/ssd300_2class40000.pth')

vgg = net.vgg
extras = net.extras
loc = net.loc
conf = net.conf


for i, l in enumerate(conf):
	print(i, '->', l)



