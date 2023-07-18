import random
import os

import matplotlib.pyplot as plt
import numpy as np

from skimage import io
from skimage.transform import resize

import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

import torchvision
from torchvision import ops

from utils import *
from model import *


class ObjectDetectionDataset(Dataset):
    """
    A Pytorch Dataset Class to load the images and their corresponding annotations.
    :returns:
        images (torch.Tensor): Size (B, C, H. W)
        gt_bboxes (torch.Tensor): Size (B, max_objects, 4)
        gt_classes (torch.Tensor): Size (B, max_objects)
    """
    def __init__(self, annotation_path, img_dir, img_size, name2idx):
        self.annotaion_path = annotation_path
        self.img_dir = img_dir
        self.img_size = img_size
        self.name2idx = name2idx
        self.img_data_all, self.gt_bboxes_all, self.gt_classes_all = self.get_data()

    def __len__(self):
        return self.img_data_all.size(dim=0)

    def __getitem__(self, idx):
        return self.img_data_all[idx], self.gt_bboxes_all[idx], self.gt_classes_all[idx]

    def get_data(self):
        img_data_all = []
        gt_idx_all = []

        gt_boxes_all, gt_classes_all, img_paths = parse_annotation(self.annotaion_path, self.img_dir, self.img_size)

        for i, img_path in enumerate(img_paths):
            if (not img_path) or (not os.path.exists(img_path)):
                continue

            img = io.imread(img_path)
            img = resize(img, self.img_size)

            img_tensor = torch.from_numpy(img).permute(2, 0, 1)

            gt_classes = gt_classes_all[i]
            gt_idx = torch.Tensor([self.name2idx[name] for name in gt_classes])

            img_data_all.append(img_tensor)
            gt_idx_all.append(gt_idx)

        gt_bboxes_pad = pad_sequence(gt_boxes_all, batch_first=True, padding_value=-1)
        gt_classes_pad = pad_sequence(gt_idx_all, batch_first=True, padding_value=-1)

        img_data_stacked = torch.stack(img_data_all, dim=0)

        return img_data_stacked.to(dtype=torch.float32), gt_bboxes_pad, gt_classes_pad


img_width = 640
img_height = 480
annotation_path = "data/annotations.xml"
image_dir = os.path.join("data", "images")
name2idx = {"pad": -1, "camel": 0, "bird": 1}
idx2name = {v:k for k, v in name2idx.items()}

od_dataset = ObjectDetectionDataset(annotation_path, image_dir, (img_height, img_width), name2idx)
od_dataloader = DataLoader(od_dataset, batch_size=2)


# Grabbing a batch for demonstration
for img_batch, gt_bboxes_batch, gt_classes_batch in od_dataloader:
    img_data_all = img_batch
    gt_bboxes_all = gt_bboxes_batch
    gt_classes_all = gt_classes_batch
    break

img_data_all = img_data_all[:2]
gt_bboxes_all = gt_bboxes_all[:2]
gt_classes_all = gt_classes_all[:2]

if __name__ == "__main__":
    # Display Images and Bounding Boxes
    gt_class_1 = gt_classes_all[0].long()
    gt_class_1 = [idx2name[idx.item()] for idx in gt_class_1]

    gt_class_2 = gt_classes_all[1].long()
    gt_class_2 = [idx2name[idx.item()] for idx in gt_class_2]

    nrows, ncols = (1, 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
    fig, axes = display_img(img_data_all, fig, axes)
    fig, _ = display_bbox(gt_bboxes_all[0], fig, axes[0], classes=gt_class_1)
    fig, _ = display_bbox(gt_bboxes_all[1], fig, axes[1], classes=gt_class_2)
    plt.show()




























