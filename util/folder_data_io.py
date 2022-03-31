# -*- coding: utf-8 -*- 
# 
# @Time    : 2019/7/6
# @Author  : cyliu7
# 
# Simple IO
# 

import torch.utils.data as data

from PIL import Image
import os
import os.path
import random


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir,test_img_list, class_to_idx):
    images = []
    dir = os.path.abspath(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for fname in os.listdir(d):
            if fname in test_img_list:
                path = os.path.join(d, fname)
                item = (path, class_to_idx[target])
                images.append(item)
    return images


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root,list, transform=None, target_transform=None):
        classes, class_to_idx = find_classes(root)
        # cyliu7 edited
        #samples = make_dataset(root, class_to_idx, extensions)
        samples = make_dataset(root,list, class_to_idx)
        assert len(samples) > 0

        self.root = root
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
         
        sample = Image.open(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

    def __len__(self):
        return len(self.samples)

class DatasetList(data.Dataset):
    """A generic data loader where the samples are arranged in this way:
    """

    def __init__(self, samples, transform=None, target_transform=None):

        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
         
        sample = Image.open(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

    def __len__(self):
        return len(self.samples)


def get_trainval_dataset(root,list, transform=None, target_transform=None, val_transform=None, val_target_transform=None, train_ratio = 0.7):
    classes, class_to_idx = find_classes(root)

    train_images = []
    val_images = []
    dir = os.path.abspath(root)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        cur_list = os.listdir(d)
        cur_list_split = [i for i in cur_list if i not in list]
        random.shuffle(cur_list_split)
        train_num = int(train_ratio * len(cur_list_split)) + 1
        for fname in cur_list_split[:train_num]:
            path = os.path.join(d, fname)
            item = (path, class_to_idx[target])
            train_images.append(item)
        for fname in cur_list_split[train_num:]:
            path = os.path.join(d, fname)
            item = (path, class_to_idx[target])
            val_images.append(item)

    return DatasetList(train_images, transform, target_transform), DatasetList(val_images, transform, target_transform)