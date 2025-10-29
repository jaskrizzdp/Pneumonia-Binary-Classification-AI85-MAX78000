###################################################################################################
#
# Chest X-Ray Pneumonia Dataset Loader
#
###################################################################################################

import os
import sys

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import albumentations as album
import cv2

import ai8x

input_dims = (64, 64, 1)
num_classes = 2

class ChestXRay(Dataset):
    """
    Pneumonia vs Normal Chest X-Ray Dataset.

    Folder structure expected:
      data/chest_xray/train/normal
      data/chest_xray/train/pneumonia
      data/chest_xray/val/normal
      data/chest_xray/val/pneumonia
      data/chest_xray/test/normal
      data/chest_xray/test/pneumonia
    """

    labels = ['normal', 'pneumonia']
    label_to_id_map = {k: v for v, k in enumerate(labels)}
    label_to_folder_map = {'normal': 'normal', 'pneumonia': 'pneumonia'}

    def __init__(self, root_dir, d_type, transform=None,
                 resize_size=(128, 128), augment_data=False):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, 'chest_xray', d_type)

        if not self.__check_data_exist():
            self.__print_download_manual()
            sys.exit("Dataset not found!")

        self.__get_image_paths()

        self.album_transform = None
        if d_type == 'train' and augment_data:
            self.album_transform = album.Compose([
                album.GaussNoise(var_limit=(1.0, 20.0), p=0.25),
                album.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                album.ColorJitter(p=0.5),
                album.SmallestMaxSize(max_size=int(1.2*min(resize_size))),
                album.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                album.RandomCrop(height=resize_size[0], width=resize_size[1]),
                album.HorizontalFlip(p=0.5),
                album.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
            ])
        else:
            self.album_transform = album.Compose([
                album.SmallestMaxSize(max_size=int(1.2*min(resize_size))),
                album.CenterCrop(height=resize_size[0], width=resize_size[1]),
                album.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
            ])

        self.transform = transform

    def __check_data_exist(self):
        return os.path.isdir(self.data_dir)

    def __print_download_manual(self):
        print("******************************************")
        print("Please download the Chest X-Ray Pneumonia dataset from Kaggle:")
        print("https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("Unzip and place the folders under 'data/chest_xray/'")
        print("Make sure your folders are named exactly: 'normal' and 'pneumonia'")
        print("******************************************")

    def __get_image_paths(self):
        self.data_list = []

        for label in self.labels:
            image_dir = os.path.join(self.data_dir, self.label_to_folder_map[label])
            for file_name in sorted(os.listdir(image_dir)):
                file_path = os.path.join(image_dir, file_name)
                if os.path.isfile(file_path):
                    self.data_list.append((file_path, self.label_to_id_map[label]))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        label = torch.tensor(self.data_list[index][1], dtype=torch.int64)

        image_path = self.data_list[index][0]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.album_transform:
            image = self.album_transform(image=image)["image"]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_chestxray_dataset(data, load_train, load_test):
    """
    Load the Chest X-Ray dataset.
    Returns each datasample in 128x128 size.
    """
    (data_dir, args) = data

    transform = transforms.Compose([
        transforms.ToTensor(),
        ai8x.normalize(args=args),
    ])

    if load_train:
        train_dataset = ChestXRay(root_dir=data_dir, d_type='train',
                                  transform=transform, augment_data=True)
    else:
        train_dataset = None

    if load_test:
        test_dataset = ChestXRay(root_dir=data_dir, d_type='test', transform=transform)
    else:
        test_dataset = None

    return train_dataset, test_dataset


# datasets = [
#     {
#         'name': 'chest_xray',
#         'input': (3, 128, 128),
#         'output': ('normal', 'pneumonia'),
#         'loader': get_chestxray_dataset,
#     },

datasets = [
    {
        'name': 'pneumonia',
        'input_dims': (64, 64, 1),
        'output': ('normal', 'pneumonia'),
        'loader': get_chestxray_dataset,
    },
]


]
