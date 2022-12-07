import os
import shutil
import math

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config


class DirectorySetup:
    def __init__(self, raw_path: str, test_size: float=0.1, val_size: float=0.1) -> None:
        self.raw_path = os.path.join(raw_path)
        self.test_size = test_size
        self.val_size = val_size
        self.train_size = 1 - (test_size + val_size)

        self.n_files = 0
        self.n_classes = 0

        self.data_dir_names = ["train", "test", "valid"]
        self.data_dir_props = [self.train_size, self.test_size, self.val_size]

    def setup(self, preprocessed_name: str):
        self.preprocessed_dir = preprocessed_name
        if not os.path.isdir(self.preprocessed_dir):
            os.mkdir(self.preprocessed_dir)

        self.classes = os.listdir(self.raw_path)
        self.n_classes = len(self.classes)

        self._create_directories()
        self._populate_directories()

    def _create_directories(self):

        for data_type in self.data_dir_names:
            type_path = os.path.join(self.preprocessed_dir, data_type)
            if not os.path.isdir(type_path): os.mkdir(type_path)

            for cls in self.classes:
                ## create the data class path and check if a dir is already exists
                cls_path = os.path.join(self.preprocessed_dir, data_type, cls)
                if not os.path.isdir(cls_path): os.mkdir(cls_path)

    def _get_raw_class_path(self, cls: str):
        return os.path.join(self.raw_path, cls)

    def _get_class_path(self, typ: str, cls: str):
        return os.path.join(self.preprocessed_dir, typ, cls)

    def _get_class_size(self, typ, cls, prop):
        return int(math.floor(len(os.listdir(os.path.join(self.raw_path, cls))) * prop))

    def _populate_directories(self):

        self.raw_images = {
            cls: os.listdir(self._get_raw_class_path(cls))
                 for cls in self.classes
        }

        self._sizes = {
            typ: {
                cls: self._get_class_size(typ, cls, prop)
                     for cls in self.classes
            } for typ, prop in zip(self.data_dir_names, self.data_dir_props)
        }

        print(self._sizes)

        self._images = {
            typ: {
                cls: []
                     for cls in self.classes
            } for typ in self.data_dir_names
        }

        start = 0
        for typ in self.data_dir_names:
            start = 0
            for cls in self.classes:
                size = self._sizes[typ][cls]
                start = 0

                for i in range(start, start+size):
                    try:
                        img = self.raw_images[cls][i]
                        src = os.path.join(self._get_raw_class_path(cls), img)
                        dst = os.path.join(self._get_class_path(typ, cls), img)
                        self._images[typ][cls].append(img)

                        shutil.copy(src, dst)

                    except Exception as e:
                        # print(str(e))
                        pass

            start += size

        self.n_files = sum( [sum([ len(self._images[typ][cls]) for cls in self.classes ]) for typ in self.data_dir_names] )

    def __repr__(self):
        return "Found {} belongs to {} Classes".format(self.n_files, self.n_classes)


def create_dataloaders(train_dir: str, valid_dir: str, test_dir: str):

    train_ds = datasets.ImageFolder(
        train_dir,
        transform=transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.RandomRotation(45.),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]),
    )

    valid_ds = datasets.ImageFolder(
        valid_dir,
        transform=transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),

        ]),
    )

    test_ds = datasets.ImageFolder(
        test_dir,
        transform=transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]),
    )

    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    test_dl  = DataLoader(test_ds , batch_size=config.BATCH_SIZE, shuffle=True)

    return train_dl, valid_dl, test_dl

