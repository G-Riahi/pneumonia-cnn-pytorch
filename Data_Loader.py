import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PneumoniaMNIST


class PneumoniaDataModule:

    def __init__(
        self,
        batch_size=32,
        image_size=128,
    ):
        self.batch_size = batch_size
        self.image_size = image_size

        self.mean = None
        self.std = None

        self.transform = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.pos_weight = None

    def compute_pixel_mean_std(self):

        temp_train = PneumoniaMNIST(
            split="train",
            transform=transforms.ToTensor(),
            download=True,
            size=self.image_size,
            as_rgb=False
        )

        sum_pixels = 0
        sum_pixels_squared = 0

        for i, _ in temp_train:
            sum_pixels += i.sum()
            sum_pixels_squared += (i**2).sum()

        total_pixels = len(temp_train)*self.image_size*self.image_size

        self.mean = sum_pixels/total_pixels
        variance = (sum_pixels_squared/total_pixels)-(self.mean**2) #Konig-huygens formula
        self.std = torch.sqrt(variance)

    def update_transform(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[self.mean],
                std=[self.std]
            )
        ])

    def setup(self):

        # -------------------------
        # Updating mean, std and transform
        # -------------------------

        self.compute_pixel_mean_std()
        self.update_transform()

        # -------------------------
        # Datasets
        # -------------------------

        self.train_dataset = PneumoniaMNIST(
            split="train",
            transform=self.transform,
            download=True,
            size=self.image_size,
            as_rgb=False
        )

        self.val_dataset = PneumoniaMNIST(
            split="val",
            transform=self.transform,
            download=True,
            size=self.image_size,
            as_rgb=False
        )

        self.test_dataset = PneumoniaMNIST(
            split="test",
            transform=self.transform,
            download=True,
            size=self.image_size,
            as_rgb=False
        )

        # -------------------------
        # DataLoaders
        # -------------------------

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        # -------------------------
        # Class imbalance
        # -------------------------

        labels = self.train_dataset.labels.squeeze()

        negative_count = np.sum(labels == 0)
        positive_count = np.sum(labels == 1)

        self.pos_weight = negative_count / positive_count

    def get_class_distribution(self):

        labels = self.train_dataset.labels.squeeze()
        
        return {
            "normal": int(np.sum(labels == 0)),
            "pneumonia": int(np.sum(labels == 1))
        }