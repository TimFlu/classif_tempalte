import torch
from torch.utils.data import Dataset
import os
import logging
logger = logging.getLogger(__name__)
import pandas as pd
import numpy as np
from PIL import Image
from torchvision.datasets.folder import default_loader
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Subset

class CustomDataset(Dataset):
    def __init__(self, root, csv_path, targets, transform=None):
        self.root_path = root
        self.csv_file = csv_path
        self.transforms = transform
        self.targets = targets

        # Load the path to img and corresponding labels from csv. 
        self.label_data = pd.read_csv(csv_path)
        self.path_to_images = self.label_data['path_to_image'].values
        self.labels = self.label_data[targets].replace({-1:0}).fillna(0).values
        
        # Check if the dataset is binary or multiclass
        self.binary = len(self.targets) == 1
        
        # Calculate weights
        if self.binary:
            logger.info('datasets.py: Binary classification detected')
            self.case = 'binary'
            # Binary classification weights
            positive_counts = self.labels.sum()  # Total positive labels
            total_samples = len(self.labels)
            self.class_weights = ([
                total_samples / (2.0 * positive_counts)]
                if positive_counts > 0
                else [1.0]
            )  # Handle division by zero
        else:
            # Multiclass classification weights
            if self.labels.ndim == 1:  # Ensure labels are class indices
                logger.info('datasets.py: Multiclass classification detected')
                self.case = 'multi_class'
                class_counts = np.bincount(self.labels)
                self.class_weights = [
                    max(class_counts) / cls if cls > 0 else 1.0
                    for cls in class_counts
                ]  # Handle division by zero
            else:
                # Multi-label classification weights
                logger.info('datasets.py: Multi-label classification detected')
                self.case = 'multi_label'
                positive_counts = self.labels.sum(axis=0)  # Sum positives per class
                total_samples = self.labels.shape[0]
                self.class_weights = np.array(
                    [
                        total_samples / (2.0 * count) if count > 0 else 1.0
                        for count in positive_counts
                    ]
                )  # Handle division by zero



    def __len__(self):
        # Return the size of the dataset
        return len(self.label_data)

    def __getitem__(self, idx):
        path_to_image = os.path.join(self.root_path, self.path_to_images[idx])

        # Either jpg or png
        image = default_loader(path_to_image.replace(".jpg", ".png"))
        
        if self.transforms is not None:
            image = self.transforms(image)

        return image, self.labels[idx]



class Cifar10(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        # Load CIFAR-10 data using torchvision's CIFAR10 class
        self.cifar_data = CIFAR10(root=root, train=train, download=download, transform=transform)

        self.data = self.cifar_data.data
        self.labels = np.array(self.cifar_data.targets)

        self.classes = self.cifar_data.classes
        self.binary = len(self.classes) == 2
        
        # Calculate weights
        if self.binary:
            logger.info('datasets.py: Binary classification detected')
            self.case = 'binary'
            # Binary classification weights
            positive_counts = self.labels.sum()  # Total positive labels
            total_samples = len(self.labels)
            self.class_weights = ([
                total_samples / (2.0 * positive_counts)]
                if positive_counts > 0
                else [1.0]
            )  # Handle division by zero
        else:
            # Multiclass classification weights
            if self.labels.ndim == 1:  # Ensure labels are class indices
                logger.info('datasets.py: Multiclass classification detected')
                self.case = 'multi_class'
                class_counts = np.bincount(self.labels)
                self.class_weights = [
                    max(class_counts) / cls if cls > 0 else 1.0
                    for cls in class_counts
                ]  # Handle division by zero
            else:
                # Multi-label classification weights
                logger.info('datasets.py: Multi-label classification detected')
                self.case = 'multi_label'
                positive_counts = self.labels.sum(axis=0)  # Sum positives per class
                total_samples = self.labels.shape[0]
                self.class_weights = np.array(
                    [
                        total_samples / (2.0 * count) if count > 0 else 1.0
                        for count in positive_counts
                    ]
                )  # Handle division by zero

    def __len__(self):
        # Return the size of the dataset
        return len(self.cifar_data)

    def __getitem__(self, idx):
        # Get the data item (image and label) at the specified index
        image, label = self.cifar_data[idx]
        return image, label

class Cifar100(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        # Load CIFAR-10 data using torchvision's CIFAR10 class
        self.cifar_data = CIFAR100(root=root, train=train, download=download, transform=transform)

        self.data = self.cifar_data.data
        self.labels = np.array(self.cifar_data.targets)

        self.classes = self.cifar_data.classes
        self.binary = len(self.classes) == 2
        
        # Calculate weights
        if self.binary:
            logger.info('datasets.py: Binary classification detected')
            self.case = 'binary'
            # Binary classification weights
            positive_counts = self.labels.sum()  # Total positive labels
            total_samples = len(self.labels)
            self.class_weights = ([
                total_samples / (2.0 * positive_counts)]
                if positive_counts > 0
                else [1.0]
            )  # Handle division by zero
        else:
            # Multiclass classification weights
            if self.labels.ndim == 1:  # Ensure labels are class indices
                logger.info('datasets.py: Multiclass classification detected')
                self.case = 'multi_class'
                class_counts = np.bincount(self.labels)
                self.class_weights = [
                    max(class_counts) / cls if cls > 0 else 1.0
                    for cls in class_counts
                ]  # Handle division by zero
            else:
                # Multi-label classification weights
                logger.info('datasets.py: Multi-label classification detected')
                self.case = 'multi_label'
                positive_counts = self.labels.sum(axis=0)  # Sum positives per class
                total_samples = self.labels.shape[0]
                self.class_weights = np.array(
                    [
                        total_samples / (2.0 * count) if count > 0 else 1.0
                        for count in positive_counts
                    ]
                )  # Handle division by zero

    def __len__(self):
        # Return the size of the dataset
        return len(self.cifar_data)

    def __getitem__(self, idx):
        # Get the data item (image and label) at the specified index
        image, label = self.cifar_data[idx]
        return image, label

class Cifar100Binary(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        # Load CIFAR-10 data using torchvision's CIFAR10 class
        self.cifar_data = CIFAR100(root=root, train=train, download=download, transform=transform)

        self.data = self.cifar_data.data
        self.labels = np.array(self.cifar_data.targets)
        indices = np.where((self.labels == 0) | (self.labels == 1))[0]
        self.data = self.data[indices]
        self.labels = self.labels[indices]

        self.transform = transform

        self.classes = self.cifar_data.classes[:2]
        self.binary = len(self.classes) == 2
        
        # Calculate weights
        if self.binary:
            logger.info('datasets.py: Binary classification detected')
            self.case = 'binary'
            # Binary classification weights
            positive_counts = self.labels.sum()  # Total positive labels
            total_samples = len(self.labels)
            self.class_weights = ([
                total_samples / (2.0 * positive_counts)]
                if positive_counts > 0
                else [1.0]
            )  # Handle division by zero
        else:
            # Multiclass classification weights
            if self.labels.ndim == 1:  # Ensure labels are class indices
                logger.info('datasets.py: Multiclass classification detected')
                self.case = 'multi_class'
                class_counts = np.bincount(self.labels)
                self.class_weights = [
                    max(class_counts) / cls if cls > 0 else 1.0
                    for cls in class_counts
                ]  # Handle division by zero
            else:
                # Multi-label classification weights
                logger.info('datasets.py: Multi-label classification detected')
                self.case = 'multi_label'
                positive_counts = self.labels.sum(axis=0)  # Sum positives per class
                total_samples = self.labels.shape[0]
                self.class_weights = np.array(
                    [
                        total_samples / (2.0 * count) if count > 0 else 1.0
                        for count in positive_counts
                    ]
                )  # Handle division by zero
    def __len__(self):
        # Return the size of the dataset
        return len(self.labels)

    def __getitem__(self, idx):
        # Get the data item (image and label) at the specified index
        image = self.data[idx]
        label = self.labels[idx]

        label = torch.tensor(label, dtype=torch.float32).view(1)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label