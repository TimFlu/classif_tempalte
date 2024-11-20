import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from PIL import Image
from torchvision.datasets.folder import default_loader


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

        self.binary = len(self.targets) == 1
        
        # Calculate weights for each class
        self.class_counts = np.unique(self.labels, return_counts=True)[1]
        self.class_weights = [max(self.class_counts) / cls for cls in self.class_counts]

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

