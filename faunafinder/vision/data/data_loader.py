"""A dataloader implementation in pytorch"""
import os
import glob
import json
import random

from PIL import Image
from torch.utils.data import Dataset, Subset


def split_dataset(dataset, split_ratio=0.8):
    # determine size of training and validation sets
    train_size = int(split_ratio * len(dataset))

    # create random indices for training and validation sets
    train_indices = random.sample(range(len(dataset)), train_size)
    val_indices = list(set(range(len(dataset))) - set(train_indices))

    # create subsets for training and validation
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    return train_subset, val_subset


class ImageDataset(Dataset):
    """A dataset for loading images and their labels from a root directory.

    Images are assumed to be organized in subdirectories of the root directory, with
    each subdirectory representing a different label.

    Args:
        root_dir (str): The root directory of the dataset.
        transform (callable, optional): A transformation to apply to each image.
    """
    def __init__(self, root_dir, num_classes, transform=None):
        self.root_dir = root_dir
        self.num_classes = num_classes
        self.transform = transform

        # load the class label mapping
        with open(os.path.join(root_dir, 'class_labels.json'), 'r') as f:
            self.target_dict = json.load(f)

        # get list of all image file paths
        self.image_paths = glob.glob(os.path.join(root_dir, '**/*.jpg'), recursive=True)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        # load image and apply transformation
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # get label from image path
        label = image_path.split(os.sep)[-2]
        label = self.target_dict[label]

        return image, label
