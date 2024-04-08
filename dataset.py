import os
import torch
import PIL

from torch.utils.data import Dataset, random_split
from typing import Optional, Callable 

class CCSNDataset(Dataset):
    def __init__(self, img_label_pairs: list):
        self._transform = None
        self.img_label_pairs = img_label_pairs

    @property
    def transform(self):
        return self._transform 

    @transform.setter
    def transform(self, transform):
        self._transform = transform

    def __len__(self):
        return len(self.img_label_pairs)

    def __getitem__(self, idx):
        img_path, label = self.img_label_pairs[idx]
        img = PIL.Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label

def get_dataset(dataset_dir: str, train_transform: Optional[Callable] = None, 
                val_test_transform: Optional[Callable] = None, train_ratio: float = 0.85):
    # sort the dataset to enforce non-randomness
    class_names = sorted(os.listdir(dataset_dir))
    train_img_label_pairs = []
    val_img_label_pairs = []
    test_img_label_pairs = []
    for label, class_name in enumerate(class_names):
        img_names = sorted(os.listdir(os.path.join(dataset_dir, class_name)))
        img_paths = [os.path.join(dataset_dir, class_name, img_name) for img_name in img_names]
        img_label_pairs = [(img_path, label) for img_path in img_paths]

        train_len = int(len(img_paths) * train_ratio)
        train_img_label_pairs += img_label_pairs[:train_len]
        val_img_label_pairs += img_label_pairs[train_len:]
        # CCSN is a small dataset, just use the entire dataset as the test set
        test_img_label_pairs += img_label_pairs

    train_dataset = CCSNDataset(train_img_label_pairs)
    val_dataset = CCSNDataset(val_img_label_pairs)
    test_dataset = CCSNDataset(test_img_label_pairs)
    train_dataset.transform = train_transform
    val_dataset.transform = val_test_transform
    test_dataset.transform = val_test_transform
    return train_dataset, val_dataset, test_dataset
