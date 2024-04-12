import yaml
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def json_to_array(json_str):
    """ Convert a JSON string back into a numpy array. """
    try:
        data = json.loads(json_str)
        return np.array(data, dtype=np.uint8)
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e} - JSON string: {json_str}")
        return np.zeros((1, 1, 3), dtype=np.uint8)  # Placeholder for failed decode

def load_and_prepare_data(csv_file, transform, is_test=False):
    print('-' * 20)
    print('Started processing images...')

    data = pd.read_csv(csv_file)
    data['resized_cropped_face'] = data['resized_cropped_face2'].apply(json_to_array)
    
    images = []
    labels = []
    
    for _, row in data.iterrows():
        image = Image.fromarray(row['resized_cropped_face'], 'RGB')
        tensor_image = transform(image)
        images.append(tensor_image)
        labels.append(int(row['original_class']))  # Ensure this extracts integer labels

    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)

    
    print('Finished processing images.')
    print('-' * 20)
    if is_test:
        return images, labels
    else:
        # Splitting the dataset
        train_imgs, val_imgs, train_labels, val_labels = train_test_split(
            images, labels, test_size=0.2, random_state=42)
        return train_imgs, train_labels, val_imgs, val_labels
    
class TestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 2])
        image = Image.open(img_name)
        label = int(self.dataframe.iloc[idx, 0])

        if self.transform:
            image = self.transform(image)

        return image, label

def load_test_data(csv_file, root_dir, transform):
    # Create the dataset
    test_dataset = TestDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
    
    # Create the DataLoader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Assuming a batch size of 1 for individual test image loading
    return test_loader

