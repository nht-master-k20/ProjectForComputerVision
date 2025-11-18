from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd

def convert_df_to_dataset(df):
    images = []
    labels = []

    for idx, row in df.iterrows():
        img = Image.open(row['image_path']).convert("RGB")
        img = transforms.ToTensor()(img)
        images.append(img)
        labels.append(row['malignant'])

    images = torch.stack(images)
    labels = torch.tensor(labels)

    dataset = TensorDataset(images, labels)
    return dataset

def get_dataset(train_path, val_path, test_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    train_dataset = convert_df_to_dataset(train_df)
    val_dataset = convert_df_to_dataset(val_df)
    test_dataset = convert_df_to_dataset(test_df)

    return train_dataset, val_dataset, test_dataset



