import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

class ISICDataset(Dataset):
    def __init__(self, df, img_size=256):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = row['image_path']
        label = row['malignant']

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (self.img_size, self.img_size))

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)