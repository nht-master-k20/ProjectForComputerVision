import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import numpy as np


class ISICDataset(Dataset):
    # Thêm is_train vào __init__
    def __init__(self, df, img_size=300, is_train=False):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.is_train = is_train  # Lưu lại trạng thái

        # CẤU HÌNH AUGMENTATION
        if self.is_train:
            # Augmentation cho TRAIN (Xoay, lật, màu sắc...)
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=30, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            # Transform cho VAL/TEST (Chỉ Resize & Normalize)
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = row['image_path']

        # Xử lý nhãn
        if 'malignant' in row:
            label = row['malignant']
        else:
            label = 0

        # Đọc ảnh
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Áp dụng Transform (Augment hoặc không tùy vào is_train)
        augmented = self.transform(image=img)
        img_tensor = augmented['image']

        return img_tensor, torch.tensor(label, dtype=torch.long)