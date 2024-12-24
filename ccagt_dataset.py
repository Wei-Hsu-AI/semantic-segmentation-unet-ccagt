import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class CCAgT_Dataset(Dataset):
    def __init__(self, img_dir, mask_dir, split='train', split_ratio=(0.7, 0.1, 0.2), transform=None):
        """
        自動生成數據集文件路徑並分割為訓練、驗證、測試集。
        :param img_dir: 圖像的根目錄
        :param mask_dir: 標籤的根目錄
        :param split: 'train', 'val', 'test' 指定使用的數據部分
        :param split_ratio: 訓練、驗證和測試的比例
        :param seed: 隨機種子，用於可重現性
        :param transform: 數據增強轉換
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # self.mask_transform = mask_transform

        # 獲取所有文件的相對路徑列表
        self.file_list = self._generate_file_list(img_dir, mask_dir)

        # 根據比例分割數據集
        train_files, val_files, test_files = self._split_dataset(self.file_list, split_ratio)

        if split == 'train':
            self.file_list = train_files
        elif split == 'val':
            self.file_list = val_files
        elif split == 'test':
            self.file_list = test_files
        else:
            raise ValueError("Invalid split value. Choose from 'train', 'val', or 'test'.")

    def _generate_file_list(self, img_dir, mask_dir):
        """
        自動生成影像與標籤的文件列表。
        :param img_dir: 影像目錄
        :param mask_dir: 標籤目錄
        :return: 文件名列表
        """
        categories = os.listdir(img_dir)  # 獲取所有子目錄（分類）
        file_list = []

        for category in categories:
            img_category_path = os.path.join(img_dir, category)
            mask_category_path = os.path.join(mask_dir, category)

            if os.path.isdir(img_category_path) and os.path.isdir(mask_category_path):
                # 獲取該分類下的所有文件名
                img_files = os.listdir(img_category_path)
                img_files.sort()  # 確保影像與標籤順序一致

                for img_file in img_files:
                    img_name = os.path.splitext(img_file)[0] 
                    mask_file = os.path.join(mask_category_path, img_name) + '.png'
                    if os.path.exists(mask_file):
                        file_list.append(os.path.join(category, img_name))

        return file_list

    def _split_dataset(self, file_list, split_ratio):
        """
        按比例分割數據集。
        :param file_list: 文件列表
        :param split_ratio: 訓練、驗證、測試的比例
        :param seed: 隨機種子
        :return: 分割後的訓練、驗證、測試文件列表
        """
        np.random.shuffle(file_list)

        n_total = len(file_list)
        n_train = int(split_ratio[0] * n_total)
        n_val = int(split_ratio[1] * n_total)

        train_files = file_list[:n_train]
        val_files = file_list[n_train:n_train + n_val]
        test_files = file_list[n_train + n_val:]

        return train_files, val_files, test_files

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.img_dir, img_name) + '.jpg'
        mask_path = os.path.join(self.mask_dir, img_name) + '.png'

        # 加載影像與標籤
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # 應用數據增強
        if self.transform:
            img, mask = self.transform(img, mask)
            # mask = self.mask_transform(mask)
            
        return img, mask