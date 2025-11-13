import os
import pandas as pd
from glob import glob
from typing import List, Tuple, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import torchvision.transforms as transforms


class PersonImageDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 label_excel: str,
                 transform=None):
        """
        Args:
            root_dir: 数据集根目录（如 './data'）
            label_excel: label.xlsx 完整路径
            transform: 图像预处理（如包含 ToTensor）
        """
        self.root_dir = root_dir
        self.transform = transform
        self.label_df = pd.read_excel(label_excel)
        self.samples, self.person_counts = self._load_samples()

    def _load_samples(self) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
        samples = []
        person_counts = defaultdict(int)
        name_to_label = dict(zip(self.label_df['name'], self.label_df['label']))

        for person_name in os.listdir(self.root_dir):
            person_dir = os.path.join(self.root_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            if person_name not in name_to_label:
                print(f"Warning: {person_name} not found in label.xlsx, skipping...")
                continue
            label = int(name_to_label[person_name])
            if label not in {0, 1}:
                print(f"Warning: Invalid label {label} for {person_name}, skipping...")
                continue

            # 递归查找所有 *_JPG 目录（如 5_JPG, 502_JPG）
            jpg_dirs = glob(os.path.join(person_dir, '**', '*_JPG'), recursive=True)

            for jpg_dir in jpg_dirs:
                img_paths = glob(os.path.join(jpg_dir, '*.jpg')) + \
                            glob(os.path.join(jpg_dir, '*.JPG'))
                for img_path in img_paths:
                    samples.append((img_path, label))
                    person_counts[person_name] += 1

        if not samples:
            raise ValueError("No valid image samples found!")

        # === 打印统计信息 ===
        valid_persons = len(set(name_to_label.keys()) & set(os.listdir(self.root_dir)))
        print(f"Loaded {len(samples)} images from {valid_persons} persons.")
        print("=" * 50)
        print("=== 每个人图片数量统计 ===")
        for name in sorted(person_counts.keys()):
            print(f"  {name}: {person_counts[name]} 张")
        print(f"  总计: {len(samples)} 张 ({len(person_counts)} 人)")
        print("=" * 50)

        return samples, dict(person_counts)

    def get_person_counts(self) -> Dict[str, int]:
        """返回 {人名: 图片数量} 字典"""
        return self.person_counts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# =============================================
# 主程序：直接运行可测试
# =============================================
if __name__ == "__main__":
    # ==================== 配置路径 ====================
    ROOT_DIR = "./data"           # 修改为你的实际路径
    LABEL_EXCEL = "./label.xlsx"  # 修改为你的 label.xlsx 路径

    # ==================== 数据增强 ====================
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),  # 必须加！否则 img 是 PIL 图像
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])

    # ==================== 加载数据集 ====================
    dataset = PersonImageDataset(
        root_dir=ROOT_DIR,
        label_excel=LABEL_EXCEL,
        transform=transform
    )

    # ==================== 编程方式获取统计 ====================
    counts = dataset.get_person_counts()
    print("编程获取统计:", counts)

    # ==================== 测试第一张图 ====================
    img, label = dataset[0]
    print(f"First image shape: {img.shape}, Label: {label}")

    # ==================== 创建 DataLoader ====================
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 测试一个 batch
    for batch_imgs, batch_labels in dataloader:
        print(f"Batch image shape: {batch_imgs.shape}, Labels: {batch_labels[:5].tolist()}...")
        break