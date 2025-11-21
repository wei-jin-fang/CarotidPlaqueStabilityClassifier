#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试数据处理：验证掐头去尾功能是否正常工作
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import PersonSequenceDataset
import torchvision.transforms as transforms

def test_data_processing():
    """测试数据处理功能"""

    print("="*70)
    print("测试数据处理功能：掐头去尾保留中间100张")
    print("="*70)

    # 数据路径（请根据实际情况修改）
    data_root = "/seu_nvme/home/shendian/220256451/datasets/Carotid_artery/Carotid_artery"
    label_excel = "/seu_nvme/home/shendian/220256451/datasets/Carotid_artery/label.xlsx"

    # 如果路径不存在，使用默认路径
    if not os.path.exists(data_root):
        data_root = "./data"
    if not os.path.exists(label_excel):
        label_excel = "./label.xlsx"

    print(f"\n使用的数据路径:")
    print(f"  数据目录: {data_root}")
    print(f"  标签文件: {label_excel}")
    print()

    # 定义变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 测试不同的配置
    configs = [
        {"keep_middle_n": 100, "min_imgs_required": 100, "desc": "保留中间100张，最低100张"},
        {"keep_middle_n": 80, "min_imgs_required": 80, "desc": "保留中间80张，最低80张"},
        {"keep_middle_n": 50, "min_imgs_required": 50, "desc": "保留中间50张，最低50张"},
    ]

    for i, config in enumerate(configs):
        print(f"\n{'='*70}")
        print(f"测试配置 {i+1}: {config['desc']}")
        print(f"{'='*70}")

        try:
            dataset = PersonSequenceDataset(
                root_dir=data_root,
                label_excel=label_excel,
                transform=transform,
                keep_middle_n=config['keep_middle_n'],
                min_imgs_required=config['min_imgs_required']
            )

            print(f"\n✓ 数据集加载成功!")
            print(f"  总样本数: {len(dataset)}")

            if len(dataset) > 0:
                # 测试获取第一个样本
                images, label = dataset[0]
                print(f"  第一个样本形状: {images.shape}")
                print(f"  第一个样本标签: {label.item()}")
                print(f"  每个样本图片数: {config['keep_middle_n']}")

            # 获取样本信息
            person_info = dataset.get_person_info()
            if person_info:
                sample_name = list(person_info.keys())[0]
                sample_info = person_info[sample_name]
                print(f"\n示例样本: {sample_name}")
                print(f"  图片数: {sample_info['count']}")
                print(f"  标签: {sample_info['label']}")

        except FileNotFoundError as e:
            print(f"\n✗ 错误: 文件未找到")
            print(f"  {str(e)}")
            print(f"\n提示: 请修改此脚本中的数据路径:")
            print(f"  data_root = '你的数据目录'")
            print(f"  label_excel = '你的标签文件路径'")
            break
        except Exception as e:
            print(f"\n✗ 发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            break

    print(f"\n{'='*70}")
    print("测试完成!")
    print(f"{'='*70}")

if __name__ == "__main__":
    test_data_processing()
