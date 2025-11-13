# -*- coding: utf-8 -*-
"""
推理代码 + GradCAM 可视化 (完整版)

功能:
1. 加载训练好的模型进行推理
2. 评估与训练时完全相同的测试集(使用相同的随机种子和数据划分)
3. 为指定人物/文件夹生成 GradCAM 可视化
4. 单张图片推理和可视化

用法:
  # 1. 评估测试集 (使用训练时的数据划分)
  python inference_gradcam.py \
      --train-dir ./output/train_20250105_123456 \
      --eval-test

  # 2. 为指定人物生成 GradCAM
  python inference_gradcam.py \
      --train-dir ./output/train_20250105_123456 \
      --visualize-person ./data/person_name \
      --max-vis 5

  # 3. 单张图片推理
  python inference_gradcam.py \
      --model ./output/train_20250105_123456/models/best_model.pth \
      --image ./test.jpg \
      --output ./result.png
"""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import pandas as pd
from glob import glob
from collections import defaultdict
import cv2
import warnings
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import seaborn as sns
from datetime import datetime
import json
import shutil
import argparse
warnings.filterwarnings("ignore")


# ====================== 1. 自定义 collate_fn ======================
def person_sequence_collate_fn(batch):
    """
    batch: List[(images_tensor, label)]
           images_tensor: [N_i, 3, 224, 224]
           label: scalar
    返回:
        images_list: List[Tensor] 每个 Tensor 形状不同
        labels: Tensor [B]
    """
    images_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return images_list, labels


# ====================== 2. 按人组织的数据集 ======================
class PersonSequenceDataset(Dataset):
    def __init__(self, root_dir: str, label_excel: str, transform=None, max_imgs_per_person: int = None):
        self.root_dir = root_dir
        self.transform = transform
        self.max_imgs = max_imgs_per_person
        self.persons = self._load_persons(label_excel)

    def _load_persons(self, label_excel) -> List[Dict]:
        df = pd.read_excel(label_excel)
        name_to_label = dict(zip(df['name'], df['label']))
        persons = []

        for person_name in os.listdir(self.root_dir):
            person_dir = os.path.join(self.root_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
            if person_name not in name_to_label:
                print(f"Skip {person_name}: not in label.xlsx")
                continue
            label = int(name_to_label[person_name])
            if label not in {0, 1}:
                continue

            img_paths = []
            for jpg_dir in glob(os.path.join(person_dir, '**', '*_JPG'), recursive=True):
                for ext in ('*.jpg', '*.JPG'):
                    img_paths.extend(glob(os.path.join(jpg_dir, ext)))

            if not img_paths:
                continue

            if self.max_imgs and len(img_paths) > self.max_imgs:
                img_paths = random.sample(img_paths, self.max_imgs)

            persons.append({
                'name': person_name,
                'paths': img_paths,
                'label': label
            })

        print(f"Loaded {len(persons)} persons.")
        total_imgs = sum(len(p['paths']) for p in persons)
        print(f"Total images: {total_imgs}")
        return persons

    def __len__(self):
        return len(self.persons)

    def __getitem__(self, idx):
        person = self.persons[idx]
        imgs = []
        for path in person['paths']:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        images_tensor = torch.stack(imgs)  # [N, 3, 224, 224]
        label = torch.tensor(person['label'], dtype=torch.long)
        return images_tensor, label

    def get_person_info(self):
        return {p['name']: {'count': len(p['paths']), 'label': p['label']} for p in self.persons}


# ====================== 3. 模型：ResNet + Attention Fusion ======================
class ResNetAttentionFusion(nn.Module):
    def __init__(self, pretrained=True, num_classes=2, dropout=0.5):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # [*, 512, 7, 7]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

        # Grad-CAM
        self.feature_maps = None
        self.gradients = None
        resnet.layer4[-1].register_forward_hook(self._save_feature_maps)
        resnet.layer4[-1].register_backward_hook(self._save_gradients)

    def _save_feature_maps(self, module, input, output):
        self.feature_maps = output

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def forward(self, x):
        """
        x: List[Tensor] 每个: [N_i, 3, 224, 224]
           或单个 Tensor: [N, 3, 224, 224]
        """
        if not isinstance(x, list):
            x = [x]

        # 提取所有特征 - 使用批量处理避免 BatchNorm 问题
        all_feats = []
        batch_sizes = []

        # 将所有序列合并成一个大 batch 处理
        concat_imgs = torch.cat(x, dim=0)  # [sum(N_i), 3, 224, 224]

        # 确保输入是连续的且格式正确
        if not concat_imgs.is_contiguous():
            concat_imgs = concat_imgs.contiguous()

        # 统一批量提取特征
        with torch.cuda.amp.autocast(enabled=False):  # 禁用混合精度避免 cuDNN 问题
            concat_feats = self.feature_extractor(concat_imgs)  # [sum(N_i), 512, 7, 7]
            concat_pooled = self.avgpool(concat_feats).view(concat_imgs.size(0), -1)  # [sum(N_i), 512]

        # 按原始序列拆分
        start_idx = 0
        for seq in x:
            B = seq.size(0)
            batch_sizes.append(B)
            pooled = concat_pooled[start_idx:start_idx+B]  # [N_i, 512]
            all_feats.append(pooled)
            start_idx += B

        # Attention 融合
        fused_list = []
        for feats in all_feats:  # feats: [N_i, 512]
            if feats.size(0) == 1:
                fused = feats.squeeze(0)
            else:
                attn = self.attention(feats).squeeze(-1)      # [N_i]
                weights = torch.softmax(attn, dim=0).unsqueeze(-1)  # [N_i, 1]
                fused = torch.sum(feats * weights, dim=0)      # [512]
            fused_list.append(fused)

        fused = torch.stack(fused_list)  # [B, 512]
        logits = self.classifier(fused)  # [B, 2]
        return logits, fused

    def get_cam(self, img_tensor):
        self.eval()
        device = next(self.parameters()).device

        # 确保输入格式正确
        img_tensor = img_tensor.unsqueeze(0).to(device)
        if not img_tensor.is_contiguous():
            img_tensor = img_tensor.contiguous()

        # 禁用混合精度
        with torch.cuda.amp.autocast(enabled=False):
            _ = self.feature_extractor(img_tensor)
            pooled = self.avgpool(self.feature_maps).view(1, -1)
            logits = self.classifier(pooled)

        pred = logits.argmax(dim=1)
        self.zero_grad()
        logits[0, pred].backward()

        if self.gradients is None:
            return None

        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1).squeeze(0)
        cam = torch.relu(cam).cpu().detach().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam011 = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam011


# ====================== 4. 测试集评估 ======================
def evaluate_test_set(model, test_dataset, device, class_names=['Class 0', 'Class 1'], save_dir='test_results'):
    """
    在测试集上评估模型性能,计算各种指标并可视化
    使用逐样本推理避免 batch_size=1 问题

    参数:
        model: 训练好的模型
        test_dataset: 测试数据集 (Subset 或原始 Dataset)
        device: 计算设备
        class_names: 类别名称列表
        save_dir: 结果保存目录

    返回:
        metrics: 包含所有指标的字典
    """
    model.eval()

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 收集所有预测结果
    all_preds = []
    all_labels = []
    all_probs = []  # 用于 ROC 曲线

    print("\n" + "="*60)
    print("开始测试集评估 (逐样本推理)...")
    print("="*60)
    print(f"测试集样本数: {len(test_dataset)}")

    # 逐个样本推理,避免 batch_size=1 的问题
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            # 获取单个样本
            seq_imgs, label = test_dataset[idx]  # seq_imgs: [N_imgs, 3, 224, 224]

            # 转移到设备
            seq_imgs = seq_imgs.to(device)  # [N_imgs, 3, 224, 224]
            label = label.to(device)  # scalar

            # 前向传播 - 传入 list 格式
            try:
                logits, _ = model([seq_imgs])  # 注意: 包装成 list
                probs = torch.softmax(logits, dim=1)
                pred = logits.argmax(dim=1)

                # 收集结果
                all_preds.append(pred.cpu().item())
                all_labels.append(label.cpu().item())
                all_probs.append(probs[0, 1].cpu().item())  # 类别1的概率

                # 打印进度
                if (idx + 1) % 10 == 0 or (idx + 1) == len(test_dataset):
                    print(f"  进度: {idx + 1}/{len(test_dataset)} ({(idx+1)/len(test_dataset)*100:.1f}%)")

            except Exception as e:
                print(f"⚠️  样本 {idx} 推理失败: {str(e)[:100]}")
                continue

    # 转换为 numpy 数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 检查是否有有效数据
    if len(all_preds) == 0:
        print("\n⚠️  警告: 没有有效的测试数据,跳过评估")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc': 0.0,
            'confusion_matrix': np.array([[0, 0], [0, 0]]),
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }

    # ==================== 计算各种指标 ====================
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    # 计算 AUC (如果两个类别都存在)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
        print("Warning: Cannot calculate AUC (only one class present)")

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 分类报告
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names,
                                   zero_division=0)

    # ==================== 打印结果 ====================
    print("\n" + "="*60)
    print("测试集评估结果")
    print("="*60)
    print(f"总样本数: {len(all_labels)}")
    print(f"类别分布: {class_names[0]}={np.sum(all_labels==0)}, {class_names[1]}={np.sum(all_labels==1)}")
    print("-"*60)
    print(f"准确率 (Accuracy):  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1 分数 (F1-Score): {f1:.4f}")
    print(f"AUC-ROC:           {auc:.4f}")
    print("="*60)

    print("\n详细分类报告:")
    print(report)

    # ==================== 可视化 ====================

    # 1. 混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ 混淆矩阵已保存: {cm_path}")

    # 2. ROC 曲线
    if auc > 0:
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        roc_path = os.path.join(save_dir, 'roc_curve.png')
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ ROC 曲线已保存: {roc_path}")

    # 3. 指标柱状图
    metrics_dict = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC': auc
    }

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_dict.keys(), metrics_dict.values(),
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                   alpha=0.8, edgecolor='black')
    plt.ylim([0, 1.1])
    plt.ylabel('Score', fontsize=12)
    plt.title('Test Set Performance Metrics', fontsize=14, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.grid(axis='y', alpha=0.3)

    # 在柱子上标注数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    metrics_path = os.path.join(save_dir, 'metrics_bar.png')
    plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 指标柱状图已保存: {metrics_path}")

    # 4. 预测概率分布
    plt.figure(figsize=(10, 6))

    # 分别绘制两个类别的概率分布
    probs_class0 = all_probs[all_labels == 0]
    probs_class1 = all_probs[all_labels == 1]

    plt.hist(probs_class0, bins=30, alpha=0.6, label=f'{class_names[0]} (True)',
             color='blue', edgecolor='black')
    plt.hist(probs_class1, bins=30, alpha=0.6, label=f'{class_names[1]} (True)',
             color='red', edgecolor='black')

    plt.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
    plt.xlabel(f'Predicted Probability for {class_names[1]}', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    prob_dist_path = os.path.join(save_dir, 'probability_distribution.png')
    plt.savefig(prob_dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 概率分布图已保存: {prob_dist_path}")

    # 5. 保存详细结果到 CSV
    results_df = pd.DataFrame({
        'True_Label': all_labels,
        'Predicted_Label': all_preds,
        'Probability_Class1': all_probs,
        'Correct': all_labels == all_preds
    })
    csv_path = os.path.join(save_dir, 'detailed_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"✓ 详细结果已保存: {csv_path}")

    # 6. 保存指标摘要到文本文件
    summary_path = os.path.join(save_dir, 'metrics_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("测试集评估结果摘要\n")
        f.write("="*60 + "\n\n")
        f.write(f"总样本数: {len(all_labels)}\n")
        f.write(f"类别分布: {class_names[0]}={np.sum(all_labels==0)}, {class_names[1]}={np.sum(all_labels==1)}\n\n")
        f.write(f"准确率 (Accuracy):  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"精确率 (Precision): {precision:.4f}\n")
        f.write(f"召回率 (Recall):    {recall:.4f}\n")
        f.write(f"F1 分数 (F1-Score): {f1:.4f}\n")
        f.write(f"AUC-ROC:           {auc:.4f}\n\n")
        f.write("="*60 + "\n")
        f.write("混淆矩阵:\n")
        f.write("="*60 + "\n")
        f.write(f"                 Predicted {class_names[0]}  Predicted {class_names[1]}\n")
        f.write(f"True {class_names[0]:8s}     {cm[0,0]:6d}          {cm[0,1]:6d}\n")
        f.write(f"True {class_names[1]:8s}     {cm[1,0]:6d}          {cm[1,1]:6d}\n\n")
        f.write("="*60 + "\n")
        f.write("详细分类报告:\n")
        f.write("="*60 + "\n")
        f.write(report)
    print(f"✓ 指标摘要已保存: {summary_path}")

    print("\n" + "="*60)
    print(f"所有结果已保存到目录: {save_dir}")
    print("="*60 + "\n")

    # 返回指标字典
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


# ====================== 5. 单人 GradCAM 可视化 ======================
def visualize_gradcam_for_person(model, person_dir, label_excel, transform, device,
                                  max_vis=5, output_dir='gradcam_person', class_names=['0', '1']):
    """
    为指定人物文件夹生成 GradCAM 可视化

    参数:
        model: 训练好的模型
        person_dir: 人物文件夹路径
        label_excel: 标签文件
        transform: 图片预处理
        device: 计算设备
        max_vis: 最多可视化多少张图片
        output_dir: 输出目录
        class_names: 类别名称
    """
    model.eval()

    # 获取人物名称
    person_name = os.path.basename(person_dir.rstrip('/'))

    # 读取标签
    df = pd.read_excel(label_excel)
    name_to_label = dict(zip(df['name'], df['label']))

    if person_name not in name_to_label:
        print(f"\n⚠️  错误: 人物 '{person_name}' 不在标签文件中")
        return

    true_label = int(name_to_label[person_name])

    # 查找所有图片
    img_paths = []
    for jpg_dir in glob(os.path.join(person_dir, '**', '*_JPG'), recursive=True):
        for ext in ('*.jpg', '*.JPG'):
            img_paths.extend(glob(os.path.join(jpg_dir, ext)))

    if not img_paths:
        print(f"\n⚠️  错误: 在 '{person_dir}' 中未找到图片")
        return

    print(f"\n{'='*60}")
    print(f"为人物 '{person_name}' 生成 GradCAM 可视化")
    print(f"{'='*60}")
    print(f"  真实标签: {true_label} ({class_names[true_label]})")
    print(f"  图片数量: {len(img_paths)}")
    print(f"  可视化数量: {min(max_vis, len(img_paths))}")

    # 加载所有图片进行融合预测
    imgs = []
    for path in img_paths:
        img = Image.open(path).convert('RGB')
        img_t = transform(img)
        imgs.append(img_t)

    input_list = [torch.stack(imgs).to(device)]  # [1, N, 3, 224, 224]

    # 融合预测
    with torch.no_grad():
        logits, _ = model(input_list)
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(logits.argmax(dim=1).item())

    print(f"  融合预测: {pred} ({class_names[pred]})")
    print(f"  预测概率: [{prob[0]:.3f}, {prob[1]:.3f}]")
    print("-"*60)

    # 创建输出目录
    person_output_dir = os.path.join(output_dir, person_name)
    os.makedirs(person_output_dir, exist_ok=True)

    # 为前 N 张图片生成 GradCAM
    num_vis = min(max_vis, len(img_paths))
    print(f"\n生成 GradCAM 可视化...")

    for i in range(num_vis):
        cam = model.get_cam(imgs[i])
        if cam is not None:
            save_path = os.path.join(person_output_dir,
                                    f"img{i+1:02d}_pred{pred}_true{true_label}.png")
            visualize_cam(img_paths[i], cam, pred, prob, class_names, save_path)
            print(f"  [{i+1}/{num_vis}] 已保存: {save_path}")

    print(f"\n{'='*60}")
    print(f"✓ 所有可视化已保存到: {person_output_dir}")
    print(f"{'='*60}\n")


def visualize_cam(img_path, cam, pred_class, prob, class_names, save_path):
    """生成 GradCAM 可视化图片"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original", fontsize=12)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("GradCAM Heatmap", fontsize=12)
    plt.imshow(cam, cmap='jet')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"Overlay\nPred: {pred_class} ({class_names[pred_class]})\nProb: [{prob[0]:.3f}, {prob[1]:.3f}]",
              fontsize=11)
    plt.imshow(img)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ====================== 6. 单张图片推理 ======================
def predict_single_image(model, image_path, transform, device, class_names=['0', '1'], output_path=None):
    """
    单张图片推理和可视化

    参数:
        model: 训练好的模型
        image_path: 图片路径
        transform: 预处理
        device: 计算设备
        class_names: 类别名称
        output_path: 输出路径
    """
    model.eval()

    # 加载图片
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)

    # 推理
    with torch.no_grad():
        img_batch = img_tensor.unsqueeze(0).to(device)
        logits, _ = model([img_batch])
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(logits.argmax(dim=1).item())

    print(f"\n{'='*60}")
    print(f"单张图片推理结果")
    print(f"{'='*60}")
    print(f"  图片路径: {image_path}")
    print(f"  预测类别: {pred} ({class_names[pred]})")
    print(f"  预测概率: [{prob[0]:.4f}, {prob[1]:.4f}]")
    print(f"{'='*60}\n")

    # 生成 GradCAM
    cam = model.get_cam(img_tensor)

    if cam is None:
        print("⚠️  GradCAM 生成失败")
        return

    # 保存可视化
    if output_path is None:
        output_path = f"gradcam_single_{os.path.basename(image_path)}"

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    visualize_cam(image_path, cam, pred, prob, class_names, output_path)
    print(f"✓ 可视化已保存: {output_path}\n")


# ====================== 7. 主函数 ======================
def main():
    parser = argparse.ArgumentParser(
        description='GradCAM 推理和可视化工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ==================== 模式选择 ====================
    mode_group = parser.add_argument_group('运行模式 (三选一)')
    mode_group.add_argument('--eval-test', action='store_true',
                           help='评估测试集 (使用训练时的数据划分和随机种子)')
    mode_group.add_argument('--visualize-person', type=str, default=None,
                           help='为指定人物文件夹生成 GradCAM (例如: ./data/person_name)')
    mode_group.add_argument('--image', type=str, default=None,
                           help='单张图片推理和可视化 (例如: ./test.jpg)')

    # ==================== 训练目录配置 ====================
    train_group = parser.add_argument_group('训练目录配置 (用于加载模型和配置)')
    train_group.add_argument('--train-dir', type=str, default=None,
                            help='训练输出目录 (例如: ./output/train_20250105_123456)')
    train_group.add_argument('--model', type=str, default=None,
                            help='模型文件路径 (如果不指定,从 train-dir 自动加载 best_model.pth)')

    # ==================== 数据配置 ====================
    data_group = parser.add_argument_group('数据配置')
    data_group.add_argument('--data-dir', type=str, default='./data',
                           help='数据根目录')
    data_group.add_argument('--label-excel', type=str, default='./label.xlsx',
                           help='标签 Excel 文件')
    data_group.add_argument('--class-names', type=str, nargs=2, default=['0', '1'],
                           help='类别名称 (两个类别)')

    # ==================== 可视化配置 ====================
    vis_group = parser.add_argument_group('可视化配置')
    vis_group.add_argument('--max-vis', type=int, default=5,
                          help='为每个人最多可视化多少张图片')
    vis_group.add_argument('--output', type=str, default=None,
                          help='输出路径 (用于单张图片模式)')
    vis_group.add_argument('--output-dir', type=str, default='./inference_results',
                          help='输出目录 (用于测试集评估和人物可视化)')

    # ==================== 其他配置 ====================
    other_group = parser.add_argument_group('其他配置')
    other_group.add_argument('--device', type=str, default='auto',
                            choices=['auto', 'cuda', 'cpu'],
                            help='计算设备')

    args = parser.parse_args()

    # ==================== 参数验证 ====================
    # 检查是否至少选择了一个模式
    if not any([args.eval_test, args.visualize_person, args.image]):
        parser.error('必须指定一个运行模式: --eval-test, --visualize-person, 或 --image')

    # 如果使用 eval-test 或 visualize-person,必须提供 train-dir
    if (args.eval_test or args.visualize_person) and not args.train_dir:
        parser.error('使用 --eval-test 或 --visualize-person 时必须指定 --train-dir')

    # 如果使用 image 模式,必须提供 model
    if args.image and not args.model and not args.train_dir:
        parser.error('使用 --image 时必须指定 --model 或 --train-dir')

    # ==================== 设备配置 ====================
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"\n使用设备: {device}\n")

    # ==================== 数据预处理 ====================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # ==================== 加载模型 ====================
    # 确定模型路径
    if args.model:
        model_path = args.model
    elif args.train_dir:
        model_path = os.path.join(args.train_dir, 'models', 'best_model.pth')
    else:
        parser.error('必须指定 --model 或 --train-dir')

    if not os.path.exists(model_path):
        print(f"\n⚠️  错误: 模型文件不存在: {model_path}")
        return

    print(f"加载模型: {model_path}")
    model = ResNetAttentionFusion(pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✓ 模型加载成功\n")

    # ==================== 模式 1: 评估测试集 ====================
    if args.eval_test:
        # 加载训练配置
        config_path = os.path.join(args.train_dir, 'logs', 'config.json')
        split_path = os.path.join(args.train_dir, 'logs', 'data_split.json')

        if not os.path.exists(config_path):
            print(f"\n⚠️  错误: 训练配置文件不存在: {config_path}")
            return

        if not os.path.exists(split_path):
            print(f"\n⚠️  错误: 数据划分文件不存在: {split_path}")
            return

        # 读取配置
        with open(config_path, 'r') as f:
            config = json.load(f)

        with open(split_path, 'r') as f:
            split_info = json.load(f)

        print("="*60)
        print("加载训练配置和数据划分")
        print("="*60)
        print(f"  训练目录: {args.train_dir}")
        print(f"  随机种子: {config.get('random_seed', 42)}")
        print(f"  数据划分: Train={split_info['train_size']}, Val={split_info['val_size']}, Test={split_info['test_size']}")
        print("="*60 + "\n")

        # 使用相同的随机种子
        random.seed(config.get('random_seed', 42))

        # 加载完整数据集
        print("加载数据集...")
        full_dataset = PersonSequenceDataset(
            root_dir=config.get('root_dir', args.data_dir),
            label_excel=config.get('label_excel', args.label_excel),
            transform=transform,
            max_imgs_per_person=config.get('max_imgs_per_person', None)
        )

        # 创建测试子数据集
        test_indices = split_info['test_indices']
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

        print(f"\n✓ 成功加载测试集 (使用训练时的数据划分)")
        print(f"  测试集样本数: {len(test_dataset)}\n")

        # 评估测试集
        output_dir = os.path.join(args.output_dir, 'test_evaluation')
        test_metrics = evaluate_test_set(
            model=model,
            test_dataset=test_dataset,
            device=device,
            class_names=args.class_names,
            save_dir=output_dir
        )

        print(f"\n✓ 测试集评估完成!")
        print(f"  准确率: {test_metrics['accuracy']:.4f}")
        print(f"  F1 分数: {test_metrics['f1']:.4f}")
        print(f"  AUC: {test_metrics['auc']:.4f}\n")

    # ==================== 模式 2: 人物可视化 ====================
    elif args.visualize_person:
        if not os.path.exists(args.visualize_person):
            print(f"\n⚠️  错误: 人物文件夹不存在: {args.visualize_person}")
            return

        # 从训练配置中读取 label_excel
        config_path = os.path.join(args.train_dir, 'logs', 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            label_excel = config.get('label_excel', args.label_excel)
        else:
            label_excel = args.label_excel

        output_dir = os.path.join(args.output_dir, 'person_gradcam')

        visualize_gradcam_for_person(
            model=model,
            person_dir=args.visualize_person,
            label_excel=label_excel,
            transform=transform,
            device=device,
            max_vis=args.max_vis,
            output_dir=output_dir,
            class_names=args.class_names
        )

    # ==================== 模式 3: 单张图片 ====================
    elif args.image:
        if not os.path.exists(args.image):
            print(f"\n⚠️  错误: 图片文件不存在: {args.image}")
            return

        output_path = args.output
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(args.image))[0]
            output_path = os.path.join(args.output_dir, 'single_image',
                                      f"{base_name}_gradcam.png")

        predict_single_image(
            model=model,
            image_path=args.image,
            transform=transform,
            device=device,
            class_names=args.class_names,
            output_path=output_path
        )


# ====================== 8. 程序入口 ======================
if __name__ == "__main__":
    main()
'''
1. 评估训练时的测试集
python inference_gradcam.py \
      --train-dir ./output/train_20251105_164453 \
      --eval-test

2. 为某个人生成 GradCAM

  python inference_gradcam.py \
      --train-dir ./output/train_20251105_164453 \
      --visualize-person ./data/古宏林 \
      --max-vis 5

  python inference_gradcam.py \
      --model ./output/train_20250105_143022/models/best_model.pth \
      --image ./test.jpg

  python inference_gradcam.py \
      --model /home/jinfang/project/jindongmai_classfication/test/CatsDogsSample/best_fusion_model.pth \
      --image /home/jinfang/project/jindongmai_classfication/test/CatsDogsSample/testdog.jpg
'''