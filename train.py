# -*- coding: utf-8 -*-
"""
完整训练 + 融合 + Grad-CAM
解决：不同人图片数量不一致 → DataLoader stack 错误
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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ====================== 0. 创建训练文件夹 ======================
def create_training_folder(base_dir="."):
    """
    创建一个带时间戳的训练文件夹,用于保存本次训练的所有材料

    返回:
        exp_dir: 实验目录路径
        sub_dirs: 各子目录的字典
    """
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"train_{timestamp}"
    exp_dir = os.path.join(base_dir, exp_name)

    # 创建主目录和子目录
    sub_dirs = {
        'root': exp_dir,
        'models': os.path.join(exp_dir, 'models'),          # 保存模型
        'logs': os.path.join(exp_dir, 'logs'),              # 保存训练日志
        'test_results': os.path.join(exp_dir, 'test_results'),  # 测试结果
        'gradcam': os.path.join(exp_dir, 'gradcam'),        # GradCAM 可视化
        'plots': os.path.join(exp_dir, 'plots'),            # 训练曲线等图表
    }

    for dir_path in sub_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    print("="*60)
    print(f"创建训练文件夹: {exp_dir}")
    print("="*60)
    print(f"  - 模型保存: {sub_dirs['models']}")
    print(f"  - 训练日志: {sub_dirs['logs']}")
    print(f"  - 测试结果: {sub_dirs['test_results']}")
    print(f"  - GradCAM: {sub_dirs['gradcam']}")
    print(f"  - 图表: {sub_dirs['plots']}")
    print("="*60 + "\n")

    return exp_dir, sub_dirs


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
        print("="*60)
        for p in sorted(persons, key=lambda x: x['name']):
            print(f"  {p['name']:10s}: {len(p['paths']):4d} 张 | Label: {p['label']}")
        print("="*60)
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
        '''
          list(resnet.children())[:-2] 表示取 ResNet18 的所有层,除了最后2层:

        ResNet18 的完整结构:
        ResNet18.children() = [
            0: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)     # 初始卷积
            1: BatchNorm2d(64)
            2: ReLU(inplace=True)
            3: MaxPool2d(kernel_size=3, stride=2, padding=1)
            4: layer1 (残差块组1)
            5: layer2 (残差块组2)
            6: layer3 (残差块组3)
            7: layer4 (残差块组4)  ← 这是最后一个卷积层
            8: AdaptiveAvgPool2d((1, 1))  ← 被去掉
            9: Linear(512, 1000)           ← 被去掉
  ]
        '''
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
        '''
        feature_maps 内容:
        - 形状: [1, 512, 7, 7]
        - 512 个通道,每个通道是 7×7 的特征图
        '''

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]
        '''
        gradients 内容:
        - 形状: [1, 512, 7, 7]
        - 表示每个特征图的每个位置对预测结果的影响
        - 梯度越大,说明该位置对预测越重要
        '''

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
        # print(f"Concat imgs shape: {concat_imgs.shape}")# torch.Size([1271, 3, 224, 224])

        # 确保输入是连续的且格式正确
        if not concat_imgs.is_contiguous():
            concat_imgs = concat_imgs.contiguous()

        # 统一批量提取特征
        with torch.cuda.amp.autocast(enabled=False):  # 禁用混合精度避免 cuDNN 问题
            concat_feats = self.feature_extractor(concat_imgs)  # [sum(N_i), 512, 7, 7]
            # AdaptiveAvgPool2d((1, 1)),对每个 7×7 的特征图求平均 → 1×1 → [sum(N_i), 512, 1, 1]   
            concat_pooled = self.avgpool(concat_feats).view(concat_imgs.size(0), -1)  # [sum(N_i), 512]

        # 按原始序列拆分
        start_idx = 0
        for seq in x:
            B = seq.size(0)
            batch_sizes.append(B)
            pooled = concat_pooled[start_idx:start_idx+B]  # [N_i, 512]
            all_feats.append(pooled)
            start_idx += B

        # Attention 融合  针对每个人都可以有不同数量的图片进行操作
        fused_list = []
        for feats in all_feats:  # feats: [N_i, 512]
            if feats.size(0) == 1:
                fused = feats.squeeze(0)
            else:
                attn = self.attention(feats).squeeze(-1)      #[N_i, 512]-[N_i,1]- [N_i]
                # print(f"{attn.shape}")#torch.Size([N_i])
                weights = torch.softmax(attn, dim=0).unsqueeze(-1)
                # print(f"{weights.shape}")#torch.Size([N_i],1)
                fused = torch.sum(feats * weights, dim=0)      #[5, 512]* [5, 1]= [5, 512]--》sum= [512]
                # print(f"{fused.shape}")#torch.Size(512)
                
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
            pooled = self.avgpool(self.feature_maps).view(1, -1) #用钩子得到 feature_maps
            logits = self.classifier(pooled)  # classifier 只返回 logits

        pred = logits.argmax(dim=1)
        self.zero_grad()
        logits[0, pred].backward()

        if self.gradients is None:
            return None
        
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
          # gradients: [1, 512, 7, 7] → weights: [1, 512, 1, 1]
        cam = torch.sum(weights * self.feature_maps, dim=1).squeeze(0)
        cam = torch.relu(cam).cpu().detach().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam011 = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam011


# ====================== 4. 训练函数 ======================
def train_model(model, train_loader, val_loader, device, sub_dirs, epochs=20, lr=1e-4):
    """
    训练模型并保存训练历史

    参数:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 设备
        sub_dirs: 训练目录字典
        epochs: 训练轮数
        lr: 学习率
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 记录训练历史
    history = {
        'train_loss': [],
        'val_acc': [],
        'val_loss': [],
        'learning_rate': []
    }

    best_acc = 0.0
    best_epoch = 0

    print("\n" + "="*60)
    print("开始训练...")
    print("="*60)

    for epoch in range(epochs):
        # ==================== 训练阶段 ====================
        model.train()
        total_loss = 0.0
        for seq_list, labels in train_loader:
            seq_list = [img.to(device) for img in seq_list]
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(seq_list)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ==================== 验证阶段 ====================
        model.eval()
        correct, total = 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for seq_list, labels in val_loader:
                # 跳过 batch_size=1 的情况 (BatchNorm 会报错)
                if labels.size(0) == 1:
                    continue

                seq_list = [img.to(device) for img in seq_list]
                labels = labels.to(device)
                logits, _ = model(seq_list)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        # 避免除零错误
        if len(val_loader) == 0 or total == 0:
            avg_val_loss = 0.0
            val_acc = 0.0
        else:
            avg_val_loss = val_loss / max(1, len(val_loader))
            val_acc = correct / max(1, total)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(avg_val_loss)
        history['learning_rate'].append(current_lr)

        # 打印进度
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            best_model_path = os.path.join(sub_dirs['models'], "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  → 保存最佳模型 (Val Acc: {best_acc:.4f})")

        scheduler.step()

    print("="*60)
    print(f"训练完成! 最佳验证准确率: {best_acc:.4f} (Epoch {best_epoch})")
    print("="*60 + "\n")

    # ==================== 保存训练历史 ====================
    # 1. 保存为 CSV
    history_df = pd.DataFrame(history)
    history_df['epoch'] = range(1, epochs + 1)
    history_csv_path = os.path.join(sub_dirs['logs'], 'training_history.csv')
    history_df.to_csv(history_csv_path, index=False)
    print(f"✓ 训练历史已保存: {history_csv_path}")

    # 2. 保存为 JSON
    history_json_path = os.path.join(sub_dirs['logs'], 'training_history.json')
    with open(history_json_path, 'w') as f:
        json.dump(history, f, indent=4)

    # 3. 绘制训练曲线
    plot_training_curves(history, epochs, sub_dirs['plots'])

    # 4. 保存最终模型
    final_model_path = os.path.join(sub_dirs['models'], "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"✓ 最终模型已保存: {final_model_path}")

    return history, best_acc


def plot_training_curves(history, epochs, save_dir):
    """
    绘制训练曲线

    参数:
        history: 训练历史字典
        epochs: 总轮数
        save_dir: 保存目录
    """
    epochs_range = range(1, epochs + 1)

    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 损失曲线
    axes[0, 0].plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)

    # 2. 准确率曲线
    axes[0, 1].plot(epochs_range, history['val_acc'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    best_acc = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_acc) + 1
    axes[0, 1].axhline(y=best_acc, color='r', linestyle='--',
                       label=f'Best: {best_acc:.4f} @ Epoch {best_epoch}')
    axes[0, 1].legend(fontsize=10)

    # 3. 学习率曲线
    axes[1, 0].plot(epochs_range, history['learning_rate'], 'purple', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_yscale('log')

    # 4. 损失vs准确率
    axes[1, 1].plot(history['val_loss'], history['val_acc'], 'o-',
                    color='orange', linewidth=2, markersize=5)
    axes[1, 1].set_xlabel('Validation Loss', fontsize=12)
    axes[1, 1].set_ylabel('Validation Accuracy', fontsize=12)
    axes[1, 1].set_title('Val Loss vs Val Accuracy', fontsize=14, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ 训练曲线已保存: {save_path}")


# ====================== 5. 推理 + Grad-CAM ======================
def predict_and_visualize(model, dataset, person_name, device, max_vis=3):
    model.eval()
    person = next((p for p in dataset.persons if p['name'] == person_name), None)
    if not person:
        print(f"Person {person_name} not found!")
        return

    imgs = []
    paths = []
    for path in person['paths']:
        img = Image.open(path).convert('RGB')
        img_t = transform(img)
        imgs.append(img_t)
        paths.append(path)

    input_list = [torch.stack(imgs).to(device)]  # [1, N, 3, 224, 224]

    with torch.no_grad():
        logits, _ = model(input_list)
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(logits.argmax(dim=1).item())

    print(f"\n=== 融合预测：{person_name} ===")
    print(f"真实标签: {person['label']} | 预测: {pred} | 概率: [{prob[0]:.3f}, {prob[1]:.3f}]")

    os.makedirs("gradcam_fusion", exist_ok=True)
    for i in range(min(max_vis, len(imgs))):
        cam = model.get_cam(imgs[i])
        if cam is not None:
            save_path = f"gradcam_fusion/{person_name}_img{i+1}_pred{pred}.png"
            visualize_cam(paths[i], cam, save_path)


def visualize_cam(img_path, cam, save_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM")
    plt.imshow(img)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ====================== 测试集评估 ======================
def evaluate_test_set(model, test_dataset, device, class_names=['Class 0', 'Class 1'], save_dir='test_results'):
    """
    在测试集上评估模型性能,计算各种指标并可视化

    注意: 为了避免 batch_size=1 的 BatchNorm 问题,
          测试时不使用 DataLoader,而是逐个样本推理

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


# ====================== 6. 保存训练配置 ======================
def save_config(config, save_path):
    """保存训练配置"""
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"✓ 训练配置已保存: {save_path}")


# ====================== 7. 参数解析 ======================
def parse_args():
    """
    解析命令行参数

    返回:
        args: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='训练 ResNet + Attention Fusion 模型,支持 GradCAM 可视化',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据配置
    data_group = parser.add_argument_group('数据配置')
    data_group.add_argument('--root-dir', type=str, default='./data',
                           help='数据根目录')
    data_group.add_argument('--label-excel', type=str, default='./label.xlsx',
                           help='标签 Excel 文件路径')
    data_group.add_argument('--class-names', type=str, nargs=2, default=['0', '1'],
                           help='类别名称 (两个类别)')
    data_group.add_argument('--max-imgs-per-person', type=int, default=1000,
                           help='每人最多使用的图片数量')

    # 训练配置
    train_group = parser.add_argument_group('训练配置')
    train_group.add_argument('--batch-size', type=int, default=2,
                            help='批次大小')
    train_group.add_argument('--epochs', type=int, default=2,
                            help='训练轮数')
    train_group.add_argument('--lr', '--learning-rate', type=float, default=1e-4,
                            dest='learning_rate', help='学习率')
    train_group.add_argument('--weight-decay', type=float, default=1e-5,
                            help='权重衰减系数')

    # 数据划分
    split_group = parser.add_argument_group('数据划分')
    split_group.add_argument('--train-ratio', type=float, default=0.5,
                            help='训练集比例')
    split_group.add_argument('--val-ratio', type=float, default=0.3,
                            help='验证集比例')
    split_group.add_argument('--test-ratio', type=float, default=0.2,
                            help='测试集比例')

    # 其他配置
    other_group = parser.add_argument_group('其他配置')
    other_group.add_argument('--seed', '--random-seed', type=int, default=42,
                            dest='random_seed', help='随机种子')
    other_group.add_argument('--device', type=str, default='auto',
                            choices=['auto', 'cuda', 'cpu'],
                            help='计算设备')
    other_group.add_argument('--num-workers', type=int, default=0,
                            help='DataLoader 的工作进程数')
    other_group.add_argument('--output-dir', type=str, default='./output',
                            help='输出根目录')

    # 模式选择
    mode_group = parser.add_argument_group('运行模式')
    mode_group.add_argument('--train', action='store_true', default=True,
                           help='是否执行训练')
    mode_group.add_argument('--no-train', action='store_false', dest='train',
                           help='跳过训练')
    mode_group.add_argument('--test', action='store_true', default=True,
                           help='是否执行测试')
    mode_group.add_argument('--no-test', action='store_false', dest='test',
                           help='跳过测试')
    mode_group.add_argument('--load-model', type=str, default=None,
                           help='加载已有模型进行测试 (跳过训练)')

    args = parser.parse_args()

    # 自动选择设备
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 验证数据划分比例
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        parser.error(f'数据划分比例之和必须为 1.0, 当前为 {total_ratio:.2f}')

    return args


# ====================== 8. 主程序 ======================
if __name__ == "__main__":
    # ==================== 解析命令行参数 ====================
    args = parse_args()

    # 转换为字典方便保存和打印
    config_dict = vars(args)

    DEVICE = torch.device(args.device)

    print("\n" + "="*60)
    print("训练配置")
    print("="*60)
    for key, value in config_dict.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")

    # ==================== 创建训练文件夹 ====================
    exp_dir, sub_dirs = create_training_folder(base_dir=args.output_dir)

    # 保存配置
    config_path = os.path.join(sub_dirs['logs'], 'config.json')
    save_config(config_dict, config_path)

    # ==================== Transform ====================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # ==================== 加载数据集 ====================
    print("加载数据集...")
    full_dataset = PersonSequenceDataset(
        root_dir=args.root_dir,
        label_excel=args.label_excel,
        transform=transform,
        max_imgs_per_person=args.max_imgs_per_person
    )

    # ==================== 8:1:1 划分数据集 ====================
    print(f"\n数据集划分 (Train:Val:Test = {args.train_ratio:.1f}:{args.val_ratio:.1f}:{args.test_ratio:.1f})...")

    indices = list(range(len(full_dataset)))
    random.seed(args.random_seed)
    random.shuffle(indices)

    n_total = len(indices)
    n_train = int(args.train_ratio * n_total)
    n_val = int(args.val_ratio * n_total)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    print(f"  训练集: {len(train_idx)} 个样本")
    print(f"  验证集: {len(val_idx)} 个样本")
    print(f"  测试集: {len(test_idx)} 个样本")
    print(f"  总计:   {n_total} 个样本\n")

    # 保存数据划分信息
    split_info = {
        'train_indices': train_idx,
        'val_indices': val_idx,
        'test_indices': test_idx,
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx)
    }
    split_path = os.path.join(sub_dirs['logs'], 'data_split.json')
    with open(split_path, 'w') as f:
        json.dump(split_info, f, indent=4)
    print(f"✓ 数据划分信息已保存: {split_path}\n")

    # 创建子数据集
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

    # ==================== DataLoader ====================
    # 注意: batch_size 不能为 1,否则 BatchNorm 会报错
    # 如果数据集太小,自动调整 batch_size
    effective_batch_size = min(args.batch_size, max(2, len(train_idx) // 2))

    if effective_batch_size < args.batch_size:
        print(f"⚠️  数据集较小,自动调整 batch_size: {args.batch_size} → {effective_batch_size}\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=person_sequence_collate_fn,
        pin_memory=False,
        drop_last=True  # 丢弃最后一个不完整的 batch,避免 batch_size=1
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=person_sequence_collate_fn,
        pin_memory=False,
        drop_last=False  # 验证集保留所有数据
    )

    # 测试集不再使用 DataLoader,改用逐样本推理
    # 这样可以完美处理任意数量的测试样本,包括只有1个样本的情况
    print("ℹ️  测试集将使用逐样本推理模式 (无 batch_size 限制)\n")

    # ==================== 创建模型 ====================
    print("创建模型...")
    model = ResNetAttentionFusion(pretrained=True, num_classes=2).to(DEVICE)
    print(f"✓ 模型已加载到设备: {DEVICE}\n")

    # ==================== 训练或加载模型 ====================
    if args.load_model:
        # 加载已有模型
        print(f"加载已有模型: {args.load_model}")
        model.load_state_dict(torch.load(args.load_model, map_location=DEVICE))
        print(f"✓ 模型加载成功\n")
        best_val_acc = 0.0  # 未知
    elif args.train:
        # 训练模型
        history, best_val_acc = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=DEVICE,
            sub_dirs=sub_dirs,
            epochs=args.epochs,
            lr=args.learning_rate
        )
        # 训练后加载最佳模型
        best_model_path = os.path.join(sub_dirs['models'], "best_model.pth")
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    else:
        print("跳过训练 (使用 --train 启用训练)\n")
        best_val_acc = 0.0

    # ==================== 测试集评估 ====================
    if args.test and len(test_idx) > 0:
        print("\n" + "="*60)
        print("在测试集上评估模型...")
        print("="*60 + "\n")

        # 评估 - 直接传入 dataset,不使用 DataLoader
        test_metrics = evaluate_test_set(
            model=model,
            test_dataset=test_dataset,  # 传入 dataset 而非 loader
            device=DEVICE,
            class_names=args.class_names,
            save_dir=sub_dirs['test_results']
        )

        # ==================== 保存最终结果摘要 ====================
        final_results = {
            'best_val_acc': float(best_val_acc),
            'test_accuracy': float(test_metrics['accuracy']),
            'test_precision': float(test_metrics['precision']),
            'test_recall': float(test_metrics['recall']),
            'test_f1': float(test_metrics['f1']),
            'test_auc': float(test_metrics['auc']),
            'training_epochs': args.epochs,
            'device': args.device
        }

        results_path = os.path.join(exp_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=4)

        print("\n" + "="*60)
        print("最终结果摘要")
        print("="*60)
        print(f"  最佳验证准确率: {best_val_acc:.4f}")
        print(f"  测试集准确率:   {test_metrics['accuracy']:.4f}")
        print(f"  测试集 F1:      {test_metrics['f1']:.4f}")
        print(f"  测试集 AUC:     {test_metrics['auc']:.4f}")
        print("="*60)
        print(f"\n✓ 所有训练材料已保存到: {exp_dir}")
        print("="*60 + "\n")
    else:
        print("\n跳过测试评估 (使用 --test 启用测试)\n")

    # ==================== GradCAM 可视化示例 (可选) ====================
    # 如果想要为某个样本生成 GradCAM
    # test_person = "dog1"
    # predict_and_visualize(model, full_dataset, test_person, DEVICE, max_vis=3)