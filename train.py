# -*- coding: utf-8 -*-
"""
完整训练 + 融合 + Grad-CAM
解决：不同人图片数量不一致 → DataLoader stack 错误
"""
# 文件开头统一写
from tqdm import tqdm
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
from accelerate import Accelerator

warnings.filterwarnings("ignore")
import os


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
    def __init__(self, root_dir: str, label_excel: str, transform=None, max_imgs_per_person: int = None,
                 keep_middle_n: int = 100, min_imgs_required: int = 100, verbose: bool = True):
        """
        参数:
            root_dir: 数据根目录
            label_excel: 标签 Excel 文件路径
            transform: 数据增强
            max_imgs_per_person: 已废弃，保留用于兼容性
            keep_middle_n: 保留中间的 N 张图片 (掐头去尾)
            min_imgs_required: 最少需要的图片数量，不足则舍弃该样本
            verbose: 是否打印数据加载统计信息（多进程训练时只在主进程设为True）
        """
        self.root_dir = root_dir
        self.transform = transform
        self.keep_middle_n = keep_middle_n
        self.min_imgs_required = min_imgs_required
        self.verbose = verbose
        self.persons = self._load_persons(label_excel)

    def _load_persons(self, label_excel) -> List[Dict]:
        df = pd.read_excel(label_excel)
        name_to_label = dict(zip(df['name'], df['label']))
        persons = []
        skipped_persons = []  # 记录被跳过的样本

        for person_name in os.listdir(self.root_dir):
            person_dir = os.path.join(self.root_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
            if person_name not in name_to_label:
                if self.verbose:
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

            # 对图片路径进行排序，确保顺序一致
            img_paths = sorted(img_paths)

            # 检查图片数量是否满足最低要求
            total_imgs = len(img_paths)
            if total_imgs < self.min_imgs_required:
                skipped_persons.append({
                    'name': person_name,
                    'count': total_imgs,
                    'label': label
                })
                continue

            # 掐头去尾，保留中间的 keep_middle_n 张图片
            start_idx = (total_imgs - self.keep_middle_n) // 2
            end_idx = start_idx + self.keep_middle_n
            img_paths_selected = img_paths[start_idx:end_idx]

            persons.append({
                'name': person_name,
                'paths': img_paths_selected,
                'label': label,
                'original_count': total_imgs  # 保存原始图片数量
            })

        # 打印统计信息
        if self.verbose:
            print("="*60)
            print(f"数据加载统计:")
            print("="*60)
            print(f"✓ 成功加载: {len(persons)} 个样本")
            print(f"✗ 跳过样本: {len(skipped_persons)} 个 (图片数 < {self.min_imgs_required})")
            print(f"每个样本保留: {self.keep_middle_n} 张图片 (掐头去尾)")

            if skipped_persons:
                print("\n跳过的样本详情:")
                for p in sorted(skipped_persons, key=lambda x: x['name']):
                    print(f"  ✗ {p['name']:15s}: {p['count']:4d} 张 (< {self.min_imgs_required}) | Label: {p['label']}")

            print("\n成功加载的样本:")
            print(f"{'样本名称':15s} {'原始张数':>8s} {'保留张数':>8s} {'标签':>6s}")
            print("-"*60)
            for p in sorted(persons, key=lambda x: x['name']):
                print(f"{p['name']:15s} {p['original_count']:8d} {len(p['paths']):8d} {p['label']:6d}")

            # 统计每个类别的样本数
            label_counts = {}
            for p in persons:
                label_counts[p['label']] = label_counts.get(p['label'], 0) + 1

            print("-"*60)
            print(f"总计: {len(persons)} 个样本, {len(persons) * self.keep_middle_n} 张图片")
            print(f"类别分布: ", end="")
            for label, count in sorted(label_counts.items()):
                print(f"Label {label}: {count} 个  ", end="")
            print("\n" + "="*60)

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


# ====================== 3. 模型：ResNet34 + 序列建模（BiLSTM + Temporal Conv + Attention）======================
class ResNetSequenceClassifier(nn.Module):
    def __init__(self, pretrained=True, num_classes=2, dropout=0.5, freeze_early_layers=True):
        """
        序列建模版本：考虑帧间关系的颈动脉斑块分类模型

        架构：
        1. ResNet34: 提取每帧的空间特征 (冻结layer1, layer2)
        2. BiLSTM: 捕获双向时序依赖关系
        3. Temporal Conv1D: 提取局部时序模式
        4. Multi-Head Attention: 学习帧间重要性
        5. 深度分类器: 融合特征进行分类

        参数:
            pretrained: 是否使用预训练权重
            num_classes: 分类数量
            dropout: Dropout比率
            freeze_early_layers: 是否冻结早期层
        """
        super().__init__()

        # ============ 1. 空间特征提取器：ResNet34 ============
        resnet = models.resnet34(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # [*, 512, 7, 7]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # [*, 512]

        # 冻结早期层
        if freeze_early_layers:
            for param in resnet.layer1.parameters():
                param.requires_grad = False
            for param in resnet.layer2.parameters():
                param.requires_grad = False
            print("✓ 冻结 ResNet34 的 layer1 和 layer2")

        # ============ 2. 序列建模：BiLSTM ============
        self.lstm_hidden_size = 256
        self.bilstm = nn.LSTM(
            input_size=512,
            hidden_size=self.lstm_hidden_size,
            num_layers=2,               # 2层LSTM
            batch_first=True,           # 输入格式: [batch, seq_len, feature]
            bidirectional=True,         # 双向LSTM
            dropout=dropout * 0.6       # LSTM层间dropout
        )
        # BiLSTM输出: 512维 (256 forward + 256 backward)

        # ============ 3. 时序卷积：捕获局部时序模式 ============
        # 使用1D卷积在时间维度上提取特征
        self.temporal_conv = nn.Sequential(
            # 卷积核大小3: 捕获连续3帧的模式
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),

            # 卷积核大小5: 捕获连续5帧的模式
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # ============ 4. Multi-Head Self-Attention ============
        self.num_heads = 8
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=512,              # BiLSTM的输出维度
            num_heads=self.num_heads,
            dropout=dropout * 0.3,
            batch_first=True
        )

        # ============ 5. 时序注意力（学习每帧的重要性）============
        self.temporal_attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(256, 1)
        )

        # ============ 6. 特征融合层 ============
        # 融合BiLSTM特征(512) + 时序卷积特征(128)
        self.fusion = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.5)
        )

        # ============ 7. 深度分类器 ============
        self.classifier = nn.Sequential(
            # 第1层
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),

            # 第2层
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.7),

            # 第3层
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout * 0.5),

            # 输出层
            nn.Linear(128, num_classes)
        )

        # ============ Grad-CAM 钩子 ============
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
        序列建模的前向传播

        输入:
            x: List[Tensor], 每个Tensor形状为 [N_i, 3, 224, 224]
               N_i是第i个样本的序列长度（可能不同）

        输出:
            logits: [Batch, num_classes] 分类logits
            fused: [Batch, 512] 融合后的特征（用于可视化等）
        """
        if not isinstance(x, list):
            x = [x]

        # ============ 步骤1: 批量提取空间特征 ============
        # 将所有序列合并成一个大batch处理（避免BatchNorm问题）
        concat_imgs = torch.cat(x, dim=0)  # [sum(N_i), 3, 224, 224]

        if not concat_imgs.is_contiguous():
            concat_imgs = concat_imgs.contiguous()

        # 提取CNN特征
        with torch.cuda.amp.autocast(enabled=False):
            concat_feats = self.feature_extractor(concat_imgs)  # [sum(N_i), 512, 7, 7]
            concat_pooled = self.avgpool(concat_feats).view(concat_imgs.size(0), -1)  # [sum(N_i), 512]

        # ============ 步骤2: 拆分回各个序列 ============
        sequence_features = []
        start_idx = 0
        for seq in x:
            seq_len = seq.size(0)  # 当前序列长度
            seq_feats = concat_pooled[start_idx:start_idx+seq_len]  # [seq_len, 512]
            sequence_features.append(seq_feats)
            start_idx += seq_len

        # ============ 步骤3: 序列建模 ============
        lstm_outputs = []
        temporal_conv_outputs = []

        for seq_feats in sequence_features:  # seq_feats: [seq_len, 512]
            seq_len = seq_feats.size(0)

            if seq_len == 1:
                # 单帧情况：跳过LSTM和时序卷积
                lstm_out = seq_feats  # [1, 512]
                temp_conv_out = torch.zeros(1, 128, device=seq_feats.device)  # [1, 128]
            else:
                # -------- BiLSTM --------
                seq_feats_unsq = seq_feats.unsqueeze(0)  # [1, seq_len, 512]
                lstm_out, _ = self.bilstm(seq_feats_unsq)  # [1, seq_len, 512]
                lstm_out = lstm_out.squeeze(0)  # [seq_len, 512]

                # -------- Multi-Head Self-Attention --------
                # 学习序列中不同帧之间的关系
                attn_out, _ = self.multihead_attn(
                    lstm_out.unsqueeze(0),  # query: [1, seq_len, 512]
                    lstm_out.unsqueeze(0),  # key: [1, seq_len, 512]
                    lstm_out.unsqueeze(0)   # value: [1, seq_len, 512]
                )  # output: [1, seq_len, 512]
                attn_out = attn_out.squeeze(0)  # [seq_len, 512]

                # 残差连接
                lstm_out = lstm_out + attn_out  # [seq_len, 512]

                # -------- Temporal Attention（加权融合序列）--------
                attn_scores = self.temporal_attention(lstm_out).squeeze(-1)  # [seq_len]
                attn_scores = torch.clamp(attn_scores, min=-10, max=10)  # 数值稳定性
                attn_weights = torch.softmax(attn_scores, dim=0).unsqueeze(-1)  # [seq_len, 1]
                lstm_out = torch.sum(lstm_out * attn_weights, dim=0, keepdim=True)  # [1, 512]

                # -------- Temporal Convolution --------
                # 提取局部时序模式（连续帧的变化）
                seq_feats_t = seq_feats.t().unsqueeze(0)  # [1, 512, seq_len]
                temp_conv_out = self.temporal_conv(seq_feats_t)  # [1, 128, seq_len]
                temp_conv_out = torch.mean(temp_conv_out, dim=2)  # [1, 128] 时序平均池化

            lstm_outputs.append(lstm_out.squeeze(0) if lstm_out.dim() > 1 else lstm_out)
            temporal_conv_outputs.append(temp_conv_out.squeeze(0) if temp_conv_out.dim() > 1 else temp_conv_out)

        # ============ 步骤4: 堆叠batch ============
        lstm_features = torch.stack(lstm_outputs)  # [Batch, 512]
        temporal_features = torch.stack(temporal_conv_outputs)  # [Batch, 128]

        # ============ 步骤5: 特征融合 ============
        fused = torch.cat([lstm_features, temporal_features], dim=1)  # [Batch, 640]
        fused = self.fusion(fused)  # [Batch, 512]

        # ============ 步骤6: 分类 ============
        logits = self.classifier(fused)  # [Batch, num_classes]

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
def train_model(model, train_loader, val_loader, device, sub_dirs, epochs=20, lr=1e-4, weight_decay=1e-5, patience=10, accelerator=None):
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
        weight_decay: 权重衰减系数
        patience: 早停耐心值（验证集无提升的最大epoch数）
        accelerator: Accelerator 对象用于分布式训练
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
    patience_counter = 0  # 早停计数器

    # 使用 accelerator.print() 确保只在主进程打印
    if accelerator:
        accelerator.print("\n" + "="*60)
        accelerator.print("开始训练...")
        accelerator.print("="*60)
    else:
        print("\n" + "="*60)
        print("开始训练...")
        print("="*60)

    from tqdm import tqdm

    for epoch in range(epochs):
        # ---------------- 训练 ----------------
        model.train()
        total_loss = 0.0
        avg_train_loss = 0.0   # 默认值

        # 只在主进程显示进度条
        if accelerator is None or accelerator.is_main_process:
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        else:
            pbar = train_loader

        for seq_list, labels in pbar:
            # accelerate 会自动处理设备转移
            if accelerator is None:
                seq_list = [img.to(device) for img in seq_list]
                labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(seq_list)
            loss = criterion(logits, labels)

            # 检查损失是否为 NaN
            if torch.isnan(loss) or torch.isinf(loss):
                if accelerator is None or accelerator.is_main_process:
                    print(f"\n⚠️  警告: 检测到 NaN/Inf 损失，跳过此批次")
                continue

            # 使用 accelerator.backward() 或标准 backward()
            if accelerator:
                accelerator.backward(loss)
            else:
                loss.backward()

            # 梯度裁剪，防止梯度爆炸
            if accelerator:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            # 只在主进程更新进度条
            if accelerator is None or accelerator.is_main_process:
                if hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                    'avg_loss': f'{total_loss/(pbar.n+1):.4f}'})

        # 整个 epoch 的平均训练损失
        avg_train_loss = total_loss / len(train_loader) if len(train_loader) else 0.0

        # ---------------- 验证 ----------------
        model.eval()
        correct = total = 0
        val_loss = 0.0
        with torch.no_grad():
            for seq_list, labels in val_loader:
                if labels.size(0) == 1:        # 跳过 batch_size=1
                    continue

                if accelerator is None:
                    seq_list = [img.to(device) for img in seq_list]
                    labels = labels.to(device)

                logits, _ = model(seq_list)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        # 在多GPU环境下同步指标
        if accelerator and accelerator.num_processes > 1:
            # 使用 all_reduce 来同步指标，兼容旧版本 PyTorch
            correct_tensor = torch.tensor(correct, dtype=torch.float32, device=accelerator.device)
            total_tensor = torch.tensor(total, dtype=torch.float32, device=accelerator.device)
            val_loss_tensor = torch.tensor(val_loss, dtype=torch.float32, device=accelerator.device)

            # 在所有进程间求和
            torch.distributed.all_reduce(correct_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(val_loss_tensor, op=torch.distributed.ReduceOp.SUM)

            correct = int(correct_tensor.item())
            total = int(total_tensor.item())
            val_loss = val_loss_tensor.item()

        avg_val_loss = val_loss / max(1, len(val_loader)) if total else 0.0
        val_acc = correct / max(1, total) if total else 0.0
        current_lr = optimizer.param_groups[0]['lr']

        # ---------------- 记录 & 打印 ----------------
        history['train_loss'].append(avg_train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(avg_val_loss)
        history['learning_rate'].append(current_lr)

        # 只在主进程打印
        if accelerator is None or accelerator.is_main_process:
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f}")

        # ---------------- 保存最佳模型 & 早停检查 ----------------
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0  # 重置早停计数器

            # 使用 accelerator.save() 保存模型
            if accelerator is None or accelerator.is_main_process:
                if accelerator:
                    # 保存解包的模型
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save(unwrapped_model.state_dict(), os.path.join(sub_dirs['models'], "best_model.pth"))
                else:
                    torch.save(model.state_dict(), os.path.join(sub_dirs['models'], "best_model.pth"))
                print(f"  → 保存最佳模型 (Val Acc: {best_acc:.4f})")
        else:
            patience_counter += 1
            if accelerator is None or accelerator.is_main_process:
                print(f"  ⚠️  验证准确率未提升 ({patience_counter}/{patience})")

            # 早停检查
            if patience_counter >= patience:
                if accelerator is None or accelerator.is_main_process:
                    print(f"\n⛔ 早停！连续 {patience} 个epoch验证准确率未提升")
                    print(f"最佳验证准确率: {best_acc:.4f} (Epoch {best_epoch})")
                break

        scheduler.step()

    # 只在主进程打印
    if accelerator is None or accelerator.is_main_process:
        print("="*60)
        print(f"训练完成! 最佳验证准确率: {best_acc:.4f} (Epoch {best_epoch})")
        print("="*60 + "\n")

    # ==================== 保存训练历史 ====================
    # 只在主进程保存
    if accelerator is None or accelerator.is_main_process:
        # 1. 保存为 CSV
        actual_epochs = len(history['train_loss'])  # 实际训练的epoch数（可能因早停而少于设定值）
        history_df = pd.DataFrame(history)
        history_df['epoch'] = range(1, actual_epochs + 1)
        history_csv_path = os.path.join(sub_dirs['logs'], 'training_history.csv')
        history_df.to_csv(history_csv_path, index=False)
        print(f"✓ 训练历史已保存: {history_csv_path}")

        # 2. 保存为 JSON
        history_json_path = os.path.join(sub_dirs['logs'], 'training_history.json')
        with open(history_json_path, 'w') as f:
            json.dump(history, f, indent=4)

        # 3. 绘制训练曲线
        plot_training_curves(history, actual_epochs, sub_dirs['plots'])

        # 4. 保存最终模型
        final_model_path = os.path.join(sub_dirs['models'], "final_model.pth")
        if accelerator:
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), final_model_path)
        else:
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

    # 1. 训练集损失
    axes[0, 0].plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)

    # 2. 验证集损失
    axes[0, 1].plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].set_title('Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(alpha=0.3)

    # 3. 验证准确率曲线
    axes[1, 0].plot(epochs_range, history['val_acc'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Accuracy', fontsize=12)
    axes[1, 0].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    best_acc = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_acc) + 1
    axes[1, 0].axhline(y=best_acc, color='r', linestyle='--',
                       label=f'Best: {best_acc:.4f} @ Epoch {best_epoch}')
    axes[1, 0].legend(fontsize=10)

    # 4. 学习率曲线
    axes[1, 1].plot(epochs_range, history['learning_rate'], 'purple', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ 训练曲线已保存: {save_path}")


# ====================== 5. 推理 + Grad-CAM ======================
def predict_and_visualize(model, dataset, person_name, device, transform, max_vis=3):
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
    data_group.add_argument('--root-dir', type=str, default='/seu_nvme/home/shendian/220256451/datasets/Carotid_artery/Carotid_artery',
                           help='数据根目录')
    data_group.add_argument('--label-excel', type=str, default='/seu_nvme/home/shendian/220256451/datasets/Carotid_artery/label.xlsx',
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
    train_group.add_argument('--weight-decay', type=float, default=1e-4,
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
    other_group.add_argument('--device', type=str, default='cuda',
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
    # ==================== 初始化 Accelerator ====================
    accelerator = Accelerator()

    # ==================== 解析命令行参数 ====================
    args = parse_args()

    # 转换为字典方便保存和打印
    config_dict = vars(args)

    # 使用 accelerator.device 替代手动设备管理
    DEVICE = accelerator.device

    # 只在主进程打印
    accelerator.print("\n" + "="*60)
    accelerator.print("训练配置")
    accelerator.print("="*60)
    accelerator.print(f"  当前进程: {accelerator.process_index + 1}/{accelerator.num_processes}")
    accelerator.print(f"  使用设备: {accelerator.device}")
    accelerator.print(f"  混合精度: {accelerator.mixed_precision}")
    for key, value in config_dict.items():
        accelerator.print(f"  {key}: {value}")
    accelerator.print("="*60 + "\n")

    # ==================== 创建训练文件夹 ====================
    # 只在主进程创建文件夹
    if accelerator.is_main_process:
        exp_dir, sub_dirs = create_training_folder(base_dir=args.output_dir)
    else:
        # 等待主进程创建完成,然后获取路径
        exp_dir = None
        sub_dirs = None

    # 广播文件夹路径到所有进程
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        # 从主进程同步目录信息
        import time
        time.sleep(2)  # 等待主进程创建完成
        # 获取最新的训练文件夹
        output_dirs = sorted([d for d in os.listdir(args.output_dir) if d.startswith('train_')], reverse=True)
        if output_dirs:
            exp_dir = os.path.join(args.output_dir, output_dirs[0])
            sub_dirs = {
                'root': exp_dir,
                'models': os.path.join(exp_dir, 'models'),
                'logs': os.path.join(exp_dir, 'logs'),
                'test_results': os.path.join(exp_dir, 'test_results'),
                'gradcam': os.path.join(exp_dir, 'gradcam'),
                'plots': os.path.join(exp_dir, 'plots'),
            }

    accelerator.wait_for_everyone()

    # 只在主进程保存配置
    if accelerator.is_main_process:
        config_path = os.path.join(sub_dirs['logs'], 'config.json')
        save_config(config_dict, config_path)

    # ==================== Transform ====================
    # 训练集 Transform - 针对医学超声图像的保守增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),          # 随机裁剪（模拟位置偏差）
        transforms.RandomHorizontalFlip(p=0.5),     # 水平翻转（左右颈动脉对称）
        transforms.RandomRotation(degrees=10),      # 轻微旋转±10度（保守，避免破坏结构）
        transforms.RandomAffine(                    # 轻微仿射变换
            degrees=0,
            translate=(0.05, 0.05),  # 平移±5%（更保守）
            scale=(0.95, 1.05)       # 缩放95%-105%（更保守）
        ),
        # 医学图像特有增强 - 非常轻微的颜色调整
        transforms.ColorJitter(
            brightness=0.1,    # 亮度±10%（模拟超声探头压力差异）
            contrast=0.1,      # 对比度±10%（模拟设备设置差异）
            saturation=0.0,    # 不调整饱和度（超声图像通常是灰度）
            hue=0.0           # 不调整色调
        ),
        # 可选：添加轻微高斯模糊（模拟图像质量差异）
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        transforms.ToTensor(),
        # 随机添加轻微噪声（在ToTensor之后，通过Lambda实现）
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.02 if torch.rand(1) > 0.5 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 验证集/测试集 Transform - 仅基础处理
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # ==================== 加载数据集 ====================
    accelerator.print("加载数据集...")

    # 使用主进程的随机种子确保所有进程数据划分一致
    with accelerator.main_process_first():
        # 先创建临时数据集获取所有样本信息（用于划分）
        temp_dataset = PersonSequenceDataset(
            root_dir=args.root_dir,
            label_excel=args.label_excel,
            transform=val_transform,  # 临时使用val_transform
            max_imgs_per_person=args.max_imgs_per_person,
            verbose=accelerator.is_main_process  # 只在主进程打印统计信息
        )
        all_persons = temp_dataset.persons  # 获取所有person信息

    # ==================== 划分数据集 ====================
    accelerator.print(f"\n数据集划分 (Train:Val:Test = {args.train_ratio:.1f}:{args.val_ratio:.1f}:{args.test_ratio:.1f})...")

    indices = list(range(len(all_persons)))
    random.seed(args.random_seed)
    random.shuffle(indices)

    n_total = len(indices)
    n_train = int(args.train_ratio * n_total)
    n_val = int(args.val_ratio * n_total)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    accelerator.print(f"  训练集: {len(train_idx)} 个样本")
    accelerator.print(f"  验证集: {len(val_idx)} 个样本")
    accelerator.print(f"  测试集: {len(test_idx)} 个样本")
    accelerator.print(f"  总计:   {n_total} 个样本\n")

    # 根据索引划分persons列表
    train_persons = [all_persons[i] for i in train_idx]
    val_persons = [all_persons[i] for i in val_idx]
    test_persons = [all_persons[i] for i in test_idx]

    # 创建三个独立的数据集，使用不同的transform
    with accelerator.main_process_first():
        # 训练集 - 使用强数据增强
        train_dataset = PersonSequenceDataset(
            root_dir=args.root_dir,
            label_excel=args.label_excel,
            transform=train_transform,
            max_imgs_per_person=args.max_imgs_per_person,
            verbose=False  # 不打印统计信息（persons会被替换）
        )
        train_dataset.persons = train_persons  # 替换为训练集的persons

        # 验证集 - 使用基础transform
        val_dataset = PersonSequenceDataset(
            root_dir=args.root_dir,
            label_excel=args.label_excel,
            transform=val_transform,
            max_imgs_per_person=args.max_imgs_per_person,
            verbose=False  # 不打印统计信息（persons会被替换）
        )
        val_dataset.persons = val_persons  # 替换为验证集的persons

        # 测试集 - 使用基础transform
        test_dataset = PersonSequenceDataset(
            root_dir=args.root_dir,
            label_excel=args.label_excel,
            transform=val_transform,
            max_imgs_per_person=args.max_imgs_per_person,
            verbose=False  # 不打印统计信息（persons会被替换）
        )
        test_dataset.persons = test_persons  # 替换为测试集的persons

    accelerator.print(f"✓ 数据集创建完成:")
    accelerator.print(f"  训练集: {len(train_dataset)} 个样本 (使用数据增强)")
    accelerator.print(f"  验证集: {len(val_dataset)} 个样本")
    accelerator.print(f"  测试集: {len(test_dataset)} 个样本\n")

    # 只在主进程保存数据划分信息
    if accelerator.is_main_process:
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

    # ==================== DataLoader ====================
    # 注意: batch_size 不能为 1,否则 BatchNorm 会报错
    # 如果数据集太小,自动调整 batch_size
    effective_batch_size = min(args.batch_size, max(2, len(train_dataset) // 2))

    if effective_batch_size < args.batch_size:
        accelerator.print(f"⚠️  数据集较小,自动调整 batch_size: {args.batch_size} → {effective_batch_size}\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=person_sequence_collate_fn,
        pin_memory=True,
        drop_last=False  # 丢弃最后一个不完整的 batch,避免 batch_size=1
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=person_sequence_collate_fn,
        pin_memory=True,
        drop_last=False  # 验证集保留所有数据
    )

    # 测试集不再使用 DataLoader,改用逐样本推理
    # 这样可以完美处理任意数量的测试样本,包括只有1个样本的情况
    accelerator.print("ℹ️  测试集将使用逐样本推理模式 (无 batch_size 限制)\n")

    # ==================== 创建模型 ====================
    accelerator.print("创建模型...")
    accelerator.print("使用 ResNet34 + 序列建模（BiLSTM + Temporal Conv + Multi-Head Attention）")
    model = ResNetSequenceClassifier(
        pretrained=True,
        num_classes=2,
        dropout=0.5,              # Dropout比率
        freeze_early_layers=True  # 冻结layer1和layer2
    )

    # ==================== 准备优化器和调度器 ====================
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ==================== 使用 Accelerator 准备模型和数据加载器 ====================
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    accelerator.print(f"✓ 模型已加载到设备: {accelerator.device}")
    accelerator.print(f"✓ 使用 {accelerator.num_processes} 个进程进行训练\n")

    # ==================== 训练或加载模型 ====================
    if args.load_model:
        # 加载已有模型
        accelerator.print(f"加载已有模型: {args.load_model}")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(torch.load(args.load_model, map_location=DEVICE))
        accelerator.print(f"✓ 模型加载成功\n")
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
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=100,  # 早停耐心值
            accelerator=accelerator
        )
        # 等待所有进程完成训练
        accelerator.wait_for_everyone()

        # 训练后加载最佳模型
        if accelerator.is_main_process:
            best_model_path = os.path.join(sub_dirs['models'], "best_model.pth")
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    else:
        accelerator.print("跳过训练 (使用 --train 启用训练)\n")
        best_val_acc = 0.0

    # ==================== 测试集评估 ====================
    # 只在主进程进行测试
    if args.test and len(test_idx) > 0 and accelerator.is_main_process:
        accelerator.print("\n" + "="*60)
        accelerator.print("在测试集上评估模型...")
        accelerator.print("="*60 + "\n")

        # 评估 - 直接传入 dataset,不使用 DataLoader
        # 需要使用解包的模型进行测试
        unwrapped_model = accelerator.unwrap_model(model)
        test_metrics = evaluate_test_set(
            model=unwrapped_model,
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
            'device': str(accelerator.device),
            'num_processes': accelerator.num_processes
        }

        results_path = os.path.join(exp_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=4)

        accelerator.print("\n" + "="*60)
        accelerator.print("最终结果摘要")
        accelerator.print("="*60)
        accelerator.print(f"  最佳验证准确率: {best_val_acc:.4f}")
        accelerator.print(f"  测试集准确率:   {test_metrics['accuracy']:.4f}")
        accelerator.print(f"  测试集 F1:      {test_metrics['f1']:.4f}")
        accelerator.print(f"  测试集 AUC:     {test_metrics['auc']:.4f}")
        accelerator.print("="*60)
        accelerator.print(f"\n✓ 所有训练材料已保存到: {exp_dir}")
        accelerator.print("="*60 + "\n")
    else:
        if not (args.test and len(test_idx) > 0):
            accelerator.print("\n跳过测试评估 (使用 --test 启用测试)\n")

    # ==================== GradCAM 可视化示例 (可选) ====================
    # 如果想要为某个样本生成 GradCAM
    # test_person = "dog1"
    # predict_and_visualize(model, full_dataset, test_person, DEVICE, max_vis=3)
