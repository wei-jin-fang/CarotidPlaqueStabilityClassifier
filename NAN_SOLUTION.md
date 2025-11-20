# NaN 问题解决指南

## 🔴 问题现象

训练过程中出现 `loss=nan`（Not a Number），表示训练数值不稳定。

```
Epoch 1/50 | Train Loss: 0.4974 | Val Loss: 2.1927 | Val Acc: 0.0000
Epoch 2/50 | Train Loss: 0.7663 | Val Loss: nan | Val Acc: 1.0000
Epoch 3/50 | Train Loss: nan | Val Loss: nan | Val Acc: 1.0000
```

## 🔍 NaN 出现的原因

### 1. **梯度爆炸** (最常见)
- 学习率过高导致参数更新幅度过大
- 梯度值超出浮点数表示范围

### 2. **数值溢出**
- 混合精度训练 (fp16/bf16) 容易导致数值溢出
- Attention 机制中的 softmax 计算产生极端值

### 3. **BatchNorm 问题**
- 批次太小（batch_size=1）导致统计量不准确
- 多GPU训练时数据分布不均匀

### 4. **数据问题**
- 输入数据存在异常值（如无穷大、NaN）
- 标签错误

### 5. **模型初始化问题**
- 权重初始化不当

## ✅ 已实施的解决方案

我已经在代码中添加了以下修复措施：

### 1. **梯度裁剪** (train.py:383-387)
```python
# 梯度裁剪，防止梯度爆炸
if accelerator:
    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
else:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**作用**: 限制梯度的最大范数为 1.0，防止梯度爆炸

### 2. **Attention 数值稳定性** (train.py:262-263)
```python
attn = self.attention(feats).squeeze(-1)
# 添加数值稳定性：裁剪 attention 分数防止极端值
attn = torch.clamp(attn, min=-10, max=10)
weights = torch.softmax(attn, dim=0).unsqueeze(-1)
```

**作用**: 限制 attention 分数在 [-10, 10] 范围内，防止 softmax 产生极端值

### 3. **NaN 检测和跳过** (train.py:371-375)
```python
# 检查损失是否为 NaN
if torch.isnan(loss) or torch.isinf(loss):
    if accelerator is None or accelerator.is_main_process:
        print(f"\n⚠️  警告: 检测到 NaN/Inf 损失，跳过此批次")
    continue
```

**作用**: 实时检测 NaN/Inf，跳过有问题的批次继续训练

## 🚀 推荐使用的训练配置

### 方法 1: 使用稳定配置脚本（推荐）

```bash
./run_multi_gpu_stable.sh
```

这个脚本包含以下优化：
- ✅ 降低学习率: `5e-5` (原来是 `1e-4`)
- ✅ 关闭混合精度训练 (使用 fp32)
- ✅ 启用梯度裁剪
- ✅ 启用 NaN 检测

### 方法 2: 手动设置参数

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file accelerate_config_stable.yaml \
    train.py \
    --batch-size 4 \
    --epochs 50 \
    --lr 5e-5 \
    --weight-decay 1e-5
```

## 📊 参数调优建议

### 学习率调整

根据训练情况逐步调整：

| 现象 | 学习率调整 | 建议值 |
|------|-----------|--------|
| 快速出现 NaN (1-3 epochs) | 显著降低 | `1e-5` 或 `5e-6` |
| 中期出现 NaN (10-20 epochs) | 适度降低 | `5e-5` |
| 训练稳定但收敛慢 | 适度提高 | `1e-4` |
| 一直稳定 | 可以尝试提高 | `2e-4` |

### 批次大小调整

```bash
# 如果显存充足，可以增大 batch_size 提高稳定性
--batch-size 8   # 每个 GPU 使用 8
--batch-size 16  # 每个 GPU 使用 16
```

**注意**: batch_size 越大，训练越稳定，但需要更多显存

### 梯度裁剪强度调整

如果仍然出现 NaN，可以降低梯度裁剪的阈值：

修改 `train.py:385` 中的 `max_norm`:
```python
# 更激进的梯度裁剪
accelerator.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 从 1.0 降到 0.5
```

## 🔧 进阶解决方案

### 1. 使用更保守的优化器

修改 `train.py:1119`:
```python
# 使用 AdamW 替代 Adam，添加权重衰减
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
                        weight_decay=1e-4, eps=1e-8)
```

### 2. 使用学习率预热 (Warmup)

```python
from torch.optim.lr_scheduler import LinearLR, SequentialLR

# 创建预热调度器
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-5)

scheduler = SequentialLR(optimizer,
                        schedulers=[warmup_scheduler, main_scheduler],
                        milestones=[5])
```

### 3. 检查数据是否有异常

添加数据检查代码：
```python
# 在训练循环开始前检查数据
for seq_list, labels in train_loader:
    for seq in seq_list:
        if torch.isnan(seq).any() or torch.isinf(seq).any():
            print("⚠️  发现异常数据!")
            break
```

### 4. 使用更稳定的损失函数

```python
# 使用 label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

## 📝 训练监控建议

### 在训练时添加额外的监控

```bash
# 同时运行 tensorboard（如果安装了）
tensorboard --logdir=output/

# 在另一个终端监控 GPU
watch -n 1 nvidia-smi
```

### 查看训练日志

训练日志保存在：
```
output/train_YYYYMMDD_HHMMSS/logs/training_history.csv
```

可以使用 pandas 分析：
```python
import pandas as pd
df = pd.read_csv('output/train_YYYYMMDD_HHMMSS/logs/training_history.csv')
print(df[['epoch', 'train_loss', 'val_loss', 'val_acc']])
```

## 🎯 快速检查清单

遇到 NaN 时按照以下顺序检查：

- [ ] 1. 使用 `run_multi_gpu_stable.sh` 重新训练
- [ ] 2. 检查学习率是否过高（降低到 `5e-5` 或更低）
- [ ] 3. 确认关闭了混合精度训练
- [ ] 4. 确认启用了梯度裁剪
- [ ] 5. 检查数据中是否有异常值
- [ ] 6. 增大 batch_size（如果显存允许）
- [ ] 7. 尝试更保守的优化器设置

## 💡 常见问题

### Q1: 为什么关闭混合精度训练？

**A**: 混合精度 (fp16/bf16) 虽然能加速训练，但数值表示范围小，容易溢出。在训练稳定后可以重新启用。

### Q2: 梯度裁剪会影响模型性能吗？

**A**: 轻微的梯度裁剪 (max_norm=1.0) 通常不会影响最终性能，反而能提高训练稳定性。

### Q3: 降低学习率会让训练变慢吗？

**A**: 是的，但稳定性更重要。可以通过增加 epochs 来补偿。

### Q4: NaN 出现后能恢复吗？

**A**: 一旦模型参数变成 NaN，通常无法恢复，需要重新开始训练。这就是为什么我们添加了 NaN 检测和跳过机制。

## 📚 参考资料

- [PyTorch 梯度裁剪文档](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
- [Accelerate 混合精度训练](https://huggingface.co/docs/accelerate/usage_guides/mixed_precision)
- [训练稳定性技巧](https://docs.fast.ai/callback.tracker.html)

---

如果以上方法都无法解决问题，请检查：
1. PyTorch 版本是否过旧
2. CUDA 版本是否兼容
3. GPU 驱动是否正常
4. 数据集是否损坏
