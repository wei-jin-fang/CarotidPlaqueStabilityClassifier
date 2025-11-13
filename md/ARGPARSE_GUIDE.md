# 命令行参数使用指南

## 🎯 概述

训练代码已优化为使用 `argparse` 进行参数配置,所有参数都是**可选的**,有合理的默认值。

---

## 🚀 快速开始

### 1. 使用默认参数训练
```bash
python train_gradcam.py
```

### 2. 查看所有可用参数
```bash
python train_gradcam.py --help
```

### 3. 自定义参数训练
```bash
python train_gradcam.py \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.001 \
    --class-names cat dog
```

---

## 📋 参数详解

### **数据配置**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--root-dir` | str | `./data` | 数据根目录 |
| `--label-excel` | str | `./label.xlsx` | 标签 Excel 文件路径 |
| `--class-names` | str × 2 | `0 1` | 类别名称 (两个) |
| `--max-imgs-per-person` | int | `1000` | 每人最多使用的图片数 |

**示例:**
```bash
# 指定数据目录和类别名称
python train_gradcam.py \
    --root-dir /path/to/data \
    --label-excel /path/to/label.xlsx \
    --class-names cat dog
```

---

### **训练配置**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--batch-size` | int | `8` | 批次大小 |
| `--epochs` | int | `50` | 训练轮数 |
| `--lr` / `--learning-rate` | float | `1e-4` | 学习率 |
| `--weight-decay` | float | `1e-5` | 权重衰减系数 |

**示例:**
```bash
# 快速测试 (少轮数)
python train_gradcam.py --epochs 5 --batch-size 4

# 完整训练
python train_gradcam.py --epochs 100 --lr 0.0001

# 大批次训练 (需要更多显存)
python train_gradcam.py --batch-size 32 --lr 0.001
```

---

### **数据划分**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--train-ratio` | float | `0.8` | 训练集比例 |
| `--val-ratio` | float | `0.1` | 验证集比例 |
| `--test-ratio` | float | `0.1` | 测试集比例 |

**注意:** 三个比例之和必须为 1.0

**示例:**
```bash
# 7:2:1 划分
python train_gradcam.py \
    --train-ratio 0.7 \
    --val-ratio 0.2 \
    --test-ratio 0.1

# 9:0.5:0.5 划分 (更多训练数据)
python train_gradcam.py \
    --train-ratio 0.9 \
    --val-ratio 0.05 \
    --test-ratio 0.05
```

---

### **其他配置**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--seed` / `--random-seed` | int | `42` | 随机种子 |
| `--device` | str | `auto` | 计算设备 (`auto`/`cuda`/`cpu`) |
| `--num-workers` | int | `0` | DataLoader 工作进程数 |
| `--output-dir` | str | `.` | 输出根目录 |

**示例:**
```bash
# 强制使用 CPU
python train_gradcam.py --device cpu

# 使用多进程加载数据
python train_gradcam.py --num-workers 4

# 指定输出目录
python train_gradcam.py --output-dir ./experiments
```

---

### **运行模式**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--train` | flag | `True` | 执行训练 |
| `--no-train` | flag | - | 跳过训练 |
| `--test` | flag | `True` | 执行测试 |
| `--no-test` | flag | - | 跳过测试 |
| `--load-model` | str | `None` | 加载已有模型路径 |

**示例:**
```bash
# 只训练,不测试
python train_gradcam.py --no-test

# 只测试,不训练 (需要提供模型)
python train_gradcam.py --no-train --load-model ./train_xxx/models/best_model.pth

# 加载模型继续测试
python train_gradcam.py \
    --load-model ./previous_train/models/best_model.pth \
    --no-train
```

---

## 💡 常用场景

### **场景 1: 快速测试代码**
```bash
python train_gradcam.py \
    --epochs 2 \
    --batch-size 4 \
    --max-imgs-per-person 100
```

### **场景 2: 完整训练**
```bash
python train_gradcam.py \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.0001 \
    --class-names cat dog
```

### **场景 3: 调整数据划分**
```bash
python train_gradcam.py \
    --train-ratio 0.7 \
    --val-ratio 0.2 \
    --test-ratio 0.1
```

### **场景 4: 只评估已有模型**
```bash
python train_gradcam.py \
    --no-train \
    --load-model ./train_20250105_143025/models/best_model.pth \
    --class-names cat dog
```

### **场景 5: 多GPU训练 (需要修改代码支持)**
```bash
python train_gradcam.py \
    --device cuda \
    --batch-size 32 \
    --num-workers 4
```

### **场景 6: 使用不同随机种子**
```bash
# 实验 1
python train_gradcam.py --seed 42

# 实验 2
python train_gradcam.py --seed 123

# 实验 3
python train_gradcam.py --seed 456
```

### **场景 7: 指定输出目录**
```bash
python train_gradcam.py \
    --output-dir ./experiments/exp001 \
    --epochs 50
```

---

## 🔧 组合使用示例

### **完整的生产环境训练命令**
```bash
python train_gradcam.py \
    --root-dir /data/cats_vs_dogs \
    --label-excel /data/labels.xlsx \
    --class-names cat dog \
    --batch-size 16 \
    --epochs 100 \
    --lr 0.0001 \
    --weight-decay 1e-5 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --seed 42 \
    --device cuda \
    --num-workers 4 \
    --output-dir ./production_runs
```

### **实验对比命令**
```bash
# 实验 A: 小学习率
python train_gradcam.py --lr 0.00001 --epochs 100 --seed 42

# 实验 B: 大学习率
python train_gradcam.py --lr 0.001 --epochs 100 --seed 42

# 实验 C: 不同数据划分
python train_gradcam.py --train-ratio 0.9 --val-ratio 0.05 --test-ratio 0.05
```

---

## 📊 参数优先级

```
命令行参数 > 默认值
```

所有参数都可以通过命令行覆盖默认值。

---

## ⚠️ 注意事项

### 1. **数据划分比例**
```bash
# ✅ 正确 (总和为 1.0)
--train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

# ❌ 错误 (总和不为 1.0)
--train-ratio 0.8 --val-ratio 0.2 --test-ratio 0.2
```

### 2. **类别名称**
```bash
# ✅ 正确 (恰好两个类别)
--class-names cat dog

# ❌ 错误 (超过两个类别)
--class-names cat dog bird
```

### 3. **模型加载**
```bash
# 如果使用 --load-model,会跳过训练
# 除非同时指定 --train

# 只测试
python train_gradcam.py --load-model path/to/model.pth --no-train

# 加载模型后继续训练 (不推荐,可能覆盖原模型)
# python train_gradcam.py --load-model path/to/model.pth --train
```

### 4. **显存不足**
```bash
# 减小批次大小
python train_gradcam.py --batch-size 4

# 或减少每人图片数
python train_gradcam.py --max-imgs-per-person 500
```

---

## 📝 帮助信息

运行以下命令查看完整帮助:
```bash
python train_gradcam.py --help
```

输出示例:
```
usage: train_gradcam.py [-h] [--root-dir ROOT_DIR] [--label-excel LABEL_EXCEL]
                        [--class-names CLASS_NAMES CLASS_NAMES]
                        [--max-imgs-per-person MAX_IMGS_PER_PERSON]
                        [--batch-size BATCH_SIZE] [--epochs EPOCHS]
                        [--lr LEARNING_RATE] [--weight-decay WEIGHT_DECAY]
                        [--train-ratio TRAIN_RATIO] [--val-ratio VAL_RATIO]
                        [--test-ratio TEST_RATIO] [--seed RANDOM_SEED]
                        [--device {auto,cuda,cpu}] [--num-workers NUM_WORKERS]
                        [--output-dir OUTPUT_DIR] [--train] [--no-train]
                        [--test] [--no-test] [--load-model LOAD_MODEL]

训练 ResNet + Attention Fusion 模型,支持 GradCAM 可视化

optional arguments:
  -h, --help            show this help message and exit

数据配置:
  --root-dir ROOT_DIR   数据根目录 (default: ./data)
  --label-excel LABEL_EXCEL
                        标签 Excel 文件路径 (default: ./label.xlsx)
  --class-names CLASS_NAMES CLASS_NAMES
                        类别名称 (两个类别) (default: ['0', '1'])
  --max-imgs-per-person MAX_IMGS_PER_PERSON
                        每人最多使用的图片数量 (default: 1000)

训练配置:
  --batch-size BATCH_SIZE
                        批次大小 (default: 8)
  --epochs EPOCHS       训练轮数 (default: 50)
  --lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        学习率 (default: 0.0001)
  --weight-decay WEIGHT_DECAY
                        权重衰减系数 (default: 1e-05)

数据划分:
  --train-ratio TRAIN_RATIO
                        训练集比例 (default: 0.8)
  --val-ratio VAL_RATIO
                        验证集比例 (default: 0.1)
  --test-ratio TEST_RATIO
                        测试集比例 (default: 0.1)

其他配置:
  --seed RANDOM_SEED, --random-seed RANDOM_SEED
                        随机种子 (default: 42)
  --device {auto,cuda,cpu}
                        计算设备 (default: auto)
  --num-workers NUM_WORKERS
                        DataLoader 的工作进程数 (default: 0)
  --output-dir OUTPUT_DIR
                        输出根目录 (default: .)

运行模式:
  --train               是否执行训练 (default: True)
  --no-train            跳过训练
  --test                是否执行测试 (default: True)
  --no-test             跳过测试
  --load-model LOAD_MODEL
                        加载已有模型进行测试 (跳过训练) (default: None)
```

---

## 🎯 最佳实践

### 1. **开发阶段**
使用小数据量和少轮数快速迭代:
```bash
python train_gradcam.py \
    --epochs 5 \
    --batch-size 4 \
    --max-imgs-per-person 100
```

### 2. **实验阶段**
系统性地测试不同超参数:
```bash
# 创建实验脚本
for lr in 0.0001 0.001 0.01; do
    python train_gradcam.py \
        --lr $lr \
        --epochs 50 \
        --output-dir ./experiments/lr_${lr}
done
```

### 3. **生产阶段**
使用完整数据和最优参数:
```bash
python train_gradcam.py \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.0001 \
    --seed 42 \
    --output-dir ./final_model
```

---

## 🔍 调试技巧

### 1. **检查参数**
```bash
# 只打印配置,不训练
python train_gradcam.py --no-train --no-test
```

### 2. **快速验证代码**
```bash
# 最小配置测试
python train_gradcam.py \
    --epochs 1 \
    --batch-size 2 \
    --max-imgs-per-person 50
```

### 3. **测试模型加载**
```bash
python train_gradcam.py \
    --load-model path/to/model.pth \
    --no-train \
    --batch-size 1
```

---

## ✅ 优势总结

✅ **无需修改代码** - 所有配置通过命令行传递
✅ **灵活性高** - 每个参数都可独立调整
✅ **无必填参数** - 所有参数都有合理默认值
✅ **易于批处理** - 方便编写脚本批量实验
✅ **完整文档** - `--help` 提供详细说明
✅ **类型安全** - argparse 自动进行类型检查

现在你可以完全通过命令行参数来控制训练过程了! 🎉
