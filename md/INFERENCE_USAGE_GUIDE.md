# GradCAM 推理脚本使用指南

`inference_gradcam.py` 是一个功能完整的推理和可视化工具,支持三种运行模式。

---

## 功能特性

✅ **模式 1: 测试集评估** - 使用与训练时完全相同的数据划分和随机种子评估测试集
✅ **模式 2: 人物可视化** - 为指定人物文件夹生成 GradCAM 热力图
✅ **模式 3: 单张图片推理** - 对单张图片进行预测和可视化

✅ 自动加载训练配置和模型
✅ 使用相同的随机种子确保可重复性
✅ 生成详细的评估报告和可视化图表
✅ 支持多种输出格式 (图片、CSV、JSON、TXT)

---

## 模式 1: 评估测试集

### 功能说明

- 从训练目录加载 `config.json` 和 `data_split.json`
- 使用相同的随机种子 (`random_seed`)
- 加载与训练时完全相同的测试集样本
- 计算准确率、精确率、召回率、F1、AUC
- 生成混淆矩阵、ROC 曲线、指标柱状图、概率分布图

### 使用方法

```bash
python inference_gradcam.py \
    --train-dir ./output/train_20250105_123456 \
    --eval-test
```

### 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--train-dir` | ✅ | - | 训练输出目录 (包含 models/, logs/ 等子目录) |
| `--eval-test` | ✅ | - | 启用测试集评估模式 |
| `--data-dir` | ❌ | `./data` | 数据根目录 (通常从 config.json 自动读取) |
| `--label-excel` | ❌ | `./label.xlsx` | 标签文件 (通常从 config.json 自动读取) |
| `--class-names` | ❌ | `['0', '1']` | 类别名称 (两个类别) |
| `--output-dir` | ❌ | `./inference_results` | 输出目录 |
| `--device` | ❌ | `auto` | 计算设备 (auto/cuda/cpu) |

### 输出文件

评估结果保存在 `{output_dir}/test_evaluation/` 目录下:

```
inference_results/test_evaluation/
├── confusion_matrix.png         # 混淆矩阵
├── roc_curve.png                # ROC 曲线
├── metrics_bar.png              # 指标柱状图
├── probability_distribution.png # 概率分布
├── detailed_results.csv         # 每个样本的详细结果
└── metrics_summary.txt          # 指标摘要文本
```

### 示例输出

```
============================================================
测试集评估结果
============================================================
总样本数: 20
类别分布: 0=10, 1=10
------------------------------------------------------------
准确率 (Accuracy):  0.9500 (95.00%)
精确率 (Precision): 0.9474
召回率 (Recall):    0.9500
F1 分数 (F1-Score): 0.9487
AUC-ROC:           0.9875
============================================================
```

---

## 模式 2: 人物 GradCAM 可视化

### 功能说明

- 为指定人物文件夹中的所有图片进行融合预测
- 为前 N 张图片生成 GradCAM 热力图 (可通过 `--max-vis` 控制)
- 每张图片生成包含原图、热力图、叠加图的可视化结果

### 使用方法

```bash
python inference_gradcam.py \
    --train-dir ./output/train_20250105_123456 \
    --visualize-person ./data/person_name \
    --max-vis 5
```

### 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--train-dir` | ✅ | - | 训练输出目录 |
| `--visualize-person` | ✅ | - | 人物文件夹路径 (例如: `./data/John_Smith`) |
| `--max-vis` | ❌ | `5` | 最多可视化多少张图片 |
| `--output-dir` | ❌ | `./inference_results` | 输出目录 |
| `--class-names` | ❌ | `['0', '1']` | 类别名称 |
| `--device` | ❌ | `auto` | 计算设备 |

### 输出文件

可视化结果保存在 `{output_dir}/person_gradcam/{person_name}/` 目录下:

```
inference_results/person_gradcam/John_Smith/
├── img01_pred1_true1.png
├── img02_pred1_true1.png
├── img03_pred1_true1.png
├── img04_pred1_true1.png
└── img05_pred1_true1.png
```

文件名格式: `img{序号}_pred{预测类别}_true{真实类别}.png`

### 示例输出

```
============================================================
为人物 'John_Smith' 生成 GradCAM 可视化
============================================================
  真实标签: 1 (Class 1)
  图片数量: 127
  可视化数量: 5
  融合预测: 1 (Class 1)
  预测概率: [0.023, 0.977]
------------------------------------------------------------

生成 GradCAM 可视化...
  [1/5] 已保存: inference_results/person_gradcam/John_Smith/img01_pred1_true1.png
  [2/5] 已保存: inference_results/person_gradcam/John_Smith/img02_pred1_true1.png
  ...
```

---

## 模式 3: 单张图片推理

### 功能说明

- 对单张图片进行预测
- 生成 GradCAM 热力图
- 输出包含原图、热力图、叠加图的可视化结果

### 使用方法

```bash
# 方式 1: 指定模型路径
python inference_gradcam.py \
    --model ./output/train_20250105_123456/models/best_model.pth \
    --image ./test_image.jpg \
    --output ./result.png

# 方式 2: 指定训练目录 (自动加载 best_model.pth)
python inference_gradcam.py \
    --train-dir ./output/train_20250105_123456 \
    --image ./test_image.jpg \
    --output ./result.png
```

### 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--image` | ✅ | - | 输入图片路径 |
| `--model` | ✅* | - | 模型文件路径 |
| `--train-dir` | ✅* | - | 训练目录 (二选一) |
| `--output` | ❌ | 自动生成 | 输出路径 |
| `--output-dir` | ❌ | `./inference_results` | 输出目录 (当 --output 未指定时) |
| `--class-names` | ❌ | `['0', '1']` | 类别名称 |
| `--device` | ❌ | `auto` | 计算设备 |

注: `--model` 和 `--train-dir` 二选一即可。

### 输出文件

如果不指定 `--output`,默认保存在:
```
inference_results/single_image/test_image_gradcam.png
```

### 示例输出

```
============================================================
单张图片推理结果
============================================================
  图片路径: ./test_image.jpg
  预测类别: 1 (Class 1)
  预测概率: [0.1234, 0.8766]
============================================================

✓ 可视化已保存: ./result.png
```

---

## 完整参数列表

### 运行模式 (三选一)

```bash
--eval-test                      # 评估测试集
--visualize-person <path>        # 人物 GradCAM 可视化
--image <path>                   # 单张图片推理
```

### 训练目录配置

```bash
--train-dir <path>               # 训练输出目录
--model <path>                   # 模型文件路径 (可选,默认从 train-dir 加载)
```

### 数据配置

```bash
--data-dir <path>                # 数据根目录 (默认: ./data)
--label-excel <path>             # 标签 Excel 文件 (默认: ./label.xlsx)
--class-names <name1> <name2>    # 类别名称 (默认: 0 1)
```

### 可视化配置

```bash
--max-vis <num>                  # 每个人最多可视化多少张图片 (默认: 5)
--output <path>                  # 输出路径 (用于单张图片模式)
--output-dir <path>              # 输出目录 (默认: ./inference_results)
```

### 其他配置

```bash
--device <auto|cuda|cpu>         # 计算设备 (默认: auto)
```

---

## 使用场景示例

### 场景 1: 训练后立即评估测试集

```bash
# 训练完成后,训练目录为 ./output/train_20250105_143022
python inference_gradcam.py \
    --train-dir ./output/train_20250105_143022 \
    --eval-test
```

### 场景 2: 为测试集中的某个人生成 GradCAM

```bash
# 假设训练目录为 ./output/train_20250105_143022
# 人物文件夹为 ./data/John_Smith
python inference_gradcam.py \
    --train-dir ./output/train_20250105_143022 \
    --visualize-person ./data/John_Smith \
    --max-vis 10
```

### 场景 3: 快速测试单张新图片

```bash
# 使用最新训练的模型
python inference_gradcam.py \
    --model ./output/train_20250105_143022/models/best_model.pth \
    --image ./new_test_image.jpg \
    --class-names "Cat" "Dog"
```

### 场景 4: 批量为多个人��成 GradCAM

```bash
# 使用循环为多个人生成可视化
for person_dir in ./data/*/; do
    python inference_gradcam.py \
        --train-dir ./output/train_20250105_143022 \
        --visualize-person "$person_dir" \
        --max-vis 3
done
```

---

## 常见问题 (FAQ)

### Q1: 为什么推理的测试集结果与训练时不同?

**A:** 确保:
1. 使用 `--train-dir` 而非手动指定路径
2. 脚本会自动加载 `data_split.json` 中保存的测试集索引
3. 使用相同的 `random_seed`
4. 使用相同的 `max_imgs_per_person` 参数

### Q2: 如何知道训练时的随机种子?

**A:** 随机种子保存在 `{train_dir}/logs/config.json` 中的 `random_seed` 字段。
推理脚本会自动读取并使用。

### Q3: 单张图片模式和人物可视化模式的区别?

**A:**
- **单张图片模式**: 独立推理一张图片,无需标签文件
- **人物可视化模式**:
  - 对该人物的所有图片进行融合预测 (Attention Fusion)
  - 显示融合预测结果
  - 需要标签文件来获取真实标签

### Q4: 输出目录结构是什么样的?

**A:**
```
inference_results/
├── test_evaluation/              # 测试集评估结果
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── metrics_bar.png
│   ├── probability_distribution.png
│   ├── detailed_results.csv
│   └── metrics_summary.txt
├── person_gradcam/               # 人物 GradCAM 可视化
│   ├── John_Smith/
│   │   ├── img01_pred1_true1.png
│   │   └── ...
│   └── Jane_Doe/
│       └── ...
└── single_image/                 # 单张图片推理
    └── test_image_gradcam.png
```

### Q5: 如何自定义类别名称?

**A:**
```bash
python inference_gradcam.py \
    --train-dir ./output/train_20250105_143022 \
    --eval-test \
    --class-names "Normal" "Abnormal"
```

### Q6: 我能在 CPU 上运行吗?

**A:** 可以,使用 `--device cpu`:
```bash
python inference_gradcam.py \
    --train-dir ./output/train_20250105_143022 \
    --eval-test \
    --device cpu
```

### Q7: 如何只可视化某个人的前 3 张图片?

**A:**
```bash
python inference_gradcam.py \
    --train-dir ./output/train_20250105_143022 \
    --visualize-person ./data/John_Smith \
    --max-vis 3
```

---

## 技术细节

### 数据划分一致性

脚本通过以下方式确保测试集与训练时完全一致:

1. **加载配置**: 读取 `{train_dir}/logs/config.json`
2. **加载划分**: 读取 `{train_dir}/logs/data_split.json`
3. **设置种子**: `random.seed(config['random_seed'])`
4. **使用索引**: 直接使用 `split_info['test_indices']` 创建测试子集

### 逐样本推理

为了避免 BatchNorm 在 `batch_size=1` 时的问题,测试集评估使用逐样本推理:

```python
for idx in range(len(test_dataset)):
    seq_imgs, label = test_dataset[idx]  # seq_imgs: [N_imgs, 3, 224, 224]
    logits, _ = model([seq_imgs])        # N_imgs ≥ 2, 所以 BatchNorm 正常工作
```

### GradCAM 生成流程

1. **前向传播**: 保存 `layer4[-1]` 的特征图
2. **反向传播**: 计算预测类别对特征图的梯度
3. **权重计算**: 对梯度在空间维度求平均
4. **加权求和**: `CAM = Σ(α_k × A_k)`
5. **上采样**: 从 7×7 放大到 224×224
6. **归一化**: 归一化到 [0, 1] 范围

---

## 版本信息

- **脚本版本**: 完整版 v2.0
- **支持的模型**: ResNetAttentionFusion (ResNet18 + Attention Fusion)
- **PyTorch 版本**: ≥ 1.7
- **Python 版本**: ≥ 3.7

---

## 联系与反馈

如有问题或建议,请查看代码注释或联系开发者。
