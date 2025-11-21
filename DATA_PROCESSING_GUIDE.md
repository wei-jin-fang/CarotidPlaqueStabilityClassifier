# 数据处理修改说明

## 📝 修改内容

已对 `PersonSequenceDataset` 类进行修改，实现**掐头去尾保留中间图片**的功能。

## ✨ 新功能

### 1. 掐头去尾，保留中间 N 张图片

对于每个人的数据：
```
原始数据: 共 200 张图片
处理后:   保留中间 100 张 (去掉前 50 张和后 50 张)

计算方式:
start_idx = (200 - 100) // 2 = 50
end_idx = 50 + 100 = 150
保留: 图片[50:150]
```

### 2. 自动过滤不足图片的样本

如果某个人的图片数量 < 100 张，则**自动跳过该样本**，不参与训练。

## 🔧 参数说明

```python
PersonSequenceDataset(
    root_dir: str,              # 数据根目录
    label_excel: str,           # 标签文件
    transform=None,             # 数据增强
    max_imgs_per_person=None,   # 已废弃，保留用于兼容
    keep_middle_n=100,          # 保留中间的 N 张图片 (默认 100)
    min_imgs_required=100       # 最少需要的图片数 (默认 100)
)
```

### 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `keep_middle_n` | int | 100 | 掐头去尾后保留的图片数量 |
| `min_imgs_required` | int | 100 | 样本的最低图片数量要求，不足则舍弃 |

## 📊 输出示例

训练时会输出详细的统计信息：

```
============================================================
数据加载统计:
============================================================
✓ 成功加载: 15 个样本
✗ 跳过样本: 3 个 (图片数 < 100)
每个样本保留: 100 张图片 (掐头去尾)

跳过的样本详情:
  ✗ person_001     :   45 张 (< 100) | Label: 0
  ✗ person_002     :   78 张 (< 100) | Label: 1
  ✗ person_003     :   92 张 (< 100) | Label: 0

成功加载的样本:
样本名称           原始张数   保留张数     标签
------------------------------------------------------------
person_004              250      100      0
person_005              180      100      1
person_006              320      100      0
person_007              150      100      1
...
------------------------------------------------------------
总计: 15 个样本, 1500 张图片
类别分布: Label 0: 8 个  Label 1: 7 个
============================================================
```

## 🚀 使用方法

### 方法 1: 使用默认参数（保留中间 100 张）

直接运行训练脚本，无需修改：

```bash
./run_multi_gpu_stable.sh
```

或者：

```bash
accelerate launch train.py \
    --batch-size 4 \
    --epochs 50 \
    --lr 5e-5
```

### 方法 2: 自定义保留图片数量

如果想修改保留的图片数量，需要修改 `train.py` 中数据集初始化部分：

找到 `train.py` 大约第 1035 行：

```python
# 修改前（默认保留 100 张）
with accelerator.main_process_first():
    full_dataset = PersonSequenceDataset(
        root_dir=args.root_dir,
        label_excel=args.label_excel,
        transform=transform,
        max_imgs_per_person=args.max_imgs_per_person
    )

# 修改后（自定义保留 150 张）
with accelerator.main_process_first():
    full_dataset = PersonSequenceDataset(
        root_dir=args.root_dir,
        label_excel=args.label_excel,
        transform=transform,
        keep_middle_n=150,          # 保留中间 150 张
        min_imgs_required=150       # 最低要求 150 张
    )
```

## 💡 实现原理

### 1. 图片排序
```python
img_paths = sorted(img_paths)
```
- 对所有图片路径进行排序，确保顺序一致
- 这样可以保证掐头去尾的结果是确定的

### 2. 检查最低要求
```python
if total_imgs < self.min_imgs_required:
    skipped_persons.append({...})
    continue
```
- 如果图片数量不足，记录到 `skipped_persons` 并跳过
- 不会参与训练

### 3. 掐头去尾算法
```python
start_idx = (total_imgs - self.keep_middle_n) // 2
end_idx = start_idx + self.keep_middle_n
img_paths_selected = img_paths[start_idx:end_idx]
```

**计算示例：**
- 假设有 200 张图片，保留中间 100 张
- `start_idx = (200 - 100) // 2 = 50`
- `end_idx = 50 + 100 = 150`
- 保留索引 [50:150] 的图片

**结果：**
- 去掉前 50 张（索引 0-49）
- 去掉后 50 张（索引 150-199）
- 保留中间 100 张（索引 50-149）

## 📈 优势

### 1. 数据质量更一致
- 掐头去尾可以去除序列两端可能存在的低质量图片
- 保留中间部分通常是最稳定、质量最好的

### 2. 计算效率
- 每个样本固定 100 张图片
- Attention 机制计算量更稳定
- 训练速度更快

### 3. 避免不平衡
- 统一所有样本的图片数量
- 避免某些样本因图片过多而主导训练

## ⚠️ 注意事项

### 1. 数据集可能变小

使用这个功能后，不足 100 张图片的样本会被舍弃，可能导致：
- 训练样本减少
- 某个类别的样本数减少

**建议**: 先运行一次查看有多少样本被舍弃，评估是否可接受

### 2. 调整参数

如果被舍弃的样本太多，可以：
- 降低 `keep_middle_n` (例如改为 50 或 80)
- 降低 `min_imgs_required` (例如改为 50 或 80)

### 3. 数据顺序

确保图片文件名有序（例如按时间戳命名），这样排序后才能正确保留中间部分。

## 🔍 验证数据处理

训练前可以检查数据加载情况：

```python
# 单独测试数据加载
from train import PersonSequenceDataset
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = PersonSequenceDataset(
    root_dir='/path/to/data',
    label_excel='/path/to/label.xlsx',
    transform=transform,
    keep_middle_n=100,
    min_imgs_required=100
)

print(f"\n加载成功! 共 {len(dataset)} 个样本")
print(f"每个样本有 100 张图片")
```

## 📝 修改位置

修改的代码位置：`train.py:93-193`

主要修改：
1. 添加了 `keep_middle_n` 和 `min_imgs_required` 参数
2. 添加了图片排序逻辑
3. 添加了最低图片数量检查
4. 实现了掐头去尾算法
5. 增强了统计信息输出

---

现在运行训练时，数据会自动按照新规则处理！🎉
