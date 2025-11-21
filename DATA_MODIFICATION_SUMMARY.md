# 🎯 数据处理修改总结

## ✅ 已完成的修改

已成功修改 `train.py` 中的 `PersonSequenceDataset` 类，实现了**掐头去尾保留中间100张图片**的功能。

## 📋 核心功能

### 1. 掐头去尾保留中间图片
```python
# 对于每个人的数据
原始: 200张图片 → 处理后: 保留中间100张（去掉前50张和后50张）
原始: 150张图片 → 处理后: 保留中间100张（去掉前25张和后25张）
原始: 300张图片 → 处理后: 保留中间100张（去掉前100张和后100张）
```

### 2. 自动过滤不足样本
```python
# 图片数量不足100张的样本会被自动舍弃
原始: 50张图片 → 跳过该样本 ✗
原始: 80张图片 → 跳过该样本 ✗
原始: 100张图片 → 保留中间100张 ✓
原始: 200张图片 → 保留中间100张 ✓
```

## 🔧 代码修改位置

**文件**: `train.py`
**行号**: 93-193
**类名**: `PersonSequenceDataset`

### 新增参数

```python
PersonSequenceDataset(
    root_dir: str,              # 数据根目录
    label_excel: str,           # 标签Excel文件
    transform=None,             # 数据变换
    max_imgs_per_person=None,   # 已废弃（保留兼容性）
    keep_middle_n=100,          # 🆕 保留中间的N张图片
    min_imgs_required=100       # 🆕 最少需要的图片数量
)
```

## 🎨 关键实现

### 1. 图片排序（确保一致性）
```python
img_paths = sorted(img_paths)
```

### 2. 过滤不足样本
```python
if total_imgs < self.min_imgs_required:
    skipped_persons.append({...})  # 记录被跳过的样本
    continue  # 跳过该样本
```

### 3. 掐头去尾算法
```python
# 计算保留区间
start_idx = (total_imgs - self.keep_middle_n) // 2
end_idx = start_idx + self.keep_middle_n

# 选取中间部分
img_paths_selected = img_paths[start_idx:end_idx]
```

**算法示例：**
- 总共 200 张，保留中间 100 张
- `start_idx = (200 - 100) // 2 = 50`
- `end_idx = 50 + 100 = 150`
- 结果：保留 [50:150]，去掉前50张和后50张 ✓

## 📊 训练时输出示例

运行训练时会显示详细统计：

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

成功加载的样本:
样本名称           原始张数   保留张数     标签
------------------------------------------------------------
person_003              250      100      0
person_004              180      100      1
person_005              320      100      0
------------------------------------------------------------
总计: 15 个样本, 1500 张图片
类别分布: Label 0: 8 个  Label 1: 7 个
============================================================
```

## 🚀 使用方法

### 方法1: 直接训练（使用默认参数）

```bash
# 使用稳定配置
./run_multi_gpu_stable.sh

# 或手动指定
CUDA_VISIBLE_DEVICES=0,1 accelerate launch train.py \
    --batch-size 4 \
    --epochs 50 \
    --lr 5e-5
```

**默认行为：**
- ✅ 每个样本保留中间 100 张图片
- ✅ 图片数 < 100 的样本会被自动舍弃
- ✅ 所有样本的图片数量统一为 100 张

### 方法2: 测试数据处理（不训练）

```bash
# 运行测试脚本查看数据处理效果
python test_data_processing.py
```

这个脚本会：
- 加载数据集
- 显示有多少样本被加载/跳过
- 显示详细的统计信息
- 测试不同配置（100张、80张、50张）

## ⚙️ 自定义参数

如果需要修改保留的图片数量，编辑 `train.py` 约第 1036 行：

```python
# 原代码
full_dataset = PersonSequenceDataset(
    root_dir=args.root_dir,
    label_excel=args.label_excel,
    transform=transform,
    max_imgs_per_person=args.max_imgs_per_person  # 旧参数
)

# 修改为（自定义保留80张）
full_dataset = PersonSequenceDataset(
    root_dir=args.root_dir,
    label_excel=args.label_excel,
    transform=transform,
    keep_middle_n=80,          # 保留中间80张
    min_imgs_required=80       # 最少需要80张
)
```

## 📈 优势

### 1. 数据质量一致
- 去除序列两端可能的低质量图片
- 保留中间最稳定的部分

### 2. 训练效率
- 所有样本固定相同图片数
- Attention 机制计算量稳定
- 避免样本不平衡

### 3. 避免过拟合
- 限制图片数量
- 防止某些样本主导训练

## ⚠️ 注意事项

### 1. 样本可能减少
使用此功能后，不足100张的样本会被舍弃。建议：
- 先运行 `test_data_processing.py` 查看影响
- 如果舍弃太多，可以降低 `keep_middle_n` 和 `min_imgs_required`

### 2. 确保图片有序
图片文件名应该有序（如按时间戳），这样排序后才能正确保留中间部分。

### 3. 调整参数建议

| 场景 | keep_middle_n | min_imgs_required |
|------|---------------|-------------------|
| 样本图片都很多 (>200) | 100-150 | 100-150 |
| 样本图片中等 (100-200) | 80-100 | 80-100 |
| 样本图片较少 (<100) | 50-80 | 50-80 |

## 📚 相关文档

- **`DATA_PROCESSING_GUIDE.md`** - 详细使用指南
- **`NAN_SOLUTION.md`** - NaN 问题解决方案
- **`ACCELERATE_GUIDE.md`** - 多卡训练指南
- **`test_data_processing.py`** - 数据处理测试脚本

## 🧪 验证修改

### 快速测试
```bash
# 测试数据加载
python test_data_processing.py

# 查看会保留多少样本
# 查看会舍弃多少样本
```

### 完整训练测试
```bash
# 运行一个 epoch 看看效果
./run_multi_gpu_stable.sh
```

## 💡 示例场景

### 场景1: 标准使用（保留100张）
```python
dataset = PersonSequenceDataset(
    root_dir="./data",
    label_excel="./label.xlsx",
    transform=transform,
    keep_middle_n=100,
    min_imgs_required=100
)
```
**适用于**: 大部分样本有 100+ 张图片的情况

### 场景2: 宽松要求（保留50张）
```python
dataset = PersonSequenceDataset(
    root_dir="./data",
    label_excel="./label.xlsx",
    transform=transform,
    keep_middle_n=50,
    min_imgs_required=50
)
```
**适用于**: 图片较少，不想舍弃太多样本

### 场景3: 严格要求（保留150张）
```python
dataset = PersonSequenceDataset(
    root_dir="./data",
    label_excel="./label.xlsx",
    transform=transform,
    keep_middle_n=150,
    min_imgs_required=150
)
```
**适用于**: 样本图片都很多，想要更多数据

## 🎯 下一步

1. **测试数据处理**: 运行 `python test_data_processing.py`
2. **查看统计信息**: 确认有多少样本被保留/舍弃
3. **调整参数** (如果需要): 修改 `keep_middle_n` 和 `min_imgs_required`
4. **开始训练**: 运行 `./run_multi_gpu_stable.sh`

---

**修改完成！** 🎉

现在运行训练时，数据会自动按照新规则处理：
- ✅ 自动排序图片
- ✅ 掐头去尾保留中间100张
- ✅ 自动过滤不足100张的样本
- ✅ 显示详细统计信息
