# Accelerate 多卡训练指南

本文档介绍如何使用 Hugging Face Accelerate 库进行多 GPU 训练。

## 1. 安装依赖

首先确保已安装 accelerate 库:

```bash
pip install accelerate
```

## 2. 配置 Accelerate

有两种方式配置 accelerate:

### 方式一: 交互式配置 (推荐)

运行配置向导,按照提示选择配置:

```bash
accelerate config
```

配置过程中的关键选项:
- **计算环境**: 选择 `This machine`
- **使用哪种机器**: 选择 `multi-GPU`
- **要使用多少个进程**: 输入 GPU 数量,例如 `2` 或 `4`
- **使用 DeepSpeed**: 选择 `No`
- **使用 FullyShardedDataParallel**: 选择 `No`
- **使用 Megatron-LM**: 选择 `No`
- **混合精度训练**: 选择 `fp16` 或 `bf16` (推荐 `bf16` 如果支持)
- **主进程端口**: 默认回车即可

配置完成后会保存到 `~/.cache/huggingface/accelerate/default_config.yaml`

### 方式二: 手动创建配置文件

创建配置文件 `accelerate_config.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16  # 或 fp16 或 no
num_machines: 1
num_processes: 2  # 修改为你要使用的 GPU 数量
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## 3. 多卡训练命令

### 使用默认配置启动训练

```bash
accelerate launch train.py \
    --batch-size 4 \
    --epochs 50 \
    --lr 1e-4
```

### 使用自定义配置文件启动训练

```bash
accelerate launch --config_file accelerate_config.yaml train.py \
    --batch-size 4 \
    --epochs 50 \
    --lr 1e-4
```

### 指定使用特定的 GPU

#### 方法 1: 使用 CUDA_VISIBLE_DEVICES 环境变量

只使用 GPU 0 和 1:
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch train.py --batch-size 4 --epochs 50
```

只使用 GPU 2 和 3:
```bash
CUDA_VISIBLE_DEVICES=2,3 accelerate launch train.py --batch-size 4 --epochs 50
```

#### 方法 2: 在配置文件中指定 GPU

修改 `accelerate_config.yaml` 中的 `gpu_ids`:

```yaml
gpu_ids: 0,1  # 只使用 GPU 0 和 1
# 或
gpu_ids: all  # 使用所有 GPU
```

然后运行:
```bash
accelerate launch --config_file accelerate_config.yaml train.py --batch-size 4 --epochs 50
```

#### 方法 3: 使用命令行参数直接指定

```bash
accelerate launch \
    --mixed_precision bf16 \
    --num_processes 2 \
    --gpu_ids 0,1 \
    train.py \
    --batch-size 4 \
    --epochs 50 \
    --lr 1e-4
```

## 4. 完整训练示例

### 示例 1: 使用 2 块 GPU (GPU 0 和 1)

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch train.py \
    --root-dir /seu_nvme/home/shendian/220256451/datasets/Carotid_artery/Carotid_artery \
    --label-excel /seu_nvme/home/shendian/220256451/datasets/Carotid_artery/label.xlsx \
    --batch-size 4 \
    --epochs 50 \
    --lr 1e-4 \
    --max-imgs-per-person 1000 \
    --train-ratio 0.5 \
    --val-ratio 0.3 \
    --test-ratio 0.2 \
    --output-dir ./output
```

### 示例 2: 使用 4 块 GPU,启用混合精度训练

```bash
accelerate launch \
    --mixed_precision bf16 \
    --num_processes 4 \
    train.py \
    --batch-size 8 \
    --epochs 100 \
    --lr 1e-4 \
    --weight-decay 1e-5 \
    --output-dir ./output
```

### 示例 3: 使用特定 GPU (2 和 3),并指定更多训练参数

```bash
CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --mixed_precision fp16 \
    --num_processes 2 \
    train.py \
    --root-dir /path/to/data \
    --label-excel /path/to/label.xlsx \
    --batch-size 4 \
    --epochs 50 \
    --lr 1e-4 \
    --num-workers 4 \
    --seed 42
```

## 5. 查看和验证配置

### 查看当前 accelerate 配置

```bash
accelerate env
```

### 测试 accelerate 配置

```bash
accelerate test
```

## 6. 训练监控

### 查看 GPU 使用情况

在训练过程中,可以在另一个终端窗口监控 GPU 使用:

```bash
watch -n 1 nvidia-smi
```

或使用 gpustat (需要先安装):

```bash
pip install gpustat
watch -n 1 gpustat -cpu
```

## 7. 常见问题

### Q1: 如何选择合适的 batch_size?

- 单 GPU 训练: batch_size = 2-4
- 多 GPU 训练: 每个 GPU 的 batch_size = 2-4
- 实际的全局 batch_size = batch_size × GPU 数量

### Q2: 混合精度 (fp16/bf16) 的选择?

- **bf16** (bfloat16): 更稳定,推荐用于 Ampere 架构 (RTX 30系列、A100 等)
- **fp16** (float16): 通用性强,但可能需要调整学习率
- **no**: 使用 fp32,训练最稳定但速度慢

### Q3: 出现 CUDA out of memory 错误?

减小 batch_size 或启用混合精度训练:

```bash
accelerate launch --mixed_precision bf16 train.py --batch-size 2
```

### Q4: 如何确认多卡训练正在工作?

训练开始时会打印:
```
✓ 使用 2 个进程进行训练
当前进程: 1/2
当前进程: 2/2
```

同时使用 `nvidia-smi` 可以看到多个 GPU 都有显存占用。

## 8. 性能优化建议

1. **启用混合精度训练**: 使用 bf16 或 fp16 可以显著提升训练速度
2. **调整 num_workers**: 根据 CPU 核心数设置,通常为 4-8
3. **启用 pin_memory**: 已在代码中默认启用
4. **使用更大的 batch_size**: 多 GPU 训练时可以适当增大
5. **梯度累积** (如需要): 在配置文件中添加 `gradient_accumulation_steps`

## 9. 保存的模型文件

训练后的模型会保存在:
- `output/train_YYYYMMDD_HHMMSS/models/best_model.pth`: 验证集上最佳模型
- `output/train_YYYYMMDD_HHMMSS/models/final_model.pth`: 最终模型

这些模型可以在单 GPU 或多 GPU 环境中加载使用。

## 10. 从已保存的模型继续训练

```bash
accelerate launch train.py \
    --load-model output/train_20240101_120000/models/best_model.pth \
    --batch-size 4 \
    --epochs 50
```

---

更多信息请参考:
- [Accelerate 官方文档](https://huggingface.co/docs/accelerate)
- [Accelerate 快速入门](https://huggingface.co/docs/accelerate/quicktour)
