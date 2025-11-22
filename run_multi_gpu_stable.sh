#!/bin/bash

# ==============================================================================
# 多卡训练启动脚本（稳定版本 - 防止 NaN）
# ==============================================================================

# 设置使用的 GPU (例如: 0,1 表示使用 GPU 0 和 1)
# export CUDA_VISIBLE_DEVICES=0,1

# 训练参数 - 使用更保守的设置防止 NaN
DATA_ROOT="/home/jinfang/project/CarotidPlaqueStabilityClassifier/data/Carotid_artery"
LABEL_EXCEL="/home/jinfang/project/CarotidPlaqueStabilityClassifier/data/label.xlsx"
BATCH_SIZE=4
EPOCHS=40
LR=1e-4  # 降低学习率（从 1e-4 降到 5e-5）
OUTPUT_DIR="./output"

# 启动多卡训练（使用稳定配置，关闭混合精度）
echo "================================"
echo "开始多卡训练（稳定模式）..."
echo "================================"
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LR (降低以提高稳定性)"
echo "混合精度: 关闭 (提高数值稳定性)"
echo "梯度裁剪: 启用 (max_norm=1.0)"
echo "================================"

accelerate launch --config_file accelerate_config_stable.yaml train.py \
    --root-dir "$DATA_ROOT" \
    --label-excel "$LABEL_EXCEL" \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --max-imgs-per-person 1000 \
    --train-ratio 0.5 \
    --val-ratio 0.3 \
    --test-ratio 0.2 \
    --output-dir "$OUTPUT_DIR" \
    --num-workers 0 \
    --seed 42

echo "================================"
echo "训练完成!"
echo "================================"
