#!/bin/bash

# ==============================================================================
# 多卡训练启动脚本
# ==============================================================================

# 设置使用的 GPU (例如: 0,1 表示使用 GPU 0 和 1)
# 使用所有 GPU:
# accelerate launch train.py --batch-size 4 --epochs 50
# export CUDA_VISIBLE_DEVICES=0,1

# 训练参数
DATA_ROOT="/seu_nvme/home/shendian/220256451/datasets/Carotid_artery/Carotid_artery"
LABEL_EXCEL="/seu_nvme/home/shendian/220256451/datasets/Carotid_artery/label.xlsx"
DATA_ROOT="/home/jinfang/project/CarotidPlaqueStabilityClassifier/data"
LABEL_EXCEL="/home/jinfang/project/CarotidPlaqueStabilityClassifier/label.xlsx"
BATCH_SIZE=4
EPOCHS=50
LR=1e-4
OUTPUT_DIR="./output"

# 启动多卡训练
echo "开始多卡训练..."
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LR"
echo "================================"

accelerate launch train.py \
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
    --num-workers 4 \
    --seed 42

echo "训练完成!"
