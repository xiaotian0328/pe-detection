python train_r3d_18.py \
    --log_file="train_r3d_18_chunk10_batch16_epoch20.log" \
    --path="data/train/*/*" \
    --gpu=0 \
    --batch_size=16 \
    --chunk_size=10 \
    --learning_rate=0.0004 \
    --pos_weight=1. \
    --num_epoch=20 \
    --model_dir="weights/r3d_18_chunk10_batch16"
