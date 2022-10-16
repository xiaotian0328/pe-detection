python save_features_r3d_18.py \
    --log_file="save_features_r3d_18_chunk10_batch16_epoch10_new_splits.log" \
    --path="data/train/*/*" \
    --load_model="r3d_18_chunk10_batch16/epoch9.pth" \
    --gpu=0 \
    --batch_size=16 \
    --chunk_size=10 \
    --feature_dir="features/r3d_18_chunk10_batch16_epoch10"
