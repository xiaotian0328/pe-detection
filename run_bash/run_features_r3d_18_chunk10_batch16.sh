python save_features_r3d_18.py \
    --log_file="new/save_features_r3d_18_chunk10_batch16_epoch10_new_splits.log" \
    --path="data/train/*/*" \
    --load_model="new/r3d_18_chunk10_batch16_new_splits/epoch9.pth" \
    --gpu=3 \
    --batch_size=16 \
    --chunk_size=10 \
    --feature_dir="features/new/r3d_18_chunk10_batch16_epoch10_new_splits"
