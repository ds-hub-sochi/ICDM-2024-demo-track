# Set the path to save video
OUTPUT_DIR=''
# path to video for visualization
VIDEO_PATH=''
# path to pretrain model
MODEL_PATH=''

python3 run_videomae_vis.py \
    --mask_ratio 0.9 \
    --mask_type tube \
    --decoder_depth 4 \
    --model pretrain_videomae_base_patch16_224 \
    ${VIDEO_PATH} ${OUTPUT_DIR} ${MODEL_PATH}