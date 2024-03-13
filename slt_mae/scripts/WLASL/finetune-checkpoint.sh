OUTPUT_DIR=''
ANNO_PATH=''
DATA_PATH=''
# path to pretrain model
MODEL_PATH=''

# Check if directory exists. If not, it will create it
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir "$OUTPUT_DIR"
fi

# Copy the script into the output directory
cp "$0" "$OUTPUT_DIR/"

echo "The script has been copied to the ${OUTPUT_DIR} directory."

# We add repeated_aug (--num_sample = 2) on Kinetics-400 here,
# which could better performance while need more time for fine-tuning

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
NODE_COUNT=${NODE_COUNT:-1}  # Number of nodes
GPUS=${GPUS:-1}  # Number of GPUs in each node
SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:3}  # Other training args
RANK=0
MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-12323}  # You should set the same master_port in all the nodes

# train on 32 V100 GPUs (4 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=${GPUS} \
        --master_port ${MASTER_PORT} --nnodes=${NODE_COUNT} \
        --node_rank=${RANK} --master_addr=${MASTER_ADDR} \
    run_class_finetuning.py \
    --model vit_large_patch16_224 \
    --data_set WLASL \
    --nb_classes 2000 \
    --data_path ${DATA_PATH} \
    --anno_path ${ANNO_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --soft_targets 'path_to_wlasl_softtargets' \
    --spatial_idx -1 \
    --sampling_type 'circle' \
    --batch_size 2 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --lr 1e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 40 \
    --dist_eval \
    --test_num_segment 2 \
    --test_num_crop 3 \
    --enable_deepspeed
