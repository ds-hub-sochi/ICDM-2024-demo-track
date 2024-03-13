# Set the path to save checkpoints
OUTPUT_DIR=''
# Set the path to Kinetics train set.
DATA_PATH=''

NODE_COUNT=${NODE_COUNT:-1}  # Number of nodes
GPUS=${GPUS:-8}  # Number of GPUs in each node
SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:3}  # Other training args
RANK=0
MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-12320}  # You should set the same master_port in all the nodes

# train on 32 V100 GPUs (4 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=${GPUS} \
        --master_port ${MASTER_PORT} --nnodes=${NODE_COUNT} \
        --node_rank=${RANK} --master_addr=${MASTER_ADDR} \
        run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --model pretrain_videomae_large_patch16_224 \
        --decoder_depth 12 \
        --resume "path_to_mae_kinetic_checkpoint" \
        --batch_size 25 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_segments 1 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 0 \
        --save_ckpt_freq 40 \
        --epochs 1600 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}