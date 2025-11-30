time=`date "+%Y%m%d%H%M"`

torchrun --nproc-per-node=4 examples/recipes/qwen_vl/finetune_qwen_vl.py \
--pretrained-checkpoint /lustre/fsw/general_sa/xshang/modelscope/Qwen3-VL-30B-A3B-Instruct-Megatron \
--recipe qwen3_vl_3b_active_30b_moe_finetune_config \
--dataset-type hf \
dataset.maker_name=make_cord_v2_dataset \
model.expert_model_parallel_size=4 \
model.tensor_model_parallel_size=4 \
model.freeze_language_model=false \
model.freeze_vision_model=false \
train.global_batch_size=16 \
train.train_iters=50 \
train.eval_iters=2 \
optimizer.lr=6.0e-5 \
optimizer.min_lr=6.0e-6 \
scheduler.lr_warmup_iters=10 \
logger.log_interval=1 \
logger.tensorboard_dir=/home/xshang/Megatron-Bridge/tensorboard \
dataset.num_workers=0 \
checkpoint.save=/lustre/fsw/general_sa/xshang/modelscope/tmp 2>&1 | tee $time.log
