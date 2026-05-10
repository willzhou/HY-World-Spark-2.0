export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1   # 如果使用 InfiniBand 且不稳定，可强制走 TCP
export NCCL_DEBUG=INFO     # 输出详细日志便于定位

python -m hyworld2.worldrecon.gradio_app \
    --ckpt_path ./ckpt/HY-WorldMirror-2.0/model.safetensors \
    --use_fsdp \
    --enable_bf16 \
    --num_gpus 3 \
    --port 8081
