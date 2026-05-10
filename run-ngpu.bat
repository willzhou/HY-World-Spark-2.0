python -m hyworld2.worldrecon.gradio_app ^
    --ckpt_path .\ckpt\HY-WorldMirror-2.0\model.safetensors ^
    --use_fsdp --enable_bf16 --disable_heads gs ^
    --num_gpus 3 --port 8081