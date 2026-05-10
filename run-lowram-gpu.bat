python -m hyworld2.worldrecon.gradio_app ^
    --ckpt_path .\ckpt\HY-WorldMirror-2.0\model.safetensors ^
    --enable_bf16 ^
    --disable_heads gs ^
    --target_size 640 ^
    --port 8081