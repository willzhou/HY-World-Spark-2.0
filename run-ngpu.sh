torchrun --nproc_per_node=3 -m hyworld2.worldrecon.gradio_app --num_gpus=3 --use_fsdp --enable_bf16 --ckpt_path ./ckpt/HY-WorldMirror-2.0/model.safetensors

# python -m hyworld2.worldrecon.gradio_app --ckpt_path ./ckpt/HY-WorldMirror-2.0/model.safetensors --use_fsdp --enable_bf16 --num_gpus 3 --port 8081
# torchrun  --nproc_per_node=2 -m hyworld2.worldrecon.gradio_app --ckpt_path ./ckpt/HY-WorldMirror-2.0/model.safetensors --use_fsdp --enable_bf16 --port 8081
# python -m hyworld2.worldrecon.gradio_app --ckpt_path ./ckpt/HY-WorldMirror-2.0/model.safetensors --use_fsdp --enable_bf16 --disable_heads gs --num_gpus 3 --port 8081
