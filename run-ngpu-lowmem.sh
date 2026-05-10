# target_size=518: 32×~2000=51800 tokens → 推理显存降低约 60%
python -m hyworld2.worldrecon.gradio_app --ckpt_path ./ckpt/HY-WorldMirror-2.0/model.safetensors --use_fsdp --enable_bf16 --num_gpus 3 --target_size 518 --port 8081
