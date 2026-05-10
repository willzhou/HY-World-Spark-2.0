set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m hyworld2.worldrecon.gradio_app --ckpt_path .\ckpt\HY-WorldMirror-2.0\model.safetensors