"""
Gradio demo for WorldMirror 2.0 — HunyuanWorld World Reconstruction.

Usage:
    python -m hyworld2.worldrecon.gradio_app
    python -m hyworld2.worldrecon.gradio_app --examples_dir /path/to/examples --port 8081
"""

import argparse
import gc
import http.server
import io
import os
import shutil
import socketserver
import sys
import threading
import time
from datetime import datetime
from glob import glob
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Fix torch 2.8+ on Windows: TCPStore defaults to use_libuv=True but the
# Windows build doesn't include libuv, causing DistStoreError in torchrun.
# Monkey-patch TCPStore to default use_libuv=False so rendezvous works.
import torch.distributed as _dist
if hasattr(_dist, "TCPStore"):
    _OrigTCPStoreInit = _dist.TCPStore.__init__

    def _PatchedTCPStoreInit(self, *args, **kwargs):
        kwargs.setdefault("use_libuv", False)
        _OrigTCPStoreInit(self, *args, **kwargs)

    _dist.TCPStore.__init__ = _PatchedTCPStoreInit

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

try:
    import spaces
except ImportError:
    class _SpacesStub:
        @staticmethod
        def GPU(duration=120):
            def _dec(fn):
                return fn
            return _dec
    spaces = _SpacesStub()

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

from .pipeline import WorldMirrorPipeline
from .hyworldmirror.utils.inference_utils import (
    prepare_images_to_tensor,
    compute_adaptive_target_size,
    compute_sky_mask,
    compute_filter_mask,
    _voxel_prune_gaussians,
)
from .hyworldmirror.utils.visual_util import convert_predictions_to_glb_scene
from .hyworldmirror.utils.save_utils import (
    save_camera_params,
    save_gs_ply,
    convert_gs_to_ply,
    process_ply_to_splat,
)
from .hyworldmirror.models.utils.geometry import depth_to_world_coords_points

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_pipeline = None
_pipeline_args = {}
current_terminal_output = ""
TARGET_SIZE = 952  # overridden by --target_size


def _mp_launch_fn(rank, pipeline_kwargs, examples_dir, host, port, share, mp_port):
    """Module-level worker for multiprocessing.spawn."""
    import torch
    import torch.distributed as dist

    world_size = pipeline_kwargs.pop("_world_size")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(mp_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    dist.init_process_group(
        backend="gloo" if os.name == "nt" else "nccl",
        init_method=f"tcp://127.0.0.1:{mp_port}",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)
    _init_pipeline_args(**pipeline_kwargs)
    pipe = _get_pipeline()

    if rank == 0:
        print(f"[Gradio] Rank 0 (mp.spawn): launching Gradio")
        served_dir = Path.cwd()
        served_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Gradio] Static files directory: {served_dir}")

        # Start simple HTTP server for static files on port 8090
        import threading
        import http.server
        import socketserver

        STATIC_PORT = 8090

        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(served_dir), **kwargs)
            def log_message(self, format, *args):
                pass  # Suppress request logging
            def end_headers(self):
                self.send_header('Access-Control-Allow-Origin', '*')
                super().end_headers()

        httpd = socketserver.TCPServer(("", STATIC_PORT), QuietHandler)
        httpd.allow_reuse_address = True
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        print(f"[Static Server] Running on http://127.0.0.1:{STATIC_PORT} serving {served_dir}")

        demo = build_demo(examples_dir=examples_dir)
        try:
            demo.queue().launch(show_error=True, share=share,
                              server_name=host, server_port=port, ssr_mode=False)
        finally:
            _broadcast_string("__QUIT__", pipe.rank, src=0)
            if dist.is_initialized():
                dist.destroy_process_group()
    else:
        _worker_loop()


def _init_pipeline_args(
    pretrained_model_name_or_path="tencent/HY-World-2.0",
    config_path=None,
    ckpt_path=None,
    use_fsdp=False,
    enable_bf16=False,
    fsdp_cpu_offload=False,
    disable_heads=None,
):
    global _pipeline_args
    _pipeline_args = dict(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        config_path=config_path,
        ckpt_path=ckpt_path,
        use_fsdp=use_fsdp,
        enable_bf16=enable_bf16,
        fsdp_cpu_offload=fsdp_cpu_offload,
        disable_heads=disable_heads,
    )


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = WorldMirrorPipeline.from_pretrained(**_pipeline_args)
    elif _pipeline.sp_size <= 1:
        _pipeline.model.to(_pipeline.device)
        _pipeline.model.eval()
    return _pipeline


# ---------------------------------------------------------------------------
# Distributed helpers for multi-GPU Gradio
# ---------------------------------------------------------------------------
def _broadcast_string(s, rank, src=0):
    """Broadcast a string from src to all ranks."""
    import torch.distributed as dist
    if rank == src:
        data = s.encode("utf-8")
        length = torch.tensor([len(data)], dtype=torch.long, device="cuda")
    else:
        length = torch.tensor([0], dtype=torch.long, device="cuda")
    dist.broadcast(length, src=src)
    n = length.item()
    if rank == src:
        tensor = torch.tensor(list(data), dtype=torch.uint8, device="cuda")
    else:
        tensor = torch.empty(n, dtype=torch.uint8, device="cuda")
    dist.broadcast(tensor, src=src)
    return tensor.cpu().numpy().tobytes().decode("utf-8")


def _notify_workers(img_paths, target_size):
    """Rank 0 broadcasts inference task to worker ranks."""
    pipe = _get_pipeline()
    if pipe.sp_size <= 1:
        return
    cmd = "__INFER__\t" + str(target_size) + "\t" + "\t".join(img_paths)
    _broadcast_string(cmd, pipe.rank, src=0)


def _worker_loop():
    """Non-rank-0 process loop: wait for inference commands from rank 0."""
    import torch.distributed as dist
    from datetime import timedelta

    pipe = _get_pipeline()
    rank = pipe.rank
    print(f"[Worker rank {rank}] Entering worker loop, waiting for tasks...")

    _INF_TIMEOUT = timedelta(days=365)
    _DEF_TIMEOUT = timedelta(minutes=10)

    while True:
        dist.distributed_c10d._get_default_group()._get_backend(
            torch.device("cuda")).options._timeout = _INF_TIMEOUT

        cmd = _broadcast_string("", rank, src=0)

        dist.distributed_c10d._get_default_group()._get_backend(
            torch.device("cuda")).options._timeout = _DEF_TIMEOUT

        if cmd == "__QUIT__":
            print(f"[Worker rank {rank}] Received quit signal.")
            break

        if cmd.startswith("__INFER__\t"):
            parts = cmd.split("\t")
            target_size = int(parts[1])
            img_paths = parts[2:]
            print(f"[Worker rank {rank}] Running inference ({len(img_paths)} images, size={target_size})")
            with torch.no_grad():
                pipe._run_inference(img_paths, target_size, None, None)

    if dist.is_initialized():
        dist.destroy_process_group()


class TeeOutput:
    """Capture stdout while still printing to console."""

    def __init__(self, max_chars=10000):
        self.terminal = sys.stdout
        self.log = io.StringIO()
        self.max_chars = max_chars

    def write(self, message):
        global current_terminal_output
        self.terminal.write(message)
        self.log.write(message)
        content = self.log.getvalue()
        if len(content) > self.max_chars:
            content = "...(earlier output truncated)...\n" + content[-self.max_chars :]
            self.log = io.StringIO()
            self.log.write(content)
        current_terminal_output = self.log.getvalue()

    def flush(self):
        self.terminal.flush()

    def getvalue(self):
        return self.log.getvalue()

    def clear(self):
        global current_terminal_output
        self.log = io.StringIO()
        current_terminal_output = ""


# ---------------------------------------------------------------------------
# Model inference (uses WorldMirrorPipeline internally)
# ---------------------------------------------------------------------------
@spaces.GPU(duration=120)
def run_model(target_dir):
    """Run WorldMirror inference on images inside *target_dir/images*.

    Returns (outputs_dict, processed_visualization_data).
    """
    pipe = _get_pipeline()

    image_folder = os.path.join(target_dir, "images")
    img_paths = sorted(
        glob(os.path.join(image_folder, "*.png"))
        + glob(os.path.join(image_folder, "*.jpg"))
        + glob(os.path.join(image_folder, "*.jpeg"))
        + glob(os.path.join(image_folder, "*.webp"))
    )
    if not img_paths:
        raise ValueError("No images found in the upload directory.")
    if pipe.sp_size > 1 and len(img_paths) < pipe.sp_size:
        raise ValueError(
            f"Multi-GPU mode requires at least {pipe.sp_size} input images "
            f"(got {len(img_paths)}). Please upload more images or use fewer GPUs."
        )

    print(f"[Gradio] {len(img_paths)} images from {image_folder}")

    effective = compute_adaptive_target_size(img_paths, max_target_size=TARGET_SIZE)

    _notify_workers(img_paths, effective)

    predictions, imgs, infer_time = pipe._run_inference(
        img_paths, effective, prior_cam_path=None, prior_depth_path=None
    )
    B, S, C, H, W = imgs.shape
    print(f"[Gradio] Inference done in {infer_time:.2f}s  ({S} views, {H}x{W})")

    sky_mask = compute_sky_mask(
        img_paths, H, W, S,
        predictions=predictions,
        source="auto",
        model_threshold=0.45,
        processed_aspect_ratio=W / H,
    )

    filter_mask, gs_filter_mask = compute_filter_mask(
        predictions, imgs, img_paths, H, W, S,
        apply_confidence_mask=False,
        apply_edge_mask=True,
        apply_sky_mask=True,
        confidence_percentile=10.0,
        edge_normal_threshold=1.0,
        edge_depth_threshold=0.03,
        sky_mask=sky_mask,
        use_gs_depth=("gs_depth" in predictions),
    )

    imgs_np = imgs[0].permute(0, 2, 3, 1).detach().cpu().numpy()
    depth_np = predictions["depth"][0].detach().cpu().float().numpy()
    normal_np = predictions["normals"][0].detach().cpu().float().numpy()
    cam_poses_np = predictions["camera_poses"][0].detach().cpu().float().numpy()
    cam_intrs_np = predictions["camera_intrs"][0].detach().cpu().float().numpy()

    pts3d_np = depth_to_world_coords_points(
        predictions["depth"][0, ..., 0],
        predictions["camera_poses"][0],
        predictions["camera_intrs"][0],
    )[0].detach().cpu().float().numpy()

    outputs = {
        "images": imgs_np,
        "world_points": pts3d_np,
        "depth": depth_np,
        "normal": normal_np,
        "final_mask": filter_mask,
        "gs_filter_mask": gs_filter_mask,
        "sky_mask": sky_mask,
        "camera_poses": cam_poses_np,
        "camera_intrs": cam_intrs_np,
    }
    if "splats" in predictions:
        sp = predictions["splats"]
        outputs["splats"] = {
            k: sp[k] for k in ("means", "scales", "quats", "opacities")
        }
        for optional in ("sh", "colors", "weights"):
            if optional in sp:
                outputs["splats"][optional] = sp[optional]

    processed_data = _build_vis_data(outputs, imgs)
    torch.cuda.empty_cache()
    return outputs, processed_data


def _build_vis_data(outputs, imgs_tensor):
    """Build per-view visualization dicts from model outputs."""
    vis = {}
    S = outputs["images"].shape[0]
    for i in range(S):
        vis[i] = {
            "image": imgs_tensor[0, i].detach().cpu().numpy(),
            "points3d": outputs["world_points"][i],
            "depth": outputs["depth"][i].squeeze(),
            "normal": outputs["normal"][i],
            "mask": outputs["final_mask"][i].copy(),
        }
    return vis


# ---------------------------------------------------------------------------
# File upload helpers
# ---------------------------------------------------------------------------
def process_uploaded_files(files, time_interval=1.0):
    """Process uploaded files (images or videos) into a target_dir/images folder."""
    gc.collect()
    torch.cuda.empty_cache()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"gradio_demo_output/input_images_{timestamp}"
    images_dir = os.path.join(target_dir, "images")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(images_dir)

    image_paths = []
    if files is None:
        return target_dir, image_paths

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".3gp"}

    for file_data in files:
        src_path = str(file_data["name"] if isinstance(file_data, dict) and "name" in file_data else file_data)
        ext = os.path.splitext(src_path)[1].lower()
        base_name = os.path.splitext(os.path.basename(src_path))[0]

        if ext in video_exts:
            cap = cv2.VideoCapture(src_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            interval = max(1, int(fps * time_interval))
            idx, saved = 0, 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                idx += 1
                if idx % interval == 0:
                    dst = os.path.join(images_dir, f"{base_name}_{saved:06d}.png")
                    cv2.imwrite(dst, frame)
                    image_paths.append(dst)
                    saved += 1
            cap.release()
            print(f"Extracted {saved} frames from {os.path.basename(src_path)}")
        elif ext in {".heic", ".heif"}:
            try:
                with Image.open(src_path) as img:
                    img = img.convert("RGB")
                    dst = os.path.join(images_dir, f"{base_name}.jpg")
                    img.save(dst, "JPEG", quality=95)
                    image_paths.append(dst)
            except Exception:
                dst = os.path.join(images_dir, os.path.basename(src_path))
                shutil.copy(src_path, dst)
                image_paths.append(dst)
        else:
            dst = os.path.join(images_dir, os.path.basename(src_path))
            shutil.copy(src_path, dst)
            image_paths.append(dst)

    image_paths = sorted(image_paths)
    print(f"Processed {len(image_paths)} files to {images_dir}")
    return target_dir, image_paths


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def render_depth_visualization(depth_map, mask=None):
    if depth_map is None:
        return None
    import matplotlib.pyplot as plt
    d = depth_map.copy()
    valid = d > 0
    if mask is not None:
        valid = valid & mask
    if valid.sum() > 0:
        lo, hi = np.percentile(d[valid], 5), np.percentile(d[valid], 95)
        d[valid] = (d[valid] - lo) / (hi - lo + 1e-9)
    rgb = (plt.cm.turbo_r(d)[:, :, :3] * 255).astype(np.uint8)
    rgb[~valid] = [255, 255, 255]
    return rgb


def render_normal_visualization(normal_map, mask=None):
    if normal_map is None:
        return None
    n = normal_map.copy()
    if mask is not None:
        n[~mask] = 0
    return ((n + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)


def update_view_info(current, total, view_type="Depth"):
    return (
        f"<div style='text-align:center;padding:10px;background:#f8f8f8;color:#999;"
        f"border-radius:8px;margin-bottom:10px;'>"
        f"<strong>{view_type} View Navigation</strong> | "
        f"Current: View {current} / {total} views</div>"
    )


def _get_view(processed_data, idx):
    if not processed_data:
        return None
    keys = list(processed_data.keys())
    idx = max(0, min(idx, len(keys) - 1))
    return processed_data[keys[idx]]


def update_depth_view(pd, idx):
    v = _get_view(pd, idx)
    return render_depth_visualization(v["depth"], v.get("mask")) if v else None


def update_normal_view(pd, idx):
    v = _get_view(pd, idx)
    return render_normal_visualization(v["normal"], v.get("mask")) if v else None


def update_view_selectors(pd):
    n = max(1, len(pd) if pd else 1)
    return (
        gr.update(minimum=1, maximum=n, value=1, step=1),
        gr.update(minimum=1, maximum=n, value=1, step=1),
        update_view_info(1, n, "Depth"),
        update_view_info(1, n, "Normal"),
    )


# ---------------------------------------------------------------------------
# Main reconstruction entry point
# ---------------------------------------------------------------------------
@spaces.GPU(duration=120)
def gradio_demo(
    target_dir,
    frame_selector="All",
    show_camera=False,
    filter_sky_bg=False,
    show_mesh=False,
    filter_ambiguous=False,
):
    tee = TeeOutput()
    old_stdout = sys.stdout
    sys.stdout = tee

    try:
        if not target_dir or not os.path.isdir(target_dir):
            sys.stdout = old_stdout
            empty = [None] * 14
            empty[1] = "No valid directory. Please upload files first."
            empty[-1] = tee.getvalue()
            return tuple(empty)

        t0 = time.time()
        gc.collect()
        torch.cuda.empty_cache()

        all_files = sorted(os.listdir(os.path.join(target_dir, "images"))) if os.path.isdir(os.path.join(target_dir, "images")) else []
        frame_choices = ["All"] + [f"{i}: {f}" for i, f in enumerate(all_files)]

        print("Running WorldMirror model...")
        with torch.no_grad():
            predictions, processed_data = run_model(target_dir)

        np.savez(os.path.join(target_dir, "predictions.npz"), **{
            k: v for k, v in predictions.items() if k != "splats"
        })

        save_camera_params(predictions["camera_poses"], predictions["camera_intrs"], target_dir)
        cam_file = os.path.join(target_dir, "camera_params.json")

        if frame_selector is None:
            frame_selector = "All"
        safe = frame_selector.replace(".", "_").replace(":", "").replace(" ", "_")
        glb_path = os.path.join(target_dir, f"scene_{safe}_cam{show_camera}_mesh{show_mesh}_edges{filter_ambiguous}_sky{filter_sky_bg}.glb")
        glb_scene = convert_predictions_to_glb_scene(
            predictions,
            filter_by_frames=frame_selector,
            show_camera=show_camera,
            mask_sky_bg=filter_sky_bg,
            as_mesh=show_mesh,
            mask_ambiguous=filter_ambiguous,
        )
        glb_scene.export(file_obj=glb_path)

        gs_file = None
        if "splats" in predictions:
            sp = predictions["splats"]
            means = sp["means"][0].reshape(-1, 3).detach().cpu()
            scales = sp["scales"][0].reshape(-1, 3).detach().cpu()
            quats = sp["quats"][0].reshape(-1, 4).detach().cpu()
            colors = (sp.get("sh", sp.get("colors"))[0]).reshape(-1, 3).detach().cpu()
            opacities = sp["opacities"][0].reshape(-1).detach().cpu()
            weights = (sp["weights"][0].reshape(-1).detach().cpu()
                       if "weights" in sp else torch.ones_like(opacities))

            keep = None
            if predictions.get("gs_filter_mask") is not None:
                keep = torch.from_numpy(predictions["gs_filter_mask"].reshape(-1)).bool()
            elif predictions.get("final_mask") is not None:
                keep = torch.from_numpy(predictions["final_mask"].reshape(-1)).bool()
            if keep is not None:
                means, scales, quats = means[keep], scales[keep], quats[keep]
                colors, opacities, weights = colors[keep], opacities[keep], weights[keep]

            means, scales, quats, colors, opacities = _voxel_prune_gaussians(
                means, scales, quats, colors, opacities, weights)

            compress_gs_max_points = 5_000_000
            if compress_gs_max_points > 0 and means.shape[0] > compress_gs_max_points:
                n_before = means.shape[0]
                idx = torch.from_numpy(
                    np.random.default_rng(42).choice(
                        n_before, size=compress_gs_max_points, replace=False)
                ).long()
                means, scales, quats = means[idx], scales[idx], quats[idx]
                colors, opacities = colors[idx], opacities[idx]
                print(f"[Save] Downsample gaussians: {n_before} -> {compress_gs_max_points}")

            gs_file = os.path.join(target_dir, "gaussians.ply")
            save_gs_ply(gs_file, means, scales, quats, colors, opacities)
            print(f"[Save] gaussians.ply ({os.path.getsize(gs_file)} bytes)")

            # Copy PLY to cwd so Gradio can serve it via /file=
            import shutil as _shutil
            cwd_gs_dir = os.path.join(os.getcwd(), "gradio_served")
            os.makedirs(cwd_gs_dir, exist_ok=True)
            cwd_gs_file = os.path.join(cwd_gs_dir, "gaussians.ply")
            _shutil.copy2(gs_file, cwd_gs_file)
            gs_file = cwd_gs_file  # Return the cwd-based path
            print(f"[SPARK DEBUG] gradio_demo returning gs_file={gs_file}")

        depth_vis = update_depth_view(processed_data, 0)
        normal_vis = update_normal_view(processed_data, 0)
        d_sl, n_sl, d_info, n_info = update_view_selectors(processed_data)

        del predictions
        gc.collect()
        torch.cuda.empty_cache()

        elapsed = time.time() - t0
        log_msg = f"Reconstruction Success ({len(all_files)} frames, {elapsed:.1f}s)."
        terminal_log = tee.getvalue()
        sys.stdout = old_stdout

        return (
            glb_path,
            log_msg,
            gr.Dropdown(choices=frame_choices, value=frame_selector, interactive=True),
            processed_data,
            depth_vis,
            normal_vis,
            d_sl,
            n_sl,
            d_info,
            n_info,
            cam_file,
            gs_file,
            terminal_log,
        )

    except ValueError as e:
        terminal_log = tee.getvalue()
        sys.stdout = old_stdout
        print(f"[Pipeline] {e}")
        empty = [None] * 13
        empty[1] = f"**Error:** {e}"
        empty[-1] = terminal_log
        return tuple(empty)

    except Exception as e:
        terminal_log = tee.getvalue()
        sys.stdout = old_stdout
        print(f"Error: {e}")
        raise


# ---------------------------------------------------------------------------
# Refresh helpers (interactive parameter changes)
# ---------------------------------------------------------------------------
def refresh_3d_scene(workspace, frame_sel, show_cam, is_example,
                     filter_sky, show_mesh, filter_amb):
    if is_example == "True":
        return gr.update(), gr.update(), "Click Reconstruct first.", gr.update()
    if not workspace or workspace == "None" or not os.path.isdir(workspace):
        return gr.update(), gr.update(), "No results. Please reconstruct first.", gr.update()
    pred_file = os.path.join(workspace, "predictions.npz")
    if not os.path.exists(pred_file):
        return gr.update(), gr.update(), "Missing predictions. Reconstruct first.", gr.update()

    data = np.load(pred_file, allow_pickle=True)
    preds = {k: data[k] for k in data.keys()}

    safe = frame_sel.replace(".", "_").replace(":", "").replace(" ", "_")
    glb_name = f"scene_{safe}_cam{show_cam}_mesh{show_mesh}_edges{filter_amb}_sky{filter_sky}.glb"
    glb_path = os.path.join(workspace, glb_name)
    if not os.path.exists(glb_path):
        scene = convert_predictions_to_glb_scene(
            preds, filter_by_frames=frame_sel, show_camera=show_cam,
            mask_sky_bg=filter_sky, as_mesh=show_mesh, mask_ambiguous=filter_amb,
        )
        scene.export(file_obj=glb_path)

    gs_path = os.path.join(workspace, "gaussians.ply")
    gs_url = f"/file={gs_path}" if os.path.exists(gs_path) else None
    return glb_path, gs_path if os.path.exists(gs_path) else None, "3D scene updated.", gs_url


def refresh_views_on_filter(workspace, sky_filter, pd, d_slider, n_slider):
    if not workspace or workspace == "None" or not os.path.isdir(workspace):
        return pd, None, None
    pred_file = os.path.join(workspace, "predictions.npz")
    if not os.path.exists(pred_file):
        return pd, None, None
    try:
        data = np.load(pred_file, allow_pickle=True)
        preds = {k: data[k] for k in data.keys()}
        images_dir = os.path.join(workspace, "images")
        img_paths = sorted(
            glob(os.path.join(images_dir, "*.png"))
            + glob(os.path.join(images_dir, "*.jpg"))
            + glob(os.path.join(images_dir, "*.jpeg"))
        )
        effective = compute_adaptive_target_size(img_paths, max_target_size=TARGET_SIZE)
        imgs = prepare_images_to_tensor(img_paths, target_size=effective).detach().cpu().numpy()

        refreshed = {}
        for i in range(imgs.shape[1]):
            mask = preds["final_mask"][i].copy()
            if sky_filter:
                mask = mask & preds["sky_mask"][i]
            refreshed[i] = {
                "image": imgs[0, i],
                "points3d": preds["world_points"][i],
                "depth": preds["depth"][i].squeeze(),
                "normal": preds["normal"][i],
                "mask": mask,
            }
        di = max(0, int(d_slider or 1) - 1)
        ni = max(0, int(n_slider or 1) - 1)
        return refreshed, update_depth_view(refreshed, di), update_normal_view(refreshed, ni)
    except Exception as e:
        print(f"Error refreshing views: {e}")
        return pd, None, None


# ---------------------------------------------------------------------------
# Example scene helpers
# ---------------------------------------------------------------------------
def extract_example_scenes(base_dir):
    if not os.path.exists(base_dir):
        return []
    exts = {"jpg", "jpeg", "png", "bmp", "tiff", "tif"}
    scenes = []
    for name in sorted(os.listdir(base_dir)):
        d = os.path.join(base_dir, name)
        if not os.path.isdir(d):
            continue
        imgs = []
        for e in exts:
            imgs.extend(glob(os.path.join(d, f"*.{e}")))
            imgs.extend(glob(os.path.join(d, f"*.{e.upper()}")))
        if imgs:
            imgs.sort()
            scenes.append({"name": name, "path": d, "thumbnail": imgs[0],
                           "num_images": len(imgs), "image_files": imgs})
    return scenes


def load_example_scene(scene_name, scenes):
    cfg = next((s for s in scenes if s["name"] == scene_name), None)
    if cfg is None:
        return None, None, None, None, "Scene not found"
    td, paths = process_uploaded_files(cfg["image_files"], 1.0)
    return (None, None, td, paths,
            f"Loaded '{scene_name}' ({cfg['num_images']} images). Click Reconstruct.")


# ---------------------------------------------------------------------------
# Trivial callbacks
# ---------------------------------------------------------------------------
def clear_fields():
    return None

def update_log():
    return "Loading and Reconstructing..."

def get_terminal_output():
    return current_terminal_output

def _safe_int(v, default=1):
    try:
        return int(float(v)) if v is not None else default
    except (ValueError, TypeError):
        return default

def _nav_depth(pd, target):
    if not pd:
        return None, update_view_info(1, 1)
    n = len(pd)
    idx = max(0, min(_safe_int(target) - 1, n - 1))
    return update_depth_view(pd, idx), update_view_info(idx + 1, n)

def _nav_normal(pd, target):
    if not pd:
        return None, update_view_info(1, 1, "Normal")
    n = len(pd)
    idx = max(0, min(_safe_int(target) - 1, n - 1))
    return update_normal_view(pd, idx), update_view_info(idx + 1, n, "Normal")


def _get_splat_url(output_dir, gs_file):
    """Build URL for the PLY splat file via our static mount."""
    # Check gradio_served directory (where we copied the PLY in gradio_demo)
    served_path = os.path.join(os.getcwd(), "gradio_served", "gaussians.ply")
    print(f"[SPARK] _get_splat_url: checking {served_path}, exists={os.path.isfile(served_path)}")

    if os.path.isfile(served_path):
        # Return relative path for our StaticFiles mount at /gradio_served/
        return "gradio_served/gaussians.ply"

    # Fallback: try gs_file from gradio_demo output (absolute path)
    if gs_file and isinstance(gs_file, str) and gs_file != "None" and os.path.isfile(gs_file):
        print(f"[SPARK] Using gs_file fallback: {gs_file}")
        return gs_file

    # Fallback: try output_dir
    if output_dir and output_dir != "None" and isinstance(output_dir, str):
        ply_path = os.path.join(output_dir, "gaussians.ply")
        if os.path.isfile(ply_path):
            print(f"[SPARK] Using output_dir fallback: {ply_path}")
            return ply_path

    print(f"[SPARK ERROR] PLY not found anywhere. served={served_path}, gs_file={gs_file}, output_dir={output_dir}")
    return None


def _clear_spark():
    """Reset the Spark HTML overlay."""
    return None


def _spark_js_callback():
    """JavaScript code to load and display splat file using Spark.gl"""
    return """
    async (url) => {
        console.log("[Spark] Loading:", url);
        if (!url) {
            console.log("[Spark] No URL provided");
            return;
        }
    
        // If already initialized, just load the file
        if (window._sparkReady) {
            console.log("[Spark] Already initialized, loading:", url);
            try { await window._sparkLoadSplat([url]); } catch(e) { console.error("[Spark] Error:", e); }
            return;
        }
    
        // Show loading
        const overlay = document.getElementById("spark-overlay");
        const controls = document.getElementById("spark-controls");
        const info = document.getElementById("spark-info");
        if (overlay) overlay.innerHTML = '<div style="color:#888;">Loading 3D viewer...</div>';
        if (controls) controls.style.display = "none";
    
        try {
            // Inject import map dynamically into document head
            const importMap = {
              imports: {
                "three": "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.180.0/three.module.js",
                "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/",
                "spark": "https://sparkjs.dev/releases/spark/preview/2.0.0/spark.module.js",
                "lil-gui": "http://127.0.0.1:8090/vendor/js/lil-gui.esm.js"
              }
            };
            const im = document.createElement('script');
            im.type = 'importmap';
            im.textContent = JSON.stringify(importMap);
            document.head.appendChild(im);
            console.log("[Spark] Import map injected");
    
            console.log("[Spark] Loading Three.js...");
            const THREE = await import("three");
            const { OrbitControls } = await import("three/addons/controls/OrbitControls.js");
    
            console.log("[Spark] Loading Spark...");
            const spark = await import("spark");
            const { SplatMesh, SparkRenderer, SparkControls, constructGrid, isMobile, LN_SCALE_MIN, LN_SCALE_MAX } = spark;
    
            console.log("[Spark] Loading lil-gui...");
            const { GUI } = await import("lil-gui");
    
            // ---- Setup renderer ----
            const canvas = document.getElementById("spark-canvas");
            const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: "high-performance" });
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.setClearColor(0x000000, 1);
    
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(75, canvas.clientWidth / canvas.clientHeight || 1, 0.01, 1000);
            camera.position.set(0, 0, 1);
    
            const sparkRenderer = new SparkRenderer({ renderer });
            scene.add(sparkRenderer);
    
            function handleResize() {
                const width = canvas.clientWidth;
                const height = canvas.clientHeight;
                renderer.setSize(width, height, false);
                camera.aspect = width / height;
                camera.updateProjectionMatrix();
            }
            handleResize();
            window.addEventListener("resize", handleResize);
    
            // ---- frame Group (official pattern: coord-independent transforms) ----
            const frame = new THREE.Group();
            frame.quaternion.set(1, 0, 0, 0);
            scene.add(frame);
    
            // ---- Grid (procedural reference) ----
            const grid = new SplatMesh({
                constructSplats: (splats) => constructGrid({
                    splats,
                    extents: new THREE.Box3(new THREE.Vector3(-10, -10, -10), new THREE.Vector3(10, 10, 10)),
                }),
            });
            grid.opacity = 0;
            grid.visible = false;
            scene.add(grid);
    
            // ---- Controls ----
            const sparkControls = new SparkControls({ canvas });
    
            const orbitControls = new OrbitControls(camera, renderer.domElement);
            orbitControls.enabled = false;
            orbitControls.target.set(0, 0, 0);
            orbitControls.minDistance = 0.1;
            orbitControls.maxDistance = 10;
    
            // ---- Progress bar ----
            const progressBar = document.getElementById("spark-progress-bar");
            const progressFill = document.getElementById("spark-progress-fill");
    
            function showProgress() {
                if (progressBar) { progressBar.style.display = "block"; progressFill.style.width = "0%"; }
            }
    
            function updateProgress(progress) {
                if (progressFill) { progressFill.style.width = Math.min(100, Math.max(0, progress * 100)) + "%"; }
            }
    
            function hideProgress() {
                if (progressBar) { progressBar.style.display = "none"; }
            }
    
            function calculateUnknownProgress(loadedBytes) {
                const midpointMB = 10 * 1024 * 1024;
                return loadedBytes / (loadedBytes + midpointMB);
            }
    
            async function fetchWithProgress(url) {
                try {
                    const response = await fetch(url, { mode: "cors", cache: "default" });
                    if (!response.ok) { throw new Error("HTTP error! status: " + response.status); }
                    const contentLength = response.headers.get("content-length");
                    const total = contentLength ? parseInt(contentLength, 10) : null;
                    const reader = response.body.getReader();
                    const chunks = [];
                    let loadedBytes = 0;
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        chunks.push(value);
                        loadedBytes += value.length;
                        const progress = total ? loadedBytes / total : calculateUnknownProgress(loadedBytes);
                        updateProgress(progress);
                    }
                    const allChunks = new Uint8Array(loadedBytes);
                    let offset = 0;
                    for (const chunk of chunks) { allChunks.set(chunk, offset); offset += chunk.length; }
                    return allChunks.buffer;
                } catch (error) {
                    hideProgress();
                    throw error;
                }
            }
    
            // ---- GUI Options ----
            const guiOptions = {
                highDevicePixel: !isMobile(),
                stats: false,
                resetOnLoad: true,
                loadOffset: 0,
                coordSystem: "OpenGL",
                autoRotate: false,
                orbit: false,
                reversePointerDir: false,
                reversePointerSlide: false,
                backgroundColor: "#111111",
                viewBoundingBox: false,
                showCones: false,
                splatCount: "-",
                openFiles: () => {},
                loadFromText: "",
                loadFromTextAction: () => {},
                resetPose: () => {
                    camera.position.set(0, 0, 1);
                    camera.quaternion.set(0, 0, 0, 1);
                    camera.fov = 75;
                    resetFrameQuaternion();
                    camera.updateProjectionMatrix();
                },
            };
    
            const splatExtra = { extSplats: true, lod: false };
    
            const splatEncoding = {
                rgbMin: 0.0, rgbMax: 1.0,
                lnScaleMin: LN_SCALE_MIN, lnScaleMax: LN_SCALE_MAX,
                sh1Max: 1, sh2Max: 1, sh3Max: 1,
                lodOpacity: false,
            };
    
            function resetFrameQuaternion() {
                if (guiOptions.coordSystem === "OpenCV") {
                    frame.quaternion.set(1, 0, 0, 0);
                } else if (guiOptions.coordSystem === "OpenGL") {
                    frame.quaternion.set(0, 0, 0, 1);
                } else if (guiOptions.coordSystem === "Z-up") {
                    frame.quaternion.set(-1, 0, 0, 1).normalize();
                }
            }

            function touch() { sparkRenderer.needsUpdate = true; }
    
            // ---- lil-gui Panels ----
            const gui = new GUI({ title: "Settings", container: document.getElementById("spark-gui") });
            const secondGui = new GUI({ title: "Splats", container: document.getElementById("spark-splats-gui") }).close();
    
            // -- Camera folder --
            const cameraFolder = gui.addFolder("Camera").close();
            const cameraPose = cameraFolder.addFolder("Camera Pose").close();
            cameraPose.add(camera.position, "x", -10, 10, 0.01).name("X").listen();
            cameraPose.add(camera.position, "y", -10, 10, 0.01).name("Y").listen();
            cameraPose.add(camera.position, "z", -10, 10, 0.01).name("Z").listen();
            const rotX = cameraPose.add(camera.rotation, "x", -Math.PI, Math.PI, 0.01).name("RotateX").listen();
            const rotY = cameraPose.add(camera.rotation, "y", -Math.PI, Math.PI, 0.01).name("RotateY").listen();
            const rotZ = cameraPose.add(camera.rotation, "z", -Math.PI, Math.PI, 0.01).name("RotateZ").listen();
            cameraPose.add(camera, "fov", 1, 179, 1).name("Fov Y degrees").listen().onChange(() => {
                camera.updateProjectionMatrix();
            });
    
            cameraFolder.add(guiOptions, "resetPose").name("Reset pose");
            cameraFolder.add(guiOptions, "coordSystem", ["OpenCV", "OpenGL", "Z-up"]).name("Coordinate system")
                .listen().onChange(resetFrameQuaternion);
            cameraFolder.add(guiOptions, "autoRotate").name("Auto rotate").listen().onChange((value) => {
                if (value) { frame.rotation.y = 0; }
            });
            cameraFolder.add(guiOptions, "orbit").name("Orbit controls").listen().onChange((value) => {
                orbitControls.enabled = value;
                canvas.focus();
                rotX.enable(!value);
                rotY.enable(!value);
                rotZ.enable(!value);
            });
            cameraFolder.add(guiOptions, "reversePointerDir").name("Reverse ptr direction").onChange((value) => {
                sparkControls.pointerControls.reverseRotate = value;
                sparkControls.pointerControls.reverseScroll = value;
            });
            cameraFolder.add(guiOptions, "reversePointerSlide").name("Reverse ptr slide").onChange((value) => {
                sparkControls.pointerControls.reverseSlide = value;
                sparkControls.pointerControls.reverseSwipe = value;
            });
    
            // -- Level-of-Detail folder --
            const lodFolder = gui.addFolder("Level-of-Detail").close();
            lodFolder.add(splatExtra, "lod").name("Create Level-of-Detail on load");
            lodFolder.add(sparkRenderer, "lodSplatScale", 0.001, 3.0, 0.001).name("LoD splat scale").listen();
            lodFolder.add(guiOptions, "splatCount").name("Current splat count").listen();
            lodFolder.add(sparkRenderer, "coneFov0", 0.0, 170.0, 1.0).name("Cone Fov0").listen().onChange(updateConeMeshes);
            lodFolder.add(sparkRenderer, "coneFov", 0.0, 170.0, 1.0).name("Cone Fov").listen().onChange(updateConeMeshes);
            lodFolder.add(guiOptions, "showCones").name("Show FOV cones").listen().onChange((show) => {
                if (cone0Mesh) cone0Mesh.visible = show;
                if (coneMesh) coneMesh.visible = show;
            });
            lodFolder.add(sparkRenderer, "coneFoveate", 0.0, 1.0, 0.01).name("Cone Foveate").listen();
            lodFolder.add(sparkRenderer, "behindFoveate", 0.0, 1.0, 0.01).name("Behind Foveate").listen();
            lodFolder.add(sparkRenderer, "enableDriveLod").name("Enable LoD updates").listen();
            lodFolder.add(sparkRenderer, "lodInflate").name("Soften LoD splats").listen();

            // -- Main GUI (continued) --
            gui.add(guiOptions, "highDevicePixel").name("High DPI").onChange((value) => { setHighDpi(value); });
            gui.add(sparkRenderer, "sortRadial").name("Radial sort").listen();
            gui.add(grid, "opacity", 0, 1, 0.01).name("Grid opacity").listen().onChange((value) => {
                grid.visible = value > 0;
            });
            gui.add({ logFocalDistance: 0.0 }, "logFocalDistance", -2, 2, 0.01).name("Ln(Focal distance)").onChange((value) => {
                sparkRenderer.focalDistance = Math.exp(value);
            });
            gui.add(sparkRenderer, "apertureAngle", 0, 0.01 * Math.PI, 0.001).name("Aperture angle").listen();
            scene.background = new THREE.Color(guiOptions.backgroundColor);
            gui.addColor(guiOptions, "backgroundColor").name("Background color").onChange((value) => {
                scene.background.set(value);
            });
    
            // -- Debug folder --
            const debugFolder = gui.addFolder("Debug").close();
            debugFolder.add(guiOptions, "viewBoundingBox").name("View bounding boxes").onChange((v) => {
                frame.children.forEach((child) => {
                    if (child instanceof THREE.Box3Helper) { child.visible = v; }
                });
            });
            debugFolder.add(sparkRenderer, "maxStdDev", 0.1, 3.0, 0.01).name("Max Gsplat stddev").listen();
            debugFolder.add(sparkRenderer, "falloff", 0, 1, 0.01).name("Gaussian falloff").listen();
            debugFolder.add(sparkRenderer, "preBlurAmount", 0, 2, 0.1).name("Blur amount (no AA)").listen();
            debugFolder.add(sparkRenderer, "blurAmount", 0, 2, 0.1).name("Blur amount (AA)").listen();
            debugFolder.add({ nonAA: () => { sparkRenderer.preBlurAmount = 0.3; sparkRenderer.blurAmount = 0.0; } }, "nonAA").name("Non-AA preset");
            debugFolder.add({ AA: () => { sparkRenderer.preBlurAmount = 0.0; sparkRenderer.blurAmount = 0.3; } }, "AA").name("AA preset");
            debugFolder.add(sparkRenderer, "focalAdjustment", 0.1, 2.0, 0.1).name("Tweak focalAdjustment").listen();
            debugFolder.add(sparkRenderer, "minPixelRadius", 0, 16, 0.1).name("Min pixel radius").listen();
            debugFolder.add(sparkRenderer, "maxPixelRadius", 1, 1024, 1).name("Max pixel radius").listen();
            debugFolder.add(sparkRenderer, "minAlpha", 0, 1, 0.001).name("Min alpha").listen();
            debugFolder.add(sparkRenderer, "premultipliedAlpha").name("Premultiplied alpha").listen();
    
            // -- SplatMesh encoding folder --
            const splatFolder = gui.addFolder("SplatMesh encoding").close();
            splatFolder.add(splatExtra, "extSplats").name("Extended splats");
            splatFolder.add(splatEncoding, "rgbMin", -1, 1, 0.1).name("RGB min").onChange(touch);
            splatFolder.add(splatEncoding, "rgbMax", 0, 4, 0.1).name("RGB max").onChange(touch);
            splatFolder.add(splatEncoding, "lnScaleMin", -14, -2.5, 0.1).name("Ln scale min").onChange(touch);
            splatFolder.add(splatEncoding, "lnScaleMax", -14, 14, 0.1).name("Ln scale max").onChange(touch);
            splatFolder.add(splatEncoding, "sh1Max", -6, 6, 0.1).name("SH1 max").onChange(touch);
            splatFolder.add(splatEncoding, "sh2Max", -6, 6, 0.1).name("SH2 max").onChange(touch);
            splatFolder.add(splatEncoding, "sh3Max", -6, 6, 0.1).name("SH3 max").onChange(touch);
    
            // -- Splats GUI (left panel) --
            guiOptions.openFiles = () => {
                const fileInput = document.getElementById("spark-file-input");
                if (fileInput) fileInput.click();
            };
            guiOptions.loadFromTextAction = () => {
                if (guiOptions.loadFromText.trim()) {
                    const urls = parseURLsFromText(guiOptions.loadFromText);
                    if (urls.length > 0) { loadFiles(urls); guiOptions.loadFromText = ""; }
                    else { alert("No valid URLs found. URLs must start with http:// or https:// and end with .ply, .spz, .splat, .ksplat, .json, .zip, .sog, or .rad"); }
                }
            };

            // File input change handler
            const fileInput = document.getElementById("spark-file-input");
            if (fileInput) {
                fileInput.onchange = (event) => {
                    loadFiles([...event.target.files]);
                };
            }

            secondGui.add(guiOptions, "resetOnLoad").name("Reset on load");
            secondGui.add(guiOptions, "loadOffset", -2, 2, 0.01).name("Loading offset");
            secondGui.add(guiOptions, "openFiles").name("Select Files");
            secondGui.add(guiOptions, "loadFromText").name("Paste URL(s) here");
            secondGui.add(guiOptions, "loadFromTextAction").name("Load from URL(s)");
    
            const splatsFolder = secondGui.addFolder("Files");
    
            // ---- Cone meshes for FOV visualization ----
            let cone0Mesh = null;
            let coneMesh = null;
            function makeConeMesh(fov0) {
                const radius = 32; const height = 32; const segments = 32;
                const geometry = new THREE.ConeGeometry(radius, height, segments, 1, true);
                geometry.translate(0, -height / 2, 0);
                geometry.rotateX(Math.PI / 2);
                const edges = new THREE.EdgesGeometry(geometry);
                return new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: fov0 ? 0xffcc00 : 0x00ccff }));
            }
            function makeConeCircles(fov0) {
                const group = new THREE.Group();
                const segments = 32;
                const color = fov0 ? 0xffcc00 : 0x00ccff;
                const material = new THREE.LineBasicMaterial({ color });
                for (const z of [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]) {
                    const geom = new THREE.BufferGeometry();
                    const vertices = [];
                    for (let j = 0; j <= segments; ++j) {
                        const theta = (j / segments) * 2 * Math.PI;
                        vertices.push(Math.cos(theta) * z, Math.sin(theta) * z, -z);
                    }
                    geom.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
                    group.add(new THREE.Line(geom, material));
                }
                group.add(makeConeMesh(fov0));
                return group;
            }
            cone0Mesh = makeConeCircles(true);
            cone0Mesh.visible = false;
            scene.add(cone0Mesh);
            coneMesh = makeConeCircles(false);
            coneMesh.visible = false;
            scene.add(coneMesh);
    
            function updateConeMeshes() {
                const tan0 = Math.tan(0.5 * sparkRenderer.coneFov0 * Math.PI / 180);
                cone0Mesh.scale.set(tan0, tan0, 1);
                const tan = Math.tan(0.5 * sparkRenderer.coneFov * Math.PI / 180);
                coneMesh.scale.set(tan, tan, 1);
            }
            updateConeMeshes();
    
            // ---- High DPI ----
            function setHighDpi(value) {
                renderer.setPixelRatio(value ? window.devicePixelRatio : 1);
                const width = canvas.clientWidth;
                const height = canvas.clientHeight;
                renderer.setSize(width, height, false);
            }
            setHighDpi(guiOptions.highDevicePixel);
    
            // ---- Bounding box helper ----
            async function addBoundingBoxHelper(splatMesh) {
                await splatMesh.initialized;
                const box = splatMesh.getBoundingBox();
                const boxHelper = new THREE.Box3Helper(box, 0x00ff00);
                boxHelper.visible = guiOptions.viewBoundingBox;
                frame.add(boxHelper);
            }
    
            // ---- loadFiles function (official pattern) ----
            async function loadFiles(splatFiles) {
                if (guiOptions.resetOnLoad) {
                    const toRemove = frame.children.filter((child) => child instanceof SplatMesh || child instanceof THREE.Box3Helper);
                    for (const child of toRemove) { frame.remove(child); }
                    splatsFolder.foldersRecursive().forEach((child) => child.destroy());
                }
    
                guiOptions.autoRotate = false;
                resetFrameQuaternion();
    
                const hasUrls = splatFiles.some(function(file) { return typeof file === "string"; });
                if (hasUrls) { showProgress(); }
    
                let index = 0;
                for (const splatFile of splatFiles) {
                    try {
                        let fileName = undefined;
                        let stream = undefined;
                        let streamLength = undefined;
                        let url = undefined;
    
                        if (typeof splatFile === "string") {
                            url = splatFile;
                            // Rewrite gradio_served/ paths to use the static file server on port 8090
                            if (url.includes("gradio_served/")) {
                                url = "http://127.0.0.1:8090/" + url;
                                console.log("[Spark] Rewrote URL to static server:", url);
                            }
                            fileName = splatFile.split("/").pop().split("?")[0] || "downloaded-file";
                        } else {
                            fileName = splatFile.name;
                            stream = splatFile.stream();
                            streamLength = splatFile.size;
                        }
    
                        // Build SplatMesh init with splatExtra and splatEncoding
                        const init = { url, fileName, stream, streamLength };
                        for (const k in splatExtra) { if (splatExtra.hasOwnProperty(k)) init[k] = splatExtra[k]; }
                        init.splatEncoding = {};
                        for (const k in splatEncoding) { if (splatEncoding.hasOwnProperty(k)) init.splatEncoding[k] = splatEncoding[k]; }

                        const splatMesh = new SplatMesh(init);
                        const translate = guiOptions.loadOffset * index;
                        splatMesh.position.set(translate, 0.5 * translate, 0.1 * translate);
                        splatMesh.enableWorldToView = true;
                        await splatMesh.initialized;

                        frame.add(splatMesh);
                        // Show lil-gui panels after first splat loads
                        document.getElementById("spark-gui").classList.add("lil-ready");
                        document.getElementById("spark-splats-gui").classList.add("lil-ready");
                        const numSplats = splatMesh.splats ? (splatMesh.splats.lodSplats ? splatMesh.splats.lodSplats.numSplats : splatMesh.splats.numSplats) : null;
                        console.log("Loaded " + fileName + " with " + numSplats + " splats");
                        addBoundingBoxHelper(splatMesh);
    
                        // Per-file folder in Splats GUI
                        const splatFolder = splatsFolder.addFolder(fileName).close();
                        splatFolder.add(splatMesh, "opacity", 0, 1, 0.01).name("Opacity").listen();
                        splatFolder.add(splatMesh.position, "x", -10, 10, 0.01).name("X").listen();
                        splatFolder.add(splatMesh.position, "y", -10, 10, 0.01).name("Y").listen();
                        splatFolder.add(splatMesh.position, "z", -10, 10, 0.01).name("Z").listen();
                        splatFolder.add(splatMesh.scale, "x", 0.01, 4, 0.01).name("Scale").listen().onChange((value) => {
                            splatMesh.scale.setScalar(value);
                        });
                        splatFolder.add(splatMesh.rotation, "x", -Math.PI, Math.PI, 0.01).name("RotateX").listen();
                        splatFolder.add(splatMesh.rotation, "y", -Math.PI, Math.PI, 0.01).name("RotateY").listen();
                        splatFolder.add(splatMesh.rotation, "z", -Math.PI, Math.PI, 0.01).name("RotateZ").listen();
                        splatFolder.add(splatMesh, "maxSh", 0, 3, 1).name("Max SH").listen().onChange(() => {
                            splatMesh.updateGenerator();
                        });
                    } catch (error) {
                        console.error("Error loading splat file:", error);
                    }
                    index += 1;
                }
    
                if (hasUrls) { hideProgress(); }
                canvas.focus();
            }
    
            // ---- Drag & drop ----
            canvas.addEventListener("dragover", (e) => {
                e.preventDefault();
                canvas.style.opacity = "0.5";
            });
            canvas.addEventListener("dragleave", (e) => {
                e.preventDefault();
                canvas.style.opacity = "1";
            });
            canvas.addEventListener("drop", async (e) => {
                e.preventDefault();
                canvas.style.opacity = "1";
                const textData = e.dataTransfer.getData("text/plain");
                if (textData) {
                    const urls = parseURLsFromText(textData);
                    if (urls.length > 0) { loadFiles(urls); return; }
                }
                const files = Array.from(e.dataTransfer.files);
                const exts = [".ply", ".spz", ".splat", ".ksplat", ".zip", ".sog", ".rad"];
                const splatFiles = files.filter(function(f) {
                    return exts.some(function(ext) { return f.name.toLowerCase().endsWith(ext); });
                });
                if (splatFiles.length > 0) { loadFiles(splatFiles); }
            });
    
            function parseURLsFromText(text) {
                const supportedExtensions = [".ply", ".spz", ".splat", ".ksplat", ".zip", ".json", ".sog", ".rad"];
                const urls = [];
                const parts = text.trim().split(/[\\r\\n,;]+/);
                for (const part of parts) {
                    const trimmed = part.trim();
                    if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
                        if (supportedExtensions.some(function(ext) { return trimmed.toLowerCase().includes(ext); })) {
                            urls.push(trimmed);
                        }
                    }
                }
                return urls;
            }
    
            // ---- Expose load function ----
            window._sparkLoadSplat = loadFiles;
    
            console.log("[Spark] Editor panels initialized");
    
            // Focus canvas for keyboard controls
            canvas.focus();
    
            // ---- Render loop ----
            let lastTime = null;
            renderer.setAnimationLoop(function animate(time) {
                const deltaTime = time - (lastTime || time);
                lastTime = time;
    
                // Auto rotate via frame group
                if (guiOptions.autoRotate) {
                    if (guiOptions.coordSystem === "OpenCV") {
                        frame.rotation.y = time / 5000;
                    } else if (guiOptions.coordSystem === "OpenGL") {
                        frame.rotation.y = -time / 5000;
                    } else if (guiOptions.coordSystem === "Z-up") {
                        frame.rotation.z = -time / 5000;
                    }
                }
    
                // Update controls
                if (guiOptions.orbit) {
                    orbitControls.update();
                } else {
                    sparkControls.update(camera);
                }
    
                // Update cone mesh positions
                const lodPos = sparkRenderer.currentLod ? (sparkRenderer.currentLod.pos || new THREE.Vector3().copy(camera.position)) : new THREE.Vector3().copy(camera.position);
                const lodQuat = sparkRenderer.currentLod ? (sparkRenderer.currentLod.quat || new THREE.Quaternion().copy(camera.quaternion)) : new THREE.Quaternion().copy(camera.quaternion);
                if (cone0Mesh) { cone0Mesh.position.copy(lodPos); cone0Mesh.quaternion.copy(lodQuat); }
                if (coneMesh) { coneMesh.position.copy(lodPos); coneMesh.quaternion.copy(lodQuat); }
    
                renderer.render(scene, camera);
                guiOptions.splatCount = sparkRenderer.display ? ("" + sparkRenderer.display.numSplats) : "-";
            });
    
            // ---- Hide overlay, show controls ----
            if (overlay) overlay.style.display = "none";
            if (controls) controls.style.display = "flex";
    
            // Mark initialized
            window._sparkReady = true;
    
            // Load file immediately if URL was provided
            if (url) {
                console.log("[Spark] Loading initial URL:", url);
                await window._sparkLoadSplat([url]);
            }
    
            console.log("[Spark] Initialized");
        } catch(e) {
            console.error("[Spark] Init error:", e);
            if (overlay) overlay.innerHTML = '<div style="color:#f55;">Failed: ' + (e.message || String(e)) + '</div>';
        }
    }
    
"""

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_demo(examples_dir="./examples/worldrecon"):
    theme = gr.themes.Base()

    css = """
    .custom-log * {
        font-style: italic; font-size: 22px !important;
        background-image: linear-gradient(120deg, #a9b8f8 0%, #7081e8 60%, #4254c5 100%);
        -webkit-background-clip: text; background-clip: text;
        font-weight: bold !important; color: transparent !important; text-align: center !important;
    }
    .normal-weight-btn button, .normal-weight-btn button span { font-weight: 400 !important; }
    .terminal-output { max-height: 400px !important; overflow-y: auto !important; }
    .terminal-output textarea {
        font-family: 'Monaco','Menlo','Ubuntu Mono',monospace !important;
        font-size: 13px !important; line-height: 1.5 !important;
        color: #333 !important; background-color: #f8f9fa !important;
        max-height: 400px !important;
    }
    .example-gallery img { width: 100% !important; height: 280px !important; object-fit: contain !important; }
    .example-col, .example-col > div {
        max-height: 85vh !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }
    .example-col .gallery {
        max-height: none !important;
        overflow: visible !important;
    }
    #gs-output-container { display: none !important; height: 0 !important; overflow: hidden !important; }
    #gs-output-container > * { display: none !important; height: 0 !important; visibility: hidden !important; }
    }
    """

    with gr.Blocks(theme=theme, css=css) as demo:
        is_example = gr.Textbox(visible=False, value="None")
        processed_data_state = gr.State(value=None)
        gs_ply_state = gr.Textbox(visible=False, lines=1)

        gr.HTML("""
        <script type="importmap">
        {
          "imports": {
            "three": "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.180.0/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/",
            "spark": "https://sparkjs.dev/releases/spark/preview/2.0.0/spark.module.js",
            "lil-gui": "http://127.0.0.1:8090/vendor/js/lil-gui.esm.js"
          }
        }
        </script>
        <div style="text-align:center;">
        <h1>
            <span style="background:linear-gradient(90deg,#3b82f6,#1e40af);-webkit-background-clip:text;
            background-clip:text;color:transparent;font-weight:bold;">WorldMirror 2.0:</span>
            <span style="color:#555;">Universal 3D World Reconstruction</span>
        </h1>
        <p>
        <a href="https://arxiv.org/abs/2510.10726">📄 Paper</a> |
        <a href="https://3d-models.hunyuan.tencent.com/world/">🌐 Project</a> |
        <a href="https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror">💻 GitHub</a> |
        <a href="https://huggingface.co/tencent/HY-World-2.0">🤗 Model</a>
        </p>
        </div>
        <div style="font-size:16px;line-height:1.5;">
        <p>WorldMirror supports any combination of inputs and outputs including point clouds, camera parameters,
        depth maps, normal maps, and 3D Gaussian Splatting.</p>
        <ol>
            <li><strong>Upload</strong> video or images</li>
            <li><strong>Reconstruct</strong> to start 3D processing</li>
            <li><strong>Visualize</strong> across multiple tabs: 3DGS, Point Cloud, Depth, Normal, Camera</li>
        </ol>
        </div>
        """)

        output_path_state = gr.Textbox(visible=False, value="None")

        with gr.Row(equal_height=False):
            # ---------- Left column: Upload ----------
            with gr.Column(scale=1):
                file_upload = gr.File(file_count="multiple", label="Upload Video or Images",
                                      interactive=True, file_types=["image", "video"], height="200px")
                time_interval = gr.Slider(0.1, 10.0, value=1.0, step=0.1,
                                          label="Video Sample Interval (s)", scale=4)
                resample_btn = gr.Button("Resample", scale=1, elem_classes=["normal-weight-btn"])
                image_gallery = gr.Gallery(label="Image Preview", columns=4, height="200px",
                                           show_download_button=True, object_fit="contain", preview=True)
                terminal_output = gr.Textbox(label="Terminal Output", lines=6, max_lines=6,
                                             interactive=False, show_copy_button=True,
                                             elem_classes=["terminal-output"], autoscroll=True)

            # ---------- Center column: Visualization ----------
            with gr.Column(scale=3):
                log_output = gr.Markdown("Upload files, then click Reconstruct.", elem_classes=["custom-log"])

                with gr.Tabs():
                    with gr.Tab("3D Gaussian Splatting", id=1):
                        # Hidden Model3D to capture gs_file path for JS interop
                        gs_output = gr.Model3D(visible=True, height=1, elem_id="gs-output-container")

                        # Static HTML for Spark container (no JS execution in gr.HTML)
                        spark_viewer_html = gr.HTML(f"""
                        <div id="spark-container" style="position:relative; width:100%; height:500px;
                             background:#111; border-radius:8px; overflow:hidden; margin:0;">
                          <div id="spark-progress-bar" style="position:absolute; top:0; left:0; width:100%; height:4px;
                               background-color:rgba(0,0,0,0.1); z-index:1000; display:none;">
                            <div id="spark-progress-fill" style="height:100%;
                                 background:linear-gradient(90deg, #4CAF50, #45a049); width:0%;
                                 transition:width 0.3s ease;"></div>
                          </div>
                          <canvas id="spark-canvas" style="display:block; width:100%; height:100%; outline:none;" tabindex="0"></canvas>
                          <div id="spark-overlay" style="position:absolute; top:0; left:0; right:0; bottom:0;
                               display:flex; align-items:center; justify-content:center; flex-direction:column;
                               background:#1a1a1a; color:#888; font-family:monospace; font-size:14px;">
                            <div style="font-size:18px; margin-bottom:8px;">📡</div>
                            <div>Click "Reconstruct" to generate 3DGS</div>
                          </div>
                          <style>
                            /* Hide lil-gui panels initially */
                            #spark-gui .lil-gui, #spark-splats-gui .lil-gui {{ display: none !important; }}
                            /* White text for all lil-gui elements on dark background */
                            .lil-gui, .lil-gui *, .lil-root {{ color: #e8e8e8 !important; font-family: monospace !important; }}
                            .lil-gui .lil-title {{ color: #ffffff !important; font-weight: 600 !important; font-family: monospace !important; }}
                            .lil-gui input, .lil-gui select, .lil-gui textarea {{ color: #e0e0e0 !important; background: #2a2a2a !important; }}
                            .lil-gui button {{ color: #ffffff !important; }}
                            .lil-gui .title {{ color: #ffffff !important; }}
                            /* Show panels when splat is loaded (class added by JS) */
                            #spark-gui.lil-ready .lil-gui, #spark-splats-gui.lil-ready .lil-gui {{ display: block !important; }}
                          </style>
                          <div id="spark-gui" style="position:absolute; top:5px; right:5px; z-index:10;"></div>
                          <div id="spark-splats-gui" style="position:absolute; top:5px; left:5px; z-index:10;"></div>
                          <input id="spark-file-input" type="file" accept=".ply,.spz,.splat,.ksplat,.zip,.sog,.rad" multiple="true" style="display:none;" />
                          <div id="spark-controls" style="position:absolute; bottom:12px; left:12px; right:12px;
                               display:none; justify-content:space-between; align-items:center; color:#ccc;
                               font-family:monospace; font-size:12px;">
                            <div id="spark-info"></div>
                          </div>
                        </div>
                        """)

                    with gr.Tab("Point Cloud / Mesh", id=0):
                        reconstruction_output = gr.Model3D(label="3D Pointmap / Mesh", height=500,
                                                            zoom_speed=0.4, pan_speed=0.4)

                    with gr.Tab("Depth"):
                        depth_view_info = gr.HTML(update_view_info(1, 1))
                        depth_view_slider = gr.Slider(1, 1, step=1, value=1, label="View")
                        depth_map = gr.Image(type="numpy", label="Depth Map", format="png",
                                             interactive=False, height=340)

                    with gr.Tab("Normal"):
                        normal_view_info = gr.HTML(update_view_info(1, 1, "Normal"))
                        normal_view_slider = gr.Slider(1, 1, step=1, value=1, label="View")
                        normal_map = gr.Image(type="numpy", label="Normal Map", format="png",
                                              interactive=False, height=340)

                    with gr.Tab("Camera Parameters"):
                        with gr.Row():
                            gr.HTML("")
                            camera_params_btn = gr.DownloadButton(label="Download Camera Parameters",
                                                                   scale=1, variant="primary")
                            gr.HTML("")

                with gr.Row():
                    reconstruct_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                    clear_btn = gr.ClearButton([
                        file_upload, reconstruction_output, log_output, output_path_state,
                        image_gallery, depth_map, normal_map, depth_view_slider,
                        normal_view_slider, depth_view_info, normal_view_info,
                        camera_params_btn, gs_output, gs_ply_state,
                    ], scale=1)

                with gr.Row():
                    frame_selector = gr.Dropdown(choices=["All"], value="All",
                                                  label="Show Points of a Specific Frame")

                gr.Markdown("### Reconstruction Options (not applied to 3DGS)")
                with gr.Row():
                    show_camera = gr.Checkbox(label="Show Camera", value=True)
                    show_mesh = gr.Checkbox(label="Show Mesh", value=True)
                    filter_ambiguous = gr.Checkbox(label="Filter low confidence & edges", value=True)
                    filter_sky_bg = gr.Checkbox(label="Filter Sky Background", value=False)

            # ---------- Right column: Examples ----------
            with gr.Column(scale=1, elem_classes=["example-col"]):
                gr.Markdown("### Click to load example scenes")
                real_dir = os.path.join(examples_dir, "realistic")
                style_dir = os.path.join(examples_dir, "stylistic")
                realworld = extract_example_scenes(real_dir) if os.path.isdir(real_dir) else extract_example_scenes(examples_dir)
                stylistic = extract_example_scenes(style_dir) if os.path.isdir(style_dir) else []

                example_outputs = [reconstruction_output, gs_output, output_path_state, image_gallery, log_output]

                def _make_gallery_handler(scene_list):
                    def _handler(evt: gr.SelectData):
                        return load_example_scene(scene_list[evt.index]["name"], scene_list)
                    return _handler

                if not os.path.isdir(real_dir) and not os.path.isdir(style_dir):
                    if realworld:
                        items = [(s["thumbnail"], f"{s['name']}\n{s['num_images']} imgs") for s in realworld]
                        eg = gr.Gallery(value=items, columns=1, height=None, object_fit="contain",
                                        show_label=False, interactive=True, preview=False,
                                        allow_preview=False, elem_classes=["example-gallery"])
                        eg.select(fn=_make_gallery_handler(realworld), outputs=example_outputs)
                else:
                    with gr.Tabs():
                        with gr.Tab("Realistic"):
                            if realworld:
                                items = [(s["thumbnail"], f"{s['name']}\n{s['num_images']} imgs") for s in realworld]
                                rg = gr.Gallery(value=items, columns=1, height=None, object_fit="contain",
                                                show_label=False, interactive=True, preview=False,
                                                allow_preview=False, elem_classes=["example-gallery"])
                                rg.select(fn=_make_gallery_handler(realworld), outputs=example_outputs)
                            else:
                                gr.Markdown("No examples available")
                        with gr.Tab("Stylistic"):
                            if stylistic:
                                items = [(s["thumbnail"], f"{s['name']}\n{s['num_images']} imgs") for s in stylistic]
                                sg = gr.Gallery(value=items, columns=1, height=None, object_fit="contain",
                                                show_label=False, interactive=True, preview=False,
                                                allow_preview=False, elem_classes=["example-gallery"])
                                sg.select(fn=_make_gallery_handler(stylistic), outputs=example_outputs)
                            else:
                                gr.Markdown("No examples available")

        # ---- Events: Reconstruct ----
        reconstruct_btn.click(clear_fields, [], []).then(
            update_log, [], [log_output]
        ).then(
            gradio_demo,
            [output_path_state, frame_selector, show_camera, filter_sky_bg, show_mesh, filter_ambiguous],
            [reconstruction_output, log_output, frame_selector, processed_data_state,
             depth_map, normal_map, depth_view_slider, normal_view_slider,
             depth_view_info, normal_view_info, camera_params_btn, gs_output, terminal_output],
        ).then(
            _get_splat_url, [output_path_state, gs_output], [gs_ply_state],
        ).then(
            lambda url: url,
            [gs_ply_state],
            [gs_ply_state],
            js=_spark_js_callback(),
        ).then(lambda: "False", [], [is_example])

        # ---- Events: Refresh 3D scene on option change ----
        _scene_inputs = [output_path_state, frame_selector, show_camera, is_example,
                         filter_sky_bg, show_mesh, filter_ambiguous]
        _scene_outputs = [reconstruction_output, gs_output, log_output]
        for ctrl in (frame_selector, show_camera, show_mesh):
            ctrl.change(refresh_3d_scene, _scene_inputs, _scene_outputs).then(
                _get_splat_url, [output_path_state, gs_output], [gs_ply_state],
            ).then(
                lambda url: url,
                [gs_ply_state],
                [gs_ply_state],
                js=_spark_js_callback(),
            )

        _filter_inputs = [output_path_state, filter_sky_bg, processed_data_state,
                          depth_view_slider, normal_view_slider]
        _filter_outputs = [processed_data_state, depth_map, normal_map]

        for ctrl in (filter_sky_bg, filter_ambiguous):
            ctrl.change(refresh_3d_scene, _scene_inputs, _scene_outputs).then(
                _get_splat_url, [output_path_state, gs_output], [gs_ply_state],
            ).then(
                lambda url: url,
                [gs_ply_state],
                [gs_ply_state],
                js=_spark_js_callback(),
            ).then(
                refresh_views_on_filter, _filter_inputs, _filter_outputs)

        # ---- Events: File upload ----
        def _on_upload(files, interval):
            if not files:
                return None, None, None, ""
            tee = TeeOutput()
            old = sys.stdout
            sys.stdout = tee
            try:
                td, paths = process_uploaded_files(files, interval)
                log = tee.getvalue()
                sys.stdout = old
                return td, paths, "Upload complete. Click Reconstruct.", log
            except Exception:
                sys.stdout = old
                raise

        def _on_resample(files, interval, cur_dir):
            if not files:
                return cur_dir, None, "No files.", ""
            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
            has_video = any(
                os.path.splitext(str(f["name"] if isinstance(f, dict) else f))[1].lower() in video_exts
                for f in files
            )
            if not has_video:
                return cur_dir, None, "No video found.", ""
            if cur_dir and cur_dir != "None" and os.path.exists(cur_dir):
                shutil.rmtree(cur_dir)
            tee = TeeOutput()
            old = sys.stdout
            sys.stdout = tee
            try:
                td, paths = process_uploaded_files(files, interval)
                log = tee.getvalue()
                sys.stdout = old
                return td, paths, f"Resampled ({interval}s interval). Click Reconstruct.", log
            except Exception:
                sys.stdout = old
                raise

        file_upload.change(_on_upload, [file_upload, time_interval],
                           [output_path_state, image_gallery, log_output, terminal_output])
        resample_btn.click(_on_resample, [file_upload, time_interval, output_path_state],
                            [output_path_state, image_gallery, log_output, terminal_output])

        # ---- Events: Depth / Normal navigation ----
        depth_view_slider.change(_nav_depth, [processed_data_state, depth_view_slider],
                                  [depth_map, depth_view_info])
        normal_view_slider.change(_nav_normal, [processed_data_state, normal_view_slider],
                                   [normal_map, normal_view_info])

        # ---- Terminal polling ----
        timer = gr.Timer(value=0.5)
        timer.tick(get_terminal_output, [], [terminal_output])

        gr.HTML("""
        <hr style="margin-top:40px;margin-bottom:20px;">
        <div style="text-align:center;font-size:14px;color:#666;margin-bottom:20px;">
            <h3>Acknowledgements</h3>
            <p>🔗 <a href="https://github.com/facebookresearch/vggt">VGGT</a> |
            🔗 <a href="https://github.com/CUT3R/CUT3R">CUT3R</a></p>
        </div>
        """)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WorldMirror 2.0 Gradio Demo")
    parser.add_argument("--examples_dir", type=str, default="./examples/worldrecon",
                        help="Path to example scenes directory")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Training config YAML (used with --ckpt_path)")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="tencent/HY-World-2.0",
                        help="HuggingFace repo ID or local path")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Local checkpoint file (.ckpt / .safetensors)")
    parser.add_argument("--use_fsdp", action="store_true", default=False,
                        help="Enable FSDP multi-GPU sharding")
    parser.add_argument("--enable_bf16", action="store_true", default=False,
                        help="Enable bfloat16 mixed precision")
    parser.add_argument("--fsdp_cpu_offload", action="store_true", default=False,
                        help="Offload FSDP params to CPU")
    parser.add_argument("--disable_heads", type=str, default=None,
                        help="Comma-separated list of heads to disable (camera,depth,normal,points,gs). "
                             "E.g. --disable_heads gs to skip 3DGS output and save ~30% memory")
    parser.add_argument("--target_size", type=int, default=952,
                        help="Max inference resolution (longest edge). Lower values use less VRAM.")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs via multiprocessing.spawn. E.g. --num_gpus 3")
    parser.add_argument("--mp_port", type=int, default=29500,
                        help="Port for multi-GPU process group")
    args = parser.parse_args()

    TARGET_SIZE = args.target_size

    _init_pipeline_args(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        config_path=args.config_path,
        ckpt_path=args.ckpt_path,
        use_fsdp=args.use_fsdp,
        enable_bf16=args.enable_bf16,
        fsdp_cpu_offload=args.fsdp_cpu_offload,
        disable_heads=args.disable_heads.split(",") if args.disable_heads else None,
    )

    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1

    if is_distributed:
        pipe = _get_pipeline()
        rank = pipe.rank

        if rank == 0:
            print("[Gradio] Rank 0: launching Gradio server")
            served_dir = Path.cwd()
            served_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Gradio] Static files directory: {served_dir}")

            # Start simple HTTP server for static files on port 8090
            import threading
            import http.server
            import socketserver

            STATIC_PORT = 8090

            class QuietHandler(http.server.SimpleHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, directory=str(served_dir), **kwargs)
                def log_message(self, format, *args):
                    pass  # Suppress request logging
                def end_headers(self):
                    # Add CORS header to allow cross-origin requests
                    self.send_header('Access-Control-Allow-Origin', '*')
                    super().end_headers()

            httpd = socketserver.TCPServer(("", STATIC_PORT), QuietHandler)
            httpd.allow_reuse_address = True
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            print(f"[Static Server] Running on http://127.0.0.1:{STATIC_PORT} serving {served_dir}")

            demo = build_demo(examples_dir=args.examples_dir)
            demo.queue().launch(
                show_error=True,
                share=args.share,
                server_name=args.host,
                server_port=args.port,
                ssr_mode=False,
            )
        else:
            _worker_loop()
    elif args.num_gpus > 1:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(args.mp_port)
        os.environ["OMP_NUM_THREADS"] = "1"

        pipeline_kwargs = dict(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            config_path=args.config_path,
            ckpt_path=args.ckpt_path,
            use_fsdp=args.use_fsdp,
            enable_bf16=args.enable_bf16,
            fsdp_cpu_offload=args.fsdp_cpu_offload,
            disable_heads=args.disable_heads.split(",") if args.disable_heads else None,
            _world_size=args.num_gpus,
        )
        print(f"[Gradio] Spawning {args.num_gpus} processes")
        mp.spawn(_mp_launch_fn,
                 args=(pipeline_kwargs, args.examples_dir, args.host, args.port, args.share, args.mp_port),
                 nprocs=args.num_gpus, join=True)
    else:
        from pathlib import Path
        import threading
        import http.server
        import socketserver

        # Create gradio_served directory for static file serving
        served_dir = Path.cwd()
        served_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Gradio] Static files directory: {served_dir}")

        # Start simple HTTP server for static files on port 8090
        STATIC_PORT = 8090

        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(served_dir), **kwargs)
            def log_message(self, format, *args):
                pass  # Suppress request logging
            def end_headers(self):
                self.send_header('Access-Control-Allow-Origin', '*')
                super().end_headers()

        httpd = socketserver.TCPServer(("", STATIC_PORT), QuietHandler)
        httpd.allow_reuse_address = True
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        print(f"[Static Server] Running on http://127.0.0.1:{STATIC_PORT} serving {served_dir}")

        demo = build_demo(examples_dir=args.examples_dir)
        demo.queue().launch(
            show_error=True,
            share=args.share,
            server_name=args.host,
            server_port=args.port,
            ssr_mode=False,
        )
