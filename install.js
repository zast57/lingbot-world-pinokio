module.exports = {
    run: [
        // 1. Clone repo
        {
            method: "shell.run",
            params: {
                message: "git clone https://github.com/Robbyant/lingbot-world app"
            }
        },
        // 2. Copy patch script and Gradio app to app folder
        {
            method: "fs.copy",
            params: {
                src: "patch_attention.py",
                dest: "app/patch_attention.py"
            }
        },
        {
            method: "fs.copy",
            params: {
                src: "app_gradio.py",
                dest: "app/app_gradio.py"
            }
        },
        // 2.5 Create generate_bnb.py (BNB Quantization Logic)
        {
            method: "fs.write",
            params: {
                path: "app/generate_bnb.py",
                text: `#!/usr/bin/env python3
"""
Generate videos using bitsandbytes NF4 quantized LingBot-World models.

This script loads the NF4 quantized models and generates videos
on a single RTX 5090 GPU with reduced VRAM usage.

Usage:
    python generate_bnb.py \\
        --image examples/00/image.jpg \\
        --prompt "A cinematic video of the scene" \\
        --frame_num 81 \\
        --size 480*832
"""

import argparse
import gc
import logging
import math
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import bitsandbytes as bnb
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_1 import Wan2_1_VAE
from wan.configs.wan_i2v_A14B import i2v_A14B as cfg
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.cam_utils import (
    compute_relative_poses,
    interpolate_camera_poses,
    get_plucker_embeddings,
    get_Ks_transformed,
)
from einops import rearrange

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def replace_linear_with_nf4(model):
    """Replace all nn.Linear layers with bitsandbytes Linear4bit (NF4)."""
    replaced = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            nf4_linear = bnb.nn.Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                compute_dtype=torch.bfloat16,
                compress_statistics=True,
                quant_type='nf4',
            )
            nf4_linear.weight = bnb.nn.Params4bit(
                module.weight.data,
                requires_grad=False,
                compress_statistics=True,
                quant_type='nf4',
            )
            if module.bias is not None:
                nf4_linear.bias = torch.nn.Parameter(module.bias.data.clone())

            setattr(parent, child_name, nf4_linear)
            replaced += 1

    return model, replaced


def load_bnb_nf4_model(ckpt_dir: str, model_name: str):
    """Load model and apply bitsandbytes NF4 quantization on-the-fly.

    Note: We quantize at load time rather than loading pre-quantized weights
    because bitsandbytes quantized state dicts have special format that's
    complex to serialize/deserialize correctly.
    """
    logger.info(f"Loading and quantizing {model_name} with NF4...")

    # Load original model
    model = WanModel.from_pretrained(
        ckpt_dir,
        subfolder=model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )

    # Replace linear layers with NF4 (this also copies weights)
    model, replaced = replace_linear_with_nf4(model)
    logger.info(f"Quantized {replaced} linear layers to NF4")

    return model


class WanI2V_BNB:
    """Image-to-video generation pipeline using bitsandbytes NF4 quantized models."""

    def __init__(
        self,
        checkpoint_dir: str,
        device_id: int = 0,
        t5_cpu: bool = True,
    ):
        self.device = torch.device(f"cuda:{device_id}")
        self.config = cfg
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = cfg.num_train_timesteps
        self.boundary = cfg.boundary
        self.param_dtype = cfg.param_dtype
        self.vae_stride = cfg.vae_stride
        self.patch_size = cfg.patch_size
        self.sample_neg_prompt = cfg.sample_neg_prompt

        # Load T5 encoder
        logger.info("Loading T5 encoder...")
        self.text_encoder = T5EncoderModel(
            text_len=cfg.text_len,
            dtype=cfg.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, cfg.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, cfg.t5_tokenizer),
            shard_fn=None,
        )

        # Load VAE
        logger.info("Loading VAE...")
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, cfg.vae_checkpoint),
            device=self.device
        )

        # Load NF4 quantized diffusion models
        logger.info("Loading BNB NF4 quantized diffusion models...")
        self.low_noise_model = load_bnb_nf4_model(checkpoint_dir, cfg.low_noise_checkpoint)
        self.low_noise_model.eval().requires_grad_(False)

        self.high_noise_model = load_bnb_nf4_model(checkpoint_dir, cfg.high_noise_checkpoint)
        self.high_noise_model.eval().requires_grad_(False)

        logger.info("Model loading complete!")

    def _prepare_model_for_timestep(self, t, boundary):
        """Prepare and return the required model for the current timestep."""
        if t.item() >= boundary:
            required_model_name = 'high_noise_model'
            offload_model_name = 'low_noise_model'
        else:
            required_model_name = 'low_noise_model'
            offload_model_name = 'high_noise_model'

        required_model = getattr(self, required_model_name)
        offload_model = getattr(self, offload_model_name)

        # Offload unused model to CPU
        try:
            if next(offload_model.parameters()).device.type == 'cuda':
                offload_model.to('cpu')
                torch.cuda.empty_cache()
        except StopIteration:
            pass

        # Load required model to GPU
        try:
            if next(required_model.parameters()).device.type == 'cpu':
                required_model.to(self.device)
        except StopIteration:
            pass

        return required_model

    def generate(
        self,
        input_prompt: str,
        img: Image.Image,
        action_path: str = None,
        max_area: int = 720 * 1280,
        frame_num: int = 81,
        shift: float = 5.0,
        sampling_steps: int = 40,
        guide_scale: float = 5.0,
        n_prompt: str = "",
        seed: int = -1,
    ):
        """Generate video from image and text prompt."""
        if action_path is not None:
            c2ws = np.load(os.path.join(action_path, "poses.npy"))
            len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
            frame_num = min(frame_num, len_c2ws)
            c2ws = c2ws[:frame_num]

        guide_scale = (guide_scale, guide_scale) if isinstance(guide_scale, float) else guide_scale
        img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img_tensor.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        lat_f = (F - 1) // self.vae_stride[0] + 1
        max_seq_len = lat_f * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16, (F - 1) // self.vae_stride[0] + 1, lat_h, lat_w,
            dtype=torch.float32, generator=seed_g, device=self.device
        )

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # Encode text
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        # Camera preparation
        dit_cond_dict = None
        if action_path is not None:
            Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()
            Ks = get_Ks_transformed(Ks, 480, 832, h, w, h, w)
            Ks = Ks[0]

            len_c2ws = len(c2ws)
            c2ws_infer = interpolate_camera_poses(
                src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
                src_rot_mat=c2ws[:, :3, :3],
                src_trans_vec=c2ws[:, :3, 3],
                tgt_indices=np.linspace(0, len_c2ws - 1, int((len_c2ws - 1) // 4) + 1),
            )
            c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
            Ks = Ks.repeat(len(c2ws_infer), 1)

            c2ws_infer = c2ws_infer.to(self.device)
            Ks = Ks.to(self.device)
            c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w)
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                c1=int(h // lat_h), c2=int(w // lat_w),
            )
            c2ws_plucker_emb = c2ws_plucker_emb[None, ...]
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb, 'b (f h w) c -> b c f h w',
                f=lat_f, h=lat_h, w=lat_w
            ).to(self.param_dtype)
            dit_cond_dict = {"c2ws_plucker_emb": c2ws_plucker_emb.chunk(1, dim=0)}

        # Encode image
        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img_tensor[None].cpu(), size=(h, w), mode='bicubic'
                ).transpose(0, 1),
                torch.zeros(3, F - 1, h, w)
            ], dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        # Diffusion sampling
        with torch.amp.autocast('cuda', dtype=self.param_dtype), torch.no_grad():
            boundary = self.boundary * self.num_train_timesteps

            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False
            )
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps

            latent = noise

            arg_c = {
                'context': [context[0]],
                'seq_len': max_seq_len,
                'y': [y],
                'dit_cond_dict': dit_cond_dict,
            }

            arg_null = {
                'context': context_null,
                'seq_len': max_seq_len,
                'y': [y],
                'dit_cond_dict': dit_cond_dict,
            }

            torch.cuda.empty_cache()

            # Pre-load first model
            first_model_name = 'high_noise_model' if timesteps[0].item() >= boundary else 'low_noise_model'
            getattr(self, first_model_name).to(self.device)
            logger.info(f"Loaded {first_model_name} to GPU")

            for _, t in enumerate(tqdm(timesteps, desc="Sampling")):
                latent_model_input = [latent.to(self.device)]
                timestep = torch.stack([t]).to(self.device)

                model = self._prepare_model_for_timestep(t, boundary)
                sample_guide_scale = guide_scale[1] if t.item() >= boundary else guide_scale[0]

                noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0]
                torch.cuda.empty_cache()
                noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0]
                torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + sample_guide_scale * (noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0), t, latent.unsqueeze(0),
                    return_dict=False, generator=seed_g
                )[0]
                latent = temp_x0.squeeze(0)

            # Offload models
            self.low_noise_model.cpu()
            self.high_noise_model.cpu()
            torch.cuda.empty_cache()

            # Decode video
            videos = self.vae.decode([latent])

        del noise, latent
        gc.collect()
        torch.cuda.synchronize()

        return videos[0]


def save_video(frames: torch.Tensor, output_path: str, fps: int = 16):
    """Save video frames to file."""
    import imageio

    frames = ((frames + 1) / 2 * 255).clamp(0, 255).byte()
    frames = frames.permute(1, 2, 3, 0).cpu().numpy()

    imageio.mimwrite(output_path, frames, fps=fps, codec='libx264')
    logger.info(f"Saved video to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate videos with BNB NF4 quantized models")
    parser.add_argument("--ckpt_dir", type=str, default="lingbot-world-base-cam")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--action_path", type=str, default=None, help="Camera control path")
    parser.add_argument("--size", type=str, default="480*832", help="Output resolution")
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--sampling_steps", type=int, default=40)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument("--t5_cpu", action="store_true", default=True)
    args = parser.parse_args()

    h, w = map(int, args.size.split('*'))
    max_area = h * w

    img = Image.open(args.image).convert('RGB')

    pipeline = WanI2V_BNB(
        checkpoint_dir=args.ckpt_dir,
        t5_cpu=args.t5_cpu,
    )

    logger.info("Generating video...")
    video = pipeline.generate(
        input_prompt=args.prompt,
        img=img,
        action_path=args.action_path,
        max_area=max_area,
        frame_num=args.frame_num,
        sampling_steps=args.sampling_steps,
        guide_scale=args.guide_scale,
        seed=args.seed,
    )

    save_video(video, args.output)


if __name__ == "__main__":
    main()
`
            }
        },
        // 3. Install PyTorch
        {
            method: "script.start",
            params: {
                uri: "torch.js",
                params: {
                    venv: "env",
                    path: "app"
                }
            }
        },
        // 4. Patch attention.py to use PyTorch SDPA instead of flash_attn
        {
            method: "shell.run",
            params: {
                venv: "env",
                path: "app",
                message: "python patch_attention.py"
            }
        },
        // 5. Install all dependencies (no flash-attn needed)
        {
            method: "shell.run",
            params: {
                venv: "env",
                path: "app",
                message: [
                    "uv pip install einops easydict ftfy scipy tqdm opencv-python imageio[ffmpeg]",
                    "uv pip install diffusers transformers accelerate bitsandbytes>=0.49.0",
                    "uv pip install huggingface_hub[cli,hf_xet]"
                ]
            }
        },
        // 6. Create .pth file so Python finds the wan module
        {
            method: "shell.run",
            params: {
                venv: "env",
                path: "app",
                message: "python -c \"import site, os; sp = site.getsitepackages()[0]; open(os.path.join(sp, 'lingbot.pth'), 'w').write(os.getcwd())\""
            }
        },
        // Done
        {
            method: "notify",
            params: {
                html: "Installation complete!<br><br>Click <b>Download Models</b> (~28GB total)"
            }
        }
    ]
}
