import copy
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoTokenizer

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.lladao import (
    LLaDAO,
    LLaDAOConfig,
    LLaDAConfig,
    LLaDAModelLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)


DEFAULT_TEXT_TO_IMAGE_ARGS = {
    "cfg_text_scale": 4.0,
    "cfg_img_scale": 1.0,
    "cfg_interval": (0.4, 1.0),
    "timestep_shift": 3.0,
    "num_timesteps": 50,
    "cfg_renorm_min": 0.0,
    "cfg_renorm_type": "global",
    "image_shapes": (1024, 1024),
}

DEFAULT_IMAGE_EDIT_ARGS = {
    "cfg_text_scale": 4.0,
    "cfg_img_scale": 2.0,
    "cfg_interval": (0.0, 1.0),
    "timestep_shift": 3.0,
    "num_timesteps": 50,
    "cfg_renorm_min": 0.0,
    "cfg_renorm_type": "text_channel",
}

DEFAULT_UNDERSTANDING_ARGS = {
    "mask_id": 126336,
    "block_length": 32,
    "steps_per_block": 32,
    "max_blocks": 32,
    "temperature": 0.0,
    "cfg_scale": 0.0,
    "confidence_threshold": 0.95,
}


ImageLike = Union[str, os.PathLike[str], Image.Image]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_image(image: ImageLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return pil_img2rgb(image)
    return pil_img2rgb(Image.open(image))


def clean_response_text(text: str) -> str:
    clean_text = text
    if "</think>" in clean_text:
        clean_text = clean_text.split("</think>")[-1]
    return clean_text.replace("<|endoftext|>", "").strip()


def _build_device_map(model: LLaDAO, max_mem_per_gpu: str) -> Dict[str, Union[int, str]]:
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("CUDA device not found. LLaDA-o inference currently requires at least one GPU.")

    device_map = infer_auto_device_map(
        model,
        max_memory={index: max_mem_per_gpu for index in range(gpu_count)},
        no_split_module_classes=["LLaDAO", "LLaDAMoTDecoderLayer"],
    )

    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]

    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for module_name in same_device_modules:
        device_map[module_name] = first_device

    return device_map


class LLaDAMultimodalDemo:
    def __init__(
        self,
        model: LLaDAO,
        vae_model: Any,
        tokenizer: AutoTokenizer,
        inferencer: InterleaveInferencer,
        new_token_ids: Dict[str, int],
        understanding_transform: ImageTransform,
    ) -> None:
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.inferencer = inferencer
        self.new_token_ids = new_token_ids
        self.understanding_transform = understanding_transform

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, os.PathLike[str]],
        max_mem_per_gpu: str = "40GiB",
        offload_dir: Union[str, os.PathLike[str]] = "/tmp/offload",
    ) -> "LLaDAMultimodalDemo":
        model_path = Path(model_path).expanduser()
        checkpoint_path = model_path / "ema.safetensors.index.json"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Sharded checkpoint index not found: {checkpoint_path}")

        llm_config = LLaDAConfig.from_json_file(str(model_path / "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "LLaDAMoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(str(model_path / "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        vae_model, vae_config = load_ae(local_path=str(model_path / "ae.safetensors"))

        config = LLaDAOConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=64,
        )

        with init_empty_weights():
            language_model = LLaDAModelLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = LLaDAO(language_model, vit_model, None, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)
        understanding_transform = ImageTransform(980, 378, 14, max_pixels=2_007_040)

        device_map = _build_device_map(model, max_mem_per_gpu=max_mem_per_gpu)
        os.makedirs(offload_dir, exist_ok=True)

        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=str(checkpoint_path),
            device_map=device_map,
            offload_buffers=True,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder=str(offload_dir),
        )
        model.eval()

        inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )

        return cls(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            inferencer=inferencer,
            new_token_ids=new_token_ids,
            understanding_transform=understanding_transform,
        )

    def understand(self, image: ImageLike, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        image = load_image(image)
        run_kwargs = copy.deepcopy(DEFAULT_UNDERSTANDING_ARGS)
        run_kwargs.update(kwargs)

        mask_id = run_kwargs.pop("mask_id", None)
        if mask_id is None:
            mask_id = self.tokenizer.mask_token_id
        if mask_id is None:
            mask_id = 126336

        start_time = time.time()
        raw_text, valid_tokens, total_tokens = self.model.chat_block(
            tokenizer=self.tokenizer,
            new_token_ids=copy.deepcopy(self.new_token_ids),
            image_transform=self.understanding_transform,
            images=[image],
            prompt=prompt,
            mask_id=mask_id,
            **run_kwargs,
        )
        elapsed_seconds = time.time() - start_time

        return {
            "text": clean_response_text(raw_text),
            "raw_text": raw_text,
            "valid_tokens": valid_tokens,
            "total_tokens": total_tokens,
            "elapsed_seconds": elapsed_seconds,
        }

    def text_to_image(
        self,
        prompt: str,
        seed: Optional[int] = None,
        image_shapes: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if seed is not None:
            set_seed(seed)

        run_kwargs = copy.deepcopy(DEFAULT_TEXT_TO_IMAGE_ARGS)
        if image_shapes is not None:
            run_kwargs["image_shapes"] = image_shapes
        run_kwargs.update(kwargs)
        return self.inferencer(text=prompt, **run_kwargs)

    def edit_image(
        self,
        image: ImageLike,
        prompt: str,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if seed is not None:
            set_seed(seed)

        run_kwargs = copy.deepcopy(DEFAULT_IMAGE_EDIT_ARGS)
        run_kwargs.update(kwargs)
        return self.inferencer(image=load_image(image), text=prompt, **run_kwargs)


__all__ = [
    "DEFAULT_IMAGE_EDIT_ARGS",
    "DEFAULT_TEXT_TO_IMAGE_ARGS",
    "DEFAULT_UNDERSTANDING_ARGS",
    "LLaDAMultimodalDemo",
    "clean_response_text",
    "load_image",
    "set_seed",
]
