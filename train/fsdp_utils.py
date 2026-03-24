# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright 2025 AntGroup and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import json
import os

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors.torch import load_file, save_file

from modeling.lladao.modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from modeling.lladao.llada_navit import (
    LLaDAMoEDecoderLayer,
    LLaDADecoderLayer,
    LLaDAMoTDecoderLayer
)
from modeling.lladao.siglip_navit import SiglipEncoderLayer, SiglipVisionTransformer
from transformers import DINOv3ViTModel


class FSDPConfig:
    def __init__(
        self,
        sharding_strategy, 
        backward_prefetch, 
        cpu_offload, 
        num_replicate,
        num_shard=8,
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard


def fsdp_wrapper(original_model, fsdp_config, ignored_modules=[]):
    if fsdp_config.sharding_strategy == 'HYBRID_SHARD':
        device_mesh = init_device_mesh(
            "cuda", 
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard")
        )
    else:
        device_mesh = None
    return FSDP(
        original_model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                LLaDADecoderLayer,
                LLaDAMoEDecoderLayer,
                LLaDAMoTDecoderLayer,
                SiglipEncoderLayer,
                SiglipVisionTransformer,
                DINOv3ViTModel,
                MLPconnector,
                TimestepEmbedder,
                PositionEmbedding,
            },
        ),
        ignored_modules=ignored_modules,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
    )


class FSDPCheckpoint:
    FIXED_POS_EMBED_KEYS = (
        "latent_pos_embed.pos_embed",
        "vit_pos_embed.pos_embed",
    )

    @staticmethod
    def _find_safetensors_artifact(checkpoint_dir, stem):
        single_file_path = os.path.join(checkpoint_dir, f"{stem}.safetensors")
        index_file_path = os.path.join(checkpoint_dir, f"{stem}.safetensors.index.json")

        if os.path.exists(single_file_path):
            return single_file_path
        if os.path.exists(index_file_path):
            return index_file_path
        return None

    @staticmethod
    def _iter_shard_paths(index_file_path):
        with open(index_file_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or len(weight_map) == 0:
            raise ValueError(f"Invalid sharded safetensors index file: {index_file_path}")

        shard_names = list(dict.fromkeys(weight_map.values()))
        index_dir = os.path.dirname(index_file_path)
        for shard_name in shard_names:
            shard_path = os.path.join(index_dir, shard_name)
            if not os.path.exists(shard_path):
                raise FileNotFoundError(
                    f"Shard referenced by index file does not exist: {shard_path}"
                )
            yield shard_name, shard_path

    @staticmethod
    def _remove_fixed_pos_embeds(state_dict, logger, state_name):
        removed_keys = []
        for key in FSDPCheckpoint.FIXED_POS_EMBED_KEYS:
            if key in state_dict:
                state_dict.pop(key)
                removed_keys.append(key)
        if removed_keys:
            logger.info(f"Removed fixed position embeddings from {state_name}: {removed_keys}")

    @staticmethod
    def _load_model_from_safetensors_artifact(target_model, artifact_path, logger, state_name):
        model_keys = set(target_model.state_dict().keys())
        ignored_missing_keys = {
            key for key in FSDPCheckpoint.FIXED_POS_EMBED_KEYS if key in model_keys
        }
        loaded_keys = set()
        unexpected_keys = set()

        if artifact_path.endswith(".index.json"):
            shard_paths = list(FSDPCheckpoint._iter_shard_paths(artifact_path))
            logger.info(
                f"Loading sharded {state_name} from {artifact_path} ({len(shard_paths)} shards)."
            )
        else:
            shard_paths = [(os.path.basename(artifact_path), artifact_path)]
            logger.info(f"Loading {state_name} from {artifact_path}.")

        for shard_name, shard_path in shard_paths:
            shard_state_dict = load_file(shard_path, device="cpu")
            FSDPCheckpoint._remove_fixed_pos_embeds(shard_state_dict, logger, state_name)

            shard_keys = set(shard_state_dict.keys())
            loaded_keys.update(shard_keys & model_keys)
            unexpected_keys.update(shard_keys - model_keys)

            incompatible_keys = target_model.load_state_dict(shard_state_dict, strict=False)
            unexpected_keys.update(incompatible_keys.unexpected_keys)
            logger.info(
                f"Loaded {len(shard_state_dict)} tensors from {shard_name} into {state_name}."
            )
            del shard_state_dict

        missing_keys = sorted(model_keys - loaded_keys - ignored_missing_keys)
        if missing_keys or unexpected_keys:
            logger.info(
                f"{state_name} load summary: loaded={len(loaded_keys)}, "
                f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
            )
            if missing_keys:
                logger.info(f"{state_name} missing keys (first 20): {missing_keys[:20]}")
            if unexpected_keys:
                logger.info(
                    f"{state_name} unexpected keys (first 20): {sorted(unexpected_keys)[:20]}"
                )
        else:
            logger.info(f"{state_name} load summary: loaded all {len(loaded_keys)} tensors.")

    @staticmethod
    def fsdp_save_ckpt(
        ckpt_dir, 
        train_steps, 
        model, 
        ema_model, 
        optimizer, 
        scheduler, 
        data_status,
        logger, 
        fsdp_config,
    ):
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving checkpoint to {save_path}.")

        if ema_model is not None:
            with FSDP.state_dict_type(
                ema_model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                ema_state_dict = ema_model.state_dict()
                if dist.get_rank() == 0:
                    save_file(ema_state_dict, os.path.join(save_path, "ema.safetensors"))

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            model_state_dict = model.state_dict()
            if dist.get_rank() == 0:
                save_file(model_state_dict, os.path.join(save_path, "model.safetensors"))

        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_save_path = os.path.join(
                save_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                torch.save(optimizer.state_dict(), optimizer_save_path)
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                if dist.get_rank() < fsdp_config.num_shard:
                    torch.save(optimizer.state_dict(), optimizer_save_path)
            else:
                raise NotImplementedError

        if dist.get_rank() == 0 and scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

        if dist.get_rank() == 0 and data_status is not None:
            torch.save(data_status, os.path.join(save_path, "data_status.pt"))

        dist.barrier()
        return

    @staticmethod
    def try_load_ckpt(resume_from, logger, model, ema_model=None, resume_from_ema=False):
        if resume_from is not None and os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}.")
            model_stem = "ema" if resume_from_ema else "model"
            model_artifact_path = FSDPCheckpoint._find_safetensors_artifact(resume_from, model_stem)
            if model_artifact_path is None:
                raise FileNotFoundError(
                    f"Could not find {model_stem}.safetensors or "
                    f"{model_stem}.safetensors.index.json under {resume_from}"
                )
            FSDPCheckpoint._load_model_from_safetensors_artifact(
                model, model_artifact_path, logger, state_name="model"
            )

            if ema_model is not None:
                ema_artifact_path = FSDPCheckpoint._find_safetensors_artifact(resume_from, "ema")
                if ema_artifact_path is None:
                    logger.info(f"Replicating ema model from {model_artifact_path}.")
                    ema_artifact_path = model_artifact_path
                FSDPCheckpoint._load_model_from_safetensors_artifact(
                    ema_model, ema_artifact_path, logger, state_name="ema_model"
                )
        else:
            logger.info(f"Training from scratch.")
        return model, ema_model

    @staticmethod
    def try_load_train_state(resume_from, optimizer, scheduler, fsdp_config):
        if resume_from is not None and os.path.exists(resume_from):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_state_dict_path = os.path.join(
                resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            optimizer_state_dict = torch.load(optimizer_state_dict_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(optimizer_state_dict)
            del optimizer_state_dict

            scheduler_state_dict_path = os.path.join(resume_from, "scheduler.pt")
            scheduler_state_dict = torch.load(scheduler_state_dict_path, weights_only=True, map_location="cpu")
            scheduler.load_state_dict(scheduler_state_dict)
            del scheduler_state_dict

            train_steps = int(os.path.basename(os.path.normpath(resume_from))) + 1
            """
            data_status = [
                {
                    dataset_name: {
                        worker_id: [parquet_idx, row_group_id, row_idx],
                    },
                },
            ]
            """
            data_status_path = os.path.join(resume_from, "data_status.pt")
            if os.path.exists(data_status_path):
                data_status = torch.load(data_status_path, weights_only=True, map_location="cpu")
                local_rank = dist.get_rank()
                if local_rank < len(data_status):
                    data_status = data_status[local_rank]
                else:
                    data_status = None
            else:
                data_status = None
        else:
            train_steps = 0
            data_status = None
        return optimizer, scheduler, train_steps, data_status


def grad_checkpoint_check_fn(module):
    module_options = (
        LLaDADecoderLayer, 
        SiglipEncoderLayer, 
        MLPconnector, 
        LLaDAMoEDecoderLayer, 
        LLaDAMoTDecoderLayer
    )
    return isinstance(module, module_options)


def fsdp_ema_setup(ema_model, fsdp_config, ignored_modules=[]):
    for param in ema_model.parameters():
        param.requires_grad = False

    ema_model = fsdp_wrapper(ema_model, fsdp_config, ignored_modules=ignored_modules)
    return ema_model


@torch.no_grad()
def fsdp_ema_update(ema_model, model, decay=0.9999):
    ema_handles = traversal_utils._get_fsdp_handles(ema_model)
    new_handles = traversal_utils._get_fsdp_handles(model)
    assert len(ema_handles) == len(new_handles)
    ema_params = []
    new_params = []

    for ema_handle, new_handle in zip(ema_handles, new_handles):
        if ema_handle.flat_param is not None and new_handle.flat_param.requires_grad:
            ema_params.append(ema_handle.flat_param.data)
            new_params.append(new_handle.flat_param.data.to(dtype=ema_handle.flat_param.dtype))

    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, new_params, alpha=1 - decay)
