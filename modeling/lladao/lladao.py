# Copyright 2025 AntGroup and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from data.data_utils import (
    create_sparse_mask, 
    get_flattened_position_ids_extrapolate, 
    get_flattened_position_ids_interpolate,
    patchify, 
    prepare_attention_mask_per_sample
)
from .llada_navit import NaiveCache
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding

from tqdm import tqdm

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise
def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

def build_mlp(hidden_size, projector_dim, z_dim):
    mlp = nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )
    # Initialize each linear layer
    for layer in mlp:
        if isinstance(layer, nn.Linear):
            # Initialize weights and biases
            #nn.init.constant_(layer.weight, 0)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
    
    return mlp

class LLaDAOConfig(PretrainedConfig):
    def __init__(
        self,
        visual_gen=True,
        visual_und=True,
        visual_gen_repa=False,
        visual_gen_reg=False,
        repa_output_depth=0,
        llm_config=None,
        vit_config=None,
        vae_config=None,
        latent_patch_size=2,
        max_latent_size=32,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.visual_gen_repa = visual_gen_repa
        self.visual_gen_reg = visual_gen_reg
        self.repa_output_depth = repa_output_depth
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.vae_config = vae_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift


class LLaDAO(PreTrainedModel):
    config_class = LLaDAOConfig
    base_model_prefix = 'lladao'

    def __init__(self, language_model, vit_model, repa_model, config: LLaDAOConfig):
        super().__init__(config)
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads

        if config.visual_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.latent_downsample = config.vae_config.downsample * config.latent_patch_size
            self.max_latent_size = config.max_latent_size
            self.latent_channel = config.vae_config.z_channels
            self.patch_latent_dim = self.latent_patch_size ** 2 * self.latent_channel
            self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
            self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)

        if config.visual_und:
            self.vit_model = vit_model
            self.vit_patch_size = config.vit_config.patch_size
            self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
            self.vit_hidden_size = config.vit_config.hidden_size
            self.connector = MLPconnector(self.vit_hidden_size, self.hidden_size, config.connector_act)
            self.vit_pos_embed = PositionEmbedding(self.vit_max_num_patch_per_side, self.hidden_size)
        
        if config.visual_gen_repa:
            self.repa_model = repa_model
            self.repa_hidden_size = repa_model.config.hidden_size
            self.repa_mlp = build_mlp(self.hidden_size, self.hidden_size * 2, self.repa_hidden_size) # Default set to 2x mid hidden size.
        
        if config.visual_gen_reg:
            self.reg_input_mlp = nn.Linear(self.repa_model.config.hidden_size, self.hidden_size)
            self.reg_output_mlp = nn.Linear(self.hidden_size, self.repa_model.config.hidden_size)
            # Use zero initialization for weights and biases
            #nn.init.constant_(self.reg_input_mlp.weight, 0)
            nn.init.xavier_uniform_(self.reg_input_mlp.weight)
            nn.init.constant_(self.reg_input_mlp.bias, 0)
            nn.init.constant_(self.reg_output_mlp.weight, 0)
            nn.init.constant_(self.reg_output_mlp.bias, 0)

        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config
        self._init_weights()

    def _init_weights(self):
        if self.config.visual_gen:
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)

    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        # for visual understanding
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        # for visual generation
        padded_images: Optional[torch.Tensor] = None,
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
        mse_loss_indexes: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            sequence_length: length of sequence.
            packed_text_ids: 1-D int tensor, packed text token ids.
            packed_text_indexes: 1-D int tensor, packed text token indexes in sequence.
            sample_lens: A list of N ints, length of each sample in packed_sequence.
            nested_attention_masks: A list of N 2-D float tensor,  where 0.0 means attention and 
                -inf means ignore.
            packed_position_ids: packed 1-D positions, an image has only one global position shared
                by all latent tokens.

            packed_vit_tokens: packed patchified image tokens for vit model.
            packed_vit_position_ids: 1-D int tensor, the position of each token for vit model.
            packed_vit_token_indexes: 1-D int tensor, packed vit token indexes in sequence.
            vit_token_seqlens: 1-D int tensor, the length of each image tokens for vit model.
            packed_label_ids: 1-D int tensor, packed label token ids.
            ce_loss_indexes: 1-D bool tensor, where to compute ce loss.

            padded_latent: padded latent from VAE encoder.
            patchified_vae_latent_shapes: A list of (h, w) tuples, patchfied latent shapes of each image.
            packed_latent_position_ids: 1-D int tensor, the position of each token for latent.
            packed_vae_token_indexes: 1-D int tensor, padded image token indexes in sequence.
            packed_timesteps: 1-D float tensor, flow timesteps. 0 indicates use clean image.
            mse_loss_indexes: 1-D bool tensor, where to compute mse loss.
        """
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        if nested_attention_masks is None:
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, 
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
        else:
            attention_mask = nested_attention_masks

        if self.config.visual_und:
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
            cu_seqlens = cu_seqlens.to(torch.int32)
            max_seqlen = torch.max(vit_token_seqlens).item()
            packed_vit_token_embed = self.vit_model(
                packed_pixel_values=packed_vit_tokens, 
                packed_flattened_position_ids=packed_vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            packed_vit_token_embed = self.connector(packed_vit_token_embed)
            vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
            packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed
        
        if self.config.visual_gen_repa:
            # using repa for visual generation
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                with torch.no_grad(): # no grad for repa model
                    repa_outputs = self.repa_model(pixel_values=padded_images)
                    patch_features_flat = repa_outputs.last_hidden_state[:, 1 + self.repa_model.config.num_register_tokens:, :]
                    cls_features = repa_outputs.last_hidden_state[:, 0, :] # batch, hidden_size
                    patch_features = patch_features_flat.unflatten(1, (padded_images.shape[2] // self.repa_model.config.patch_size, padded_images.shape[3] // self.repa_model.config.patch_size))
                    repa_features = []
                    if self.config.visual_gen_reg:
                        for feature, (h, w), cls_feature in zip(patch_features, patchified_vae_latent_shapes, cls_features):
                            repa_features.append(
                                torch.cat([cls_feature.unsqueeze(0), feature[:h, :w].reshape(-1, feature.shape[-1])], dim=0) 
                            ) # If reg, need to add cls_feature 
                    else:
                        for feature, (h, w) in zip(patch_features, patchified_vae_latent_shapes):
                            repa_features.append(feature[:h, :w].reshape(-1, feature.shape[-1]))

        if self.config.visual_gen:
            p = self.latent_patch_size
            packed_latent = []
            for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                packed_latent.append(latent)
            packed_latent_clean = torch.cat(packed_latent, dim=0)

            if self.config.visual_gen_reg:
                cls_timesteps = []
                current_timestep_idx = 0
                for (h, w) in patchified_vae_latent_shapes:
                    cls_timesteps.append(packed_timesteps[current_timestep_idx])
                    current_timestep_idx += h * w
                cls_timesteps = torch.stack(cls_timesteps, dim=0).to(packed_timesteps.device) # batch

                noise_cls = torch.randn_like(cls_features) # batch, hidden_size
                cls_timesteps = torch.sigmoid(cls_timesteps) # batch
                cls_timesteps = self.timestep_shift * cls_timesteps / (1 + (self.timestep_shift - 1) * cls_timesteps) # batch 
                packed_cls = (1 - cls_timesteps[:, None]) * cls_features + cls_timesteps[:, None] * noise_cls # batch, hidden_size
                packed_cls_latent = self.reg_input_mlp(packed_cls) # batch, hidden_size of llm
                packed_cls_timestep_embeds = self.time_embedder(cls_timesteps) # batch, hidden_size of llm
                packed_cls_latent = packed_cls_latent + packed_cls_timestep_embeds # batch, hidden_size of llm

            noise = torch.randn_like(packed_latent_clean)
            packed_timesteps = torch.sigmoid(packed_timesteps)
            packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
            packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise
            packed_timestep_embeds = self.time_embedder(packed_timesteps)
            latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
            packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + latent_token_pos_emb
            if self.config.visual_gen_reg:
                all_tokens = []
                cls_idx = 0
                patch_start = 0
                for h, w in patchified_vae_latent_shapes:
                    # Add cls token
                    all_tokens.append(packed_cls_latent[cls_idx])  # [hidden_size]
                    # Add patch tokens
                    patch_end = patch_start + h * w
                    all_tokens.extend(packed_latent[patch_start:patch_end])  # [h*w, hidden_size]
                    cls_idx += 1
                    patch_start = patch_end
                # Convert to tensor and assign
                packed_sequence[packed_vae_token_indexes] = torch.stack(all_tokens, dim=0)
            else:
                packed_sequence[packed_vae_token_indexes] = packed_latent
                    

        extra_inputs = {}
        if self.use_moe or self.config.visual_gen_repa:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes=torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_vae_token_indexes,
            )
        
        if self.config.visual_gen_repa: 
            extra_inputs.update(
                output_depth = self.config.repa_output_depth,
            )

        last_hidden_state, repa_intermediate_output = self.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )
        repa_proj_loss = None
        if repa_intermediate_output is not None and self.config.visual_gen_repa:
            repa_intermediate_output = self.repa_mlp(repa_intermediate_output)
            # 2. Split according to patchified_vae_latent_shapes
            repa_intermediate_features = []
            start_idx = 0
            
            for h, w in patchified_vae_latent_shapes:
                if self.config.visual_gen_reg:
                    num_tokens = h * w + 1
                else:
                    num_tokens = h * w
                end_idx = start_idx + num_tokens
                img_features = repa_intermediate_output[start_idx:end_idx]
                repa_intermediate_features.append(img_features)
                start_idx = end_idx
            
            # 3. Compute contrastive loss (maintaining original averaging logic)
            repa_proj_loss = 0.
            total_images = len(repa_intermediate_features)  # Number of images (equivalent to bsz)
            
            for z_j, z_tilde_j in zip(repa_intermediate_features, repa_features):
                # Normalization
                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                
                # Compute negative cosine similarity for each token [num_tokens]
                token_similarities = -(z_j * z_tilde_j).sum(dim=-1)
                
                # First average all tokens for the current image to a scalar
                image_loss = token_similarities.mean()
                
                # Accumulate loss for each image
                repa_proj_loss += image_loss
            
            # Then average over all images
            if total_images > 0:
                repa_proj_loss /= total_images
            

        mse = None
        reg_loss = None
        if self.config.visual_gen:
            if self.config.visual_gen_reg:
                cls_mse_indexes = []
                patch_mse_indexes = []
                current_idx_in_mse_list = 0
                img_idx = 0  # Added: track current image index
                for (h, w) in patchified_vae_latent_shapes:
                    num_patches = h * w

                    # Added: check if current image is valid
                    if img_idx < len(cls_timesteps) and cls_timesteps[img_idx] > 0:
                        # Assume mse_loss_indexes has enough elements
                        if current_idx_in_mse_list < len(mse_loss_indexes):
                            # 1. CLS token index is the first of current segment
                            cls_mse_indexes.append(mse_loss_indexes[current_idx_in_mse_list])
                            
                            # 2. Patch token indices are the next h*w elements
                            start = current_idx_in_mse_list + 1
                            end = start + num_patches
                            patch_mse_indexes.extend(mse_loss_indexes[start:end])
                        # Update index, prepare for next image
                        current_idx_in_mse_list += 1 + num_patches
                    
                    img_idx += 1  # Added: update image index
                
                cls_mse_indexes = torch.tensor([idx.item() for idx in cls_mse_indexes], device=last_hidden_state.device)
                patch_mse_indexes = torch.tensor([idx.item() for idx in patch_mse_indexes], device=last_hidden_state.device)
                
                # Process CLS and patch predictions separately
                cls_mse_preds = self.reg_output_mlp(last_hidden_state[cls_mse_indexes]) # batch, hidden_size of repa
                patch_mse_preds = self.llm2vae(last_hidden_state[patch_mse_indexes]) # batch, hidden_size of vae
                
                # CLS token target
                cls_target = noise_cls - cls_features  # v_t = noise - clean, batch hidden_size of repa
                cls_has_mse = cls_timesteps > 0
                cls_mse = (cls_mse_preds - cls_target[cls_has_mse]) ** 2
                
                # Patch tokens target
                patch_target = noise - packed_latent_clean
                patch_has_mse = packed_timesteps > 0
                patch_mse = (patch_mse_preds - patch_target[patch_has_mse]) ** 2
                
                # Merge losses
                mse = patch_mse 
                reg_loss = cls_mse.mean(dim=-1).mean()
            else:
                packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
                target = noise - packed_latent_clean # NOTE: v_t=dx_t/dt=x_1-x_0, pointing from data to noise
                has_mse = packed_timesteps > 0
                mse = (packed_mse_preds - target[has_mse]) ** 2

        ce = None
        if ce_loss_indexes is not None:
            packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")

        return dict(mse=mse, ce=ce, repa=repa_proj_loss, reg=reg_loss)


    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False, # NOTE: text is not causal for dllm
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vit_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2), 
                self.vit_patch_size, 
                max_num_patches_per_side=self.vit_max_num_patch_per_side
            )
            vit_tokens = patchify(image_tensor, self.vit_patch_size)
            packed_vit_tokens.append(vit_tokens)
            num_img_tokens = vit_tokens.shape[0]
            packed_vit_position_ids.append(vit_position_ids)
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()
        packed_vit_token_embed = self.vit_model(
            packed_pixel_values=packed_vit_tokens, 
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_token_embed = self.connector(packed_vit_token_embed)
        pos_emb = self.vit_pos_embed(packed_vit_position_ids)
        packed_vit_token_embed = packed_vit_token_embed + pos_emb
        if packed_vit_token_embed.dtype != packed_sequence.dtype:
            packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
        packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, timestep=0):
        patchified_vae_latent_shapes, packed_vae_position_ids = list(), list()
        packed_vae_token_indexes = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        vae_image_tensors = list()
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vae_image_tensors.append(image_tensor)
            vae_posiiton_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2),
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)
            H, W = image_tensor.shape[1:]
            h = H // self.latent_downsample
            w = W // self.latent_downsample
            patchified_vae_latent_shapes.append((h, w))

            num_img_tokens = w * h
            packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]
        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size))
        for i, image_tensor in enumerate(vae_image_tensors):
            padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

        generation_input = {
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vae(
        self,
        vae_model,
        past_key_values: NaiveCache,
        padded_images: torch.Tensor,
        patchified_vae_latent_shapes: List,
        packed_vae_position_ids: torch.LongTensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.Tensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        padded_latent = vae_model.encode(padded_images)

        p = self.latent_patch_size
        packed_latent = list()
        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
            packed_latent.append(latent)
        packed_latent = torch.cat(packed_latent, dim=0)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        if packed_latent.dtype != packed_sequence.dtype:
            packed_latent = packed_latent.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids):
        packed_text_ids, packed_text_indexes = list(), list()
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = list(), list(), list()
        packed_init_cls_noises = list()
        packed_position_ids, packed_seqlens, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            vae_posiiton_ids = self.get_flattened_position_ids(
                H, W,
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)

            h, w = H // self.latent_downsample, W // self.latent_downsample
            if self.config.visual_gen_reg:
                packed_init_cls_noises.append(
                    torch.randn(1, self.repa_model.config.hidden_size)
                )
                packed_init_noises.append(
                    torch.randn(h * w, self.latent_channel * self.latent_patch_size ** 2)
                )
                # both cls token and image tokens
                num_image_tokens = h * w + 1
            else: 
                num_image_tokens = h * w
                packed_init_noises.append(
                    torch.randn(num_image_tokens, self.latent_channel * self.latent_patch_size ** 2)
                )

            packed_vae_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))
            packed_seqlens.append(num_image_tokens + 2)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_init_cls_noises": torch.cat(packed_init_cls_noises, dim=0) if self.config.visual_gen_reg else None,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes):
        packed_position_ids, packed_indexes, packed_key_value_indexes = list(), list(), list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            h, w = H // self.latent_downsample, W // self.latent_downsample
            if self.config.visual_gen_reg:
                num_image_tokens = h * w + 1
            else:
                num_image_tokens = h * w
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))

        generation_input = {
            "cfg_packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "cfg_key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "cfg_packed_query_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "cfg_packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_image(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        packed_init_cls_noises: Optional[torch.Tensor] = None,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
    ):
        x_t = packed_init_noises
        cls_t = packed_init_cls_noises

        timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts =  timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):

            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            v_t, cls_v_t = self._forward_flow(
                x_t=x_t,
                cls_t=cls_t,
                timestep=timestep, 
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                # cfg_text
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                # cfg_img
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
            )

            x_t = x_t - v_t.to(x_t.device) * dts[i] # velocity pointing from data to noise
            if cls_t is not None:
                cls_t = cls_t - cls_v_t.to(cls_t.device) * dts[i]
        if cls_t is not None:
            unpacked_latent = x_t.split((packed_seqlens - 3).tolist()) # remove cls
        else:
            unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        return unpacked_latent

    @torch.no_grad
    def _forward_flow(
        self,
        x_t: torch.Tensor,
        cls_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_key_values_lens: Optional[torch.Tensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep)
        x_t = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        if cls_t is not None:
            cls_t = self.reg_input_mlp(cls_t) + self.time_embedder(timestep[0:1]) # 1, hidden_size of llm
            if cls_t.dtype != packed_sequence.dtype:
                cls_t = cls_t.to(packed_sequence.dtype)
            packed_sequence[packed_vae_token_indexes] = torch.cat([cls_t, x_t], dim=0)
        else:
            packed_sequence[packed_vae_token_indexes] = x_t

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )
        if cls_t is not None:
            v_t = self.llm2vae(output.packed_query_sequence[packed_vae_token_indexes[1:]]) # (h*w, dim of vae)
            cls_v_t = self.reg_output_mlp(output.packed_query_sequence[packed_vae_token_indexes[:1]]) # (1, dim of repa)
            
        else: 
            v_t = self.llm2vae(output.packed_query_sequence)
            v_t = v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            cfg_text_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            if cls_t is not None: 
                cfg_text_v_t = self.llm2vae(cfg_text_output.packed_query_sequence[packed_vae_token_indexes[1:]])
                cfg_text_cls_v_t = self.reg_output_mlp(cfg_text_output.packed_query_sequence[packed_vae_token_indexes[:1]])

            else:
                cfg_text_v_t = self.llm2vae(cfg_text_output.packed_query_sequence)
                cfg_text_v_t = cfg_text_v_t[packed_vae_token_indexes]

        if cfg_img_scale > 1.0:
            cfg_img_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            if cls_t is not None:
                cfg_img_v_t = self.llm2vae(cfg_img_output.packed_query_sequence[packed_vae_token_indexes[1:]])
                cfg_img_cls_v_t = self.reg_output_mlp(cfg_img_output.packed_query_sequence[packed_vae_token_indexes[:1]])

            else:
                cfg_img_v_t = self.llm2vae(cfg_img_output.packed_query_sequence)
                cfg_img_v_t = cfg_img_v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t_text = v_t_text_ * scale
                if cls_t is not None:
                    cls_v_t_text_ = cfg_text_cls_v_t + cfg_text_scale * (cls_v_t - cfg_text_cls_v_t)
                    norm_cls_v_t = torch.norm(cls_v_t, dim=-1, keepdim=True)
                    norm_cls_v_t_text_ = torch.norm(cls_v_t_text_, dim=-1, keepdim=True)
                    cls_scale = (norm_cls_v_t / (norm_cls_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                    cls_v_t_text = cls_v_t_text_ * cls_scale

                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                    if cls_t is not None:
                        cls_v_t = cfg_img_cls_v_t + cfg_img_scale * (cls_v_t_text - cfg_img_cls_v_t)
                else:
                    v_t = v_t_text
                    if cls_t is not None:
                        cls_v_t = cls_v_t_text

            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)

                # Process cls part
                if cls_t is not None:
                    cls_v_t_text_ = cfg_text_cls_v_t + cfg_text_scale * (cls_v_t - cfg_text_cls_v_t)
        
                
                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                    if cls_t is not None:
                        cls_v_t_ = cfg_img_cls_v_t + cfg_img_scale * (cls_v_t_text_ - cfg_img_cls_v_t)
                else:
                    v_t_ = v_t_text_
                    if cls_t is not None:
                        cls_v_t_ = cls_v_t_text_

                # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
                if cfg_renorm_type == "global":
                    norm_v_t = torch.norm(v_t)
                    norm_v_t_ = torch.norm(v_t_)
                    if cls_t is not None:
                        norm_cls_v_t = torch.norm(cls_v_t)
                        norm_cls_v_t_ = torch.norm(cls_v_t_)
                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                    if cls_t is not None:
                        norm_cls_v_t = torch.norm(cls_v_t, dim=-1, keepdim=True)
                        norm_cls_v_t_ = torch.norm(cls_v_t_, dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not supported")
                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t = v_t_ * scale
                # Process renorm for cls part
                if cls_t is not None:
                    cls_scale = (norm_cls_v_t / (norm_cls_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                    cls_v_t = cls_v_t_ * cls_scale
        # Return results
        if cls_t is not None:
            return v_t, cls_v_t
        else:
            return v_t, None


    def forward_for_generation_with_cache(
        self,
        past_key_values: NaiveCache,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        update_cache: bool = False,
    ):
        """
        Support generation forward pass with cache
        """
        device = packed_text_ids.device  # Get device
        
        # Build query sequence
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_query_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_query_sequence[packed_text_indexes] = packed_text_embedding
        
        # Prepare query_indexes (position in the entire KV sequence)
        packed_query_indexes = []
        kv_start = 0
        
        for kv_len, sample_len in zip(key_values_lens.tolist(), sample_lens):
            query_start = kv_start + kv_len  # New logic
            query_end = kv_start + kv_len + sample_len
            packed_query_indexes.extend(range(query_start, query_end))
            kv_start += (kv_len + sample_len)  # Update offset
        
        packed_query_indexes = torch.tensor(packed_query_indexes, dtype=torch.long, device=device)  # Specify device
        query_lens = torch.tensor(sample_lens, dtype=torch.int, device=device)  # Specify device
        
        # Process MoE
        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}
        
        # Use inference mode (supports KV cache)
        output = self.language_model.forward_inference(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_cache,
            is_causal=False,
            **extra_inputs,
        )
        
        # Get logits
        logits = self.language_model.lm_head(output.packed_query_sequence)
        
        return logits

    @torch.no_grad
    def _generate_with_full_cache(
        self,
        past_key_values: NaiveCache,
        cached_kvlens: List[int],
        initial_sequence: torch.Tensor,  # Sequence to generate (initially all mask)
        gen_position_ids: torch.LongTensor,
        gen_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        bos_token_id: int,
        eos_token_id: int,
        steps: int = 128,
        block_length: int = 32,
        temperature: float = 0.,
        cfg_scale: float = 0.,
        remasking: str = 'low_confidence',
        mask_id: int = 126336,
        confidence_threshold: float = None,  # New parameter: threshold-based confidence sampling
    ):
        """
        Generate text from fully cached context
        All prompts and images are already in cache
        
        Args:
            confidence_threshold: Confidence threshold, None means use step-based sampling,
                                  when set to a value between 0-1, tokens with confidence exceeding this threshold will be directly sampled
        """
        device = initial_sequence.device
        gen_length = initial_sequence.shape[0]
        x = initial_sequence.clone()
        
        # packed_key_value_indexes contains all cached KV
        packed_key_value_indexes = []
        curr = 0
        for cached_len in cached_kvlens:
            packed_key_value_indexes.extend(range(curr, curr + cached_len))
            curr += cached_len
        packed_key_value_indexes = torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device)
        
        # key_values_lens is total length of cache+query
        key_values_lens = torch.tensor(
            cached_kvlens,  # Contains only cached
            dtype=torch.int,
            device=device
        )
        
        # prompt_mask all False (because entire sequence is to be generated)
        prompt_mask = torch.zeros_like(x, dtype=torch.bool)
        
        # Block generation
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks
        
        for num_block in range(num_blocks):
            block_start = num_block * block_length
            block_end = (num_block + 1) * block_length
            block_mask_index = (x[block_start:block_end] == mask_id).unsqueeze(0)
            
            # Determine sampling strategy based on whether to use threshold method
            if confidence_threshold is None:
                # Step-based sampling method (original logic)
                num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
                
                for i in range(steps_per_block):
                    # mask_index marks which positions are still mask
                    mask_index = (x == mask_id) & (~prompt_mask)
                    
                    # Current sequence token ids
                    current_sequence_ids = x[gen_text_indexes]
                    
                    # CFG processing
                    if cfg_scale > 0.:
                        # Create unconditional version: keep only bos, others are mask
                        uncond_sequence_ids = torch.full_like(current_sequence_ids, mask_id)
                        uncond_sequence_ids[0] = bos_token_id
                        
                        # Concatenate conditional and unconditional
                        cfg_packed_text_ids = torch.cat([current_sequence_ids, uncond_sequence_ids], dim=0)
                        cfg_packed_text_indexes = torch.cat([
                            gen_text_indexes, 
                            gen_text_indexes + gen_length
                        ], dim=0)
                        cfg_sample_lens = sample_lens + sample_lens
                        cfg_packed_position_ids = torch.cat([
                            gen_position_ids, 
                            gen_position_ids
                        ], dim=0)
                        
                        # Copy cache (CFG needs two copies)
                        cfg_past_key_values = copy.deepcopy(past_key_values)
                        cfg_cached_kvlens = cached_kvlens + cached_kvlens
                        
                        # KV indexes for CFG
                        cfg_packed_key_value_indexes = []
                        curr = 0
                        for cached_len in cfg_cached_kvlens:
                            cfg_packed_key_value_indexes.extend(range(curr, curr + cached_len))
                            curr += cached_len
                        cfg_packed_key_value_indexes = torch.tensor(
                            cfg_packed_key_value_indexes, dtype=torch.long, device=device
                        )
                        cfg_key_values_lens = torch.tensor(
                            cfg_cached_kvlens,  # Contains only cached
                            dtype=torch.int,
                            device=device
                        )
                        
                        # Call model
                        logits = self.forward_for_generation_with_cache(
                            past_key_values=cfg_past_key_values,
                            sequence_length=gen_length * 2,
                            packed_text_ids=cfg_packed_text_ids,
                            packed_text_indexes=cfg_packed_text_indexes,
                            sample_lens=cfg_sample_lens,
                            packed_position_ids=cfg_packed_position_ids,
                            packed_key_value_indexes=cfg_packed_key_value_indexes,
                            key_values_lens=cfg_key_values_lens,
                            nested_attention_masks=None,
                            split_lens=[gen_length, gen_length],
                            attn_modes=['full', 'full'],
                            update_cache=False,
                        )
                        
                        # Separate conditional and unconditional logits
                        logits = logits.reshape(2, -1, logits.shape[-1])
                        cond_logits, uncond_logits = logits[0], logits[1]
                        logits = uncond_logits + (cfg_scale + 1) * (cond_logits - uncond_logits)
                        
                    else:
                        # No CFG case
                        logits = self.forward_for_generation_with_cache(
                            past_key_values=past_key_values,
                            sequence_length=gen_length,
                            packed_text_ids=current_sequence_ids,
                            packed_text_indexes=gen_text_indexes,
                            sample_lens=sample_lens,
                            packed_position_ids=gen_position_ids,
                            packed_key_value_indexes=packed_key_value_indexes,
                            key_values_lens=key_values_lens,
                            nested_attention_masks=None,
                            split_lens=[gen_length],
                            attn_modes=['full'],
                            update_cache=False,
                        )
                    
                    # Add Gumbel noise
                    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                    x0 = torch.argmax(logits_with_noise, dim=-1)
                    
                    # Compute confidence
                    if remasking == 'low_confidence':
                        p = F.softmax(logits, dim=-1)
                        x0_p = torch.squeeze(
                            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                    elif remasking == 'random':
                        x0_p = torch.rand_like(x0, dtype=torch.float, device=x0.device)
                    else:
                        raise NotImplementedError(remasking)
                    
                    # Limit confidence range - only for positions before current block
                    x0_p[block_end:] = -np.inf
                    
                    # Update sequence
                    x0 = x0.to(x.device)
                    x0_p = x0_p.to(x.device)
                    x0 = torch.where(mask_index, x0, x)
                    confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))
                    
                    # Select tokens to transfer (based on confidence)
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                    _, select_index = torch.topk(confidence, k=num_transfer_tokens[0, i])
                    transfer_index[select_index] = True
                    x[transfer_index] = x0[transfer_index]
            else:
                # Threshold-based sampling method (new logic)
                # Set max iterations to prevent infinite loop
                max_iterations = steps_per_block * 2
                
                for iteration in range(max_iterations):
                    # Check if current block still has mask
                    block_mask_count = (x[block_start:block_end] == mask_id).sum().item()
                    if block_mask_count == 0:
                        break  # Current block fully sampled
                    
                    # mask_index marks which positions are still mask
                    mask_index = (x == mask_id) & (~prompt_mask)
                    
                    # Current sequence token ids
                    current_sequence_ids = x[gen_text_indexes]
                    
                    # CFG processing
                    if cfg_scale > 0.:
                        # Create unconditional version: keep only bos, others are mask
                        uncond_sequence_ids = torch.full_like(current_sequence_ids, mask_id)
                        uncond_sequence_ids[0] = bos_token_id
                        
                        # Concatenate conditional and unconditional
                        cfg_packed_text_ids = torch.cat([current_sequence_ids, uncond_sequence_ids], dim=0)
                        cfg_packed_text_indexes = torch.cat([
                            gen_text_indexes, 
                            gen_text_indexes + gen_length
                        ], dim=0)
                        cfg_sample_lens = sample_lens + sample_lens
                        cfg_packed_position_ids = torch.cat([
                            gen_position_ids, 
                            gen_position_ids
                        ], dim=0)
                        
                        # Copy cache (CFG needs two copies)
                        cfg_past_key_values = copy.deepcopy(past_key_values)
                        cfg_cached_kvlens = cached_kvlens + cached_kvlens
                        
                        # KV indexes for CFG
                        cfg_packed_key_value_indexes = []
                        curr = 0
                        for cached_len in cfg_cached_kvlens:
                            cfg_packed_key_value_indexes.extend(range(curr, curr + cached_len))
                            curr += cached_len
                        cfg_packed_key_value_indexes = torch.tensor(
                            cfg_packed_key_value_indexes, dtype=torch.long, device=device
                        )
                        cfg_key_values_lens = torch.tensor(
                            cfg_cached_kvlens,  # Contains only cached
                            dtype=torch.int,
                            device=device
                        )
                        
                        # Call model
                        logits = self.forward_for_generation_with_cache(
                            past_key_values=cfg_past_key_values,
                            sequence_length=gen_length * 2,
                            packed_text_ids=cfg_packed_text_ids,
                            packed_text_indexes=cfg_packed_text_indexes,
                            sample_lens=cfg_sample_lens,
                            packed_position_ids=cfg_packed_position_ids,
                            packed_key_value_indexes=cfg_packed_key_value_indexes,
                            key_values_lens=cfg_key_values_lens,
                            nested_attention_masks=None,
                            split_lens=[gen_length, gen_length],
                            attn_modes=['full', 'full'],
                            update_cache=False,
                        )
                        
                        # Separate conditional and unconditional logits
                        logits = logits.reshape(2, -1, logits.shape[-1])
                        cond_logits, uncond_logits = logits[0], logits[1]
                        logits = uncond_logits + (cfg_scale + 1) * (cond_logits - uncond_logits)
                        
                    else:
                        # No CFG case
                        logits = self.forward_for_generation_with_cache(
                            past_key_values=past_key_values,
                            sequence_length=gen_length,
                            packed_text_ids=current_sequence_ids,
                            packed_text_indexes=gen_text_indexes,
                            sample_lens=sample_lens,
                            packed_position_ids=gen_position_ids,
                            packed_key_value_indexes=packed_key_value_indexes,
                            key_values_lens=key_values_lens,
                            nested_attention_masks=None,
                            split_lens=[gen_length],
                            attn_modes=['full'],
                            update_cache=False,
                        )
                    
                    # Add Gumbel noise
                    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                    x0 = torch.argmax(logits_with_noise, dim=-1)
                    
                    # Compute confidence
                    if remasking == 'low_confidence':
                        p = F.softmax(logits, dim=-1)
                        x0_p = torch.squeeze(
                            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                    elif remasking == 'random':
                        x0_p = torch.rand_like(x0, dtype=torch.float, device=x0.device)
                    else:
                        raise NotImplementedError(remasking)
                    
                    # Limit confidence range - only within current block range
                    x0_p[:block_start] = -np.inf
                    x0_p[block_end:] = -np.inf
                    
                    # Update sequence - only at mask positions
                    x0 = x0.to(x.device)
                    x0_p = x0_p.to(x.device)
                    x0 = torch.where(mask_index, x0, x)
                    confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))
                    
                    # Threshold-based sampling: select tokens with confidence exceeding threshold
                    transfer_index = (confidence >= confidence_threshold) & mask_index
                    
                    # If no token exceeds threshold, select the one with highest confidence
                    if not transfer_index.any():
                        # Get highest confidence for mask positions in current block
                        block_confidence = confidence.clone()
                        block_confidence[:block_start] = -np.inf
                        block_confidence[block_end:] = -np.inf
                        _, best_idx = torch.topk(block_confidence, k=1)
                        transfer_index[best_idx] = True
                    
                    x[transfer_index] = x0[transfer_index]
        
        return x.unsqueeze(0)  # Add batch dimension
    
    @torch.no_grad
    def generate_text_mask_prediction(
        self,
        sequence_length: int,
        # special token ids
        eos_token_id: int,
        bos_token_id: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        # visual understanding inputs (if any)
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        # generation parameters
        steps: int = 128,
        gen_length: int = 128,
        block_length: int = 32,
        temperature: float = 0.,
        cfg_scale: float = 0.,
        remasking: str = 'low_confidence',
        mask_id: int = 126336,
        confidence_threshold: float = None,  # New parameter: threshold-based confidence sampling
    ):
        """
        Generate text using mask prediction method
        
        Args:
            confidence_threshold: Confidence threshold, None means use step-based sampling,
                                  when set to a value between 0-1, tokens with confidence exceeding this threshold will be directly sampled
        """
        device = packed_text_ids.device
        
        # Create initial sequence, fill generation part with mask_id
        x = torch.full((sequence_length + gen_length,), mask_id, dtype=torch.long, device=device)
        x[packed_text_indexes] = packed_text_ids
        x[sequence_length] = bos_token_id  # Place bos token at the start of generation part
        #x[-1] = eos_token_id # Place eos token at the end of sequence
        
        # Extend position_ids for generated tokens
        # Assign consecutive position ids for generated tokens
        max_pos_id = packed_position_ids.max().item()
        gen_position_ids = torch.arange(
            max_pos_id + 1, max_pos_id + 1 + gen_length, 
            dtype=packed_position_ids.dtype, 
            device=device
        )
        extended_packed_position_ids = torch.cat([packed_position_ids, gen_position_ids], dim=0)
        
        # Create text_indexes for generation part
        gen_text_indexes = torch.arange(
            sequence_length, sequence_length + gen_length, 
            dtype=packed_text_indexes.dtype, 
            device=device
        )
        extended_packed_text_indexes = torch.cat([packed_text_indexes, gen_text_indexes], dim=0)
        
        # Extend sample_lens, split_lens, attn_modes
        for i in range(len(sample_lens)):
            sample_lens[i] += gen_length
        extended_sample_lens = sample_lens

        if split_lens is not None:
            extended_split_lens = split_lens + [gen_length]
        else:
            extended_split_lens = None
        if attn_modes is not None:
            extended_attn_modes = attn_modes + ['full']
        else:
            extended_attn_modes = None
        
        # Now sum the extended results
        if extended_split_lens is not None:
            total_len = sum(extended_split_lens)
            extended_split_lens = [total_len]
        if extended_attn_modes is not None:
            extended_attn_modes = ['full']
        
        # Record prompt positions (first sequence_length positions are prompt part)
        prompt_mask = torch.zeros_like(x, dtype=torch.bool)
        prompt_mask[:sequence_length] = True
        
        # Block generation
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks
        
        for num_block in range(num_blocks):
            block_start = sequence_length + num_block * block_length
            block_end = sequence_length + (num_block + 1) * block_length
            block_mask_index = (x[block_start:block_end] == mask_id).unsqueeze(0)  # Add batch dimension
            
            # Determine sampling strategy based on whether to use threshold method
            if confidence_threshold is None:
                # Step-based sampling method (original logic)
                num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
                
                for i in range(steps_per_block):
                    # mask_index only contains mask positions in generation part, excluding prompt part
                    mask_index = (x == mask_id) & (~prompt_mask)
                    
                    # Prepare complete sequence token ids for current step
                    current_sequence_ids = x[extended_packed_text_indexes]
                    
                    # CFG processing
                    if cfg_scale > 0.:
                        # should be checked again...
                        # Create unconditional version: also mask the prompt part
                        un_x = x.clone()
                        un_x[packed_text_indexes] = mask_id  # Only mask original prompt, do not mask generated part
                        uncond_sequence_ids = un_x[extended_packed_text_indexes]
                        
                        # Prepare two inputs for CFG
                        cfg_packed_text_ids = torch.cat([current_sequence_ids, uncond_sequence_ids], dim=0)
                        cfg_packed_text_indexes = torch.cat([
                            extended_packed_text_indexes, 
                            extended_packed_text_indexes + sequence_length + gen_length
                        ], dim=0)
                        cfg_sample_lens = extended_sample_lens + extended_sample_lens
                        cfg_packed_position_ids = torch.cat([
                            extended_packed_position_ids, 
                            extended_packed_position_ids
                        ], dim=0)
                        cfg_sequence_length = (sequence_length + gen_length) * 2
                        
                        if extended_split_lens is not None:
                            cfg_split_lens = extended_split_lens + extended_split_lens
                        else:
                            cfg_split_lens = None
                            
                        if extended_attn_modes is not None:
                            cfg_attn_modes = extended_attn_modes + extended_attn_modes
                        else:
                            cfg_attn_modes = None
                        
                        # Process visual inputs (if any)
                        if packed_vit_tokens is not None:
                            cfg_packed_vit_token_indexes = torch.cat([
                                packed_vit_token_indexes, 
                                packed_vit_token_indexes + sequence_length + gen_length
                            ], dim=0)
                            cfg_packed_vit_tokens = torch.cat([packed_vit_tokens, packed_vit_tokens], dim=0)
                            cfg_packed_vit_position_ids = torch.cat([packed_vit_position_ids, packed_vit_position_ids], dim=0)
                            cfg_vit_token_seqlens = torch.cat([vit_token_seqlens, vit_token_seqlens], dim=0)
                        else:
                            cfg_packed_vit_token_indexes = None
                            cfg_packed_vit_tokens = None
                            cfg_packed_vit_position_ids = None
                            cfg_vit_token_seqlens = None
                        
                        # Call model
                        outputs = self.forward_for_generation(
                            sequence_length=cfg_sequence_length,
                            packed_text_ids=cfg_packed_text_ids,
                            packed_text_indexes=cfg_packed_text_indexes,
                            sample_lens=cfg_sample_lens,
                            packed_position_ids=cfg_packed_position_ids,
                            nested_attention_masks=nested_attention_masks,
                            split_lens=cfg_split_lens,
                            attn_modes=cfg_attn_modes,
                            packed_vit_tokens=cfg_packed_vit_tokens,
                            packed_vit_token_indexes=cfg_packed_vit_token_indexes,
                            packed_vit_position_ids=cfg_packed_vit_position_ids,
                            vit_token_seqlens=cfg_vit_token_seqlens,
                        )
                        
                        # Separate CFG results
                        logits = outputs.reshape(2, -1, outputs.shape[-1])
                        cond_logits, uncond_logits = logits[0], logits[1]
                        logits = uncond_logits + (cfg_scale + 1) * (cond_logits - uncond_logits)
                        
                    else:
                        # No CFG case
                        outputs = self.forward_for_generation(
                            sequence_length=sequence_length + gen_length,
                            packed_text_ids=current_sequence_ids,
                            packed_text_indexes=extended_packed_text_indexes,
                            sample_lens=extended_sample_lens,
                            packed_position_ids=extended_packed_position_ids,
                            nested_attention_masks=nested_attention_masks,
                            split_lens=extended_split_lens,
                            attn_modes=extended_attn_modes,
                            packed_vit_tokens=packed_vit_tokens,
                            packed_vit_token_indexes=packed_vit_token_indexes,
                            packed_vit_position_ids=packed_vit_position_ids,
                            vit_token_seqlens=vit_token_seqlens,
                        )
                        logits = outputs
                    
                    # Add Gumbel noise
                    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                    x0 = torch.argmax(logits_with_noise, dim=-1)
                    
                    # Compute confidence
                    if remasking == 'low_confidence':
                        p = F.softmax(logits, dim=-1)
                        x0_p = torch.squeeze(
                            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                    elif remasking == 'random':
                        x0_p = torch.rand_like(x0, dtype=torch.float, device=x0.device)
                    else:
                        raise NotImplementedError(remasking)
                    
                    # Limit confidence range - only operate within current generation range
                    x0_p[block_end:] = -np.inf
                    
                    # Update sequence - only update mask positions
                    x0 = torch.where(mask_index, x0, x)
                    confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))
                    
                    # Select tokens to transfer
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                    _, select_index = torch.topk(confidence, k=num_transfer_tokens[0, i])
                    transfer_index[select_index] = True
                    x[transfer_index] = x0[transfer_index]
            else:
                # Threshold-based sampling method (new logic)
                # Set max iterations to prevent infinite loop
                max_iterations = steps_per_block * 2
                
                for iteration in range(max_iterations):
                    # Check if current block still has mask
                    block_mask_count = (x[block_start:block_end] == mask_id).sum().item()
                    if block_mask_count == 0:
                        break  # Current block fully sampled
                    
                    # mask_index only contains mask positions in generation part, excluding prompt part
                    mask_index = (x == mask_id) & (~prompt_mask)
                    
                    # Prepare complete sequence token ids for current step
                    current_sequence_ids = x[extended_packed_text_indexes]
                    
                    # CFG processing
                    if cfg_scale > 0.:
                        # Create unconditional version: also mask the prompt part
                        un_x = x.clone()
                        un_x[packed_text_indexes] = mask_id  # Only mask original prompt, do not mask generated part
                        uncond_sequence_ids = un_x[extended_packed_text_indexes]
                        
                        # Prepare two inputs for CFG
                        cfg_packed_text_ids = torch.cat([current_sequence_ids, uncond_sequence_ids], dim=0)
                        cfg_packed_text_indexes = torch.cat([
                            extended_packed_text_indexes, 
                            extended_packed_text_indexes + sequence_length + gen_length
                        ], dim=0)
                        cfg_sample_lens = extended_sample_lens + extended_sample_lens
                        cfg_packed_position_ids = torch.cat([
                            extended_packed_position_ids, 
                            extended_packed_position_ids
                        ], dim=0)
                        cfg_sequence_length = (sequence_length + gen_length) * 2
                        
                        if extended_split_lens is not None:
                            cfg_split_lens = extended_split_lens + extended_split_lens
                        else:
                            cfg_split_lens = None
                            
                        if extended_attn_modes is not None:
                            cfg_attn_modes = extended_attn_modes + extended_attn_modes
                        else:
                            cfg_attn_modes = None
                        
                        # Process visual inputs (if any)
                        if packed_vit_tokens is not None:
                            cfg_packed_vit_token_indexes = torch.cat([
                                packed_vit_token_indexes, 
                                packed_vit_token_indexes + sequence_length + gen_length
                            ], dim=0)
                            cfg_packed_vit_tokens = torch.cat([packed_vit_tokens, packed_vit_tokens], dim=0)
                            cfg_packed_vit_position_ids = torch.cat([packed_vit_position_ids, packed_vit_position_ids], dim=0)
                            cfg_vit_token_seqlens = torch.cat([vit_token_seqlens, vit_token_seqlens], dim=0)
                        else:
                            cfg_packed_vit_token_indexes = None
                            cfg_packed_vit_tokens = None
                            cfg_packed_vit_position_ids = None
                            cfg_vit_token_seqlens = None
                        
                        # Call model
                        outputs = self.forward_for_generation(
                            sequence_length=cfg_sequence_length,
                            packed_text_ids=cfg_packed_text_ids,
                            packed_text_indexes=cfg_packed_text_indexes,
                            sample_lens=cfg_sample_lens,
                            packed_position_ids=cfg_packed_position_ids,
                            nested_attention_masks=nested_attention_masks,
                            split_lens=cfg_split_lens,
                            attn_modes=cfg_attn_modes,
                            packed_vit_tokens=cfg_packed_vit_tokens,
                            packed_vit_token_indexes=cfg_packed_vit_token_indexes,
                            packed_vit_position_ids=cfg_packed_vit_position_ids,
                            vit_token_seqlens=cfg_vit_token_seqlens,
                        )
                        
                        # Separate CFG results
                        logits = outputs.reshape(2, -1, outputs.shape[-1])
                        cond_logits, uncond_logits = logits[0], logits[1]
                        logits = uncond_logits + (cfg_scale + 1) * (cond_logits - uncond_logits)
                        
                    else:
                        # No CFG case
                        outputs = self.forward_for_generation(
                            sequence_length=sequence_length + gen_length,
                            packed_text_ids=current_sequence_ids,
                            packed_text_indexes=extended_packed_text_indexes,
                            sample_lens=extended_sample_lens,
                            packed_position_ids=extended_packed_position_ids,
                            nested_attention_masks=nested_attention_masks,
                            split_lens=extended_split_lens,
                            attn_modes=extended_attn_modes,
                            packed_vit_tokens=packed_vit_tokens,
                            packed_vit_token_indexes=packed_vit_token_indexes,
                            packed_vit_position_ids=packed_vit_position_ids,
                            vit_token_seqlens=vit_token_seqlens,
                        )
                        logits = outputs
                    
                    # Add Gumbel noise
                    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                    x0 = torch.argmax(logits_with_noise, dim=-1)
                    
                    # Compute confidence
                    if remasking == 'low_confidence':
                        p = F.softmax(logits, dim=-1)
                        x0_p = torch.squeeze(
                            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                    elif remasking == 'random':
                        x0_p = torch.rand_like(x0, dtype=torch.float, device=x0.device)
                    else:
                        raise NotImplementedError(remasking)
                    
                    # Limit confidence range - only within current block range
                    x0_p[:block_start] = -np.inf
                    x0_p[block_end:] = -np.inf
                    
                    # Update sequence - only at mask positions
                    x0 = torch.where(mask_index, x0, x)
                    confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))
                    
                    # Threshold-based sampling: select tokens with confidence exceeding threshold
                    transfer_index = (confidence >= confidence_threshold) & mask_index
                    
                    # If no token exceeds threshold, select the one with highest confidence
                    if not transfer_index.any():
                        # Get highest confidence for mask positions in current block
                        block_confidence = confidence.clone()
                        block_confidence[:block_start] = -np.inf
                        block_confidence[block_end:] = -np.inf
                        _, best_idx = torch.topk(block_confidence, k=1)
                        transfer_index[best_idx] = True
                    
                    x[transfer_index] = x0[transfer_index]
        
        return x.unsqueeze(0)  # Add batch dimension to match original interface
        
    def forward_for_generation(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
    ):
        """
        Forward pass for generation, returns logits
        """
        # Build input sequence
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding
        # Process attention mask
        if nested_attention_masks is None:
            nested_attention_masks = []
            nested_attention_masks.append(
                prepare_attention_mask_per_sample(split_lens, attn_modes).to(packed_text_embedding.device)
            )
            # sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            # seqlen = sum(sample_lens)
            # block_mask = create_block_mask(
            #     sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, 
            #     device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            # )
            attention_mask = nested_attention_masks
        else:
            attention_mask = nested_attention_masks
        # Process visual inputs
        if self.config.visual_und and packed_vit_tokens is not None:
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
            cu_seqlens = cu_seqlens.to(torch.int32)
            max_seqlen = torch.max(vit_token_seqlens).item()
            packed_vit_token_embed = self.vit_model(
                packed_pixel_values=packed_vit_tokens, 
                packed_flattened_position_ids=packed_vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            packed_vit_token_embed = self.connector(packed_vit_token_embed)
            vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
            packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed
        # Prepare MoE inputs
        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes = torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=None,  # Not needed during generation
            )
        # Call language model
        last_hidden_state, _ = self.language_model.forward_train(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )
        # Get logits
        logits = self.language_model.lm_head(last_hidden_state)
        return logits
        
    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        new_token_ids,
        image_transform,
        images,
        prompt,
        max_length: int = 128,
        steps: int = 128,
        block_length: int = 32,
        temperature: float = 0.,
        cfg_scale: float = 0.,
        remasking: str = 'low_confidence',
        mask_id: int = None,
        use_cache: bool = False,  # New parameter: control whether to use cache
        confidence_threshold: float = None,  # New parameter: threshold-based confidence sampling, None means use step-based sampling
    ):
        device = next(self.parameters()).device
        
        if mask_id is None:
            mask_id = new_token_ids.get('mask_token_id', tokenizer.mask_token_id)
        if isinstance(new_token_ids, dict):
            for k, v in new_token_ids.items():
                if torch.is_tensor(v):
                    new_token_ids[k] = v.to(device)
        
        # ===== Use cache path =====
        if use_cache:
            # Initialize cache
            self.eval()
            past_key_values = NaiveCache(self.config.llm_config.num_hidden_layers)
            newlens = [0]
            new_rope = [0]
            
            # Step 1: Cache all images
            for image in images:
                generation_input, newlens, new_rope = self.prepare_vit_images(
                    curr_kvlens=newlens,
                    curr_rope=new_rope, 
                    images=[image], 
                    transforms=image_transform,
                    new_token_ids=new_token_ids,
                )
                for k, v in generation_input.items():
                    if torch.is_tensor(v):
                        generation_input[k] = v.to(device)
                
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    past_key_values = self.forward_cache_update_vit(past_key_values, **generation_input)
            
            # Step 2: Cache prompt
            generation_input, newlens, new_rope = self.prepare_prompts(
                curr_kvlens=newlens,
                curr_rope=new_rope, 
                prompts=[prompt],
                tokenizer=tokenizer, 
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)
            
            # Step 3: Prepare generation stage inputs
            gen_length = max_length
            x = torch.full((gen_length,), mask_id, dtype=torch.long, device=device)
            x[0] = new_token_ids['bos_token_id']
            
            # Generation part position_ids
            current_pos = new_rope[0]
            gen_position_ids = torch.arange(
                current_pos, current_pos + gen_length, 
                dtype=torch.long, 
                device=device
            )
            
            # Generation part text_indexes
            gen_text_indexes = torch.arange(0, gen_length, dtype=torch.long, device=device)
            
            # sample_lens only contains generation part length
            sample_lens = [gen_length]
            
            # Step 4: mask prediction generation
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                generated_sequence = self._generate_with_full_cache(
                    past_key_values=past_key_values,
                    cached_kvlens=newlens,
                    initial_sequence=x,
                    gen_position_ids=gen_position_ids,
                    gen_text_indexes=gen_text_indexes,
                    sample_lens=sample_lens,
                    bos_token_id=new_token_ids['bos_token_id'],
                    eos_token_id=new_token_ids['eos_token_id'],
                    steps=steps,
                    block_length=block_length,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    remasking=remasking,
                    mask_id=mask_id,
                    confidence_threshold=confidence_threshold,
                )
            
            # Decode output
            output = tokenizer.decode(generated_sequence[0], skip_special_tokens=False)
            
        # ===== No cache path =====
        else:
            self.train()
            # Prepare input sequence
            packed_text_ids = []
            packed_text_indexes = []
            packed_vit_tokens = []
            packed_vit_token_indexes = []
            packed_vit_position_ids = []
            vit_token_seqlens = []
            packed_position_ids = []
            split_lens = []
            attn_modes = []
            
            current_idx = 0
            current_pos = 0
            current_split_len = 0
            
            # Process images
            for image in images:
                # Add image start token
                packed_text_ids.append(new_token_ids['start_of_image'])
                packed_text_indexes.append(current_idx)
                packed_position_ids.append(current_pos)
                current_idx += 1
                current_split_len += 1
                
                # Process image
                image_tensor = image_transform(image)
                vit_position_ids = self.get_flattened_position_ids(
                    image_tensor.size(1), image_tensor.size(2), 
                    self.vit_patch_size, 
                    max_num_patches_per_side=self.vit_max_num_patch_per_side
                )
                vit_tokens = patchify(image_tensor, self.vit_patch_size)
                
                packed_vit_tokens.append(vit_tokens)
                packed_vit_position_ids.append(vit_position_ids)
                num_img_tokens = vit_tokens.shape[0]
                vit_token_seqlens.append(num_img_tokens)
                
                vit_start_idx = current_idx
                packed_vit_token_indexes.extend(range(vit_start_idx, vit_start_idx + num_img_tokens))
                packed_position_ids.extend([current_pos] * num_img_tokens)
                current_idx += num_img_tokens
                current_split_len += num_img_tokens
                
                # Add image end token
                packed_text_ids.append(new_token_ids['end_of_image'])
                packed_text_indexes.append(current_idx)
                packed_position_ids.append(current_pos)
                current_idx += 1
                current_split_len += 1
                
                # Complete image split
                split_lens.append(current_split_len)
                attn_modes.append('full')
                current_split_len = 0
                current_pos += 1
            
            # Process text
            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            
            for token_id in text_ids:
                packed_text_ids.append(token_id)
                packed_text_indexes.append(current_idx)
                packed_position_ids.append(current_pos)
                current_idx += 1
                current_split_len += 1
                current_pos += 1
            
            # Complete text split
            split_lens.append(current_split_len)
            attn_modes.append('full')
            
            sequence_length = current_idx
            
            # Convert to tensors
            packed_text_ids = torch.tensor(packed_text_ids, dtype=torch.long, device=device)
            packed_text_indexes = torch.tensor(packed_text_indexes, dtype=torch.long, device=device)
            packed_position_ids = torch.tensor(packed_position_ids, dtype=torch.long, device=device)
            
            if packed_vit_tokens:
                packed_vit_tokens = torch.cat(packed_vit_tokens, dim=0).to(device)
                packed_vit_token_indexes = torch.tensor(packed_vit_token_indexes, dtype=torch.long, device=device)
                packed_vit_position_ids = torch.cat(packed_vit_position_ids, dim=0).to(device)
                vit_token_seqlens = torch.tensor(vit_token_seqlens, dtype=torch.int, device=device)
            else:
                packed_vit_tokens = None
                packed_vit_token_indexes = None
                packed_vit_position_ids = None
                vit_token_seqlens = None

            # Generate text
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                generated_sequence = self.generate_text_mask_prediction(
                    sequence_length=sequence_length,
                    bos_token_id=new_token_ids['bos_token_id'],
                    eos_token_id=new_token_ids['eos_token_id'],
                    packed_text_ids=packed_text_ids,
                    packed_text_indexes=packed_text_indexes,
                    sample_lens=[sequence_length],
                    packed_position_ids=packed_position_ids,
                    split_lens=split_lens,
                    attn_modes=attn_modes,
                    packed_vit_tokens=packed_vit_tokens,
                    packed_vit_token_indexes=packed_vit_token_indexes,
                    packed_vit_position_ids=packed_vit_position_ids,
                    vit_token_seqlens=vit_token_seqlens,
                    steps=steps,
                    gen_length=max_length,
                    block_length=block_length,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    remasking=remasking,
                    mask_id=mask_id,
                    confidence_threshold=confidence_threshold,
                )
            
            # Decode output
            generated_tokens = generated_sequence[0, sequence_length:]
            output = tokenizer.decode(generated_tokens, skip_special_tokens=False)
        
        return output

    @torch.no_grad()
    def chat_block(
        self,
        tokenizer,
        new_token_ids,
        image_transform,
        images,
        prompt,
        block_length: int = 32,
        steps_per_block: int = 32,
        max_blocks: int = 16,  # Maximum number of blocks, prevent infinite loop
        temperature: float = 0.,
        cfg_scale: float = 0.,
        remasking: str = 'low_confidence',
        mask_id: int = None,
        confidence_threshold: float = None,  # New parameter: threshold-based confidence sampling, None means use step-based sampling
    ):
        """
        Block-based iterative generation, stops automatically when EOS is encountered.
        
        Difference from chat:
        - chat: Generate fixed length max_length at once
        - chat_block: Generate one block at a time, cache generated content and continue if no EOS
        
        Args:
            tokenizer: Tokenizer
            new_token_ids: Special token ids
            image_transform: Image transform
            images: Image list
            prompt: Input prompt
            block_length: Length of each block
            steps_per_block: Generation steps per block
            max_blocks: Maximum number of blocks (prevent infinite loop)
            temperature: Sampling temperature
            cfg_scale: CFG scale
            remasking: Remasking strategy
            mask_id: Mask token id
        
        Returns:
            Generated text
        """
        input_device = self.language_model.model.embed_tokens.weight.device
        
        if mask_id is None:
            mask_id = new_token_ids.get('mask_token_id', tokenizer.mask_token_id)
        if isinstance(new_token_ids, dict):
            for k, v in new_token_ids.items():
                if torch.is_tensor(v):
                    new_token_ids[k] = v.to(input_device)
        
        self.eval()
        
        # ===== Step 1: Initialize cache, cache images and prompt =====
        past_key_values = NaiveCache(self.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]
        
        # Cache all images
        for image in images:
            generation_input, newlens, new_rope = self.prepare_vit_images(
                curr_kvlens=newlens,
                curr_rope=new_rope, 
                images=[image], 
                transforms=image_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(input_device)
            
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.forward_cache_update_vit(past_key_values, **generation_input)
        
        # Cache prompt
        generation_input, newlens, new_rope = self.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[prompt],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(input_device)
        
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)
        
        # ===== Step 2: Iteratively generate blocks =====
        eos_token_id = new_token_ids['eos_token_id']
        bos_token_id = new_token_ids['bos_token_id']
        
        generated_tokens = []  # Store all generated tokens
        valid_generated = 0
        total_generated = 0
        current_rope_pos = new_rope[0]
        is_first_block = True
        
        for block_idx in range(max_blocks):
            # Prepare current block input
            x = torch.full((block_length,), mask_id, dtype=torch.long, device=input_device)
            
            # First token of first block is BOS
            if is_first_block:
                x[0] = bos_token_id
                is_first_block = False
            
            # Generation part position_ids
            gen_position_ids = torch.arange(
                current_rope_pos, current_rope_pos + block_length, 
                dtype=torch.long, 
                device=input_device
            )
            
            # Generation part text_indexes
            gen_text_indexes = torch.arange(0, block_length, dtype=torch.long, device=input_device)
            
            # sample_lens only contains current block length
            sample_lens = [block_length]
            
            # Generate current block (steps = steps_per_block, block_length = block_length)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                block_output = self._generate_with_full_cache(
                    past_key_values=past_key_values,
                    cached_kvlens=newlens,
                    initial_sequence=x,
                    gen_position_ids=gen_position_ids,
                    gen_text_indexes=gen_text_indexes,
                    sample_lens=sample_lens,
                    bos_token_id=bos_token_id,
                    eos_token_id=eos_token_id,
                    steps=steps_per_block,
                    block_length=block_length,  # Entire block as a generation unit
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    remasking=remasking,
                    mask_id=mask_id,
                    confidence_threshold=confidence_threshold,
                )
            
            # block_output shape: [1, block_length]
            block_tokens = block_output[0]  # [block_length]
            total_generated += block_tokens.numel()
            
            # Check if EOS is contained
            eos_positions = (block_tokens == eos_token_id).nonzero(as_tuple=True)[0]
            
            if len(eos_positions) > 0:
                # Find first EOS position, truncate and finish
                first_eos_pos = eos_positions[0].item()
                generated_tokens.append(block_tokens[:first_eos_pos + 1])  # Include EOS
                valid_generated += first_eos_pos + 1
                break
            else:
                # No EOS, save current block tokens
                generated_tokens.append(block_tokens)
                valid_generated += block_tokens.numel()
                
                # Cache generated block as context for next block
                # Reuse prepare_prompts logic format, manually construct input
                curr_kvlen = newlens[0]
                
                # Construct generation_input with same format as prepare_prompts output
                packed_key_value_indexes = list(range(curr_kvlen))
                packed_text_indexes = list(range(curr_kvlen, curr_kvlen + block_length))
                packed_text_position_ids = list(range(current_rope_pos, current_rope_pos + block_length))
                
                generation_input = {
                    "text_token_lens": torch.tensor([block_length], dtype=torch.int, device=input_device),
                    "packed_text_ids": block_tokens.clone(),
                    "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long, device=input_device),
                    "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=input_device),
                    "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=input_device),
                    "key_values_lens": torch.tensor([curr_kvlen], dtype=torch.int, device=input_device),
                }
                
                # Update cache (exactly same as caching prompt)
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)
                
                # Update state
                newlens = [curr_kvlen + block_length]
                current_rope_pos += block_length
        
        # ===== Step 3: Concatenate all generated tokens and decode =====
        if generated_tokens:
            all_tokens = torch.cat(generated_tokens, dim=0)
            output = tokenizer.decode(all_tokens, skip_special_tokens=False)
        else:
            output = ""
        
        return output, valid_generated, total_generated
