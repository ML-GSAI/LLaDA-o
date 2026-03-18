# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright 2025 AntGroup and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import random
import json

import numpy as np
import torch

from .data_utils import (
    get_flattened_position_ids_interpolate,
    get_flattened_position_ids_extrapolate, 
    len2weight,
    patchify, 
    prepare_attention_mask_per_sample, 
)
from .dataset_info import DATASET_INFO, DATASET_REGISTRY
from .transforms import ImageTransform
from .video_utils import FrameSampler


class DataConfig:
    def __init__(
        self, 
        grouped_datasets, 
        text_cond_dropout_prob=0.1,
        vit_cond_dropout_prob=0.4,
        vae_cond_dropout_prob=0.1,
        vae_image_downsample=16,
        max_latent_size=32,
        vit_patch_size=14,
        max_num_patch_per_side=70,
        visual_und=False,
        visual_gen=False,
        visual_gen_reg=False,
        visual_und_sft=False,
        ada_len=False,
        ada_len_split=False,
        visual_und_always_mask_last=False,
        merge_vit_text_segments=False,
        loss_reduction='square',
    ):
        self.grouped_datasets = grouped_datasets
        self.text_cond_dropout_prob = text_cond_dropout_prob
        self.vit_cond_dropout_prob = vit_cond_dropout_prob
        self.vit_patch_size = vit_patch_size
        self.max_num_patch_per_side = max_num_patch_per_side
        self.vae_cond_dropout_prob = vae_cond_dropout_prob
        self.vae_image_downsample = vae_image_downsample
        self.max_latent_size = max_latent_size
        self.visual_und = visual_und
        self.visual_gen = visual_gen
        self.visual_gen_reg = visual_gen_reg
        self.visual_und_sft = visual_und_sft
        self.ada_len = ada_len
        self.ada_len_split = ada_len_split
        self.visual_und_always_mask_last = visual_und_always_mask_last
        self.merge_vit_text_segments = merge_vit_text_segments
        self.loss_reduction = loss_reduction


class PackedDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        data_config, 
        tokenizer, 
        special_tokens,
        local_rank, 
        world_size, 
        num_workers,
        expected_num_tokens=32768, 
        max_num_tokens_per_sample=16384,
        max_num_tokens=36864,
        prefer_buffer_before=16384,
        max_buffer_size=50,
        interpolate_pos=False,
        use_flex=False,
        data_status=None,
    ):
        super().__init__()
        self.masked_token_id = 126336 # masked token for LLaDA and LLaDA-1.5
        self.expected_num_tokens = expected_num_tokens
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        self.prefer_buffer_before = prefer_buffer_before
        self.max_num_tokens = max_num_tokens
        self.max_buffer_size = max_buffer_size
        self.tokenizer = tokenizer
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.use_flex = use_flex
        for k, v in special_tokens.items():
            setattr(self, k, v)

        grouped_datasets, is_mandatory, grouped_weights, is_auxiliary = self.build_datasets(
            data_config.grouped_datasets, data_status
        )
        self.grouped_datasets = grouped_datasets
        self.dataset_iters = [iter(dataset) for dataset in grouped_datasets]
        self.is_mandatory = is_mandatory
        self.is_auxiliary = is_auxiliary
        self.grouped_weights = grouped_weights
        self.data_config = data_config
        self.interpolate_pos = interpolate_pos
        if self.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

    def build_datasets(self, datasets_metainfo, data_status):
        datasets = []
        is_mandatory = []
        grouped_weights = []
        is_auxiliary = []
        for grouped_dataset_name, dataset_args in datasets_metainfo.items():
            is_mandatory.append(dataset_args.pop('is_mandatory', False))
            is_auxiliary.append(dataset_args.pop('is_auxiliary', False)) # Auxiliary datasets use weight 0.0!
            grouped_weights.append(dataset_args.pop('weight', 0.0))

            if 'frame_sampler_args' in dataset_args.keys():
                frame_sampler = FrameSampler(**dataset_args.pop('frame_sampler_args'))
                dataset_args['frame_sampler'] = frame_sampler
            if 'image_transform_args' in dataset_args.keys():
                transform = ImageTransform(**dataset_args.pop('image_transform_args'))
                dataset_args['transform'] = transform
            if 'vit_image_transform_args' in dataset_args.keys():
                vit_transform = ImageTransform(**dataset_args.pop('vit_image_transform_args'))
                dataset_args['vit_transform'] = vit_transform

            assert 'dataset_names' in dataset_args.keys()
            dataset_names = dataset_args.pop('dataset_names')
            dataset_args['data_dir_list'] = []
            for item in dataset_names:
                if self.local_rank == 0:
                    print(f'Preparing Dataset {grouped_dataset_name}/{item}')
                meta_info = DATASET_INFO[grouped_dataset_name][item]
                dataset_args['data_dir_list'].append(meta_info['data_dir'])

                if "parquet_info_path" in meta_info.keys():
                    if 'parquet_info' not in dataset_args.keys():
                        dataset_args['parquet_info'] = {}
                    with open(meta_info['parquet_info_path'], 'r') as f:
                        parquet_info = json.load(f)
                    dataset_args['parquet_info'].update(parquet_info)

                if 'json_dir' in meta_info.keys():
                    # parquet/tar with json
                    if 'json_dir_list' not in dataset_args.keys():
                        dataset_args['json_dir_list'] = [meta_info['json_dir']]
                    else:
                        dataset_args['json_dir_list'].append(meta_info['json_dir'])

                if 'jsonl_path' in meta_info.keys():
                    # jsonl with jpeg
                    if 'jsonl_path_list' not in dataset_args.keys():
                        dataset_args['jsonl_path_list'] = [meta_info['jsonl_path']]
                    else:
                        dataset_args['jsonl_path_list'].append(meta_info['jsonl_path'])

            resume_data_status = dataset_args.pop('resume_data_status', True)
            if data_status is not None and grouped_dataset_name in data_status.keys() and resume_data_status:
                data_status_per_group = data_status[grouped_dataset_name]
            else:
                data_status_per_group = None
            dataset = DATASET_REGISTRY[grouped_dataset_name](
                dataset_name=grouped_dataset_name,
                tokenizer=self.tokenizer,
                local_rank=self.local_rank,
                world_size=self.world_size,
                num_workers=self.num_workers,
                data_status=data_status_per_group,
                **dataset_args
            )
            datasets.append(dataset)

        return datasets, is_mandatory, grouped_weights, is_auxiliary

    def set_epoch(self, seed):
        for dataset in self.grouped_datasets:
            dataset.set_epoch(seed)
        
    def add_noise(self, input_ids, eps=1e-3, always_mask_last=False):
        l = len(input_ids)
        t = random.random()
        p_mask = (1 - eps) * t + eps
        masked_indices = [random.random() < p_mask for _ in range(l)]
        
        # If the last token always needs to be masked
        if always_mask_last and l > 0:
            masked_indices[-1] = True
        
        # fill [MASK] token
        noisy_input_ids = [self.masked_token_id if mask else token_id 
                           for mask, token_id in zip(masked_indices, input_ids)]
        return noisy_input_ids, masked_indices, p_mask

    def set_sequence_status(self):
        sequence_status = dict(
            curr                        = 0,
            sample_lens                 = list(),
            packed_position_ids         = list(),
            nested_attention_masks      = list(),
            split_lens                  = list(),
            attn_modes                  = list(),
            packed_text_ids             = list(), 
            packed_text_indexes         = list(),
            packed_label_ids            = list(),
            ce_loss_indexes             = list(),
            ce_loss_weights             = list(),
            vae_image_tensors           = list(), 
            packed_latent_position_ids  = list(),
            vae_latent_shapes           = list(), 
            packed_vae_token_indexes    = list(), 
            packed_timesteps            = list(), 
            mse_loss_indexes            = list(),
            packed_vit_tokens           = list(), 
            vit_token_seqlens           = list(),
            packed_vit_position_ids     = list(),
            packed_vit_token_indexes    = list(), 
        )
        return sequence_status

    def to_tensor(self, sequence_status):
        data = dict(
            sequence_length=sum(sequence_status['sample_lens']),
            sample_lens=sequence_status['sample_lens'],
            packed_text_ids=torch.tensor(sequence_status['packed_text_ids']),
            packed_text_indexes=torch.tensor(sequence_status['packed_text_indexes']),
            packed_position_ids=torch.tensor(sequence_status['packed_position_ids']),
        )
        if not self.use_flex:
            data['nested_attention_masks'] = sequence_status['nested_attention_masks']
        else:
            sequence_len = data['sequence_length']
            if sequence_len > self.max_num_tokens:
                reconstructed_masks = []
                all_splits = sequence_status['split_lens']
                all_modes = sequence_status['attn_modes']
                
                cursor = 0
                for sample_len in sequence_status['sample_lens']:
                    sample_specific_splits = []
                    current_sum = 0
                    start_index = cursor
                    while current_sum < sample_len:
                        current_sum += all_splits[cursor]
                        cursor += 1
                    
                    sample_specific_splits = all_splits[start_index:cursor]
                    sample_specific_modes = all_modes[start_index:cursor]

                    attention_mask = prepare_attention_mask_per_sample(
                        sample_specific_splits,
                        sample_specific_modes
                    )
                    reconstructed_masks.append(attention_mask)
                
                data['nested_attention_masks'] = reconstructed_masks

            else:
                pad_len = self.max_num_tokens - sequence_len
                if pad_len < 0: pad_len = 0
                    
                data['split_lens'] = sequence_status['split_lens'] + [pad_len]
                data['attn_modes'] = sequence_status['attn_modes'] + ['full'] # from causal to full for dllm
                data['sample_lens'] += [pad_len]

        # if the model has a convnet vae (e.g., as visual tokenizer)
        if len(sequence_status['vae_image_tensors']) > 0:
            image_tensors = sequence_status.pop('vae_image_tensors')
            image_sizes = [item.shape for item in image_tensors]
            max_image_size = [max(item) for item in list(zip(*image_sizes))]
            padded_images = torch.zeros(size=(len(image_tensors), *max_image_size))
            for i, image_tensor in enumerate(image_tensors):
                padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

            data['padded_images'] = padded_images
            data['patchified_vae_latent_shapes'] = sequence_status['vae_latent_shapes']
            data['packed_latent_position_ids'] = torch.cat(sequence_status['packed_latent_position_ids'], dim=0)
            data['packed_vae_token_indexes'] = torch.tensor(sequence_status['packed_vae_token_indexes'])

        # if the model has a vit (e.g., as visual tokenizer)
        if len(sequence_status['packed_vit_tokens']) > 0:
            data['packed_vit_tokens'] = torch.cat(sequence_status['packed_vit_tokens'], dim=0)
            data['packed_vit_position_ids'] = torch.cat(sequence_status['packed_vit_position_ids'], dim=0)
            data['packed_vit_token_indexes'] = torch.tensor(sequence_status['packed_vit_token_indexes'])
            data['vit_token_seqlens'] = torch.tensor(sequence_status['vit_token_seqlens'])

        # if the model is required to perform visual generation
        if len(sequence_status['packed_timesteps']) > 0:
            data['packed_timesteps'] = torch.tensor(sequence_status['packed_timesteps'])
            data['mse_loss_indexes'] = torch.tensor(sequence_status['mse_loss_indexes'])

        # if the model is required to perform text generation
        if len(sequence_status['packed_label_ids']) > 0:
            data['packed_label_ids'] = torch.tensor(sequence_status['packed_label_ids'])
            data['ce_loss_indexes'] = torch.tensor(sequence_status['ce_loss_indexes'])
            data['ce_loss_weights'] = torch.tensor(sequence_status['ce_loss_weights'])

        return data

    def next_power_of_2_strict(self, n):
        # Logic: Get the bit length of n, then left shift by 1 to get the next strictly greater power of 2
        # For example, the next power of 2 greater than 5 is 8
        if n <= 0: return 1
        return 1 << n.bit_length()

    def determine_split(self, total_length, min_size=16, max_size=512):
        """
        Input: Total length
        Output: Selected number of chunks K, and the size of each chunk
        """
        
        # 1. Find all valid K values (must be powers of 2)
        #    Constraint: min_size <= total_length / K <= max_size
        #    Rearranging => total_length / max_size <= K <= total_length / min_size

        # Handle short sequences: if total length is less than min block size, return without splitting
        if total_length <= min_size:
            return 1, [total_length]
        
        min_k = total_length / max_size
        max_k = total_length / min_size
        
        valid_k_list = []
        
        # Iterate through possible powers of 2: 1, 2, 4, 8, ...
        # Actually just need to try from 1 to max_k
        k = 1
        while k <= max_k:
            if k >= min_k:
                valid_k_list.append(k)
            k *= 2  # Next power of 2
        
        if not valid_k_list:
            raise ValueError(f"Length {total_length} cannot satisfy the split constraint [{min_size}, {max_size}]")

        # 2. Calculate raw weights (1/2K)
        #    Note: No need to manually normalize to 1, random.choices will handle relative weights
        weights = [1 / (2 * k) for k in valid_k_list]
        
        # 3. Randomly select a K based on weights
        #    k=1 weight=0.5, k=2 weight=0.25... larger K is harder to be selected
        selected_k = random.choices(valid_k_list, weights=weights, k=1)[0]
        
        # 4. Calculate specific chunk sizes (handling non-divisible cases)
        base_size = total_length // selected_k
        remainder = total_length % selected_k
        
        # Generate chunk list
        chunks = []
        for i in range(selected_k):
            # Distribute remainder to first few chunks to ensure total length is preserved
            size = base_size + (1 if i < remainder else 0)
            chunks.append(size)
            
        return selected_k, chunks

    def __iter__(self):
        total_weights = sum(self.grouped_weights)
        assert total_weights > 0.0
        group_cumprobs = [sum(self.grouped_weights[:i + 1]) / total_weights 
                          for i in range(len(self.grouped_weights))]
        sequence_status = self.set_sequence_status()
        batch_data_indexes = []

        buffer = []
        while True:
            # Ensure at least one sample from each group
            if sequence_status['curr'] == 0:
                for group_index, group_iter in enumerate(self.dataset_iters):
                    if self.is_mandatory[group_index]:
                        while True:
                            sample = next(group_iter)
                            # if a sample is too long, skip it
                            num_tokens = sample['num_tokens'] + 2 * len(sample['sequence_plan'])
                            if num_tokens < self.max_num_tokens_per_sample:
                                sequence_status = self.pack_sequence(sample, sequence_status)
                                batch_data_indexes.append(sample['data_indexes'])
                                break
                            else:
                                #print(f"skip a sample with length {num_tokens}")
                                continue

            if sequence_status['curr'] < self.prefer_buffer_before and len(buffer) > 0:
                sample = buffer.pop(0)
                sample_from_buffer = True
            else:
                # sample normally across all groups
                n = random.random()
                group_index = 0
                for i, cumprob in enumerate(group_cumprobs):
                    if n < cumprob:
                        group_index = i
                        break
                sample = next(self.dataset_iters[group_index])
                sample_from_buffer = False

            # if a sample is too long, skip it
            num_tokens = sample['num_tokens'] + 2 * len(sample['sequence_plan'])
            if num_tokens > self.max_num_tokens_per_sample:
                # print(f"skip a sample with length {num_tokens}")
                continue

            if sequence_status['curr'] + num_tokens > self.max_num_tokens:
                if len(buffer) < self.max_buffer_size and not sample_from_buffer:
                    buffer.append(sample)
                else:
                    #print(f"Yielding data with length {sum(sequence_status['sample_lens'])}")
                    data = self.to_tensor(sequence_status)
                    data['batch_data_indexes'] = batch_data_indexes
                    yield data
                    sequence_status = self.set_sequence_status()
                    batch_data_indexes = []
                continue

            sequence_status = self.pack_sequence(sample, sequence_status)
            batch_data_indexes.append(sample['data_indexes'])

            if sequence_status['curr'] >= self.expected_num_tokens:
                data = self.to_tensor(sequence_status)
                data['batch_data_indexes'] = batch_data_indexes
                yield data
                sequence_status = self.set_sequence_status()
                batch_data_indexes = []

    def pack_sequence(self, sample, sequence_status):
        image_tensor_list = sample['image_tensor_list']
        text_ids_list = sample['text_ids_list']
        sequence_plan = sample['sequence_plan']

        split_lens, attn_modes = list(), list()
        curr = sequence_status['curr']
        curr_rope_id = 0
        sample_lens = 0

        for item in sequence_plan:
            split_start = item.get('split_start', True)
            if split_start:
                curr_split_len = 0

            if item['type'] == 'text':
                text_ids = text_ids_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.text_cond_dropout_prob:
                    continue
            
                if self.data_config.visual_und_sft:
                    if item['loss'] == 1:
                        # use multimodal mdm sft loss 
                        # new version different from llada v. 
                        # Each round of conversation has both noisy and clean versions. The next round cannot see
                        # the noisy version of previous rounds. Each round also randomly adds some eos prediction targets.
                        # For single-turn dialogues, there's a 25% probability to add eos targets with length 0 to len(text_ids).
                        # Otherwise, the probability is 10%.
                        text_ids = text_ids + [self.eos_token_id]
                        is_single_turn = (item.get('round', 0) == 1) # Whether it's a single-turn dialogue
                        eos_prob = 0.15 if is_single_turn else 0.1 # For single-turn dialogues, probability of adding eos target is 0.15, otherwise 0.1
                        if random.random() < eos_prob:
                            num_eos_to_add = random.randint(0, len(text_ids))
                            pad_text_ids = text_ids + [self.eos_token_id] * num_eos_to_add # add some eos target
                        else:
                            pad_text_ids = text_ids # No additional eos target
                        while True:
                            masked_text_ids, masked_indices, p_mask = self.add_noise(pad_text_ids, always_mask_last=self.data_config.visual_und_always_mask_last) # padding and masking
                            if sum(masked_indices) > 0:
                                break
                        noisy_text_ids = [self.bos_token_id] + masked_text_ids # add padded and masked text
                        
                        # Calculate absolute position indices where masked_indices is True. Use list comprehension
                        # to filter indices with True values and add the current offset curr
                        target_indices = [curr + i + 1 for i, is_masked in enumerate(masked_indices) if is_masked] # BOS is not predicted, only mask token positions are predicted
                        sequence_status['ce_loss_indexes'].extend(target_indices)

                        target_labels = [token for token, is_masked in zip(pad_text_ids, masked_indices) if is_masked]
                        sequence_status['packed_label_ids'].extend(target_labels)

                        num_masked = sum(masked_indices)
                        weight = len2weight(len(masked_text_ids), self.data_config.loss_reduction) / p_mask
                        sequence_status['ce_loss_weights'].extend([weight] * num_masked)
                        
                        sequence_status['packed_text_ids'].extend(noisy_text_ids)
                        sequence_status['packed_text_indexes'].extend(range(curr, curr + len(noisy_text_ids)))
                        curr += len(noisy_text_ids)
                        attn_modes.append("noise")
                        sequence_status['packed_position_ids'].extend(range(curr_rope_id, curr_rope_id + len(noisy_text_ids)))
                        
                        # Only multi-turn dialogues need clean version for subsequent rounds
                        if not is_single_turn:
                            clean_text_ids = [self.bos_token_id] + text_ids # add clean text
                            sequence_status['packed_text_ids'].extend(clean_text_ids)
                            sequence_status['packed_text_indexes'].extend(range(curr, curr + len(clean_text_ids)))
                            curr += len(clean_text_ids)
                            attn_modes.append("full")
                            sequence_status['packed_position_ids'].extend(range(curr_rope_id, curr_rope_id + len(clean_text_ids)))
                            curr_split_len = [len(noisy_text_ids), len(clean_text_ids)]
                            curr_rope_id += len(clean_text_ids)
                        else:
                            # Single-turn dialogue doesn't need clean version
                            curr_split_len = [len(noisy_text_ids)]
                            curr_rope_id += len(noisy_text_ids)

                    elif item['loss'] == 0:
                        shifted_text_ids = [self.bos_token_id] + text_ids + [self.eos_token_id]
                        sequence_status['packed_text_ids'].extend(shifted_text_ids)
                        sequence_status['packed_text_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                        curr += len(shifted_text_ids)
                        curr_split_len += len(shifted_text_ids)

                        # update sequence status
                        attn_modes.append("full")
                        sequence_status['packed_position_ids'].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                        curr_rope_id += curr_split_len
                else:
                    # use mdm pretrained loss
                    ada_len_split_pad = False
                    if item['loss'] == 1: 
                        # should use mdm loss
                        # llada v version each response both have chance to be masked
                        text_ids = text_ids + [self.eos_token_id]
                        if self.data_config.ada_len:
                            # --- Adaptive Length (AdaLen) Strategy ---
                            # Initialize augmented_text_ids with the original list (Identity case)
                            augmented_text_ids = text_ids
                            rand_val = random.random()

                            # 1. EOS Injection (Extension) - Prob: 0.1
                            if rand_val < 0.1:
                                # Randomly add k in [1, current_len] EOS tokens
                                num_to_add = random.randint(1, len(text_ids))
                                augmented_text_ids = text_ids + [self.eos_token_id] * num_to_add

                            # 2. Random Truncation - Prob: 0.1 (accumulated < 0.2)
                            elif rand_val < 0.2:
                                # Only truncate if sequence is long enough (> 16)
                                if len(text_ids) > 16:
                                    keep_len = random.randint(1, len(text_ids) - 1)
                                    augmented_text_ids = text_ids[:keep_len]
                            
                            # Apply the augmentation result
                            text_ids = augmented_text_ids

                        elif self.data_config.ada_len_split:
                            augmented_text_ids = text_ids
                            rand_val = random.random()

                            if rand_val < 0.2:
                                # num_to_add = self.next_power_of_2_strict(len(text_ids) + 1) - len(text_ids) - 1 # -1 because have bos_token
                                current_len = len(text_ids) + 1 # + 1 because have bos_token
                                next_pow2 = self.next_power_of_2_strict(current_len)
                                threshold = next_pow2 * 3 // 4
                                target_len = next_pow2 if current_len > threshold else threshold
                                num_to_add = target_len - current_len
                                augmented_text_ids = text_ids + [self.eos_token_id] * num_to_add
                                ada_len_split_pad = True
                            
                            text_ids = augmented_text_ids

                        while True:
                            masked_text_ids, masked_indices, p_mask = self.add_noise(text_ids, always_mask_last=self.data_config.visual_und_always_mask_last)
                            if sum(masked_indices) > 0:
                                break
                        shifted_text_ids = [self.bos_token_id] + masked_text_ids
                        # Calculate absolute position indices where masked_indices is True. Use list comprehension
                        # to filter indices with True values and add the current offset curr
                        target_indices = [curr + i + 1 for i, is_masked in enumerate(masked_indices) if is_masked] # BOS token is not predicted, only mask token positions are predicted
                        sequence_status['ce_loss_indexes'].extend(target_indices)
                        num_masked = sum(masked_indices)
                        weight = len2weight(len(masked_text_ids), self.data_config.loss_reduction) / p_mask
                        sequence_status['ce_loss_weights'].extend([weight] * num_masked)
                        target_labels = [
                            token for token, is_masked in zip(text_ids, masked_indices) if is_masked
                        ]
                        sequence_status['packed_label_ids'].extend(target_labels)
                    elif item['loss'] == 0:
                        shifted_text_ids = [self.bos_token_id] + text_ids +[self.eos_token_id]
                    else:
                        raise ValueError(f"Invalid loss type: {item['loss']}")

                    sequence_status['packed_text_ids'].extend(shifted_text_ids)
                    sequence_status['packed_text_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                    curr += len(shifted_text_ids)
                    curr_split_len += len(shifted_text_ids)

                    # update sequence status
                    if self.data_config.ada_len_split and ada_len_split_pad and item['loss'] == 1:
                        # If ada_len_split is selected and also padded to power of 2 (0.2 probability)
                        # This is the noisy training text.
                        ada_len_split_k, ada_len_split_chunks = self.determine_split(len(shifted_text_ids))
                        for _ in range(ada_len_split_k):
                            attn_modes.append("full")
                    else:
                        attn_modes.append("full")
                    sequence_status['packed_position_ids'].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                    curr_rope_id += curr_split_len
                    if self.data_config.ada_len_split and ada_len_split_pad and item['loss'] == 1:
                        curr_split_len = ada_len_split_chunks

            elif item['type'] == 'vit_image':
                image_tensor = image_tensor_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.vit_cond_dropout_prob:
                    curr_rope_id += 1
                    continue

                # add a <|startofimage|> token
                sequence_status['packed_text_ids'].append(self.start_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                # preprocess image
                vit_tokens = patchify(image_tensor, self.data_config.vit_patch_size)
                num_img_tokens = vit_tokens.shape[0]
                sequence_status['packed_vit_token_indexes'].extend(range(curr, curr + num_img_tokens))
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                sequence_status['packed_vit_tokens'].append(vit_tokens)
                sequence_status['vit_token_seqlens'].append(num_img_tokens)
                sequence_status['packed_vit_position_ids'].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1), image_tensor.size(2),
                        self.data_config.vit_patch_size, 
                        max_num_patches_per_side=self.data_config.max_num_patch_per_side
                    )
                )

                # add a <|endofimage|> token
                sequence_status['packed_text_ids'].append(self.end_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                if item['special_token_loss'] == 1: # <|endofimage|> may have loss
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # update sequence status
                attn_modes.append("full")
                sequence_status['packed_position_ids'].extend([curr_rope_id] * curr_split_len)
                curr_rope_id += 1

            elif item['type'] == 'vae_image':
                image_tensor = image_tensor_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.vae_cond_dropout_prob:
                    # FIXME fix vae dropout in video2video setting.
                    curr_rope_id += 1
                    continue

                # add a <|startofimage|> token
                sequence_status['packed_text_ids'].append(self.start_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                # preprocess image
                sequence_status['vae_image_tensors'].append(image_tensor)
                sequence_status['packed_latent_position_ids'].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1), image_tensor.size(2),
                        self.data_config.vae_image_downsample, 
                        max_num_patches_per_side=self.data_config.max_latent_size
                    )
                )
                H, W = image_tensor.shape[1:]
                h = H // self.data_config.vae_image_downsample
                w = W // self.data_config.vae_image_downsample
                sequence_status['vae_latent_shapes'].append((h, w))

                if self.data_config.visual_gen_reg:
                    num_img_tokens = w * h + 1 # add 1 for reg token
                else:
                    num_img_tokens = w * h
                
                sequence_status['packed_vae_token_indexes'].extend(range(curr, curr + num_img_tokens))
                if item['loss'] == 1:
                    sequence_status['mse_loss_indexes'].extend(range(curr, curr + num_img_tokens))
                    if split_start:
                        timestep = np.random.randn()
                else:
                    timestep = float('-inf')
                if self.data_config.visual_gen_reg:
                    sequence_status['packed_timesteps'].extend([timestep] * (num_img_tokens -1)) # we do not need to add the cls token for timestep
                else: 
                    sequence_status['packed_timesteps'].extend([timestep] * num_img_tokens)
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                # add a <|endofimage|> token
                sequence_status['packed_text_ids'].append(self.end_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                # <|endofimage|> may have loss
                if item['special_token_loss'] == 1:
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # update sequence status
                if split_start:
                    if item['loss'] == 1 and 'frame_delta' not in item.keys():
                        attn_modes.append("noise")
                    else:
                        attn_modes.append("full")
                sequence_status['packed_position_ids'].extend([curr_rope_id] * (num_img_tokens + 2))
                if 'frame_delta' in item.keys():
                    curr_rope_id += item['frame_delta']
                elif item['loss'] == 0:
                    curr_rope_id += 1

            if item.get('split_end', True):
                if isinstance(curr_split_len, list):
                    # If it's a split list (e.g., [16, 16, 16, 16])
                    split_lens.extend(curr_split_len)      # Record the length of each split chunk
                    sample_lens += sum(curr_split_len)     # Accumulate total length
                else:
                    split_lens.append(curr_split_len)
                    sample_lens += curr_split_len

        sequence_status['curr'] = curr
        sequence_status['sample_lens'].append(sample_lens)
        if getattr(self.data_config, 'merge_vit_text_segments', False):
            if all(item['type'] in ['text', 'vit_image'] for item in sequence_plan) and split_lens:
                # Merge all segment lengths into a single total, following LLaDA-V's approach. For pure text or vit+text mixed input
                total_len = sum(split_lens)
                split_lens = [total_len]
                # Use unified attention mode (example uses 'full')
                attn_modes = ['full']

        # prepare attention mask
        if not self.use_flex:
            sequence_status['nested_attention_masks'].append(
                prepare_attention_mask_per_sample(split_lens, attn_modes)
            )
        else:
            sequence_status['split_lens'].extend(split_lens)
            sequence_status['attn_modes'].extend(attn_modes)

        return sequence_status


class SimpleCustomBatch:
    def __init__(self, batch):
        data = batch[0]
        self.batch_data_indexes = data['batch_data_indexes']
        self.sequence_length = data["sequence_length"]
        self.sample_lens = data["sample_lens"]
        self.packed_text_ids = data["packed_text_ids"]
        self.packed_text_indexes = data["packed_text_indexes"]
        self.packed_position_ids = data["packed_position_ids"]

        self.use_flex = "nested_attention_masks" not in data.keys()

        if self.use_flex:
            self.split_lens = data["split_lens"]
            self.attn_modes = data["attn_modes"]
        else:
            self.nested_attention_masks = data["nested_attention_masks"]

        if "padded_images" in data.keys():
            self.padded_images = data["padded_images"]
            self.patchified_vae_latent_shapes = data["patchified_vae_latent_shapes"]
            self.packed_latent_position_ids = data["packed_latent_position_ids"]
            self.packed_vae_token_indexes = data["packed_vae_token_indexes"]

        if "packed_vit_tokens" in data.keys():
            self.packed_vit_tokens = data["packed_vit_tokens"]
            self.packed_vit_position_ids = data["packed_vit_position_ids"]
            self.packed_vit_token_indexes = data["packed_vit_token_indexes"]
            self.vit_token_seqlens = data["vit_token_seqlens"]

        if "packed_timesteps" in data.keys():
            self.packed_timesteps = data["packed_timesteps"]
            self.mse_loss_indexes = data["mse_loss_indexes"]

        if "packed_label_ids" in data.keys():
            self.packed_label_ids = data["packed_label_ids"]
            self.ce_loss_indexes = data["ce_loss_indexes"]
            self.ce_loss_weights = data["ce_loss_weights"]

    def pin_memory(self):
        self.packed_text_ids = self.packed_text_ids.pin_memory()
        self.packed_text_indexes = self.packed_text_indexes.pin_memory()
        self.packed_position_ids = self.packed_position_ids.pin_memory()

        if not self.use_flex:
            self.nested_attention_masks = [item.pin_memory() for item in self.nested_attention_masks]

        if hasattr(self, 'padded_images'):
            self.padded_images = self.padded_images.pin_memory()
            self.packed_vae_token_indexes = self.packed_vae_token_indexes.pin_memory()
            self.packed_latent_position_ids = self.packed_latent_position_ids.pin_memory()

        if hasattr(self, 'packed_timesteps'):
            self.packed_timesteps = self.packed_timesteps.pin_memory()
            self.mse_loss_indexes = self.mse_loss_indexes.pin_memory()

        if hasattr(self, 'packed_vit_tokens'):
            self.packed_vit_tokens = self.packed_vit_tokens.pin_memory()
            self.packed_vit_position_ids = self.packed_vit_position_ids.pin_memory()
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.pin_memory()
            self.vit_token_seqlens = self.vit_token_seqlens.pin_memory()

        if hasattr(self, 'packed_label_ids'):
            self.packed_label_ids = self.packed_label_ids.pin_memory()
            self.ce_loss_indexes = self.ce_loss_indexes.pin_memory()
            self.ce_loss_weights = self.ce_loss_weights.pin_memory()

        return self

    def cuda(self, device):
        self.packed_text_ids = self.packed_text_ids.to(device)
        self.packed_text_indexes = self.packed_text_indexes.to(device)
        self.packed_position_ids = self.packed_position_ids.to(device)

        if not self.use_flex:
            self.nested_attention_masks = [item.to(device) for item in self.nested_attention_masks]

        if hasattr(self, 'padded_images'):
            self.padded_images = self.padded_images.to(device)
            self.packed_vae_token_indexes = self.packed_vae_token_indexes.to(device)
            self.packed_latent_position_ids = self.packed_latent_position_ids.to(device)

        if hasattr(self, 'packed_timesteps'):
            self.packed_timesteps = self.packed_timesteps.to(device)
            self.mse_loss_indexes = self.mse_loss_indexes.to(device)

        if hasattr(self, 'packed_vit_tokens'):
            self.packed_vit_tokens = self.packed_vit_tokens.to(device)
            self.packed_vit_position_ids = self.packed_vit_position_ids.to(device)
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.to(device)
            self.vit_token_seqlens = self.vit_token_seqlens.to(device)

        if hasattr(self, 'packed_label_ids'):
            self.packed_label_ids = self.packed_label_ids.to(device)
            self.ce_loss_indexes = self.ce_loss_indexes.to(device)
            self.ce_loss_weights = self.ce_loss_weights.to(device)

        return self

    def to_dict(self):
        data = dict(
            sequence_length = self.sequence_length,
            sample_lens = self.sample_lens,
            packed_text_ids = self.packed_text_ids,
            packed_text_indexes = self.packed_text_indexes,
            packed_position_ids = self.packed_position_ids,
            batch_data_indexes = self.batch_data_indexes,
        )

        if not self.use_flex:
            data['nested_attention_masks'] = self.nested_attention_masks
        else:
            data['split_lens'] = self.split_lens
            data['attn_modes'] = self.attn_modes

        if hasattr(self, 'padded_images'):
            data['padded_images'] = self.padded_images
            data['patchified_vae_latent_shapes'] = self.patchified_vae_latent_shapes
            data['packed_latent_position_ids'] = self.packed_latent_position_ids
            data['packed_vae_token_indexes'] = self.packed_vae_token_indexes

        if hasattr(self, 'packed_vit_tokens'):
            data['packed_vit_tokens'] = self.packed_vit_tokens
            data['packed_vit_position_ids'] = self.packed_vit_position_ids
            data['packed_vit_token_indexes'] = self.packed_vit_token_indexes
            data['vit_token_seqlens'] = self.vit_token_seqlens

        if hasattr(self, 'packed_timesteps'):
            data['packed_timesteps'] = self.packed_timesteps
            data['mse_loss_indexes'] = self.mse_loss_indexes

        if hasattr(self, 'packed_label_ids'):
            data['packed_label_ids'] = self.packed_label_ids
            data['ce_loss_indexes'] = self.ce_loss_indexes
            data['ce_loss_weights'] = self.ce_loss_weights

        return data


def collate_wrapper():
    def collate_fn(batch):
        return SimpleCustomBatch(batch)
    return collate_fn
