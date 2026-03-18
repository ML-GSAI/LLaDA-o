# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright 2025 AntGroup and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import pyarrow.parquet as pq
import random
import webdataset as wds
from PIL import Image

from .distributed_iterable_dataset import DistributedIterableDataset
from .webdata_utils import get_webdataset_paths
from .data_utils import pil_img2rgb

Image.MAX_IMAGE_PIXELS = 20_000_000


class T2IWdsIterableDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, data_dir_list, num_used_data=None, 
        local_rank=0, world_size=1, num_workers=8, data_status=None,
    ):
        """
        data_dir_list: list of data directories contains webdataset files
        num_used_data: list of number of sampled data paths for each data directory
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data)
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data):
        return get_webdataset_paths(data_dir_list, num_used_data)

    def process_images(self, sample):
        """Process images in the sample, keeping consistent with original logic"""
        image_keys = ['png', 'jpg', 'jpeg']
        
        for key in image_keys:
            if key in sample:
                value = sample[key]
                
                # Process according to type
                if isinstance(value, bytes):
                    pil_image = Image.open(io.BytesIO(value))
                elif isinstance(value, Image.Image):
                    pil_image = value
                else:
                    continue  # Skip unrecognized types
                
                sample[key] = pil_img2rgb(pil_image)
        
        return sample

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        
        # Handle resuming from checkpoint
        if self.data_status is not None:
            shard_start_id = self.data_status[worker_id][0]
            sample_start_id = self.data_status[worker_id][1] + 1
        else:
            shard_start_id = 0
            sample_start_id = 0
        
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at shard#{shard_start_id}, sample#{sample_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[shard_start_id:]
            
            for shard_idx, shard_path in enumerate(data_paths_per_worker_, start=shard_start_id):
                try:
                    # Create WebDataset pipeline - using wds.slice
                    dataset = wds.WebDataset(shard_path, nodesplitter=wds.split_by_worker, shardshuffle=False).decode('pil').map(self.process_images)

                    # If need to skip samples, use wds.slice
                    if sample_start_id > 0:
                        pipeline = dataset.compose(wds.slice(sample_start_id, None))
                    else:
                        pipeline = dataset
                    
                    sample_idx = sample_start_id
                    for wds_sample in pipeline:
                        try:
                            # Get image
                            if 'jpg' in wds_sample:
                                image = wds_sample['jpg']
                            elif 'png' in wds_sample:
                                image = wds_sample['png']
                            elif 'good_image.jpg' in wds_sample:
                                image = wds_sample['good_image.jpg']
                            else:
                                print(f'No image found in sample#{sample_idx}, {shard_path}')
                                sample_idx += 1
                                continue
                            
                            # Transform image
                            image_tensor = self.transform(image)
                            height, width = image_tensor.shape[1:]
                            num_tokens = width * height // transform_stride ** 2

                            # Get caption
                            try:
                                if 'txt' in wds_sample:
                                    caption_text = wds_sample['txt']
                                    caption_dict = {"caption": caption_text}
                                elif 'json' in wds_sample:
                                    caption_json = wds_sample['json']
                                    if 'prompt' in caption_json:
                                        caption_dict = {"caption": caption_json['prompt']}
                                    else:
                                        assert len(caption_json['conversations']) == 2, "Caption should have two parts"
                                        caption_dict = {"caption": caption_json['conversations'][1]['value']}
                                elif 'prompt.txt' in wds_sample:
                                    caption_text = wds_sample['prompt.txt']
                                    caption_dict = {"caption": caption_text}
                                else:
                                    print(f'No caption found in sample#{sample_idx}, {shard_path}')
                                    sample_idx += 1
                                    continue
                            except Exception as e:
                                print(f'Error parsing caption: {e} in sample#{sample_idx}, {shard_path}')
                                sample_idx += 1
                                continue

                            # Process caption tokenization
                            if isinstance(caption_dict, dict):
                                caps_token = [self.tokenizer.encode(str(v)) for _, v in caption_dict.items()]
                            else:
                                # If caption_dict is actually a string
                                caps_token = [self.tokenizer.encode(str(caption_dict))]
                            
                            if len(caps_token) == 0:
                                print(f'No valid caption in sample#{sample_idx}, {shard_path}')
                                caption_token = self.tokenizer.encode(' ')
                            else:
                                caption_token = random.choice(caps_token)

                            # Build sequence plan
                            sequence_plan, text_ids_list = [], []
                            text_ids = caption_token
                            num_tokens += len(caption_token)
                            text_ids_list.append(text_ids)
                            
                            sequence_plan.append({
                                'type': 'text',
                                'enable_cfg': 1,
                                'loss': 0,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })
                        
                            sequence_plan.append({
                                'type': 'vae_image',
                                'enable_cfg': 0,
                                'loss': 1,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

                            # Build final sample
                            sample = dict(
                                image_tensor_list=[image_tensor], 
                                text_ids_list=text_ids_list,
                                num_tokens=num_tokens,
                                sequence_plan=sequence_plan,
                                data_indexes={
                                    "data_indexes": [shard_idx, sample_idx],
                                    "worker_id": worker_id,
                                    "dataset_name": self.dataset_name,
                                }
                            )
                            yield sample
                            sample_idx += 1

                        except Exception as e:
                            print(f'Error processing sample#{sample_idx}: {e} in {shard_path}')
                            sample_idx += 1
                            continue
                    
                except Exception as e:
                    print(f'Error opening shard {shard_path}: {e}')
                    continue
                
                # Reset sample_start_id since the next shard starts from the beginning
                sample_start_id = 0
            
            # Reset shard_start_id to start a new epoch
            shard_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")