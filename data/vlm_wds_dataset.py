# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright 2025 AntGroup and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import io
import re
import traceback
from PIL import Image, ImageFile, PngImagePlugin

import webdataset as wds
from .distributed_iterable_dataset import DistributedIterableDataset
from .webdata_utils import get_webdataset_paths
from .data_utils import pil_img2rgb

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

class SftVLMWdsIterableDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, 
        frame_sampler, data_dir_list, num_used_data=None, 
        local_rank=0, world_size=1, num_workers=8, data_status=None, 
    ):
        """
        data_dir_list: list of image directories containing the images of each jsonl file
        num_used_data: list of number of sampled data points for each jsonl
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.frame_sampler = frame_sampler
        self.data_status = data_status
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data)
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data):
        return get_webdataset_paths(data_dir_list, num_used_data)

    def change_format(self, data, num_images):
        elements = []
        rounds = len(data['conversations']) // 2
        # Step 1: Add all images first
        if num_images > 0:
            for _ in range(num_images):
                elements.append({'type': 'image'})
        
        for conversation in data['conversations']:
            if conversation['from'] == 'human':
                user_text = conversation['value'].replace('<image>', '').strip() # Remove all <image> tags
                
                if user_text:  # If there is still content
                    elements.append({
                        'type': 'text',
                        'has_loss': 0,
                        'text': user_text,
                        'round': rounds,
                    })
            elif conversation['from'] == 'gpt':
                elements.append({
                    'type': 'text',
                    'has_loss': 1,
                    'text': conversation['value'],
                    'round': rounds,
                })
        return elements

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
                    continue  # Skip unknown types
                
                sample[key] = pil_img2rgb(pil_image)
        
        return sample

    def _remove_empty_images(self, sample):
        image_keys = ["jpg", "jpeg", "png"]
        for k in image_keys:
            if k in sample:
                val = sample[k]
                # Delete this key if value is empty
                if val is None or (isinstance(val, bytes) and len(val) == 0):
                    del sample[k]
        return sample

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()

        # Handle checkpoint resume
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
                    # Create WebDataset pipeline - use wds.slice
                    dataset = wds.WebDataset(shard_path, nodesplitter=wds.split_by_worker, shardshuffle=False).map(self._remove_empty_images).decode('pil').map(self.process_images)

                    # If need to skip samples, use wds.slice
                    if sample_start_id > 0:
                        pipeline = dataset.compose(wds.slice(sample_start_id, None))
                    else:
                        pipeline = dataset

                    sample_idx = sample_start_id
                    for wds_sample in pipeline: 
                        num_tokens = 0
                        image_tensor_list = []
                        text_ids_list = []
                        sequence_plan = []

                        try:
                            # Get images, video is not supported yet
                            if 'jpg' in wds_sample:
                                image = wds_sample['jpg']  # Could be a list or a single image.
                            elif 'png' in wds_sample:
                                image = wds_sample['png']
                            else:
                                # print(f'No image found in sample#{sample_idx}, {shard_path}')
                                sample_idx += 1
                                continue

                            if type(image) != list:
                                image = [image]
                            
                            for img in image:
                                image_tensor = self.transform(img, img_num=len(image))
                                image_tensor_list.append(image_tensor)
                                height, width = image_tensor.shape[1:]
                                num_tokens += width * height // transform_stride ** 2
                            
                            elements = self.change_format(wds_sample['json'], len(image_tensor_list))
                        
                            for item in elements:
                                if item['type'] == 'text':
                                    text_data = item['text']
                                    text_ids = self.tokenizer.encode(text_data)
                                    if len(text_ids) > 0:
                                        text_ids_list.append(text_ids)
                                        num_tokens += len(text_ids)
                                        current_plan = {
                                            'type': 'text',
                                            'enable_cfg': 0,
                                            'loss': item['has_loss'],
                                            'special_token_loss': 0,
                                            'special_token_label': None,
                                            'round': item.get('round', 0),  # Default to 0 if no round field
                                        }
                                        sequence_plan.append(current_plan)
                                elif item['type'] == 'image':
                                    current_plan = {
                                        'type': 'vit_image',
                                        'enable_cfg': 0,
                                        'loss': 0,
                                        'special_token_loss': 0,
                                        'special_token_label': None,
                                    }
                                    sequence_plan.append(current_plan)

                            has_loss = [item['loss'] for item in sequence_plan]
                            if sum(has_loss) == 0:
                                print(f'No loss defined, skipped.')
                                sample_idx += 1
                                continue

                            yield dict(
                                image_tensor_list=image_tensor_list,
                                text_ids_list=text_ids_list,
                                sequence_plan=sequence_plan,
                                num_tokens=num_tokens,
                                data_indexes={
                                    "data_indexes": [shard_idx, sample_idx],
                                    "worker_id": worker_id,
                                    "dataset_name": self.dataset_name,
                                }
                            )
                            sample_idx += 1

                        except Exception as e:
                            print(f'Error processing sample#{sample_idx}: {e} in {shard_path}')
                            sample_idx += 1
                            continue

                except Exception as e:
                    print(f'Error opening shard {shard_path}: {e}')
                    continue
                
                # Reset sample_start_id, as next shard starts from beginning
                sample_start_id = 0
            
            # Reset shard_start_id, start new epoch
            shard_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
