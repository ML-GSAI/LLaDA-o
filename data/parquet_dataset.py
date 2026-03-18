# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright 2025 AntGroup and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import io
import re
import traceback
from PIL import Image, ImageFile, PngImagePlugin

import pyarrow.parquet as pq
from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
from .parquet_utils import get_parquet_data_paths, init_arrow_pf_fs

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

class SftParIterableDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, 
        frame_sampler, data_dir_list, num_used_data=None, 
        local_rank=0, world_size=1, num_workers=8, data_status=None, 
    ):
        """
        data_dir_list: list of image directories containing the images of each parquet file
        num_used_data: list of number of sampled data points for each parquet
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.frame_sampler = frame_sampler
        self.data_status = data_status
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data)
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data):
        return get_parquet_data_paths(data_dir_list, num_used_data)

    def change_format(self, data, num_images):
        """
        Convert multiple conversation formats to the internal elements list format.
        
        Supported formats:
        1. [{'user': '...', 'assistant': '...'}]
        2. [{'from': 'human/gpt', 'value': '...'}]
        3. [{'role': 'user/assistant', 'content': '...'}]
        
        If the first user input does not contain <image>, add corresponding number of <image> tags
        at the beginning based on num_images.

        Args:
            data (list): Conversation list containing dictionaries in various formats.
            num_images (int): Number of images included in the conversation.

        Returns:
            list: Converted elements list.
        """
        elements = []
        if len(data) > 0 and 'user' in data[0] and data[0].get('user') is not None:
            rounds = len(data)
        else:
            rounds = len(data) // 2
        
        # Step 1: Add all images first
        if num_images > 0:
            for _ in range(num_images):
                elements.append({'type': 'image'})

        # Iterate through each conversation turn in the list
        for conversation_turn in data:
            # Extract user and assistant text content uniformly
            user_text = None
            assistant_text = None
            
            # Format 1: {'user': '...', 'assistant': '...'}
            if 'user' in conversation_turn and conversation_turn.get('user') is not None:
                user_text = conversation_turn['user']
                assistant_text = conversation_turn.get('assistant', '')
            
            # Format 2: {'from': 'human/gpt', 'value': '...'}
            elif 'from' in conversation_turn and conversation_turn.get('from') is not None:
                if conversation_turn['from'] in ['human', 'user']:
                    user_text = conversation_turn.get('value', '')
                elif conversation_turn['from'] in ['gpt', 'assistant']:
                    assistant_text = conversation_turn.get('value', '')
            
            # Format 3: {'role': 'user/assistant', 'content': '...'}
            elif 'role' in conversation_turn and conversation_turn.get('role') is not None:
                if conversation_turn['role'] == 'user':
                    user_text = conversation_turn.get('content', '')
                elif conversation_turn['role'] == 'assistant':
                    assistant_text = conversation_turn.get('content', '')
            
            # Process user input
            if user_text:
                # Remove all <image> tags
                user_text = user_text.replace('<image>', '').strip()
                if user_text:  # If there is still content
                    elements.append({
                        'type': 'text',
                        'has_loss': 0,
                        'text': user_text,
                        'round': rounds,
                    })
            # Process assistant output
            if assistant_text:
                elements.append({
                    'type': 'text',
                    'has_loss': 1,
                    'text': assistant_text,
                    'round': rounds,
                })
        
        return elements

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            parquet_start_id = self.data_status[worker_id][0]
            row_group_start_id = self.data_status[worker_id][1]
            row_start_id = self.data_status[worker_id][2] + 1
        else:
            parquet_start_id = 0
            row_group_start_id = 0
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at parquet#{parquet_start_id}, rg#{row_group_start_id}, row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            for parquet_idx, parquet_file_path in enumerate(data_paths_per_worker_, start=parquet_start_id):
                try: 
                    fs = init_arrow_pf_fs(parquet_file_path)
                    with fs.open_input_file(parquet_file_path) as f:
                        fr = pq.ParquetFile(f)
                        row_group_ids = list(range(fr.num_row_groups))
                        row_group_ids_ = row_group_ids[row_group_start_id:]

                        for row_group_id in row_group_ids_:
                            df = fr.read_row_group(row_group_id).to_pandas()
                            df = df.iloc[row_start_id:]

                            for row_idx, row in df.iterrows():
                                num_tokens = 0
                                image_tensor_list = []
                                text_ids_list = []
                                sequence_plan = []

                                try: 
                                    num_images = 0                        
                                    # Process images: support both 'images' (list) and 'image' (single) formats
                                    if 'images' in row and row['images'] is not None and len(row['images']) > 0:
                                        # Format 1: images list
                                        images = row['images']
                                        num_images = len(images)
                                        for image in images:  # images is a list, each image have a bytes field
                                            rgb_image = pil_img2rgb(Image.open(io.BytesIO(image['bytes']))) # transfer to rgb image
                                            image_tensor = self.transform(rgb_image, img_num=num_images)
                                            image_tensor_list.append(image_tensor)
                                            height, width = image_tensor.shape[1:]
                                            num_tokens += width * height // transform_stride ** 2
                                    
                                    elif 'image' in row and row['image'] is not None:
                                        # Format 2: single image dict
                                        image = row['image']
                                        num_images = 1
                                        rgb_image = pil_img2rgb(Image.open(io.BytesIO(image['bytes'])))  # transfer to rgb image
                                        image_tensor = self.transform(rgb_image, img_num=num_images)
                                        image_tensor_list.append(image_tensor)
                                        height, width = image_tensor.shape[1:]
                                        num_tokens += width * height // transform_stride ** 2
                                    
                                    
                                    # Process text: support both 'texts' and 'conversations' keys
                                    if 'texts' in row:
                                        elements = self.change_format(row['texts'], num_images)
                                    elif 'conversations' in row:
                                        elements = self.change_format(row['conversations'], num_images)
                                    else:
                                        # No conversation text, skip this sample
                                        continue

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
                                                    'round': item.get('round', 0),  # If no round field, default to 0
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
                                        continue

                                    yield dict(
                                        image_tensor_list=image_tensor_list,
                                        text_ids_list=text_ids_list,
                                        sequence_plan=sequence_plan,
                                        num_tokens=num_tokens,
                                        data_indexes={
                                            "data_indexes": [parquet_idx, row_group_id, row_idx],
                                            "worker_id": worker_id,
                                            "dataset_name": self.dataset_name,
                                        }
                                    )

                                except Exception as e:
                                    print(f'Error processing sample#{row_group_id} #{row_idx}: {e} in {parquet_file_path}')
                                    continue
                            row_start_id = 0
                        row_group_start_id = 0
                except Exception as e:
                    print(f'Error opening parquet {parquet_file_path}: {e}')
                    continue
            parquet_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")