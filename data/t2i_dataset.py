# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import pyarrow.parquet as pq
import random
from PIL import Image

from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
from .parquet_utils import get_parquet_data_paths, init_arrow_pf_fs

Image.MAX_IMAGE_PIXELS = 20_000_000


class T2IIterableDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, data_dir_list, num_used_data, 
        local_rank=0, world_size=1, num_workers=8, data_status=None,
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data)
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data):
        return get_parquet_data_paths(data_dir_list, num_used_data)

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
                            try:
                                image_byte = row['image']['bytes']
                                image = pil_img2rgb(Image.open(io.BytesIO(image_byte)))
                            except Exception as e:
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue
                            image_tensor = self.transform(image)
                            height, width = image_tensor.shape[1:]
                            num_tokens += width * height // transform_stride ** 2

                            # First check if there is a separate caption key
                            valid_captions = []
                            if 'caption' in row:
                                # If caption key exists and is valid, use it directly
                                if row['caption'] is not None and str(row['caption']).strip():
                                    valid_captions.append(str(row['caption']).strip())
                                else:
                                    # print(f'Invalid caption in rg#{row_group_id}, {parquet_file_path}')
                                    continue
                            else:
                                # Otherwise, define caption columns to process
                                caption_columns_to_process = [
                                    'caption_composition', 'caption_composition_cn',
                                    'caption_entity', 'caption_entity_cn', 
                                    'caption_text', 'caption_text_cn',
                                    'caption_imaginative', 'caption_imaginative_cn',
                                    'caption_style', 'caption_style_cn',
                                    'caption_abstract', 'caption_abstract_cn',
                                    'caption_detail', 'caption_detail_cn',
                                    'caption_original', 'caption_original_cn',
                                ]
                                # Collect all valid captions
                                for column_name in caption_columns_to_process:
                                    try:
                                        caption_value = row.get(column_name)
                                        # Check if caption exists and is not empty
                                        if caption_value is not None and str(caption_value).strip():
                                            valid_captions.append(str(caption_value).strip())
                                    except Exception as e:
                                        print(f'Caption processing warning {column_name}: {e} in rg#{row_group_id}, {parquet_file_path}')
                                        continue
                            # Tokenize captions
                            if len(valid_captions) == 0:
                                print(f'No valid caption found in rg#{row_group_id}, {parquet_file_path}')
                                caption_token = self.tokenizer.encode(' ')  # Use space as default token
                            else:
                                # Randomly select one from valid captions
                                selected_caption = random.choice(valid_captions)
                                caption_token = self.tokenizer.encode(selected_caption)

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

                            sample = dict(
                                image_tensor_list=[image_tensor], 
                                text_ids_list=text_ids_list,
                                num_tokens=num_tokens,
                                sequence_plan=sequence_plan,
                                data_indexes={
                                    "data_indexes": [parquet_idx, row_group_id, row_idx],
                                    "worker_id": worker_id,
                                    "dataset_name": self.dataset_name,
                                }
                            )
                            yield sample

                        row_start_id = 0
                    row_group_start_id = 0
            parquet_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
