# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright 2025 AntGroup and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import pyarrow.parquet as pq

from ..distributed_iterable_dataset import DistributedIterableDataset
from ..parquet_utils import get_parquet_data_paths, init_arrow_pf_fs
from ..webdata_utils import get_webdataset_paths
import webdataset as wds
from PIL import Image
from ..data_utils import pil_img2rgb
import io

class InterleavedBaseIterableDataset(DistributedIterableDataset):

    def _init_data(self):
        data = {
            'sequence_plan': [],
            'text_ids_list': [],
            'image_tensor_list': [],
            'num_tokens': 0,
        }
        return data

    def _add_text(self, data, text, need_loss, enable_cfg=True):
        text_ids = self.tokenizer.encode(text)
        data['num_tokens'] += len(text_ids)
        data['text_ids_list'].append(text_ids)
        data['sequence_plan'].append(
            {
                'type': 'text',
                'enable_cfg': int(enable_cfg),
                'loss': int(need_loss),
                'special_token_loss': 0,
                'special_token_label': None,
            }
        )
        return data

    def _add_image(self, data, image, need_loss, need_vae, need_vit, enable_cfg=True):
        assert need_loss or need_vae or need_vit

        if need_loss:
            data['sequence_plan'].append(
                {
                    'type': 'vae_image', 
                    'enable_cfg': 0, 
                    'loss': 1, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                }
            )

            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor)

        if need_vae:
            data['sequence_plan'].append(
                {
                    'type': 'vae_image', 
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                }
            )

            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor.clone())

        if need_vit:
            data['sequence_plan'].append(
                {
                    'type': 'vit_image',
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                },
            )
            vit_image_tensor = self.vit_transform(image)
            height, width = vit_image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.vit_transform.stride ** 2
            data['image_tensor_list'].append(vit_image_tensor)

        return data

    def _add_video(self, data, frames, frame_indexes, need_loss, need_vae, enable_cfg=True):
        assert int(need_loss) + int(need_vae) == 1

        if need_loss:
            for idx, (image, frame_idx) in enumerate(zip(frames, frame_indexes)):
                current_sequence_plan = {
                    'type': 'vae_image', 
                    'enable_cfg': 0, 
                    'loss': 1, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'split_start': idx == 0,
                    'split_end': idx == len(frames) - 1,
                }
                if idx < len(frame_indexes) - 1:
                    current_sequence_plan['frame_delta'] = frame_indexes[idx + 1] - frame_idx
                data['sequence_plan'].append(current_sequence_plan)
                image_tensor = self.transform(image)
                height, width = image_tensor.shape[1:]
                data['image_tensor_list'].append(image_tensor)
                data['num_tokens'] += width * height // self.transform.stride ** 2

        elif need_vae:
            for idx, (image, frame_idx) in enumerate(zip(frames, frame_indexes)):
                current_sequence_plan = {
                    'type': 'vae_image', 
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'split_start': idx == 0,
                    'split_end': idx == len(frames) - 1,
                }
                if idx < len(frame_indexes) - 1:
                    current_sequence_plan['frame_delta'] = frame_indexes[idx + 1] - frame_idx
                data['sequence_plan'].append(current_sequence_plan)
                image_tensor = self.transform(image)
                height, width = image_tensor.shape[1:]
                data['image_tensor_list'].append(image_tensor)
                data['num_tokens'] += width * height // self.transform.stride ** 2

        return data


class ParquetStandardIterableDataset(DistributedIterableDataset):

    def __init__(
        self, dataset_name, transform, tokenizer, vit_transform, 
        data_dir_list, num_used_data,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        vit_transform: input transform for vit model.
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data)
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data):
        return get_parquet_data_paths(data_dir_list, num_used_data)

    def parse_row(self, row):
        raise NotImplementedError

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
                            try:
                                data = self.parse_row(row)
                                if len(data) == 0:
                                    continue
                                data['data_indexes'] = {
                                    "data_indexes": [parquet_idx, row_group_id, row_idx],
                                    "worker_id": worker_id,
                                    "dataset_name": self.dataset_name,
                                }
                            except Exception as e:
                                print(f'Error {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue
                            yield data

                        row_start_id = 0
                    row_group_start_id = 0
            parquet_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
class WebdatasetStandardIterableDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, vit_transform, 
        data_dir_list, num_used_data,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        vit_transform: input transform for vit model.
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data)
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data):
        return get_webdataset_paths(data_dir_list, num_used_data)
    def process_images(self, sample):
        """Process images in the sample, supporting keys with image format extensions"""
        image_extensions = ['png', 'jpg', 'jpeg']
        
        # Iterate through all keys in the sample
        for key in list(sample.keys()):  # Use list() to avoid modifying dict during iteration
            # Check if the key contains any image format extension
            if any(ext in key.lower() for ext in image_extensions):
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

    def parse_sample(self, sample):
        raise NotImplementedError

    def rename_editing_keys(self, sample):
        """Rename keys for editing data"""
        new_sample = {}
        
        for key, value in sample.items():
            if key == 'src_img':
                new_sample['src_img.jpg'] = value
            elif key == 'edited_img':
                new_sample['edited_img.jpg'] = value
            elif key == 'edited_prompt':
                new_sample['edited_prompt.txt'] = value
            else:
                new_sample[key] = value  # Keep other keys unchanged
        
        return new_sample
    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        
        # Handle resuming from checkpoint
        if self.data_status is not None:
            shard_start_id = self.data_status[worker_id][0]
            sample_start_id = self.data_status[worker_id][1] + 1
        else:
            shard_start_id = 0
            sample_start_id = 0
        
        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at shard#{shard_start_id}, sample#{sample_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[shard_start_id:]
            
            for shard_idx, shard_path in enumerate(data_paths_per_worker_, start=shard_start_id):
                try:
                    # Create WebDataset pipeline - using wds.slice
                    dataset = wds.WebDataset(shard_path, nodesplitter=wds.split_by_worker, shardshuffle=False).map(self.rename_editing_keys).decode('pil').map(self.process_images)

                    # If need to skip samples, use wds.slice
                    if sample_start_id > 0:
                        pipeline = dataset.compose(wds.slice(sample_start_id, None))
                    else:
                        pipeline = dataset
                    
                    sample_idx = sample_start_id
                    for wds_sample in pipeline:
                        try:
                            data = self.parse_sample(wds_sample)
                            if len(data) == 0:
                                sample_idx += 1
                                continue 
                            data['data_indexes'] = {
                                "data_indexes": [shard_idx, sample_idx],
                                "worker_id": worker_id,
                                "dataset_name": self.dataset_name,
                            }
                            yield data
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
