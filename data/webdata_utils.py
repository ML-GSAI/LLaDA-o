# Copyright 2025 AntGroup and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import os
import subprocess
import logging

import torch.distributed as dist

logger = logging.getLogger(__name__)

def get_webdataset_paths(data_dir_list, num_sampled_data_paths=None, rank=0, world_size=1):
    num_data_dirs = len(data_dir_list)
    
    if world_size > 1:
        chunk_size = (num_data_dirs + world_size - 1) // world_size
        start_idx = rank * chunk_size
        end_idx = min(start_idx + chunk_size, num_data_dirs)
        local_data_dir_list = data_dir_list[start_idx:end_idx]
        
        if num_sampled_data_paths is not None:
            local_num_sampled_data_paths = num_sampled_data_paths[start_idx:end_idx]
        else:
            local_num_sampled_data_paths = None
    else:
        local_data_dir_list = data_dir_list
        local_num_sampled_data_paths = num_sampled_data_paths

    local_data_paths = []
    for i, data_dir in enumerate(local_data_dir_list):
        data_paths_per_dir = []
        for root, dirs, files in os.walk(data_dir):
            for name in files:
                if name.endswith(".tar"):
                    data_paths_per_dir.append(os.path.join(root, name))
        
        if local_num_sampled_data_paths is not None and local_num_sampled_data_paths[i] is not None and isinstance(local_num_sampled_data_paths[i], int):
            num_data_path = local_num_sampled_data_paths[i]
            if len(data_paths_per_dir) > 0:
                repeat = num_data_path // len(data_paths_per_dir)
                data_paths_per_dir = data_paths_per_dir * (repeat + 1)
                local_data_paths.extend(data_paths_per_dir[:num_data_path])
        else:
            local_data_paths.extend(data_paths_per_dir)

    if world_size > 1:
        gather_list = [None] * world_size
        dist.all_gather_object(gather_list, local_data_paths)

        combined_chunks = []
        for chunk_list in gather_list:
            if chunk_list is not None:
                combined_chunks.extend(chunk_list)
    else:
        combined_chunks = local_data_paths

    return combined_chunks