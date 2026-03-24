# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os

from .interleave_datasets import UnifiedEditIterableDataset, UnifiedEditWebdatasetIterableDataset
from .t2i_dataset import T2IIterableDataset
from .t2i_wds_dataset import T2IWdsIterableDataset
from .vlm_wds_dataset import SftVLMWdsIterableDataset
from .vlm_parquet_dataset import SftVLMParIterableDataset
from .wds_dataset import SftWdsIterableDataset


DATASET_REGISTRY = {
    't2i_wds': T2IWdsIterableDataset,
    't2i_parquet': T2IIterableDataset,
    'vlm_wds': SftVLMWdsIterableDataset,
    'vlm_parquet': SftVLMParIterableDataset,
    'llm_wds': SftWdsIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'unified_wds_edit': UnifiedEditWebdatasetIterableDataset
}

LOCAL_HF_DATA_ROOT = os.environ.get("LLADAO_DATA_ROOT", "/path/to/local/huggingface_datasets")
T2I_2M_DIR = os.environ.get(
    "LLADAO_T2I_2M_DIR",
    os.path.join(LOCAL_HF_DATA_ROOT, "text-to-image-2M"),
)
VLM_BEE_DIR = os.environ.get(
    "LLADAO_VLM_BEE_DIR",
    os.path.join(LOCAL_HF_DATA_ROOT, "Honey-Data-15M"),
)

DATASET_INFO = {
    't2i_wds': {
        't2i_2m':{
            'data_dir': T2I_2M_DIR,
        }, # https://huggingface.co/datasets/jackyhate/text-to-image-2M
    },
    'vlm_parquet': {
        'vlm_bee': {
            'data_dir': VLM_BEE_DIR,
        }, # https://huggingface.co/datasets/Open-Bee/Honey-Data-15M
    },
}
