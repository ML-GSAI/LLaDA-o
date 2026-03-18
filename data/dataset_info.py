from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset
from .t2i_wds_dataset import T2IWdsIterableDataset
from .vlm_wds_dataset import SftVLMWdsIterableDataset
from .vlm_parquet_dataset import SftVLMParIterableDataset
from .wds_dataset import SftWdsIterableDataset

DATASET_REGISTRY = {
    't2i_wds': T2IWdsIterableDataset,
    'vlm_wds': SftWdsIterableDataset,
    'vlm_parquet': SftVLMParIterableDataset,
}


DATASET_INFO = {
    'vlm_wds': {
        'vlm_pretrain': {
            'data_dir': ''
        },
    },
}