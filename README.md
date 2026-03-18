# LLaDA-o

Official implementation of **LLaDA-o**, an effective and length-adaptive omni diffusion model for unified multimodal generation and understanding.

[Paper](https://arxiv.org/abs/2603.01068) | [Model](https://huggingface.co/GSAI-ML/LLaDA-o)

## Introduction

LLaDA-o extends diffusion language modeling to a unified multimodal setting in which text and visual signals are represented and processed within a shared generative framework. The repository is designed to support both multimodal inference and training, with an emphasis on interleaved reasoning over language and images. In the current release, the codebase includes:

- A reusable multimodal inference pipeline in [`demo_pipeline.py`](./demo_pipeline.py)
- An interactive notebook, [`multimodal_demo.ipynb`](./multimodal_demo.ipynb), for end-to-end inference
- Core modeling components for LLaDA-o under [`modeling/`](./modeling)
- Dataset and preprocessing utilities under [`data/`](./data)
- A training entry point in [`train/pretrain_unified_navit.py`](./train/pretrain_unified_navit.py)

The provided inference workflow is centered on a single shared model instance that can be reused across multiple multimodal tasks. In particular, the notebook demonstrates:

- Text-to-image generation
- Image understanding
- Image editing
- Batch text-to-image generation from [`prompt.txt`](./prompt.txt)

## Highlights

- **Unified multimodal inference**: the same demo pipeline supports generation and understanding within one interface.
- **Reproducible notebook workflow**: [`multimodal_demo.ipynb`](./multimodal_demo.ipynb) offers a self-contained inference entry point with explicit configuration cells and saved outputs.
- **Local checkpoint loading**: the pipeline loads LLaDA-o from a local directory, making it straightforward to use checkpoints downloaded from Hugging Face.
- **Training-ready codebase**: the repository includes data modules, model definitions, and a pretraining script for further experimentation.

## Installation

We recommend creating a dedicated Python environment before installing dependencies.

```bash
git clone https://github.com/GSAI-ML/LLaDA-o.git
cd LLaDA-o
pip install -r requirements.txt
pip install --upgrade pyarrow
pip install webdataset
pip install transformers==4.56.2
```

The same setup steps are also recorded in [`init_env.sh`](./init_env.sh).

## Checkpoint Preparation

For inference, first download the released model checkpoint from Hugging Face:

- [GSAI-ML/LLaDA-o](https://huggingface.co/GSAI-ML/LLaDA-o)

After downloading, place the checkpoint in a local directory. The inference pipeline expects the directory to contain the LLaDA-o configuration files, tokenizer assets, the VAE checkpoint, and the sharded model index. In particular, the following files are required by [`demo_pipeline.py`](./demo_pipeline.py):

```text
<LOCAL_MODEL_PATH>/
|-- ae.safetensors
|-- ema.safetensors.index.json
|-- llm_config.json
|-- vit_config.json
|-- tokenizer.json / tokenizer.model / tokenizer_config.json
`-- shard files referenced by ema.safetensors.index.json
```

If `ema.safetensors.index.json` is missing, model loading will fail at initialization time.

## Inference with `multimodal_demo.ipynb`

The recommended way to run inference in this repository is through [`multimodal_demo.ipynb`](./multimodal_demo.ipynb). The notebook provides a unified and reproducible workflow for the main multimodal capabilities currently exposed by the codebase.

### 1. Launch Jupyter

From the repository root:

```bash
jupyter notebook
```

Then open [`multimodal_demo.ipynb`](./multimodal_demo.ipynb).

### 2. Set the local model path

In the configuration cell, replace `MODEL_PATH` with the local path of the downloaded Hugging Face checkpoint:

```python
MODEL_PATH = os.environ.get("LLADAO_MODEL_PATH", "/path/to/local/GSAI-ML-LLaDA-o")
```

You may either:

- edit `MODEL_PATH` directly in the notebook, or
- set the environment variable `LLADAO_MODEL_PATH`

The notebook will print a reminder if the placeholder path has not been changed.

### 3. Run the notebook sequentially

The notebook is organized into four practical stages:

1. **Load Model**: initializes `LLaDAMultimodalDemo.from_pretrained(...)` from the local checkpoint directory.
2. **Text-to-Image**: generates a reference image from a textual prompt.
3. **Image Understanding**: uses the generated image as input and produces a textual description.
4. **Image Editing**: edits the reference image according to a new instruction while preserving its overall visual identity.

An additional section supports **batch text-to-image generation** from [`prompt.txt`](./prompt.txt), saving outputs to `demo_outputs/batch_text_to_image/`.

### 4. Outputs

By default, notebook outputs are written to `demo_outputs/`, including:

- `01_text_to_image.png`
- `02_understanding.txt`
- `03_image_edit.png`
- batched images under `demo_outputs/batch_text_to_image/`

### 5. Practical notes

- The notebook should be launched from the repository root, or from a directory named `LLaDA-o` that contains the repository files.
- The current inference path requires at least one CUDA-capable GPU.
- `MAX_MEM_PER_GPU` and `OFFLOAD_DIR` can be adjusted in the notebook if you need to tune memory placement during checkpoint dispatch.
- If you would like to use your own image for understanding or editing, the notebook supports replacing the generated reference image with `load_image("/absolute/path/to/image.png")`.

## Python API

For scripted usage, the notebook workflow is backed by the reusable `LLaDAMultimodalDemo` interface in [`demo_pipeline.py`](./demo_pipeline.py).

```python
from demo_pipeline import LLaDAMultimodalDemo

demo = LLaDAMultimodalDemo.from_pretrained(
    model_path="/path/to/local/GSAI-ML-LLaDA-o",
    max_mem_per_gpu="40GiB",
    offload_dir="/tmp/lladao_offload",
)

result = demo.text_to_image("A studio-quality product photo of a glass teapot shaped like a tiny planet.")
image = result["image"]
image.save("sample.png")
```

The same interface also exposes:

- `demo.understand(image, prompt, **kwargs)`
- `demo.edit_image(image, prompt, **kwargs)`

This makes it straightforward to migrate from notebook-based experimentation to Python-based evaluation scripts.

## Repository Structure

```text
LLaDA-o/
|-- demo_pipeline.py
|-- inferencer.py
|-- multimodal_demo.ipynb
|-- prompt.txt
|-- data/
|-- modeling/
`-- train/
```

- [`demo_pipeline.py`](./demo_pipeline.py): high-level inference wrapper and default task configurations.
- [`inferencer.py`](./inferencer.py): interleaved multimodal inference logic for text and images.
- [`data/`](./data): dataset definitions, transforms, parquet/webdataset utilities, and interleaved dataset support.
- [`modeling/`](./modeling): model definitions for LLaDA, LLaDA-o, SigLIP-based vision components, and the autoencoder.
- [`train/`](./train): distributed training utilities and the main pretraining script.

## Citation

If you find this repository useful in your research, please consider citing:

```bibtex
@article{you2026lladao,
  title={LLaDA-o: An Effective and Length-Adaptive Omni Diffusion Model},
  author={You, Zebin and Zhang, Xiaolu and Zhou, Jun and Li, Chongxuan and Wen, Ji-Rong},
  journal={arXiv preprint arXiv:2603.01068},
  year={2026}
}
```
