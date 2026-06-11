# Clarity-Former

Official implementation of **Clarity-Former: Tackling Image Degradation and Geometric Diversity in Underwater Instance Segmentation**.

Clarity-Former is a one-stage underwater instance segmentation framework. This release follows the paper naming for the framework, modules, configuration keys, and file layout. See [`CLARITY_FORMER_CODE_MAPPING.md`](CLARITY_FORMER_CODE_MAPPING.md) for the paper-to-code correspondence.

## Installation

The code has been validated with Python 3.9, PyTorch 1.9.1, torchvision 0.10.1, CUDA 11.1, Detectron2 0.6, and a Linux/NVIDIA GPU environment.

Create the environment and install dependencies:

```shell
conda create -n clarity_former python=3.9 -y
conda activate clarity_former
conda install pytorch==1.9.1 torchvision==0.10.1 cudatoolkit=11.1 -c pytorch -c nvidia -y
python -m pip install -e detectron2
python setup.py build develop
```

After source changes or when moving the project to a new machine, rebuild the project extensions:

```shell
conda activate clarity_former
cd /path/to/Clarity-Former
python setup.py build develop
```

If CUDA extensions fail to build, first confirm that `nvcc`, PyTorch CUDA, and the local CUDA toolkit are version-compatible, then rerun `python setup.py build develop`.

## Dataset

The paper experiments use UIIS, USIS10K, and USIS16K. The dataset registration is in [`adet/data/datasets/underwater_instance.py`](adet/data/datasets/underwater_instance.py).

Dataset sources:

- UIIS: [LiamLian0727/WaterMask](https://github.com/LiamLian0727/WaterMask)
- USIS10K: [LiamLian0727/USIS10K](https://github.com/LiamLian0727/USIS10K)
- USIS16K: [LinHong-HIT/USIS16K](https://github.com/LinHong-HIT/USIS16K)

If you use the dataset archive distributed with this release, unpack it and point the project to the extracted dataset root:

```shell
mkdir -p /path/to/datasets
# Replace the archive path with the location where you downloaded the dataset package.
tar -xzf /path/to/ClarityFormer_UIIS_USIS10K_USIS16K.tar.gz -C /path/to/datasets
export CLARITY_FORMER_DATASETS_ROOT=/path/to/datasets
```

The extracted layout should be:

```text
/path/to/datasets/
  UIIS/
  USIS10K/
  USIS16K/
```

Registered splits:

| Dataset | Train split | Val/Test split | Classes |
|---|---|---|---|
| UIIS | `uiis_train` | `uiis_val` | 7 |
| USIS10K | `usis10k_train` | `usis10k_val`, `usis10k_test` | 7 |
| USIS16K | `usis16k_train` | `usis16k_val` | 158 |

`underwater_train` and `underwater_test` are kept as aliases for `usis10k_train` and `usis10k_test`, so the default configs work out of the box with USIS10K.

For custom locations, either set the shared root above or override individual dataset roots:

```shell
export CLARITY_FORMER_UIIS_ROOT=/path/to/UIIS
export CLARITY_FORMER_USIS10K_ROOT=/path/to/USIS10K
export CLARITY_FORMER_USIS16K_ROOT=/path/to/USIS16K
```

Make sure these config values match the selected dataset class count:

- `MODEL.CLARITY_FORMER.NUM_CLASSES`
- `MODEL.BASIS_MODULE.NUM_CLASSES`
- `MODEL.SEM_SEG_HEAD.NUM_CLASSES`

## Configs

| Variant | Config |
|---|---|
| Clarity-Former-T | `configs/ClarityFormer_SWINT.yaml` |
| Clarity-Former-S | `configs/ClarityFormer_SWINS.yaml` |
| Clarity-Former-B | `configs/ClarityFormer_SWINB.yaml` |

## Weights

Released Clarity-Former checkpoints are provided as `pretrained.tar.gz`:

```text
Baidu Netdisk: https://pan.baidu.com/s/1fQ94UC5yvKFOKFJM1JpMYw?pwd=56md
Extraction code: 56md
```

Unpack the archive under the repository root so that the directory layout becomes `pretrained/ClarityFormer/` and `pretrained/SwinTransformer/`.

```shell
tar -xzf pretrained.tar.gz -C /path/to/Clarity-Former
```

The Clarity-Former checkpoints are configured for USIS16K reproduction.

| Variant | Backbone | Config | Checkpoint |
|---|---|---|---|
| Clarity-Former-T | Swin-T | `pretrained/ClarityFormer/tiny/config.yaml` | `pretrained/ClarityFormer/tiny/model_tiny.pth` |
| Clarity-Former-S | Swin-S | `pretrained/ClarityFormer/small/config.yaml` | `pretrained/ClarityFormer/small/model_small.pth` |
| Clarity-Former-B | Swin-B | `pretrained/ClarityFormer/base/config.yaml` | `pretrained/ClarityFormer/base/model_base.pth` |

Evaluate a released checkpoint with its matching config:

```shell
PYTHONWARNINGS="ignore" python tools/train_net.py \
  --config-file pretrained/ClarityFormer/small/config.yaml \
  --eval-only
```

Swin ImageNet initialization weights are stored under `pretrained/SwinTransformer/` and are intended for training or fine-tuning configs in `configs/`:

| Backbone | Initialization |
|---|---|
| Swin-T | `pretrained/SwinTransformer/swin_tiny_patch4_window7_224_detectron2.pth` |
| Swin-S | `pretrained/SwinTransformer/swin_small_patch4_window7_224_detectron2.pth` |
| Swin-B | `pretrained/SwinTransformer/swin_base_patch4_window7_224_detectron2.pth` |

For a dataset whose class count differs from the released USIS16K checkpoints, initialize from the matching Swin backbone weight instead of directly loading a released Clarity-Former checkpoint.

To convert an official Swin ImageNet checkpoint to the Detectron2 key format:

```shell
python tools/convert_swin_weights.py \
  --input /path/to/swin_small_patch4_window7_224.pth \
  --output pretrained/SwinTransformer/swin_small_patch4_window7_224_detectron2.pth
```

## Training

Train on USIS10K with Swin-S initialization:

```shell
PYTHONWARNINGS="ignore" python tools/train_net.py \
  --config-file configs/ClarityFormer_SWINS.yaml \
  --num-gpus 1 \
  MODEL.WEIGHTS pretrained/SwinTransformer/swin_small_patch4_window7_224_detectron2.pth \
  DATASETS.TRAIN "('usis10k_train',)" \
  DATASETS.TEST "('usis10k_test',)" \
  MODEL.CLARITY_FORMER.NUM_CLASSES 7 \
  MODEL.BASIS_MODULE.NUM_CLASSES 7 \
  MODEL.SEM_SEG_HEAD.NUM_CLASSES 7 \
  OUTPUT_DIR tools/output/ClarityFormer_SWINS_USIS10K
```

Train on UIIS by switching the dataset splits:

```shell
PYTHONWARNINGS="ignore" python tools/train_net.py \
  --config-file configs/ClarityFormer_SWINS.yaml \
  --num-gpus 1 \
  MODEL.WEIGHTS pretrained/SwinTransformer/swin_small_patch4_window7_224_detectron2.pth \
  DATASETS.TRAIN "('uiis_train',)" \
  DATASETS.TEST "('uiis_val',)" \
  MODEL.CLARITY_FORMER.NUM_CLASSES 7 \
  MODEL.BASIS_MODULE.NUM_CLASSES 7 \
  MODEL.SEM_SEG_HEAD.NUM_CLASSES 7 \
  OUTPUT_DIR tools/output/ClarityFormer_SWINS_UIIS
```

Train or fine-tune on USIS16K with the released label space:

```shell
PYTHONWARNINGS="ignore" python tools/train_net.py \
  --config-file configs/ClarityFormer_SWINS.yaml \
  --num-gpus 1 \
  MODEL.WEIGHTS pretrained/SwinTransformer/swin_small_patch4_window7_224_detectron2.pth \
  DATASETS.TRAIN "('usis16k_train',)" \
  DATASETS.TEST "('usis16k_val',)" \
  MODEL.CLARITY_FORMER.NUM_CLASSES 158 \
  MODEL.BASIS_MODULE.NUM_CLASSES 158 \
  MODEL.SEM_SEG_HEAD.NUM_CLASSES 158 \
  OUTPUT_DIR tools/output/ClarityFormer_SWINS_USIS16K
```

For training from scratch, pass `MODEL.WEIGHTS ""`.

## Evaluation

Evaluate a released USIS16K checkpoint:

```shell
PYTHONWARNINGS="ignore" python tools/train_net.py \
  --config-file pretrained/ClarityFormer/small/config.yaml \
  --eval-only
```

Evaluate a custom checkpoint on USIS10K:

```shell
PYTHONWARNINGS="ignore" python tools/train_net.py \
  --config-file configs/ClarityFormer_SWINS.yaml \
  --eval-only \
  MODEL.WEIGHTS /path/to/model_final.pth \
  DATASETS.TEST "('usis10k_test',)" \
  MODEL.CLARITY_FORMER.NUM_CLASSES 7 \
  MODEL.BASIS_MODULE.NUM_CLASSES 7 \
  MODEL.SEM_SEG_HEAD.NUM_CLASSES 7
```

## Smoke Test

Run a short two-iteration training smoke test after installation:

```shell
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS="ignore" python tools/train_net.py \
  --config-file configs/ClarityFormer_SWINS.yaml \
  --num-gpus 1 \
  MODEL.WEIGHTS "" \
  SOLVER.MAX_ITER 2 \
  SOLVER.IMS_PER_BATCH 1 \
  SOLVER.CHECKPOINT_PERIOD 1000000 \
  TEST.EVAL_PERIOD 0 \
  DATASETS.TEST "()" \
  DATALOADER.NUM_WORKERS 1 \
  OUTPUT_DIR /tmp/clarity_former_train_smoke
```

## Utilities

Benchmark runtime or data loading:

```shell
python tools/benchmark-det2.py \
  --config-file configs/ClarityFormer_SWINS.yaml \
  --task eval \
  --num-gpus 1 \
  MODEL.WEIGHTS /path/to/model_final.pth
```

Estimate FLOPs:

```shell
python tools/get_flops_det2.py \
  --config-file configs/ClarityFormer_SWINS.yaml \
  --shape 800 1333 \
  MODEL.WEIGHTS ""
```

Visualize registered data:

```shell
python tools/visualize_data.py \
  --config-file configs/ClarityFormer_SWINS.yaml \
  --source annotation \
  --output-dir output/vis_data
```

## Acknowledgements

This project builds on Detectron2, AdelaiDet, Deformable DETR, and Swin Transformer.

## Citation

Please cite the Clarity-Former paper if this code is useful for your research. 

@ARTICLE{11450346,
  author={Chen, Houru and Feng, Li and Zhao, Qinglin and Sun, Yi},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Clarity-Former: Tackling Image Degradation and Geometric Diversity in Underwater Instance Segmentation}, 
  year={2026},
  volume={64},
  number={},
  pages={1-14},
  keywords={Transformers;Degradation;Feature extraction;Instance segmentation;Visualization;Shape;Computer architecture;Standards;Semantics;Image color analysis;Attention mechanisms;instance segmentation;Transformer;underwater image processing},
  doi={10.1109/TGRS.2026.3676605}}

