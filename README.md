# Clarity-Former

[English Version](#english-version)

## 中文说明

**Clarity-Former: Tackling Image Degradation and Geometric Diversity in Underwater Instance Segmentation** 的官方实现。

Clarity-Former 是一个用于水下实例分割的一阶段框架。本发布版已按照论文中的正式命名整理框架名、模块名、配置项和文件放置关系。论文与代码的对应关系见 [`CLARITY_FORMER_CODE_MAPPING.md`](CLARITY_FORMER_CODE_MAPPING.md)。

## 安装

代码已在 Python 3.9、PyTorch 1.9.1、torchvision 0.10.1、CUDA 11.1、Detectron2 0.6 和 Linux/NVIDIA GPU 环境下验证。

创建环境并安装依赖：

```shell
conda create -n clarity_former python=3.9 -y
conda activate clarity_former
conda install pytorch==1.9.1 torchvision==0.10.1 cudatoolkit=11.1 -c pytorch -c nvidia -y
python -m pip install -e detectron2
python setup.py build develop
```

修改源码或将项目移动到新机器后，重新编译项目扩展：

```shell
conda activate clarity_former
cd /path/to/Clarity-Former
python setup.py build develop
```

如果 CUDA 扩展编译失败，请先确认 `nvcc`、PyTorch CUDA 和本地 CUDA toolkit 的版本兼容，然后重新执行 `python setup.py build develop`。

## 数据集

论文实验使用 UIIS、USIS10K 和 USIS16K。数据集注册代码位于 [`adet/data/datasets/underwater_instance.py`](adet/data/datasets/underwater_instance.py)。

数据集来源：

- UIIS: [LiamLian0727/WaterMask](https://github.com/LiamLian0727/WaterMask)
- USIS10K: [LiamLian0727/USIS10K](https://github.com/LiamLian0727/USIS10K)
- USIS16K: [LinHong-HIT/USIS16K](https://github.com/LinHong-HIT/USIS16K)

如果使用本发布版整理的数据集压缩包，请解压后将项目指向解压后的数据集根目录：

```shell
mkdir -p /path/to/datasets
# 将压缩包路径替换为实际下载位置。
tar -xzf /path/to/ClarityFormer_UIIS_USIS10K_USIS16K.tar.gz -C /path/to/datasets
export CLARITY_FORMER_DATASETS_ROOT=/path/to/datasets
```

解压后的目录结构应为：

```text
/path/to/datasets/
  UIIS/
  USIS10K/
  USIS16K/
```

已注册的数据集划分：

| 数据集 | 训练划分 | 验证/测试划分 | 类别数 |
|---|---|---|---|
| UIIS | `uiis_train` | `uiis_val` | 7 |
| USIS10K | `usis10k_train` | `usis10k_val`, `usis10k_test` | 7 |
| USIS16K | `usis16k_train` | `usis16k_val` | 158 |

`underwater_train` 和 `underwater_test` 保留为 `usis10k_train` 与 `usis10k_test` 的别名，因此默认配置可以直接用于 USIS10K。

如果数据集不在同一个根目录下，也可以分别指定各数据集路径：

```shell
export CLARITY_FORMER_UIIS_ROOT=/path/to/UIIS
export CLARITY_FORMER_USIS10K_ROOT=/path/to/USIS10K
export CLARITY_FORMER_USIS16K_ROOT=/path/to/USIS16K
```

请确保以下配置项与所选数据集的类别数一致：

- `MODEL.CLARITY_FORMER.NUM_CLASSES`
- `MODEL.BASIS_MODULE.NUM_CLASSES`
- `MODEL.SEM_SEG_HEAD.NUM_CLASSES`

## 配置文件

| 模型变体 | 配置文件 |
|---|---|
| Clarity-Former-T | `configs/ClarityFormer_SWINT.yaml` |
| Clarity-Former-S | `configs/ClarityFormer_SWINS.yaml` |
| Clarity-Former-B | `configs/ClarityFormer_SWINB.yaml` |

## 权重

发布的 Clarity-Former 权重以 `pretrained.tar.gz` 提供：

- 百度网盘: [pretrained.tar.gz](https://pan.baidu.com/s/1fQ94UC5yvKFOKFJM1JpMYw?pwd=56md)
- 提取码: `56md`

将压缩包解压到项目根目录，使目录结构成为 `pretrained/ClarityFormer/` 和 `pretrained/SwinTransformer/`。

```shell
tar -xzf pretrained.tar.gz -C /path/to/Clarity-Former
```

Clarity-Former 发布权重用于 USIS16K 复现实验。

| 模型变体 | Backbone | 配置文件 | 权重文件 |
|---|---|---|---|
| Clarity-Former-T | Swin-T | `pretrained/ClarityFormer/tiny/config.yaml` | `pretrained/ClarityFormer/tiny/model_tiny.pth` |
| Clarity-Former-S | Swin-S | `pretrained/ClarityFormer/small/config.yaml` | `pretrained/ClarityFormer/small/model_small.pth` |
| Clarity-Former-B | Swin-B | `pretrained/ClarityFormer/base/config.yaml` | `pretrained/ClarityFormer/base/model_base.pth` |

使用匹配的配置文件评估发布权重：

```shell
PYTHONWARNINGS="ignore" python tools/train_net.py \
  --config-file pretrained/ClarityFormer/small/config.yaml \
  --eval-only
```

Swin ImageNet 初始化权重位于 `pretrained/SwinTransformer/`，已经是 Detectron2 key 格式，可直接用于 `configs/` 中的训练或微调配置：

| Backbone | 初始化权重 |
|---|---|
| Swin-T | `pretrained/SwinTransformer/swin_tiny_patch4_window7_224_detectron2.pth` |
| Swin-S | `pretrained/SwinTransformer/swin_small_patch4_window7_224_detectron2.pth` |
| Swin-B | `pretrained/SwinTransformer/swin_base_patch4_window7_224_detectron2.pth` |

如果目标数据集类别数与 USIS16K 发布权重不同，请使用对应的 Swin backbone 初始化权重，而不是直接加载发布的 Clarity-Former 权重。

## 训练

使用 Swin-S 初始化权重在 USIS10K 上训练：

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

在 UIIS 上训练时切换数据集划分：

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

在 USIS16K 上训练或微调：

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

如果需要从头训练，请传入 `MODEL.WEIGHTS ""`。

## 评估

评估发布的 USIS16K 权重：

```shell
PYTHONWARNINGS="ignore" python tools/train_net.py \
  --config-file pretrained/ClarityFormer/small/config.yaml \
  --eval-only
```

在 USIS10K 上评估自定义权重：

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

## 快速检查

安装后可以运行两次迭代的训练 smoke test：

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

## 工具

测试运行时间或数据加载：

```shell
python tools/benchmark-det2.py \
  --config-file configs/ClarityFormer_SWINS.yaml \
  --task eval \
  --num-gpus 1 \
  MODEL.WEIGHTS /path/to/model_final.pth
```

估算 FLOPs：

```shell
python tools/get_flops_det2.py \
  --config-file configs/ClarityFormer_SWINS.yaml \
  --shape 800 1333 \
  MODEL.WEIGHTS ""
```

可视化已注册数据：

```shell
python tools/visualize_data.py \
  --config-file configs/ClarityFormer_SWINS.yaml \
  --source annotation \
  --output-dir output/vis_data
```

## 致谢

本项目基于 Detectron2、AdelaiDet、Deformable DETR、Swin Transformer 和 OSFormer 构建。

## 引用

如果本代码对你的研究有帮助，请引用 Clarity-Former 论文：

```bibtex
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
```

<a id="english-version"></a>

## English Version

[Back to Chinese Version](#clarity-former)

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

- Baidu Netdisk: [pretrained.tar.gz](https://pan.baidu.com/s/1fQ94UC5yvKFOKFJM1JpMYw?pwd=56md)
- Extraction code: `56md`

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

Swin ImageNet initialization weights are stored under `pretrained/SwinTransformer/`, already use the Detectron2 key format, and can be used directly for training or fine-tuning configs in `configs/`:

| Backbone | Initialization |
|---|---|
| Swin-T | `pretrained/SwinTransformer/swin_tiny_patch4_window7_224_detectron2.pth` |
| Swin-S | `pretrained/SwinTransformer/swin_small_patch4_window7_224_detectron2.pth` |
| Swin-B | `pretrained/SwinTransformer/swin_base_patch4_window7_224_detectron2.pth` |

For a dataset whose class count differs from the released USIS16K checkpoints, initialize from the matching Swin backbone weight instead of directly loading a released Clarity-Former checkpoint.

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

This project builds on Detectron2, AdelaiDet, Deformable DETR, Swin Transformer, and OSFormer.

## Citation

Please cite the Clarity-Former paper if this code is useful for your research.

```bibtex
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
```
