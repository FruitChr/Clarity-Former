# Clarity-Former Code Mapping

This document maps the official paper terminology to the cleaned Project1 codebase.

Paper: **Clarity-Former: Tackling Image Degradation and Geometric Diversity in Underwater Instance Segmentation**

## Public Naming

| Scope | Public name |
|---|---|
| Framework / Detectron2 meta architecture | `ClarityFormer` |
| Config node | `MODEL.CLARITY_FORMER` |
| Python package | `adet.modeling.clarity_former` |
| Config prefix | `configs/ClarityFormer_*.yaml` |

## Paper Module to Code Mapping

| Paper component | Code component | Main files | Notes |
|---|---|---|---|
| Clarity-Former one-stage framework | `ClarityFormer` | `adet/modeling/clarity_former/clarity_former.py` | Registered as Detectron2 meta architecture `ClarityFormer`. |
| SAMA Transformer backbone | `WindowAttentionSAMA`, `SwinTransformerBlock` | `adet/modeling/backbone/swin.py` | Implements learnable temperature and diagonal attention masking. |
| Adaptive temperature in SAMA | `self.tau` | `adet/modeling/backbone/swin.py` | Scales query features before attention score computation. |
| Diagonal masking in SAMA | diagonal attention mask | `adet/modeling/backbone/swin.py` | Suppresses self-token attention by adding a large negative value on the attention diagonal. |
| Encoder-decoder module | `ClarityFormerTransformerHead` | `adet/modeling/clarity_former/clarity_former.py` | Builds the deformable encoder and decoder. |
| Deformable transformer encoder | `ClarityFormerTransformerEncoder` | `adet/modeling/clarity_former/trans_encoder.py` | Multiscale deformable self-attention over backbone features. |
| Deformable transformer decoder | `ClarityFormerTransformerDecoder` | `adet/modeling/clarity_former/trans_decoder.py` | Cross-level deformable decoding for grid-level instance features. |
| Multiscale deformable attention | `MSDeformAttn` | `adet/modeling/ops/modules/ms_deform_attn.py` | Provides sparse multiscale sampling offsets and attention weights. |
| Shape descriptor / dynamic mask parameters | `kernel_pred` branch | `adet/modeling/clarity_former/clarity_former.py` | Decoder outputs are projected to class logits and dynamic mask kernels. |
| Dynamic mask fusion | `DynamicMaskAffineFusion` | `adet/modeling/clarity_former/instance_fusion.py` | Applies the predicted dynamic kernels to shared mask features. Controlled by `MODEL.CLARITY_FORMER.DYNAMIC_MASK_NORM`. |
| Mask generation module | `VGSEMaskHead` | `adet/modeling/clarity_former/clarity_former.py` | Produces the shared high-quality mask feature map and final mask bases. |
| VGSE feature enhancement | `VGSEMaskHead` with `EASA` and `LDE` | `adet/modeling/clarity_former/neckaug.py` | Current implementation uses low/high-frequency enhancement modules as the feature refinement path. |
| Composite loss | focal classification loss and Dice mask loss | `adet/modeling/clarity_former/clarity_former.py`, `adet/modeling/clarity_former/loss.py` | Matches the paper-level formulation `L_total = L_class + 3 * L_mask`; optional semantic/edge loss is also supported. |
| Matrix NMS post-processing | `matrix_nms` | `adet/modeling/clarity_former/utils.py` | Used during inference to suppress duplicate masks. |

## Config Mapping

| Paper variant | Recommended config | Backbone setting |
|---|---|---|
| Clarity-Former-T | `configs/ClarityFormer_SWINT.yaml` | `EMBED_DIM: 96`, `DEPTHS: [2, 2, 6, 2]` |
| Clarity-Former-S | `configs/ClarityFormer_SWINS.yaml` | `EMBED_DIM: 96`, `DEPTHS: [2, 2, 18, 2]` |
| Clarity-Former-B | `configs/ClarityFormer_SWINB.yaml` | `EMBED_DIM: 128`, `DEPTHS: [2, 2, 18, 2]` |
| ResNet ablation/baseline | `configs/ClarityFormer_R50.yaml`, `configs/ClarityFormer_R101.yaml` | ResNet backbone, not the full SAMA backbone variant. |

## Dataset Notes

- `adet/data/datasets/underwater_instance.py` registers `underwater_train` and `underwater_test` for the selected underwater benchmark.
- Ensure `MODEL.CLARITY_FORMER.NUM_CLASSES`, `MODEL.BASIS_MODULE.NUM_CLASSES`, and `MODEL.SEM_SEG_HEAD.NUM_CLASSES` match the selected dataset.
- USIS10K/UIIS commonly use 7 foreground classes; USIS16K uses 158 classes.
