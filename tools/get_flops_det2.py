#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adet.config import get_cfg
import adet.modeling  # noqa: F401 - registers ClarityFormer
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger


class FlopWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        height, width = image.shape[-2:]
        self.model([{"image": image, "height": height, "width": width}])
        return image.new_empty(())


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate Clarity-Former FLOPs with fvcore.")
    parser.add_argument("--config-file", required=True, help="Path to a ClarityFormer config file.")
    parser.add_argument("--shape", type=int, nargs=2, default=[800, 1333], metavar=("H", "W"))
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Additional config options.")
    return parser.parse_args()


def main():
    setup_logger()
    args = parse_args()
    cfg = setup_cfg(args)

    model = build_model(cfg)
    model.eval()

    device = torch.device(cfg.MODEL.DEVICE)
    image = torch.rand(3, args.shape[0], args.shape[1], device=device)
    analysis = FlopCountAnalysis(FlopWrapper(model), image)
    analysis.unsupported_ops_warnings(False)

    print(flop_count_table(analysis, max_depth=3))
    print(f"Total GFLOPs: {analysis.total() / 1e9:.2f}")


if __name__ == "__main__":
    main()
