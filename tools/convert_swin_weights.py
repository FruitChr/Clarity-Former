#!/usr/bin/env python
import argparse
from collections import OrderedDict

import torch


def convert_swin_weights(input_path: str, output_path: str) -> None:
    checkpoint = torch.load(input_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    converted = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("head.") or key.startswith("norm."):
            continue
        converted[f"backbone.{key}"] = value

    torch.save(converted, output_path)
    print(f"Loaded {len(state_dict)} keys from {input_path}")
    print(f"Saved {len(converted)} converted keys to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert official Swin weights for Clarity-Former/Detectron2.")
    parser.add_argument("--input", required=True, help="Path to the official Swin checkpoint.")
    parser.add_argument("--output", required=True, help="Path to save the converted checkpoint.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_swin_weights(args.input, args.output)
