#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""Benchmark Clarity-Former training, inference, or data loading."""

import itertools
import logging
import sys
from pathlib import Path

import psutil
import torch
import tqdm
from fvcore.common.timer import Timer
from torch.nn.parallel import DistributedDataParallel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adet.config import get_cfg
from adet.data.datasets.underwater_instance import register_dataset
import adet.modeling  # noqa: F401 - registers ClarityFormer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetFromList, build_detection_test_loader, build_detection_train_loader
from detectron2.data.benchmark import DataLoaderBenchmark
from detectron2.config import instantiate
from detectron2.engine import AMPTrainer, SimpleTrainer, default_argument_parser, hooks, launch
from detectron2.modeling import build_model
from detectron2.solver import build_optimizer
from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.events import CommonMetricPrinter
from detectron2.utils.logger import setup_logger

logger = logging.getLogger("detectron2")


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    setup_logger(distributed_rank=comm.get_rank())
    return cfg


def create_data_benchmark(cfg):
    kwargs = build_detection_train_loader.from_config(cfg)
    kwargs.pop("aspect_ratio_grouping", None)
    kwargs["_target_"] = DataLoaderBenchmark
    return instantiate(kwargs)


def ram_msg():
    vram = psutil.virtual_memory()
    used = (vram.total - vram.available) / 1024**3
    total = vram.total / 1024**3
    return f"RAM Usage: {used:.2f}/{total:.2f} GB"


def benchmark_data(args):
    cfg = setup(args)
    logger.info("After spawning " + ram_msg())
    benchmark = create_data_benchmark(cfg)
    benchmark.benchmark_distributed(250, 10)
    for idx in range(10):
        logger.info(f"Iteration {idx} " + ram_msg())
        benchmark.benchmark_distributed(250, 1)


def benchmark_data_advanced(args):
    cfg = setup(args)
    benchmark = create_data_benchmark(cfg)
    if comm.get_rank() == 0:
        benchmark.benchmark_dataset(100)
        benchmark.benchmark_mapper(100)
        benchmark.benchmark_workers(100, warmup=10)
        benchmark.benchmark_IPC(100, warmup=10)
    if comm.get_world_size() > 1:
        benchmark.benchmark_distributed(100)
        logger.info("Rerun ...")
        benchmark.benchmark_distributed(100)


def benchmark_train(args):
    cfg = setup(args)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if comm.get_world_size() > 1:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )
    optimizer = build_optimizer(cfg, model)
    DetectionCheckpointer(model, optimizer=optimizer).load(cfg.MODEL.WEIGHTS)

    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 2
    data_loader = build_detection_train_loader(cfg)
    dummy_data = list(itertools.islice(data_loader, 100))

    def data_iter():
        data = DatasetFromList(dummy_data, copy=False, serialize=False)
        while True:
            yield from data

    max_iter = args.benchmark_iter
    trainer_cls = AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer
    trainer = trainer_cls(model, data_iter(), optimizer)
    trainer.register_hooks([
        hooks.IterationTimer(),
        hooks.PeriodicWriter([CommonMetricPrinter(max_iter)]),
    ])
    trainer.train(1, max_iter)


@torch.no_grad()
def benchmark_eval(args):
    cfg = setup(args)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 0
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])

    model.eval()
    logger.info("Model:\n{}".format(model))
    dummy_data = DatasetFromList(list(itertools.islice(data_loader, 100)), copy=False)

    for idx in range(5):
        model(dummy_data[idx])

    timer = Timer()
    with tqdm.tqdm(total=args.benchmark_iter) as pbar:
        for idx, data in enumerate(itertools.cycle(dummy_data)):
            if idx == args.benchmark_iter:
                break
            model(data)
            pbar.update()
    logger.info(f"{args.benchmark_iter} iters in {timer.seconds():.2f} seconds.")


def main():
    parser = default_argument_parser()
    parser.add_argument("--task", choices=["train", "eval", "data", "data_advanced"], required=True)
    parser.add_argument("--benchmark-iter", type=int, default=300)
    args = parser.parse_args()
    assert not args.eval_only

    register_dataset()
    logger.info("Environment info:\n" + collect_env_info())
    if "data" in args.task:
        print("Initial " + ram_msg())

    if args.task == "data":
        fn = benchmark_data
    elif args.task == "data_advanced":
        fn = benchmark_data_advanced
    elif args.task == "train":
        fn = benchmark_train
    else:
        fn = benchmark_eval
        assert args.num_gpus == 1 and args.num_machines == 1

    launch(fn, args.num_gpus, args.num_machines, args.machine_rank, args.dist_url, args=(args,))


if __name__ == "__main__":
    main()
