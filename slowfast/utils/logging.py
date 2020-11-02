#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Logging."""

import os
import builtins
import decimal
import functools
import logging
import os
import sys
import simplejson
import torch.distributed as dist
from datetime import datetime
from fvcore.common.file_io import PathManager

import slowfast.utils.distributed as du


def _suppress_print():
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


def setup_logger(cfg, name=None):
    logger = logging.getLogger('progress-action')
    logger.setLevel(logging.DEBUG)
    logger.propogate = False
    # don't log results for the non-master process
    if not du.is_master_proc():
        _suppress_print()
        return logger
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s")

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # multi-machine
    if cfg.NUM_GPUS != du.get_world_size():
        assert du.is_master_proc()
        num_gpus_per_machine = cfg.NUM_GPUS
        worker = du.get_rank() // cfg.NUM_GPUS
        filename = os.path.join(cfg.LOGS.DIR, f"{name}-worker-{worker}.log")
    else:
        filename = os.path.join(cfg.LOGS.DIR, f"{name}.log")
    if name is None or os.path.exists(filename):
        filename = os.path.join(
            cfg.LOGS.DIR, '{} {}.log'.format(name, datetime.now()))
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger('progress-action.' + name)


def log_json_stats(stats):
    """
    Logs json stats.
    Args:
        stats (dict): a dictionary of statistical information to log.
    """
    stats = {
        k: decimal.Decimal("{:.6f}".format(v)) if isinstance(v, float) else v
        for k, v in stats.items()
    }
    # json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
    logstr = "; ".join(["{}: {}".format(k, v) for k, v in stats.items()])
    if du.is_master_proc():
        logger = get_logger(__name__)
        logger.info(logstr)
        # logger.info("json_stats: {:s}".format(json_stats))
