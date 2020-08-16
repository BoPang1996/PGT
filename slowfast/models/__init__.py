#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .build import MODEL_REGISTRY, build_model  # noqa
from .backbones.resnet import ResNet # noqa
from .backbones.regnet import RegNet # noqa
from .backbones.slowfast import SlowFast  # noqa