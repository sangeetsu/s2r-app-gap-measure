# SPDX-License-Identifier: Apache-2.0
"""Shared code for the frame-matched sim-to-real appearance-gap measurement framework.

data/            dataset readers (CARLA cells, KITTI, Cityscapes) returning torchvision-style detection targets
detector.py      Faster R-CNN construction and checkpoint loading
eval_detection.py  mAP@IoU evaluation and result files
adaptation/      feature lookups and alignment utilities shared by the probes and the detector trainer
"""
