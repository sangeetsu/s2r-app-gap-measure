# s2r-app-gap-measure

Code and results for a frame-matched framework that measures the sim-to-real **appearance gap** of object detectors:
how much detection accuracy a change of appearance costs when everything else in the scene is held fixed.

This repository accompanies Chapters 4–6 of the dissertation *Quantifying the Sim2Real Gap through Latent Representations
and Feature Spaces* (Sangeet Sankaramangalam Ulhas, Arizona State University, 2026).

## The idea in four steps

1. **Fix the content.** A CARLA drive is recorded once as world *state* (every vehicle and pedestrian pose, blueprint,
   color and light state, and the ego pose). The same frames are then re-rendered under appearance *doses* (sun angle,
   night, fog, wetness, rain, and combinations), so a reference frame and its dosed copy differ only in appearance.
2. **Measure the cost.** A detector trained on the reference cell is evaluated on the reference frames and on each dosed
   copy. The drop in mAP@0.5 at fixed content is the appearance cost of that dose.
3. **Measure the distance without labels.** Each object is described by box-pooled features of a frozen DINOv3 ViT-L/16
   (`facebook/dinov3-vitl16-pretrain-lvd1689m`), standardized with the reference statistics, and the two renderings are
   compared with an object-level Fréchet distance (object FID), with image-level and nearest-neighbor measures alongside.
4. **Calibrate and test in advance.** A saturating calibration, drop = A_town · (1 − exp(−d / τ)), maps the object
   distance d to the expected drop. Predictions for unseen doses, towns and rounds were written to files and hashed before
   the detectors were evaluated on them, and then scored.

Chapter 4 of the dissertation reports the configuration-level measures that were tested first (distances between whole
simulated and real datasets, label-free accuracy estimators, latent geometry) and why they tracked labeled content rather
than appearance; the tools for those tests are included as well.

## Layout

```
src/s2rgap/            shared code: dataset readers (CARLA cells, KITTI, Cityscapes), Faster R-CNN construction,
                       mAP evaluation, feature lookups and alignment utilities
tools/                 every script used for the experiments (see below); run them from the repository root
tools/run_*.sh         the pipelines as they were run, one per experiment
results/               per-cell evaluation files, analysis outputs, pre-registration files and the consolidated tables
results/consolidated/  tables.json and tables.md (T1–T19): every reported statistic, recomputed from the files in results/
carla_server_profiles.example.toml   template for connecting to CARLA servers (copy to carla_server_profiles.toml)
```

Main entry points:

| Step | Script |
|---|---|
| State capture and frame-matched re-rendering under doses (doses are defined in this file) | `tools/carla_state_replay.py` |
| Original frame extraction and label projection | `tools/extract_carla_frames.py`, `tools/carla_labels.py` |
| Frozen DINOv3 feature extraction (DINOv2 and CLIP variants alongside) | `tools/extract_dinov3_features.py` |
| Object-level and image-level metrics per dose | `tools/dose_metrics.py`, `tools/dose_lowlevel.py`, `tools/dose_cmmd_kid.py` |
| Detector training and evaluation | `tools/train_detector.py`, `tools/eval_detector_on_carla_cell.py`, `tools/eval_detector_on_cityscapes.py` |
| Calibration formula, hashed predictions, scoring | `tools/predict_heldout_doses.py`, `tools/predict_heldout2.py`, `tools/predict_heldout3.py`, `tools/analyze_heldout*.py` |
| Consolidated tables with cluster-bootstrap intervals | `tools/consolidate.py` |
| Figures | `tools/make_figures.py`, `tools/make_formula_figure.py`, `tools/make_mechanism_figure.py`, `tools/make_real_figure*.py`, `tools/make_thesis_ch4_figures.py` |

Pipelines (each `tools/run_*.sh` names the scripts it calls and the files it writes):

| Experiment | Pipeline |
|---|---|
| Frame-matched dose testbed, 8 towns | `run_dose_testbed.sh`, `run_dose_metrics_8towns.sh`, `run_dose_replicates.sh` |
| Pre-registered held-out doses, rounds 1–3 | `run_dose_heldout.sh`, `run_dose_heldout2.sh`, `run_dose_heldout3.sh` |
| Other detector architectures | `run_dose_arch2.sh`, `run_dose_arch3.sh`, `run_crossfamily.sh` |
| Real-world tests (Foggy Cityscapes, RTTS, Virtual KITTI 2) | `run_foggy_cs.sh`, `run_rtts.sh`, `run_vkitti.sh`, `run_vkitti_graded.sh`, `run_vkitti_real.sh` |
| Generative real-twin pilot | `run_pilot_realtwin.sh` (generation itself is in the separate bundle below) |
| Configuration-level grids, held-out towns, learning curves, rendering quality, SHIFT replication | `run_calibration_grid.sh`, `run_anchored_grid*.sh`, `run_towns2.sh`, `run_learning_curve.sh`, `run_lowq.sh`, `run_shift_replication.sh` |

## Requirements

The experiments ran on Ubuntu 22.04 with one NVIDIA RTX A6000 and these versions:

- CARLA 0.9.16 (server and the matching `carla` Python client)
- Python 3.12, PyTorch 2.8.0 and torchvision 0.23.0 (CUDA 12.8)
- See `requirements.txt` for the other packages

DINOv3 weights are gated on Hugging Face: request access on the model page and log in (`huggingface-cli login`) before
extracting features.

Datasets are **not** included and must be obtained from their providers under their own licenses: KITTI object
detection, Cityscapes (and Foggy Cityscapes), SHIFT, Virtual KITTI 2 and RTTS. The CARLA frames, labels and features are
regenerated by the scripts.

The shell pipelines keep the paths of the original machine (for example `/mnt/features/...` and `~/datasets/...`), set at
the top of each script; edit them for a new machine. Set `PY` to choose the Python interpreter (default `python`). Some
pipelines wrap jobs in `systemd-run --user --scope` with memory limits; remove the wrapper where that is not available.
`tools/train_detector.py` also retains options from earlier work on foundation-model features; the measurement pipelines
use its plain Faster R-CNN path.

## Results and pre-registration

`results/` holds the evaluation files behind every number in Chapters 4–6 (one JSON per detector and evaluation set),
the analysis outputs of each experiment, and the prediction files of the pre-registered tests with their SHA-256 hashes
(`*.sha256`, `results/pilot_realtwin/prereg_sha256.txt`). `python tools/consolidate.py` rebuilds
`results/consolidated/tables.{json,md}` from these files. A rebuild reproduces every estimate, bootstrap interval
and leave-one-town-out R² exactly. The permutation p-values are Monte Carlo estimates (B = 2,000) and can differ slightly
between runs: in a check rebuild they differed by at most 0.024, and none crossed 0.05. The tables shipped here are the
ones the dissertation reports.

Comments in the code refer to sections of the project ledger, the timestamped record in which each test's hypothesis,
data, statistic and decision rule were written before its outcome existed. The ledger is not public at present.

## Related

- Generative real-twin bundle (Cosmos-Transfer2.5 restyling of the same CARLA frames):
  https://github.com/sangeetsu/carla-cosmos-realtwin

## Citation

```bibtex
@phdthesis{ulhas2026dissertation,
  author = {Ulhas, Sangeet Sankaramangalam},
  title  = {Quantifying the Sim2Real Gap through Latent Representations and Feature Spaces},
  school = {Arizona State University},
  year   = {2026}
}
```

## License

Code: Apache License 2.0 (see `LICENSE`). Datasets and models used by the scripts are distributed by their providers
under their own licenses.
