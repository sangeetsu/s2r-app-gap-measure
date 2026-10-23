# Consolidated tables — tools/consolidate.py, 2026-10-18T09:35:40 (B=2000, seed 0)

CI = town-cluster bootstrap 95%; p = town-clustered permutation; LOTO = leave-one-town-out calibration. Every number is recomputed from the listed results directories.


## T1. Experimental grids: real-target mAP of simulator-only detectors (mean ± sd over source cells; mAP@0.5, 3 classes)

| grid | sources | KITTI mAP (range) | Cityscapes mAP | note |
|---|---|---|---|---|
| COCO-init 24-cell grid (6 towns × 4 weathers) | 24 | 48.0 ± 3.0 (40.9–52.1) | 30.8 ± 2.2 | detector pretrained on COCO end to end |
| ImageNet-init 24-cell grid | 24 | 27.6 ± 4.7 (14.9–34.6) | 14.0 ± 2.4 | ImageNet backbone only; seed 42 |
| held-out towns (Town06/07/15 × Noon/Night) | 6 | 17.5 ± 9.3 (1.8–26.5) | 9.7 ± 5.0 | pre-registered test cells |
| fresh cells (6 towns × ClearSunset/WetNoon) | 12 | 28.4 ± 4.8 (18.8–35.1) | 15.3 ± 1.6 | pre-registered confirmation cells |
| SHIFT 15 condition cells (independent rig) | 15 | 35.7 ± 1.8 (32.3–38.3) | 25.3 ± 1.9 | 400 frames/cell, fixed 2400 it |
| frame-matched dose testbed, 8 towns × 10 doses (seed 42) | 80 | 26.1 ± 5.5 (7.7–33.6) | 14.1 ± 2.3 | Faster R-CNN R50-FPN, ImageNet init |
| dose testbed replicate (seed 17) | 80 | 26.7 ± 5.3 (7.8–34.2) | 14.2 ± 2.3 |  |
| dose testbed, FCOS (Town01 + Town10HD × 10 doses) | 20 | 22.9 ± 8.9 (0.9–31.4) | 13.8 ± 5.3 | second detector family |

sd with ddof=1 (the ledger's earlier 2.9/4.6 used ddof=0). Held-out towns include Town15, whose two cells failed to train (561/673 boxes).

Sources: results/calib_anchored, results/calib_imnet, results/calib_imnet_towns2, results/calib_imnet_confirm, results/calib_shift, results/calib_dose, results/calib_dose_rep, results/calib_dose_arch2

## T2. Noise floors: run-to-run (training-seed) variation of every outcome used in the paper

| outcome | per-run sd (pp) | scale | reliability / agreement |
|---|---|---|---|
| ImageNet grid, real mAP (8 sources re-trained, seed 17 vs 42) | KITTI 1.53 / CS 0.86 | across-source sd 4.66 / 2.41 | reliability 0.89 / 0.87; r ceiling 0.94 / 0.93 |
| ImageNet grid, within-CARLA anchored drop (184 pairs) | 1.90 | mean |Δseed| 2.07 pp | 87% of pairs within 4 pp |
| dose testbed, appearance-only drop (72 pairs, 80+80 detectors) | 1.73 | mean |Δseed| 1.84 pp; signal sd 10.5 | seed r 0.974; reliability 0.97 |
| dose testbed, real mAP (80 detectors × 2 seeds) | KITTI 1.61 / CS 1.30 | Δ real mAP (dose − ref) seed r 0.86 / 0.66 | mean |Δseed| of Δ real mAP 2.22 / 1.48 pp |

per-run sd = sd(seed17 − seed42)/√2. Correlations with a per-cell outcome cannot exceed the stated ceiling.

Sources: results/calib_imnet, results/calib_imnet_rep, results/calib_dose, results/calib_dose_rep

## T3. Pretraining hides the simulator: the same 24 CARLA cells with two detector initialisations

| init | KITTI mAP | Cityscapes mAP | KITTI town means | KITTI variance: town / weather | Cityscapes variance: town / weather | within-CARLA drop explained by the two anchors (LOO R²) |
|---|---|---|---|---|---|---|
| COCO-pretrained (backbone+FPN+heads) | 48.0 ± 3.0 | 30.8 ± 2.2 | 45.5–50.6 | 28% (p=0.116) / 33% (p=0.022) | 31% (p=0.269) / 3% (p=0.890) | 0.845 |
| ImageNet backbone only | 27.6 ± 4.7 | 14.0 ± 2.4 | 23.0–33.5 | 70% (p=<.001) / 12% (p=0.047) | 40% (p=0.135) / 0% (p=0.993) | 0.747 |

Two-way ANOVA, type-I with town before weather, 24 sources. Anchors = the detector's own in-domain mAP and the target cell's in-domain mAP.

Sources: results/calib_anchored, results/calib_imnet

## T4. Content dominates across configurations: labeled object count vs frozen-DINOv3 image FID as predictors of real transfer

| sources | n | KITTI r(log boxes, mAP) [CI] | KITTI r(FID, mAP) | KITTI partial r: FID | boxes / boxes | FID | CS r(log boxes, mAP) [CI] | CS r(FID, mAP) | CS partial r: FID | boxes / boxes | FID |
|---|---|---|---|---|---|---|---|
| 24 grid sources | 24 | +0.590 [-0.01, +0.84] | -0.639 | -0.369 / +0.220 | +0.578 [+0.13, +0.73] | -0.560 | -0.236 / +0.289 |
| 30 sources (grid + 6 held-out towns) | 30 | +0.694 [+0.06, +0.90] | -0.386 | -0.131 / +0.634 | +0.671 [+0.13, +0.86] | -0.374 | -0.112 / +0.607 |
| 12 fresh cells (unseen weathers) | 12 | +0.867 [+0.23, +0.97] | — | — / — | +0.605 [+0.08, +0.80] | — | — / — |
| town level (9 towns, 30 sources) | 9 | +0.814 (p=0.008) | -0.274 (p=0.476) | — | +0.880 (p=0.002) | -0.282 (p=0.462) | — |

Boxes = Car/Pedestrian/Cyclist labels in the 800 training frames. FID = image-level Fréchet distance in frozen DINOv3 space between the source cell and the real target. CI clusters by town.

Sources: results/calib_imnet, results/calib_imnet_towns2, results/calib_imnet_confirm, results/domain_distance_dinov3_all.json, results/calib_imnet/prereg_predictions_towns2.json

## T5. Labeled-data quantity is causal: within-cell learning curves (content and appearance fixed) vs the between-source slope

| target | CARLA within-cell slope, pp per log(boxes) [CI] | per-cell slopes (6 cells) | pp per doubling | between-source slope (30 sources) | share of between-source slope | SHIFT within-cell slope (3 conditions) |
|---|---|---|---|---|---|---|
| KITTI | +4.14 [+3.60, +4.62] | +4.15 ± 0.74 | +2.9 | +6.45 (r=+0.69) | 64% | +5.30 (SE 1.02, p=0.004) |
| Cityscapes | +0.99 [+0.21, +1.68] | +0.90 ± 1.01 | +0.7 | +3.08 (r=+0.67) | 32% | +4.02 (SE 0.29, p=<.001) |

CARLA: 6 ClearNoon cells × 100/200/400/800 frames, fixed 2400 iterations; CI = cluster bootstrap over cells. SHIFT: 3 condition cells × 100/200/400 frames.

Sources: results/calib_imnet_lc, results/calib_imnet, results/calib_imnet_towns2, results/calib_shift

## T6. Across configurations, no distance or consistency estimator predicts real transfer once content is known (24 ImageNet-init sources)

| predictor | KITTI r with real mAP [CI] | KITTI partial r | log boxes | Cityscapes r [CI] | Cityscapes partial r | log boxes |
|---|---|---|---|---|
| log labeled boxes (content) | +0.590 [-0.00, +0.84] | — | +0.578 [+0.13, +0.73] | — |
| dIV_fg usable-information gap (labeled boxes) | +0.529 [-0.08, +0.78] | +0.481 | +0.478 [-0.25, +0.72] | +0.480 |
| FM veto gap (frozen probe on detector boxes) | +0.115 [-0.57, +0.50] | -0.018 | +0.326 [-0.08, +0.52] | +0.076 |
| DoC, difference of confidences | -0.295 [-0.64, +0.18] | +0.280 | -0.324 [-0.55, +0.03] | +0.102 |
| FM–detector disagreement | +0.280 [-0.23, +0.52] | +0.041 | +0.602 [+0.30, +0.75] | +0.393 |
| image FID, frozen DINOv3 | -0.639 [-0.88, -0.26] | -0.369 | -0.560 [-0.64, -0.28] | -0.236 |
| BoS stability gap | +0.498 [+0.17, +0.83] | +0.191 | +0.027 [-0.38, +0.24] | -0.411 |
| AoL agreement gap | +0.348 [-0.28, +0.77] | +0.128 | +0.194 [-0.53, +0.54] | -0.064 |

Expected sign for a gap measure is negative (more shift → lower mAP). Only FID has the right sign, and it vanishes given object count.

Sources: results/calib_imnet, results/calib_anchored, results/domain_distance_dinov3_all.json

## T7. Pre-registered out-of-sample tests across configurations

| test | KITTI mean |error| (pp) | KITTI r(pred, obs) | Cityscapes mean |error| | Cityscapes r(pred, obs) |
|---|---|---|---|---|
| held-out towns (6 cells): FID line (pre-registered 08:12) | 9.33 | -0.317 | 4.82 | -0.418 |
| held-out towns (6 cells): log(training boxes) | 7.76 | +0.761 | 3.94 | +0.694 |
| held-out towns (6 cells): FID + log(boxes) | 8.59 | +0.263 | 4.26 | +0.618 |
| fresh cells (11): boxes only (pre-registered coefficients) | 3.15 (centred 1.91) | +0.824 | 1.80 (centred 1.30) | +0.632 |
| fresh cells (11): geometry model (pre-registered coefficients) | 2.57 (centred 2.33) | +0.778 | 1.49 (centred 1.46) | +0.615 |
| rendering quality Epic→Low (5 towns), KITTI | Epic − Low real mAP +5.52 ± 6.98 (paired t p=0.152); 4/5 towns lose | — | FID to KITTI rises at Low in 5/5 (+1.50); r(ΔFID, ΔmAP) +0.76 (wrong sign) | — |
| rendering quality Epic→Low (5 towns), Cityscapes | Epic − Low real mAP +1.82 ± 1.95 (paired t p=0.105); 4/5 towns lose | — | FID to Cityscapes rises at Low in 5/5 (+1.40); r(ΔFID, ΔmAP) +0.51 (wrong sign) | — |

Predictions were written and hashed before the detectors were trained (ledger §15.34, 15.36, 15.43, 15.40). Town15 collapsed in training (561/673 boxes) and is kept as pre-registered.

Sources: results/calib_imnet_towns2, results/calib_imnet_confirm, results/calib_imnet_lowq, results/calib_imnet/prereg_predictions_towns2*.json, results/manifold_markers/confirm_predictions.json

## T8. SHIFT replication (independent CARLA rig, 15 time-of-day × weather cells at near-matched counts)

| target | real mAP | variance shares: log boxes / time / weather | F p: time | boxes, weather | boxes+time, both | r(FID, mAP) | partial r(FID | log boxes) |
|---|---|---|---|---|---|
| KITTI | 35.7 ± 1.8 | 0% / 16% / 19% | 0.375 / 0.722 / 0.694 | +0.056 | +0.059 (F p=0.842) |
| Cityscapes | 25.3 ± 1.9 | 43% / 14% / 17% | 0.214 / 0.415 / 0.340 | -0.552 | -0.357 (F p=0.210) |

n = 15 (low power; estimation reported). Causal count slopes for SHIFT are in T5.

Sources: results/calib_shift, /mnt/features/shift_cells/cells_meta.json

## T9. R1 — appearance-only cost: mAP drop (pp) of each town's clean-trained detector on the SAME frames re-rendered under a dose (mean of 2 seeds)

| town | sun25 | sun10 | sun3 | night | fog25 | fog50 | fog75 | wet | rain |
|---|---|---|---|---|---|---|---|---|---|
| Town01 | 2.1 | 3.0 | 4.8 | 18.0 | 11.2 | 23.6 | 25.6 | 3.3 | 8.3 |
| Town02 | 0.6 | 3.4 | 4.2 | 11.4 | 7.4 | 16.3 | 18.0 | 1.1 | 6.9 |
| Town03 | 1.4 | 4.8 | 6.4 | 18.7 | 12.2 | 28.9 | 37.8 | 0.7 | 11.8 |
| Town04 | 0.4 | 1.9 | 2.5 | 7.5 | 6.4 | 10.5 | 12.9 | 0.8 | 2.8 |
| Town05 | 1.3 | -0.2 | 1.8 | 27.7 | 6.4 | 17.5 | 25.3 | 1.2 | 1.3 |
| Town06 | -0.2 | 4.1 | 7.5 | 33.6 | 18.7 | 28.9 | 36.5 | 4.4 | 10.2 |
| Town07 | 2.5 | 1.5 | 5.9 | 21.6 | 12.2 | 21.6 | 26.8 | 1.6 | 3.2 |
| Town10HD | 0.7 | 4.1 | 6.6 | 16.7 | 19.6 | 32.8 | 33.3 | 0.9 | 8.8 |
| **town mean** | **1.1** | **2.8** | **5.0** | **19.4** | **11.8** | **22.5** | **27.0** | **1.7** | **6.7** |
| town-mean CI | [+0.58, +1.71] | [+1.69, +3.85] | [+3.58, +6.21] | [+14.08, +24.80] | [+8.57, +15.28] | [+17.51, +27.54] | [+20.92, +32.41] | [+0.97, +2.69] | [+4.06, +9.16] |
| mean |seed 42 − seed 17| | 0.9 | 1.5 | 2.0 | 1.1 | 2.3 | 3.4 | 2.0 | 1.6 | 1.8 |
| town mean, boxes ≥25 px only (seed 42) | 1.4 | 2.2 | 5.8 | 24.0 | 13.2 | 27.4 | 34.0 | 2.0 | 8.7 |
| town mean, boxes ≥40 px only (seed 42) | 0.5 | 1.7 | 4.6 | 24.3 | 10.6 | 23.7 | 31.4 | -0.6 | 4.6 |

Monotone within family (Spearman ≥ .8 per town): sun 8/8 (seed 42), 8/8 (seed 17); fog 8/8, 8/8. Visibility control (seed 42, boxes <25 px / <40 px removed): monotone in sun 7 and 7 of 8 towns, fog 8 and 8 of 8 — drops grow when only nearer objects are scored.

Sources: results/calib_dose, results/calib_dose_rep, results/calib_dose_h25, results/calib_dose_h40

## T10. R2 — with content fixed, label-free metrics track the appearance-only cost (72 town×dose pairs, 8 towns)

| family | metric | needs | Spearman ρ, seed 42 [CI] | clustered p | LOTO R² (42) | ρ seed 17 | LOTO R² (17) | calibration |err| pp, k=1 → 7 towns (predict-the-mean 8.8) |
|---|---|---|---|---|---|---|---|---|
| object-level, frozen DINOv3 (GT boxes) | object FID | boxes | +0.840 [+0.78, +0.91] | <.001 | +0.62 | +0.814 | +0.57 | 5.7 → 4.7 |
| object-level, frozen DINOv3 (GT boxes) | directed NN distance dose→ref | boxes | +0.828 [+0.78, +0.89] | <.001 | +0.61 | +0.792 | +0.56 | 5.9 → 5.0 |
| object-level, frozen DINOv3 (GT boxes) | coverage of dose by ref | boxes | -0.833 [-0.89, -0.81] | <.001 | +0.59 | -0.764 | +0.53 | 6.5 → 5.1 |
| object-level, frozen DINOv3 (GT boxes) | paired object displacement | boxes | +0.843 [+0.80, +0.90] | <.001 | +0.59 | +0.807 | +0.54 | 6.1 → 5.0 |
| object-level, frozen DINOv3 (GT boxes) | task-projected displacement | boxes + probe | +0.797 [+0.69, +0.90] | <.001 | +0.40 | +0.786 | +0.35 | 6.8 → 5.8 |
| object-level, frozen DINOv3 (GT boxes) | probe flip rate | boxes + probe | +0.738 [+0.65, +0.89] | <.001 | +0.30 | +0.646 | +0.22 | 9.5 → 6.6 |
| object-level, detector-proposed boxes (label-free) | object FID | detector boxes | +0.838 [+0.78, +0.91] | <.001 | +0.62 | +0.818 | +0.57 | 5.6 → 4.7 |
| object-level, detector-proposed boxes (label-free) | directed NN distance | detector boxes | +0.841 [+0.79, +0.90] | <.001 | +0.57 | +0.778 | +0.48 | 5.9 → 5.3 |
| object-level, detector-proposed boxes (label-free) | paired object displacement | detector boxes | +0.851 [+0.81, +0.90] | <.001 | +0.53 | +0.787 | +0.44 | 6.1 → 5.3 |
| object-level, detector-proposed boxes (label-free) | probe flip rate | detector boxes + probe | +0.771 [+0.71, +0.89] | <.001 | +0.36 | +0.688 | +0.28 | 9.2 → 6.2 |
| object-level, detector-proposed boxes (label-free) | log detection-count ratio (detector itself) | the detector | -0.872 [-0.91, -0.84] | <.001 | +0.55 | -0.862 | +0.49 | 5.4 → 4.6 |
| object-level, detector-proposed boxes (label-free) | DoC (detector itself) | the detector | +0.274 [-0.02, +0.62] | 0.018 | -0.06 | +0.195 | -0.07 | 10.4 → 8.9 |
| image-level, frozen DINOv3 (no boxes) | image FID | none | +0.717 [+0.63, +0.80] | <.001 | +0.26 | +0.682 | +0.25 | 7.5 → 7.0 |
| image-level, frozen DINOv3 (no boxes) | paired image displacement | none | +0.682 [+0.55, +0.80] | <.001 | +0.29 | +0.662 | +0.28 | 7.4 → 6.8 |
| modern distribution baselines (no boxes) | CMMD, CLIP ViT-L/14-336 | none | +0.810 [+0.74, +0.88] | <.001 | +0.59 | +0.782 | +0.55 | 5.6 → 5.1 |
| modern distribution baselines (no boxes) | KID, Inception pool3 | none | +0.701 [+0.59, +0.81] | <.001 | +0.21 | +0.680 | +0.18 | 7.3 → 7.3 |
| low-level / classic (no boxes) | LPIPS-AlexNet (paired) | none | +0.784 [+0.70, +0.84] | <.001 | +0.44 | +0.761 | +0.42 | 6.6 → 5.8 |
| low-level / classic (no boxes) | Inception-pool3 FID | none | +0.752 [+0.67, +0.85] | <.001 | +0.49 | +0.727 | +0.45 | 6.5 → 5.8 |
| low-level / classic (no boxes) | |Δ RMS contrast| (paired) | none | +0.736 [+0.64, +0.88] | <.001 | +0.35 | +0.741 | +0.37 | 9.0 → 6.3 |
| low-level / classic (no boxes) | SSIM (paired) | none | -0.652 [-0.72, -0.58] | <.001 | +0.21 | -0.621 | +0.19 | 7.9 → 7.4 |
| low-level / classic (no boxes) | PSNR (paired) | none | -0.417 [-0.54, -0.27] | <.001 | +0.09 | -0.428 | +0.10 | 8.6 → 8.0 |
| low-level / classic (no boxes) | |Δ luminance| (paired) | none | -0.068 [-0.18, +0.07] | 0.565 | -0.04 | -0.043 | -0.04 | 9.4 → 9.0 |

Pre-registered bar: |ρ| ≥ .7 and LOTO R² > 0. Predict-the-mean error = 8.8 pp. All metrics are computed from the two renders only (detector-independent except the two detector rows).

Sources: results/calib_dose, results/calib_dose_rep, results/calib_dose/dose_metrics*.json, results/calib_dose/dose_lowlevel.json

## T11. R2 robustness: other feature spaces (same 4 towns, 36 pairs, seed-42 drops), a second detector family (FCOS, 2 towns, 18 pairs), and visibility-filtered drops (72 pairs)

| metric | DINOv3 ρ / LOTO R² | DINOv2 ρ / LOTO R² | CLIP ρ / LOTO R² | FCOS drop: ρ (clustered p) | ρ with drops on boxes ≥25 px / ≥40 px |
|---|---|---|---|---|---|
| object FID | +0.86 / +0.64 | +0.87 / +0.65 | +0.88 / +0.64 | +0.88 (p=<.001) | +0.86 / +0.83 |
| directed NN distance | +0.83 / +0.54 | +0.87 / +0.64 | +0.87 / +0.56 | +0.87 (p=<.001) | +0.84 / +0.81 |
| coverage | -0.87 / +0.58 | -0.88 / +0.64 | -0.88 / +0.61 | -0.91 (p=<.001) | -0.84 / -0.82 |
| paired object displacement | +0.86 / +0.55 | +0.87 / +0.62 | +0.88 / +0.60 | +0.86 (p=<.001) | +0.85 / +0.83 |
| task-projected displacement | +0.84 / +0.59 | +0.82 / +0.55 | +0.82 / +0.51 | +0.89 (p=<.001) | +0.81 / +0.78 |
| probe flip rate | +0.90 / +0.66 | +0.85 / +0.60 | +0.79 / +0.60 | +0.95 (p=<.001) | +0.75 / +0.76 |
| image FID (no boxes) | +0.74 / +0.39 | +0.77 / +0.47 | +0.85 / +0.60 | +0.78 (p=<.001) | +0.74 / +0.69 |
| paired image displacement (no boxes) | +0.74 / +0.45 | +0.78 / +0.54 | +0.86 / +0.63 | +0.75 (p=<.001) | +0.69 / +0.65 |

FCOS appearance-only drops are monotone in both towns (sun ρ [1.0, 1.0], fog ρ [1.0, 1.0]). The FCOS detectors see the SAME DINOv3 metrics (detector-independent).

Sources: results/calib_dose, results/calib_dose_arch2, results/calib_dose_h25, results/calib_dose_h40

## T12a. R3 — training on the re-rendered dose instead of the clean reference changes real-target mAP (dose-trained − reference-trained, pp)

| target | sun25 | sun10 | sun3 | night | fog25 | fog50 | fog75 | wet | rain |
|---|---|---|---|---|---|---|---|---|---|
| KITTI: Δ real mAP, mean of 8 towns × 2 seeds [CI] | +1.2 [+0.12, +2.38] | +0.6 [-1.19, +2.65] | +0.6 [-1.52, +2.34] | +1.3 [+0.40, +2.20] | -0.6 [-3.34, +1.90] | -6.5 [-9.48, -3.73] | -9.0 [-12.45, -5.78] | -0.1 [-1.94, +1.79] | -0.9 [-2.78, +0.83] |
| KITTI: FCOS (2 towns, 1 seed) | +2.8 | +4.4 | -0.3 | +4.4 | -5.9 | -3.4 | -12.0 | +0.8 | -10.0 |
| Cityscapes: Δ real mAP, mean of 8 towns × 2 seeds [CI] | +1.1 [+0.43, +1.92] | +0.4 [-0.31, +1.17] | +1.1 [+0.56, +1.59] | +1.5 [+0.63, +2.28] | +0.3 [-0.31, +0.88] | -2.1 [-3.09, -0.92] | -3.7 [-5.08, -2.34] | +0.7 [-0.07, +1.45] | +0.3 [-1.05, +1.67] |
| Cityscapes: FCOS (2 towns, 1 seed) | +1.4 | +2.4 | -0.2 | -0.6 | -4.1 | -2.0 | -7.6 | -1.3 | -6.1 |

Sources: results/calib_dose, results/calib_dose_rep, results/calib_dose_arch2

## T12b. R3 — Δ(distance to the real domain) vs Δ(real mAP) over the 72 (town, dose) pairs, Spearman ρ [CI], town-clustered p

| target | distance | detectors | all doses | without fog | fog only | sun + night |
|---|---|---|---|---|---|---|
| KITTI | DINOv3 image FID to real | seed 42 | -0.528 [-0.69, -0.41] p=<.001 | -0.315 [-0.64, +0.05] p=0.013 | -0.459 [-0.80, -0.15] p=<.001 | -0.158 [-0.61, +0.36] p=0.668 |
| KITTI | DINOv3 image FID to real | seed 17 | -0.490 [-0.65, -0.35] p=<.001 | -0.158 [-0.56, +0.28] p=0.164 | -0.580 [-0.84, -0.32] p=<.001 | +0.000 [-0.63, +0.62] p=1.000 |
| KITTI | DINOv3 image FID to real | seed mean | -0.545 [-0.70, -0.39] p=<.001 | -0.301 [-0.65, +0.12] p=0.003 | -0.490 [-0.80, -0.17] p=<.001 | -0.106 [-0.68, +0.52] p=0.865 |
| KITTI | Inception-pool3 FID to real | seed 42 | -0.534 [-0.71, -0.40] p=<.001 | -0.239 [-0.59, +0.14] p=0.092 | -0.733 [-0.93, -0.50] p=<.001 | -0.110 [-0.61, +0.43] p=0.927 |
| KITTI | Inception-pool3 FID to real | seed 17 | -0.537 [-0.71, -0.39] p=<.001 | -0.176 [-0.55, +0.26] p=0.196 | -0.798 [-0.93, -0.60] p=<.001 | -0.178 [-0.68, +0.44] p=0.529 |
| KITTI | Inception-pool3 FID to real | seed mean | -0.565 [-0.73, -0.41] p=<.001 | -0.244 [-0.61, +0.18] p=0.066 | -0.756 [-0.93, -0.53] p=<.001 | -0.147 [-0.69, +0.55] p=0.943 |
| KITTI | CMMD to real | seed 42 | -0.355 [-0.56, -0.14] p=<.001 | -0.066 [-0.33, +0.32] p=0.732 | -0.353 [-0.82, +0.15] p=0.003 | -0.116 [-0.36, +0.26] p=0.689 |
| KITTI | CMMD to real | seed 17 | -0.327 [-0.57, -0.10] p=<.001 | +0.075 [-0.36, +0.54] p=0.377 | -0.353 [-0.78, +0.10] p=0.006 | -0.023 [-0.42, +0.49] p=0.797 |
| KITTI | CMMD to real | seed mean | -0.344 [-0.57, -0.10] p=<.001 | +0.015 [-0.31, +0.50] p=0.925 | -0.343 [-0.81, +0.11] p=0.004 | -0.072 [-0.37, +0.51] p=0.817 |
| KITTI | KID to real | seed 42 | -0.448 [-0.66, -0.26] p=<.001 | -0.245 [-0.58, +0.14] p=0.089 | -0.659 [-0.87, -0.41] p=<.001 | -0.143 [-0.58, +0.43] p=0.979 |
| KITTI | KID to real | seed 17 | -0.418 [-0.60, -0.26] p=<.001 | -0.138 [-0.51, +0.31] p=0.182 | -0.715 [-0.90, -0.45] p=<.001 | -0.115 [-0.66, +0.53] p=0.697 |
| KITTI | KID to real | seed mean | -0.463 [-0.65, -0.28] p=<.001 | -0.243 [-0.60, +0.21] p=0.036 | -0.679 [-0.89, -0.40] p=<.001 | -0.149 [-0.68, +0.53] p=0.993 |
| KITTI | DINOv3 image FID to real | FCOS (18 pairs) | -0.730 p=0.001 | — | — | — |
| Cityscapes | DINOv3 image FID to real | seed 42 | -0.512 [-0.69, -0.33] p=<.001 | -0.233 [-0.53, -0.05] p=0.147 | -0.651 [-0.90, -0.35] p=<.001 | -0.088 [-0.52, +0.33] p=0.239 |
| Cityscapes | DINOv3 image FID to real | seed 17 | -0.454 [-0.57, -0.31] p=<.001 | -0.041 [-0.24, +0.27] p=0.776 | -0.657 [-0.85, -0.48] p=<.001 | +0.420 [+0.23, +0.63] p=0.002 |
| Cityscapes | DINOv3 image FID to real | seed mean | -0.537 [-0.67, -0.37] p=<.001 | -0.159 [-0.40, +0.12] p=0.401 | -0.703 [-0.87, -0.48] p=<.001 | +0.177 [-0.16, +0.54] p=0.068 |
| Cityscapes | Inception-pool3 FID to real | seed 42 | -0.502 [-0.66, -0.34] p=<.001 | -0.189 [-0.45, +0.02] p=0.194 | -0.767 [-0.93, -0.56] p=<.001 | -0.066 [-0.39, +0.26] p=0.558 |
| Cityscapes | Inception-pool3 FID to real | seed 17 | -0.464 [-0.60, -0.33] p=<.001 | -0.090 [-0.38, +0.24] p=0.379 | -0.629 [-0.80, -0.44] p=<.001 | +0.165 [-0.26, +0.56] p=0.582 |
| Cityscapes | Inception-pool3 FID to real | seed mean | -0.541 [-0.66, -0.40] p=<.001 | -0.182 [-0.41, +0.06] p=0.129 | -0.757 [-0.89, -0.56] p=<.001 | +0.034 [-0.33, +0.50] p=0.791 |
| Cityscapes | CMMD to real | seed 42 | -0.388 [-0.53, -0.15] p=0.001 | +0.049 [-0.26, +0.43] p=0.753 | -0.566 [-0.84, -0.11] p=0.012 | +0.111 [-0.18, +0.44] p=0.441 |
| Cityscapes | CMMD to real | seed 17 | -0.277 [-0.51, -0.03] p=0.002 | +0.112 [-0.26, +0.56] p=0.408 | -0.177 [-0.53, +0.18] p=0.133 | +0.146 [-0.27, +0.54] p=0.597 |
| Cityscapes | CMMD to real | seed mean | -0.359 [-0.58, -0.07] p=0.001 | +0.076 [-0.33, +0.57] p=0.449 | -0.417 [-0.72, -0.04] p=0.067 | +0.125 [-0.26, +0.53] p=0.407 |
| Cityscapes | KID to real | seed 42 | -0.435 [-0.63, -0.26] p=<.001 | -0.162 [-0.41, +0.02] p=0.253 | -0.708 [-0.95, -0.44] p=<.001 | -0.003 [-0.34, +0.34] p=0.964 |
| Cityscapes | KID to real | seed 17 | -0.406 [-0.53, -0.25] p=<.001 | -0.051 [-0.31, +0.25] p=0.640 | -0.665 [-0.82, -0.48] p=<.001 | +0.225 [-0.19, +0.65] p=0.401 |
| Cityscapes | KID to real | seed mean | -0.474 [-0.60, -0.33] p=<.001 | -0.145 [-0.37, +0.08] p=0.221 | -0.739 [-0.93, -0.55] p=<.001 | +0.097 [-0.30, +0.58] p=0.463 |
| Cityscapes | DINOv3 image FID to real | FCOS (18 pairs) | -0.554 p=0.008 | — | — | — |
| Foggy Cityscapes β=.02 | DINOv3 image FID to foggy CS | seed 42 | -0.081 [-0.36, +0.11] p=0.574 | — | — | — |

Δ = dose cell − reference cell of the same town. Expected sign negative. Foggy Cityscapes row = pre-registered F2 (fails).

Sources: results/calib_dose, results/calib_dose_rep, results/calib_dose_arch2, results/calib_dose_foggy, results/calib_dose/dose_lowlevel.json

## T13. Asymmetry: the clean→degraded direction is harder than degraded→clean (same frames, each detector scored on its own held-out frames)

| quantity | sun25 | sun10 | sun3 | night | fog25 | fog50 | fog75 | wet | rain |
|---|---|---|---|---|---|---|---|---|---|
| seed 42: clean-trained on dose frames (pp drop) | 0.9 | 3.0 | 5.3 | 19.6 | 11.5 | 23.1 | 27.7 | 2.0 | 7.1 |
| seed 42: dose-trained on clean frames (pp drop) | 0.3 | 0.5 | 3.7 | 3.3 | 4.7 | 16.0 | 18.6 | 1.8 | 0.9 |
| seed 42: towns where clean→dose is harder | 5/8 | 8/8 | 6/8 | 8/8 | 8/8 | 7/8 | 8/8 | 4/8 | 7/8 |
| seed 17: clean-trained on dose frames (pp drop) | 1.3 | 2.6 | 4.6 | 19.2 | 12.0 | 21.9 | 26.4 | 1.4 | 6.2 |
| seed 17: dose-trained on clean frames (pp drop) | 0.6 | 1.1 | 3.1 | 2.8 | 5.3 | 15.8 | 18.5 | 1.0 | 1.3 |
| seed 17: towns where clean→dose is harder | 5/8 | 6/8 | 5/8 | 8/8 | 7/8 | 6/8 | 7/8 | 4/8 | 6/8 |

Clean→dose harder in 61/72 pairs (seed 42) and 54/72 (seed 17). Symmetric distances (FID, MMD, CORAL) cannot represent this; directed coverage / NN distance can.

Sources: results/calib_dose, results/calib_dose_rep

## T14. Foggy Cityscapes (third real target): town-mean mAP of dose-trained detectors on real fog of increasing density

| real target | clear-trained | fog25-trained | fog50-trained | fog75-trained | fog75 − clear [CI] | towns fog75 > clear |
|---|---|---|---|---|---|---|
| clear (β = 0) | 14.2 | 14.4 | 12.3 | 10.4 | -3.8 [-5.52, -2.06] | 0/8 |
| β = 0.005 | 12.8 | 14.7 | 12.9 | 10.5 | -2.3 [-3.84, -0.62] | 1/8 |
| β = 0.01 | 12.0 | 14.8 | 13.4 | 11.2 | -0.8 [-2.18, +0.77] | 3/8 |
| β = 0.02 | 10.0 | 14.8 | 13.1 | 11.7 | +1.7 [-0.21, +3.43] | 6/8 |

F1 (pre-registered): fog75 beats clear-trained at β = .02 in 6/8 towns (sign test p = 0.145), benefit monotone in β = True → PASS. Mild simulated fog (fog25) is best at every real-fog level with no loss on clear images. F2/F3 (FID-to-foggy predicts the benefit) fail: see T12b.

Sources: results/calib_dose_foggy, results/calib_dose

## T15. Decomposition of real transfer (pp of real mAP): pretraining ≫ labeled content ≫ appearance

| target | COCO vs ImageNet init (same 24 cells) | spread of town means (ImageNet grid) | per doubling of labeled objects (causal) | weather share of variance (24-cell grid) | appearance at fixed content (frame-matched) |
|---|---|---|---|---|---|
| KITTI | +20.4 | 10.5 (town means 23.0–33.5) | +2.9 | 12% | fog75 -9.0; night +1.3 |
| Cityscapes | +16.8 | 4.4 (town means 11.4–15.8) | +0.7 | 0% | fog75 -3.7; night +1.5 |

Pretraining and content are between-configuration effects; the appearance column is the only within-configuration (content-fixed) manipulation.

Sources: results/calib_anchored, results/calib_imnet, results/calib_imnet_lc, results/calib_dose, results/calib_dose_rep

## T16. Held-out-dose prediction (pre-registered, predictions hashed before evaluation): metric calibrated on the 9 original doses predicts 9 unseen doses (72 pairs)

| metric | Spearman(pred, obs) [CI] | MAE (pp) | 90% PI coverage | interp. ρ / MAE | extrap. ρ / MAE | compound ρ / MAE | novel ρ / MAE |
|---|---|---|---|---|---|---|---|
| object FID (primary) | +0.855 [+0.77, +0.94] | 5.58 | 82% | +0.81 / 5.4 | +0.38 / 6.1 | +0.68 / 7.3 | +0.16 / 2.9 |
| directed NN distance | +0.831 [+0.73, +0.93] | 6.90 | 86% | +0.83 / 5.6 | +0.40 / 7.1 | +0.66 / 7.5 | -0.20 / 7.8 |
| coverage | +0.822 [+0.73, +0.92] | 6.35 | 86% | +0.78 / 5.4 | +0.69 / 7.9 | +0.68 / 6.9 | +0.11 / 6.2 |
| paired object displacement | +0.837 [+0.74, +0.93] | 6.82 | 85% | +0.82 / 5.6 | +0.43 / 7.3 | +0.68 / 7.6 | -0.17 / 7.2 |
| Inception FID (no boxes) | +0.828 [+0.70, +0.94] | 6.72 | 85% | +0.78 / 6.6 | +0.14 / 8.3 | +0.76 / 7.5 | -0.26 / 4.9 |
| LPIPS (no boxes) | +0.809 [+0.73, +0.90] | 7.28 | 85% | +0.82 / 6.9 | +0.02 / 10.9 | +0.64 / 7.4 | -0.24 / 5.8 |
| DINOv3 image FID (no boxes) | +0.800 [+0.68, +0.91] | 8.32 | 83% | +0.76 / 8.2 | +0.29 / 12.9 | +0.71 / 8.4 | -0.28 / 6.2 |

Rules (object FID): ρ ≥ .7, MAE ≤ 7.0 pp and < predict-the-calibration-mean (11.90), coverage ≥ 80%. Per dose (observed / predicted, pp): fog35 16.8 / 9.7; fog65 25.3 / 22.7; fog90 28.8 / 31.2; sun6 3.9 / 5.8; night_rain 28.7 / 29.1; night_fog50 24.1 / 25.2; sun10_wet 3.4 / 6.5; cloud90 0.1 / 3.1; dust50 2.9 / 0.2.

Sources: results/calib_dose_ho, results/calib_dose_ho_rep, results/calib_dose, results/calib_dose_rep

## T17. Real fog: RTTS (4,322 real hazy images; Car + Pedestrian mAP@0.5, difficult boxes included) — dose-trained detectors, town mean

| detectors | clear-trained | fog25 (Δ [CI]; towns better) | fog50 | fog75 |
|---|---|---|---|---|
| seed 42 | 15.7 | 20.5 (+4.8 [+3.38, +6.06]; 8/8) | 20.4 (+4.7 [+2.46, +6.98]; 8/8) | 19.0 (+3.2 [+0.53, +5.84]; 7/8) |
| seed 17 | 16.1 | 21.4 (+5.3 [+4.05, +6.72]; 8/8) | 20.8 (+4.7 [+3.33, +6.11]; 8/8) | 17.9 (+1.8 [-0.24, +3.56]; 7/8) |

Δ FID-to-RTTS vs Δ RTTS mAP over 72 pairs: ρ -0.441 [-0.70, -0.10], clustered p = <.001. Every non-clear training dose also beats clear on RTTS (seed 42: sun25 17.8, sun10 18.9, sun3 17.5, night 16.5, wet 16.0, rain 17.9); fog's gain over the non-fog mean is ≈ +3 pp.

Sources: results/calib_dose_rtts, results/calib_dose_rtts_rep

## T18. The measurement across detector families (same renders, same detector-independent DINOv3 metrics)

| detector family | towns monotone in sun and fog (R1) | object FID vs appearance-only drop, ρ (R2) | KITTI Δ mAP of fog75-trained (pp) | Δ FID-to-KITTI vs Δ KITTI mAP, ρ (R3) |
|---|---|---|---|---|
| Faster R-CNN R50 (8 towns, seed 42) | 8/8 | +0.840 | -9.5 | -0.528 (p=<.001) |
| FCOS R50 (2 towns) | 2/2 | +0.882 | -12.0 | -0.730 (p=<.001) |
| Faster R-CNN Swin-T (2 towns) | 2/2 | +0.868 | -1.3 | +0.096 (p=0.711) |

Deformable DETR (ImageNet-init) did not train on 800-frame cells (own-cell mAP 0.0; wrapper verified; §15.68). R3 holds where a training appearance costs real transfer (R50 families) and has nothing to predict where it does not (Swin-T).

Sources: results/calib_dose, results/calib_dose_arch2, results/calib_dose_arch3

## T19. Second rendering engine: Virtual KITTI 2 (Unity; frame-matched variants; Car AP@0.5) and the content-matched sim→real gap (KITTI tracking, same sequences)

| scene | held-out GT cars | morning drop | sunset | overcast | fog | rain | sim→real drop (clone det. on real frames) | object FID fog / real | KITTI Car AP Δ, fog- vs clone-trained |
|---|---|---|---|---|---|---|---|---|---|
| Scene01 | 314 | -0.1 | -0.1 | 0.0 | 27.2 | 0.7 | 1.1 | 260 / 613 | -10.4 |
| Scene02 | 215 | 0.0 | 0.0 | 0.0 | 68.3 | 8.3 | 69.0 | 389 / 1083 | -13.8 |
| Scene06 | 19 | 97.8 | 93.6 | 28.7 | 99.2 | 97.0 | 40.7 | 267 / 662 | -13.2 |
| Scene18 | 426 | 0.2 | 0.4 | 1.0 | 9.1 | 1.1 | 8.1 | 265 / 617 | -3.4 |
| Scene20 | 615 | 8.4 | 4.5 | -0.4 | 36.1 | 8.3 | 16.6 | 242 / 564 | -23.3 |

Drops = clone-trained detector, held-out frames, seed mean (pp). Object FID vs drop over 25 pairs ρ +0.595 (V2 fails). Δ FID-to-KITTI vs Δ KITTI Car AP ρ -0.608, clustered p = <.001 (V4 passes). Scene06's held-out stretch has 19 cars (unstable AP). In the reliable scenes the sim→real feature distance is 2.3–2.8× fog's while the sim→real drop is no larger than fog's.

Sources: results/calib_vkitti, results/calib_vkitti_rep
