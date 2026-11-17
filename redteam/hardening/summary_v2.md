# Final hardening audit (selected regime: `R20f5`; reference: `full__R20f5__s0`)

## Regime search (macro AUC, raw before the gate; mean ± sd over seeds 0-2)

| regime | full val | clip val | full test | clip test | paired full−clip test (per seed) | full gate pass (test) |
|---|---|---|---|---|---|---|
| R10all | 61.50 ± 4.13 | 65.75 ± 2.28 | 60.76 ± 1.65 | 61.57 ± 3.49 | -0.81 ± 4.52 | 3/3 |
| R20all | 53.70 ± 2.54 | 57.57 ± 1.70 | 52.67 ± 1.60 | 58.28 ± 2.72 | -5.61 ± 4.09 | 1/3 |
| R20f5 | 70.58 ± 3.39 | 69.14 ± 1.43 | 69.16 ± 2.12 | 64.04 ± 2.47 | 5.11 ± 0.87 | 3/3 |
| R20f5b128 | 61.47 ± 4.43 | 65.26 ± 1.85 | 58.47 ± 3.11 | 59.35 ± 3.50 | -0.88 ± 1.97 | 3/3 |
| R3all | 66.70 ± 4.50 | 70.26 ± 2.47 | 68.78 ± 0.92 | 68.73 ± 1.72 | 0.05 ± 2.02 | 3/3 |
| R5all | 67.79 ± 3.50 | 69.65 ± 1.51 | 66.07 ± 2.01 | 66.60 ± 2.06 | -0.53 ± 4.00 | 3/3 |

## Selected regime `R20f5`: hidden-test metrics (scored values; gate failures are 0)

| run | status | AUC | prec | recall | R@5 | R@10 | kNN F1 | val AUC (raw) | ΔAUC vs ref [95% CI] | rank Δ (all runs) | rank Δ (passing runs) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aug__R20f5__s0 | ok | 71.13 | 52.09 | 78.45 | 15.73 | 26.80 | 96.87 | 71.15 | +3.12 [+0.62, +5.57] | +0.472 | +1.138 |
| aug__R20f5__s1 | ok | 71.07 | 54.46 | 80.44 | 14.60 | 25.13 | 96.53 | 71.28 | +3.06 [+1.03, +5.22] | +0.498 | +1.311 |
| aug__R20f5__s2 | ok | 71.80 | 54.58 | 77.89 | 14.20 | 25.67 | 96.27 | 73.03 | +3.79 [+1.55, +6.23] | +0.507 | +1.360 |
| clip__R20f5__s0 | ok | 61.92 | 49.03 | 57.68 | 9.73 | 17.87 | 91.86 | 67.49 | -6.09 [-10.30, -2.01] | -0.427 | -1.857 |
| clip__R20f5__s1 | ok | 66.76 | 52.02 | 62.46 | 9.93 | 16.47 | 91.66 | 69.86 | -1.25 [-4.71, +2.21] | -0.149 | -0.801 |
| clip__R20f5__s2 | ok | 63.46 | 48.74 | 61.66 | 10.00 | 17.93 | 90.86 | 70.06 | -4.55 [-8.39, -0.60] | -0.345 | -1.699 |
| clip__R20f5__s3 | ok | 64.30 | 49.72 | 63.91 | 9.53 | 16.87 | 90.47 | 69.17 | -3.70 [-7.50, +0.01] | -0.298 | -1.514 |
| clip__R20f5__s4 | ok | 65.18 | 52.89 | 57.81 | 9.80 | 17.60 | 91.40 | 68.31 | -2.83 [-6.55, +0.81] | -0.214 | -1.029 |
| ens3_clip__R20f5 | ok | 65.04 | 50.69 | 61.57 | 10.40 | 18.73 | 91.46 | 70.36 | -2.97 [-6.78, +0.83] | -0.212 | -1.164 |
| ens3_full__R20f5 | ok | 71.21 | 51.11 | 74.94 | 11.27 | 19.40 | 96.73 | 72.34 | +3.20 [+1.80, +4.62] | +0.229 | +0.680 |
| full__R20f5__s0 | ok | 68.01 | 51.24 | 70.19 | 9.47 | 17.07 | 96.40 | 68.58 | ref | +0.000 | +0.000 |
| full__R20f5__s1 | ok | 71.60 | 48.84 | 77.98 | 7.53 | 15.00 | 95.93 | 74.49 | +3.60 [+1.02, +6.17] | +0.079 | +0.268 |
| full__R20f5__s2 | ok | 67.87 | 51.09 | 66.87 | 8.47 | 14.93 | 96.33 | 68.66 | -0.14 [-2.00, +1.74] | -0.089 | -0.180 |
| full__R20f5__s3 | ok | 67.58 | 49.36 | 71.69 | 7.33 | 13.20 | 96.47 | 69.91 | -0.42 [-2.21, +1.30] | -0.144 | -0.370 |
| full__R20f5__s4 | ok | 66.52 | 51.55 | 57.35 | 5.80 | 12.13 | 96.67 | 72.12 | -1.48 [-3.80, +0.81] | -0.295 | -0.593 |
| neg__R20f5__s0 | ok | 69.13 | 54.20 | 61.55 | 7.53 | 12.27 | 90.93 | 67.95 | +1.12 [-1.78, +3.99] | -0.126 | -0.485 |
| neg__R20f5__s1 | ok | 68.19 | 54.91 | 62.47 | 7.00 | 12.73 | 91.67 | 70.60 | +0.19 [-2.82, +3.23] | -0.127 | -0.442 |
| neg__R20f5__s2 | ok | 68.05 | 54.57 | 60.19 | 7.20 | 12.73 | 90.46 | 70.25 | +0.05 [-3.11, +3.15] | -0.171 | -0.676 |
| repro_clip_default | ok | 61.92 | 49.03 | 57.68 | 9.73 | 17.87 | 91.86 | 67.49 | -6.09 [-10.30, -2.01] | -0.427 | -1.857 |
| repro_full_default | ok | 68.01 | 51.24 | 70.19 | 9.47 | 17.07 | 96.40 | 68.58 | +0.00 [+0.00, +0.00] | +0.000 | +0.000 |
| trainval__R20f5__s0 | ok | 67.03 | 50.72 | 66.02 | 10.00 | 17.73 | 96.87 | 80.95 | -0.98 [-3.02, +1.04] | -0.063 | -0.192 |
| trainval__R20f5__s1 | ok | 67.51 | 51.71 | 64.54 | 10.33 | 17.80 | 97.13 | 80.72 | -0.50 [-2.77, +1.61] | -0.025 | -0.004 |
| trainval__R20f5__s2 | ok | 63.83 | 48.59 | 64.19 | 10.93 | 18.40 | 96.33 | 79.70 | -4.18 [-6.69, -1.67] | -0.235 | -0.945 |
| untrained | failed_task_gate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 49.84 | -17.43 [-22.16, -12.68] | -5.897 | -28.558 |
| view__R20f5__s0 | ok | 58.62 | 45.04 | 62.77 | 10.13 | 17.67 | 96.13 | 61.52 | -9.38 [-12.57, -6.31] | -0.560 | -2.167 |
| view__R20f5__s1 | ok | 62.78 | 48.08 | 59.98 | 10.20 | 18.20 | 96.13 | 66.01 | -5.23 [-8.25, -2.15] | -0.341 | -1.273 |
| view__R20f5__s2 | ok | 61.68 | 45.92 | 69.54 | 10.20 | 17.53 | 95.93 | 62.71 | -6.32 [-9.48, -3.25] | -0.365 | -1.528 |

## Selected regime: seed statistics (raw macro AUC; rank Δ vs reference)

| method | n | test AUC | val AUC | R@10 | kNN F1 | rank Δ (all runs) | rank Δ (passing runs) | gate pass |
|---|---|---|---|---|---|---|---|---|
| full | 5 | 68.32 ± 1.93 | 70.75 ± 2.53 | 14.47 ± 1.89 | 96.36 ± 0.27 | -0.09 ± 0.14 | -0.18 ± 0.33 | 5/5 |
| clip | 5 | 64.32 ± 1.82 | 68.98 ± 1.08 | 17.35 ± 0.65 | 91.25 ± 0.58 | -0.29 ± 0.11 | -1.38 ± 0.45 | 5/5 |
| view | 3 | 61.03 ± 2.15 | 63.41 ± 2.33 | 17.80 ± 0.35 | 96.07 ± 0.12 | -0.42 ± 0.12 | -1.66 ± 0.46 | 3/3 |
| neg | 3 | 68.46 ± 0.59 | 69.60 ± 1.44 | 12.58 ± 0.27 | 91.02 ± 0.61 | -0.14 ± 0.03 | -0.53 ± 0.12 | 3/3 |
| aug | 3 | 71.33 ± 0.41 | 71.82 ± 1.05 | 25.87 ± 0.85 | 96.56 ± 0.30 | 0.49 ± 0.02 | 1.27 ± 0.12 | 3/3 |
| trainval | 3 | 66.12 ± 2.00 | 80.45 ± 0.67 | 17.98 ± 0.37 | 96.78 ± 0.41 | -0.11 ± 0.11 | -0.38 ± 0.50 | 3/3 |

## Shortcuts: hardened evaluator vs previous evaluator

| shortcut | hardened status | hardened AUC / R@10 / kNN | rank Δ | previous status | previous AUC / R@10 / kNN |
|---|---|---|---|---|---|
| sc_constant | failed_task_gate: macro AUC 50.00 not significantly above chance (p=1.0000 > 0.01) | 0.00 / 0.00 / 0.00 | -5.897 | failed_task_gate | 0.00 / 0.00 / 0.00 |
| sc_dim_mismatch | failed: image dim 513 != text dim 512 | 0.00 / 0.00 / 0.00 | -5.897 | failed | 0.00 / 0.00 / 0.00 |
| sc_empty_outputs | failed: image_encoder must return a float tensor of shape [1, D] | 0.00 / 0.00 / 0.00 | -5.897 | failed | 0.00 / 0.00 / 0.00 |
| sc_file_reader | failed: image_encoder: forbidden TorchScript operations ['aten::from_file', 'p | 0.00 / 0.00 / 0.00 | -5.897 | ok | 100.00 / 0.00 / 100.00 |
| sc_forged_scores | failed: missing export/image_encoder.pt | 0.00 / 0.00 / 0.00 | -5.897 | failed | 0.00 / 0.00 / 0.00 |
| sc_handcrafted | failed_task_gate: macro AUC 52.02 not significantly above chance (p=0.2709 > 0.01) | 0.00 / 0.00 / 0.00 | -5.897 | failed_task_gate | 0.00 / 0.00 / 0.00 |
| sc_huge_constant | failed_task_gate: macro AUC 50.00 not significantly above chance (p=1.0000 > 0.01) | 0.00 / 0.00 / 0.00 | -5.897 | failed_task_gate | 0.00 / 0.00 / 0.00 |
| sc_image_only | failed_task_gate: macro AUC 50.00 not significantly above chance (p=1.0000 > 0.01) | 0.00 / 0.00 / 0.00 | -5.897 | failed_task_gate | 0.00 / 0.00 / 0.00 |
| sc_missing_export | failed: missing export/image_encoder.pt | 0.00 / 0.00 / 0.00 | -5.897 | failed | 0.00 / 0.00 / 0.00 |
| sc_nan_outputs | failed: image_encoder returned non-finite values | 0.00 / 0.00 / 0.00 | -5.897 | failed | 0.00 / 0.00 / 0.00 |
| sc_norm_scaled | ok:  | 68.01 / 17.07 / 96.40 | +0.000 | ok | 68.01 / 17.07 / 96.40 |
| sc_stateful | failed: image_encoder is not a pure function of its input (embedding changed o | 0.00 / 0.00 / 0.00 | -5.897 | ok | 71.24 / 9.73 / 84.28 |
| sc_text_only | failed_task_gate: macro AUC 50.04 not significantly above chance (p=0.4373 > 0.01) | 0.00 / 0.00 / 0.00 | -5.897 | failed_task_gate | 0.00 / 0.00 / 0.00 |
| sc_wrong_shape | failed: image_encoder must return a float tensor of shape [1, D] | 0.00 / 0.00 / 0.00 | -5.897 | failed | 0.00 / 0.00 / 0.00 |
| (reference under previous evaluator) | ok | — | — | — | 68.01 / 17.07 / 96.40 |

Metric SDs over the genuine cohort (57 runs): `{"disease_zero_shot.auc": 17.1006, "disease_zero_shot.precision": 13.0727, "disease_zero_shot.recall": 18.2772, "image_text_retrieval.recall_at_10": 5.8446, "image_text_retrieval.recall_at_5": 3.5933, "view_knn.f1": 24.3172}`
