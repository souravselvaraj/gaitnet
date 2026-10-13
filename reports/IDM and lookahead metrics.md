# IDM and lookahead: all metrics

Generated 2026-10-12 ~18:40 UTC from training/gaitnet/evals/{step0_variant_*,idm_*} and training/gaitnet/lookahead/full_*. Each row: 150 robots per cell, 3 cells in the 10-30% band, 1 cell at 40%. 'train' = the controller the policies were trained on (gaitnet variant); 'other' = gaitnet_core (compensates body travel, different mass and gains). Success = survived and walked >= half the commanded distance; distance = fraction of commanded distance walked (all robots); falls in % of robots; landing along = mean foot touchdown minus the chosen foothold along the walking direction. Re-runs of the same configuration differ by up to ~0.04-0.06 in rates.

### SC@645
| method | controller | speed | voids | success | survival | distance | void falls % | tip falls % | steps/tick | landing along (cm) |
|---|---|---|---|---|---|---|---|---|---|---|
| GaitNet alone | train | 0.1 | 10-30% | 0.96 | 0.96 | 0.85 | 4 | 0.0 | 0.375 | +1.3 |
| GaitNet alone | train | 0.1 | 40% | 0.82 | 0.82 | 0.80 | 18 | 0.0 | 0.373 | +1.3 |
| GaitNet alone | train | 0.2 | 10-30% | 0.38 | 0.38 | 0.60 | 62 | 0.2 | 0.390 | +2.9 |
| GaitNet alone | train | 0.2 | 40% | 0.09 | 0.13 | 0.40 | 87 | 0.0 | 0.389 | +2.8 |
| GaitNet alone (rerun) | train | 0.1 | 10-30% | 0.97 | 0.97 | 0.85 | 3 | 0.0 | 0.376 | +1.3 |
| GaitNet alone (rerun) | train | 0.1 | 40% | 0.86 | 0.86 | 0.83 | 14 | 0.0 | 0.371 | +1.3 |
| GaitNet alone (rerun) | train | 0.2 | 10-30% | 0.42 | 0.42 | 0.61 | 57 | 0.2 | 0.389 | +2.9 |
| GaitNet alone (rerun) | train | 0.2 | 40% | 0.12 | 0.14 | 0.43 | 86 | 0.0 | 0.395 | +2.8 |
| + IDM literal | train | 0.1 | 10-30% | 1.00 | 1.00 | 0.87 | 0 | 0.0 | 0.377 | -0.3 |
| + IDM literal | train | 0.1 | 40% | 0.97 | 0.97 | 0.88 | 3 | 0.7 | 0.374 | -0.2 |
| + IDM literal | train | 0.2 | 10-30% | 0.98 | 0.98 | 0.89 | 2 | 0.0 | 0.401 | -0.0 |
| + IDM literal | train | 0.2 | 40% | 0.89 | 0.92 | 0.84 | 8 | 0.0 | 0.400 | +0.0 |
| + lookahead d1 | train | 0.1 | 10-30% | 0.97 | 0.97 | 0.85 | 2 | 1.1 | 0.376 | +1.3 |
| + lookahead d1 | train | 0.1 | 40% | 0.95 | 0.95 | 0.86 | 4 | 0.7 | 0.375 | +1.3 |
| + lookahead d1 | train | 0.2 | 10-30% | 0.95 | 0.95 | 0.86 | 4 | 0.7 | 0.391 | +2.9 |
| + lookahead d1 | train | 0.2 | 40% | 0.76 | 0.79 | 0.78 | 15 | 5.3 | 0.394 | +2.8 |
| + lookahead d2 | train | 0.1 | 10-30% | 0.98 | 0.98 | 0.85 | 2 | 0.2 | 0.375 | +1.3 |
| + lookahead d2 | train | 0.1 | 40% | 0.88 | 0.88 | 0.82 | 9 | 2.7 | 0.372 | +1.3 |
| + lookahead d2 | train | 0.2 | 10-30% | 0.96 | 0.96 | 0.86 | 3 | 0.2 | 0.390 | +2.9 |
| + lookahead d2 | train | 0.2 | 40% | 0.73 | 0.76 | 0.75 | 20 | 4.0 | 0.392 | +2.8 |
| + lookahead d2 free | train | 0.1 | 10-30% | 0.98 | 0.98 | 0.85 | 1 | 0.2 | 0.361 | +1.3 |
| + lookahead d2 free | train | 0.1 | 40% | 0.90 | 0.90 | 0.83 | 8 | 2.0 | 0.336 | +1.4 |
| + lookahead d2 free | train | 0.2 | 10-30% | 0.96 | 0.96 | 0.85 | 4 | 0.2 | 0.373 | +2.9 |
| + lookahead d2 free | train | 0.2 | 40% | 0.74 | 0.77 | 0.75 | 21 | 2.0 | 0.355 | +2.9 |
| GaitNet alone | other | 0.1 | 10-30% | 0.92 | 0.92 | 0.85 | 6 | 1.8 | 0.379 | -0.6 |
| GaitNet alone | other | 0.1 | 40% | 0.61 | 0.61 | 0.75 | 28 | 10.7 | 0.380 | -0.6 |
| GaitNet alone | other | 0.2 | 10-30% | 0.77 | 0.77 | 0.79 | 14 | 8.4 | 0.406 | -0.8 |
| GaitNet alone | other | 0.2 | 40% | 0.37 | 0.41 | 0.60 | 45 | 14.0 | 0.426 | -0.8 |
| + IDM literal | other | 0.1 | 10-30% | 0.97 | 0.97 | 0.87 | 3 | 0.4 | 0.375 | +0.1 |
| + IDM literal | other | 0.1 | 40% | 0.89 | 0.89 | 0.86 | 11 | 0.7 | 0.372 | +0.1 |
| + IDM literal | other | 0.2 | 10-30% | 0.94 | 0.94 | 0.88 | 5 | 1.1 | 0.400 | +0.1 |
| + IDM literal | other | 0.2 | 40% | 0.76 | 0.79 | 0.79 | 17 | 3.3 | 0.417 | +0.1 |
| + IDM faithful | other | 0.1 | 10-30% | 0.84 | 0.84 | 0.82 | 16 | 0.4 | 0.374 | +1.5 |
| + IDM faithful | other | 0.1 | 40% | 0.49 | 0.49 | 0.69 | 51 | 0.0 | 0.373 | +1.5 |
| + IDM faithful | other | 0.2 | 10-30% | 0.36 | 0.36 | 0.58 | 63 | 0.7 | 0.388 | +2.7 |
| + IDM faithful | other | 0.2 | 40% | 0.02 | 0.05 | 0.33 | 93 | 1.3 | 0.409 | +2.7 |

### WB500@1160
| method | controller | speed | voids | success | survival | distance | void falls % | tip falls % | steps/tick | landing along (cm) |
|---|---|---|---|---|---|---|---|---|---|---|
| GaitNet alone | train | 0.1 | 10-30% | 0.92 | 0.92 | 0.74 | 8 | 0.0 | 0.171 | +1.9 |
| GaitNet alone | train | 0.1 | 40% | 0.75 | 0.75 | 0.70 | 25 | 0.0 | 0.173 | +1.9 |
| GaitNet alone | train | 0.2 | 10-30% | 0.38 | 0.38 | 0.52 | 62 | 0.0 | 0.228 | +3.7 |
| GaitNet alone | train | 0.2 | 40% | 0.01 | 0.05 | 0.26 | 95 | 0.0 | 0.261 | +3.7 |
| GaitNet alone (rerun) | train | 0.1 | 10-30% | 0.92 | 0.92 | 0.74 | 8 | 0.0 | 0.171 | +1.9 |
| GaitNet alone (rerun) | train | 0.1 | 40% | 0.75 | 0.75 | 0.70 | 25 | 0.0 | 0.173 | +1.9 |
| GaitNet alone (rerun) | train | 0.2 | 10-30% | 0.38 | 0.38 | 0.52 | 62 | 0.0 | 0.228 | +3.7 |
| GaitNet alone (rerun) | train | 0.2 | 40% | 0.01 | 0.05 | 0.26 | 95 | 0.0 | 0.261 | +3.7 |
| + IDM literal | train | 0.1 | 10-30% | 0.97 | 0.97 | 0.75 | 3 | 0.0 | 0.194 | -1.3 |
| + IDM literal | train | 0.1 | 40% | 0.92 | 0.92 | 0.75 | 8 | 0.0 | 0.196 | -1.2 |
| + IDM literal | train | 0.2 | 10-30% | 0.98 | 0.98 | 0.79 | 2 | 0.0 | 0.264 | -0.4 |
| + IDM literal | train | 0.2 | 40% | 0.89 | 0.93 | 0.75 | 7 | 0.0 | 0.290 | -0.3 |
| + lookahead d1 | train | 0.1 | 10-30% | 0.97 | 0.97 | 0.76 | 3 | 0.2 | 0.173 | +1.9 |
| + lookahead d1 | train | 0.1 | 40% | 0.89 | 0.89 | 0.74 | 10 | 1.3 | 0.176 | +1.9 |
| + lookahead d1 | train | 0.2 | 10-30% | 0.87 | 0.87 | 0.74 | 12 | 0.7 | 0.230 | +3.7 |
| + lookahead d1 | train | 0.2 | 40% | 0.51 | 0.55 | 0.60 | 39 | 6.0 | 0.266 | +3.6 |
| + lookahead d2 | train | 0.1 | 10-30% | 0.97 | 0.97 | 0.76 | 3 | 0.4 | 0.173 | +1.9 |
| + lookahead d2 | train | 0.1 | 40% | 0.87 | 0.87 | 0.73 | 12 | 1.3 | 0.177 | +1.9 |
| + lookahead d2 | train | 0.2 | 10-30% | 0.88 | 0.88 | 0.75 | 10 | 1.6 | 0.230 | +3.7 |
| + lookahead d2 | train | 0.2 | 40% | 0.47 | 0.51 | 0.58 | 45 | 4.7 | 0.264 | +3.6 |
| + lookahead d2 free | train | 0.1 | 10-30% | 0.96 | 0.96 | 0.76 | 4 | 0.0 | 0.169 | +1.9 |
| + lookahead d2 free | train | 0.1 | 40% | 0.93 | 0.93 | 0.75 | 6 | 0.7 | 0.169 | +1.9 |
| + lookahead d2 free | train | 0.2 | 10-30% | 0.85 | 0.85 | 0.74 | 14 | 0.9 | 0.224 | +3.7 |
| + lookahead d2 free | train | 0.2 | 40% | 0.56 | 0.59 | 0.64 | 35 | 5.3 | 0.250 | +3.7 |
| GaitNet alone | other | 0.1 | 10-30% | 0.89 | 0.89 | 0.82 | 9 | 2.2 | 0.209 | -1.4 |
| GaitNet alone | other | 0.1 | 40% | 0.61 | 0.61 | 0.72 | 34 | 4.7 | 0.208 | -1.3 |
| GaitNet alone | other | 0.2 | 10-30% | 0.66 | 0.66 | 0.71 | 27 | 6.7 | 0.264 | -1.4 |
| GaitNet alone | other | 0.2 | 40% | 0.27 | 0.31 | 0.53 | 56 | 13.3 | 0.292 | -1.4 |
| + IDM literal | other | 0.1 | 10-30% | 0.75 | 0.75 | 0.77 | 24 | 1.1 | 0.210 | -1.2 |
| + IDM literal | other | 0.1 | 40% | 0.48 | 0.48 | 0.70 | 49 | 3.3 | 0.208 | -1.0 |
| + IDM literal | other | 0.2 | 10-30% | 0.75 | 0.75 | 0.75 | 23 | 2.0 | 0.263 | -0.9 |
| + IDM literal | other | 0.2 | 40% | 0.47 | 0.50 | 0.62 | 44 | 6.0 | 0.289 | -0.8 |
| + IDM faithful | other | 0.1 | 10-30% | 0.71 | 0.71 | 0.80 | 27 | 2.7 | 0.192 | +1.3 |
| + IDM faithful | other | 0.1 | 40% | 0.37 | 0.37 | 0.63 | 62 | 1.3 | 0.195 | +1.5 |
| + IDM faithful | other | 0.2 | 10-30% | 0.27 | 0.27 | 0.51 | 70 | 2.9 | 0.244 | +2.3 |
| + IDM faithful | other | 0.2 | 40% | 0.07 | 0.10 | 0.32 | 87 | 2.7 | 0.275 | +2.3 |

### SDc@1035
| method | controller | speed | voids | success | survival | distance | void falls % | tip falls % | steps/tick | landing along (cm) |
|---|---|---|---|---|---|---|---|---|---|---|
| GaitNet alone | train | 0.1 | 10-30% | 0.91 | 0.91 | 0.88 | 8 | 0.9 | 0.226 | +1.6 |
| GaitNet alone | train | 0.1 | 40% | 0.71 | 0.71 | 0.80 | 25 | 4.7 | 0.291 | +1.5 |
| GaitNet alone | train | 0.2 | 10-30% | 0.68 | 0.68 | 0.78 | 32 | 0.2 | 0.326 | +3.1 |
| GaitNet alone | train | 0.2 | 40% | 0.12 | 0.15 | 0.45 | 79 | 6.0 | 0.352 | +3.1 |
| GaitNet alone (rerun) | train | 0.1 | 10-30% | 0.90 | 0.90 | 0.87 | 8 | 2.2 | 0.226 | +1.6 |
| GaitNet alone (rerun) | train | 0.1 | 40% | 0.75 | 0.75 | 0.82 | 19 | 6.0 | 0.295 | +1.5 |
| GaitNet alone (rerun) | train | 0.2 | 10-30% | 0.66 | 0.66 | 0.76 | 33 | 1.3 | 0.326 | +3.1 |
| GaitNet alone (rerun) | train | 0.2 | 40% | 0.13 | 0.17 | 0.48 | 73 | 10.0 | 0.355 | +3.1 |
| + IDM literal | train | 0.1 | 10-30% | 0.92 | 0.92 | 0.89 | 3 | 4.4 | 0.235 | -0.3 |
| + IDM literal | train | 0.1 | 40% | 0.83 | 0.83 | 0.85 | 13 | 4.0 | 0.301 | -0.3 |
| + IDM literal | train | 0.2 | 10-30% | 0.95 | 0.95 | 0.91 | 4 | 0.9 | 0.345 | +0.0 |
| + IDM literal | train | 0.2 | 40% | 0.55 | 0.58 | 0.72 | 29 | 12.7 | 0.369 | +0.1 |
| + lookahead d1 | train | 0.1 | 10-30% | 0.92 | 0.92 | 0.88 | 4 | 4.0 | 0.227 | +1.6 |
| + lookahead d1 | train | 0.1 | 40% | 0.65 | 0.65 | 0.80 | 23 | 12.0 | 0.295 | +1.5 |
| + lookahead d1 | train | 0.2 | 10-30% | 0.84 | 0.84 | 0.85 | 10 | 6.0 | 0.329 | +3.1 |
| + lookahead d1 | train | 0.2 | 40% | 0.21 | 0.24 | 0.53 | 49 | 27.3 | 0.357 | +3.0 |
| + lookahead d2 | train | 0.1 | 10-30% | 0.92 | 0.92 | 0.88 | 3 | 4.9 | 0.227 | +1.6 |
| + lookahead d2 | train | 0.1 | 40% | 0.68 | 0.68 | 0.80 | 20 | 12.0 | 0.294 | +1.5 |
| + lookahead d2 | train | 0.2 | 10-30% | 0.89 | 0.89 | 0.86 | 6 | 5.1 | 0.329 | +3.1 |
| + lookahead d2 | train | 0.2 | 40% | 0.27 | 0.30 | 0.57 | 41 | 29.3 | 0.361 | +3.0 |
| + lookahead d2 free | train | 0.1 | 10-30% | 0.95 | 0.95 | 0.88 | 2 | 2.4 | 0.223 | +1.6 |
| + lookahead d2 free | train | 0.1 | 40% | 0.74 | 0.74 | 0.80 | 19 | 7.3 | 0.286 | +1.5 |
| + lookahead d2 free | train | 0.2 | 10-30% | 0.87 | 0.87 | 0.86 | 9 | 4.2 | 0.324 | +3.1 |
| + lookahead d2 free | train | 0.2 | 40% | 0.38 | 0.41 | 0.62 | 37 | 22.0 | 0.348 | +3.0 |
| GaitNet alone | other | 0.1 | 10-30% | 0.54 | 0.54 | 0.75 | 12 | 33.8 | 0.220 | -1.0 |
| GaitNet alone | other | 0.1 | 40% | 0.26 | 0.26 | 0.58 | 27 | 46.7 | 0.264 | -0.7 |
| GaitNet alone | other | 0.2 | 10-30% | 0.52 | 0.52 | 0.72 | 16 | 32.0 | 0.340 | -1.1 |
| GaitNet alone | other | 0.2 | 40% | 0.06 | 0.09 | 0.36 | 37 | 53.3 | 0.353 | -1.0 |
| + IDM literal | other | 0.1 | 10-30% | 0.66 | 0.66 | 0.81 | 7 | 27.6 | 0.216 | -0.0 |
| + IDM literal | other | 0.1 | 40% | 0.45 | 0.45 | 0.68 | 29 | 26.0 | 0.270 | +0.1 |
| + IDM literal | other | 0.2 | 10-30% | 0.74 | 0.74 | 0.83 | 15 | 10.9 | 0.333 | -0.0 |
| + IDM literal | other | 0.2 | 40% | 0.18 | 0.21 | 0.49 | 43 | 35.3 | 0.348 | +0.2 |
| + IDM faithful | other | 0.1 | 10-30% | 0.57 | 0.57 | 0.76 | 20 | 22.4 | 0.205 | +1.9 |
| + IDM faithful | other | 0.1 | 40% | 0.34 | 0.35 | 0.63 | 48 | 17.3 | 0.258 | +1.9 |
| + IDM faithful | other | 0.2 | 10-30% | 0.48 | 0.48 | 0.69 | 44 | 8.9 | 0.314 | +2.9 |
| + IDM faithful | other | 0.2 | 40% | 0.02 | 0.05 | 0.31 | 73 | 21.3 | 0.330 | +2.9 |

### base1035
| method | controller | speed | voids | success | survival | distance | void falls % | tip falls % | steps/tick | landing along (cm) |
|---|---|---|---|---|---|---|---|---|---|---|
| GaitNet alone | train | 0.1 | 10-30% | 0.82 | 0.82 | 0.70 | 17 | 0.4 | 0.068 | +1.8 |
| GaitNet alone | train | 0.1 | 40% | 0.57 | 0.57 | 0.63 | 43 | 0.0 | 0.071 | +1.9 |
| GaitNet alone | train | 0.2 | 10-30% | 0.17 | 0.17 | 0.39 | 83 | 0.2 | 0.133 | +3.9 |
| GaitNet alone | train | 0.2 | 40% | 0.00 | 0.03 | 0.19 | 96 | 0.7 | 0.135 | +3.9 |
| + IDM literal | train | 0.1 | 10-30% | 0.94 | 0.94 | 0.72 | 6 | 0.0 | 0.089 | -1.9 |
| + IDM literal | train | 0.1 | 40% | 0.86 | 0.86 | 0.71 | 12 | 2.0 | 0.093 | -1.7 |
| + IDM literal | train | 0.2 | 10-30% | 0.93 | 0.93 | 0.78 | 5 | 2.0 | 0.175 | -0.8 |
| + IDM literal | train | 0.2 | 40% | 0.81 | 0.84 | 0.73 | 13 | 3.3 | 0.172 | -0.7 |
| GaitNet alone | other | 0.1 | 10-30% | 0.78 | 0.78 | 0.72 | 12 | 9.3 | 0.096 | -0.7 |
| GaitNet alone | other | 0.1 | 40% | 0.59 | 0.59 | 0.65 | 25 | 16.0 | 0.102 | -0.8 |
| GaitNet alone | other | 0.2 | 10-30% | 0.41 | 0.41 | 0.56 | 26 | 33.6 | 0.182 | -0.9 |
| GaitNet alone | other | 0.2 | 40% | 0.07 | 0.10 | 0.38 | 55 | 35.3 | 0.177 | -0.9 |
| + IDM literal | other | 0.1 | 10-30% | 0.73 | 0.73 | 0.70 | 16 | 11.1 | 0.096 | -0.4 |
| + IDM literal | other | 0.1 | 40% | 0.43 | 0.43 | 0.61 | 35 | 21.3 | 0.102 | -0.4 |
| + IDM literal | other | 0.2 | 10-30% | 0.41 | 0.41 | 0.56 | 31 | 28.4 | 0.177 | -0.3 |
| + IDM literal | other | 0.2 | 40% | 0.13 | 0.17 | 0.39 | 53 | 30.7 | 0.171 | -0.3 |
| + IDM faithful | other | 0.1 | 10-30% | 0.48 | 0.49 | 0.62 | 47 | 4.2 | 0.078 | +2.7 |
| + IDM faithful | other | 0.1 | 40% | 0.12 | 0.13 | 0.46 | 83 | 4.0 | 0.081 | +2.7 |
| + IDM faithful | other | 0.2 | 10-30% | 0.14 | 0.14 | 0.36 | 78 | 7.6 | 0.143 | +3.4 |
| + IDM faithful | other | 0.2 | 40% | 0.01 | 0.04 | 0.20 | 90 | 6.0 | 0.141 | +3.4 |
