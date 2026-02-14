# Evaluation report

Generated: 2025-12-15 17:45:07

Results root: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg`

## Summary

| name | split | thr | cc_min | n | mean_dice | median_dice | det_rate | mean_fp_vox | mean_fp_cc | |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| val_postproc_patch48_dwi_adc_flair | - | - | 20 | 25 | 0.0127 | 0.0058 | - | - | - | |
| test_e20_fp_ohem_bal_fromrunbest_cc20 | test | 0.5 | 20 | 25 | 0.4120 | 0.5391 | 0.720 | 257.76 | 0.92 | |
| test_e20_thr022_cc20_score_sweep/score_max_ge0.8 | test | 0.22 | 20 | 25 | 0.3103 | 0.3134 | 0.800 | 1301.96 | 0.12 | |
| test_e20_thr022_cc20_score_sweep/score_p95_ge0.65 | test | 0.22 | 20 | 25 | 0.3064 | 0.3134 | 0.760 | 1121.40 | 0.08 | |
| test_e20_thr022_cc20_score_sweep/score_p95_ge0.7 | test | 0.22 | 20 | 25 | 0.3049 | 0.3134 | 0.720 | 1082.76 | 0.00 | |
| test_e20_thr022_cc20_score_sweep/score_mean_ge0.35 | test | 0.22 | 20 | 25 | 0.3032 | 0.3134 | 0.720 | 1096.36 | 0.04 | |
| test_e20_thr022_cc20_score_sweep/score_p95_ge0.45 | test | 0.22 | 20 | 25 | 0.2965 | 0.2516 | 0.880 | 1502.28 | 0.56 | |
| test_e20_thr022_cc20_score_sweep/score_p95_ge0.6 | test | 0.22 | 20 | 25 | 0.2941 | 0.2891 | 0.800 | 1306.68 | 0.20 | |
| test_smoke2_fp_ohem_autothr_cc20 | test | 0.5 | 20 | 25 | 0.2935 | 0.2407 | 0.720 | 344.64 | 1.76 | |
| test_smoke2_fp_ohem_thr0p5_cc20 | test | 0.5,0.6,0.7 | 20 | 25 | 0.2935 | 0.2407 | 0.720 | 344.64 | 1.76 | |
| test_e20_thr022_cc20_score_sweep/score_max_ge0.85 | test | 0.22 | 20 | 25 | 0.2928 | 0.2366 | 0.760 | 1257.12 | 0.04 | |
| test_e20_thr022_cc20_score_sweep/score_max_ge0.5 | test | 0.22 | 20 | 25 | 0.2925 | 0.1881 | 0.920 | 1629.76 | 0.80 | |
| test_e20_thr022_cc20_score_sweep/score_p95_ge0.55 | test | 0.22 | 20 | 25 | 0.2923 | 0.2451 | 0.800 | 1321.96 | 0.24 | |
| test_e20_thr022_cc20_score_sweep/score_max_ge0.75 | test | 0.22 | 20 | 25 | 0.2910 | 0.2420 | 0.800 | 1401.92 | 0.24 | |
| test_e20_thr022_cc20_score_sweep/score_max_ge0.9 | test | 0.22 | 20 | 25 | 0.2901 | 0.2366 | 0.760 | 1248.96 | 0.04 | |
| test_e20_thr022_cc20_score_sweep/score_max_ge0.6 | test | 0.22 | 20 | 25 | 0.2894 | 0.1910 | 0.880 | 1537.56 | 0.48 | |
| test_e20_thr022_cc20_score_sweep/score_max_ge0.55 | test | 0.22 | 20 | 25 | 0.2891 | 0.1902 | 0.880 | 1546.72 | 0.56 | |
| test_e20_thr022_cc20_score_sweep/score_mean_ge0.3 | test | 0.22 | 20 | 25 | 0.2885 | 0.1902 | 0.840 | 1445.64 | 0.48 | |
| test_e20_thr022_cc20_score_sweep/score_p95_ge0.5 | test | 0.22 | 20 | 25 | 0.2875 | 0.1902 | 0.840 | 1431.56 | 0.40 | |
| test_e20_thr022_cc20_score_sweep/score_max_ge0.7 | test | 0.22 | 20 | 25 | 0.2875 | 0.1910 | 0.840 | 1434.24 | 0.28 | |
| test_e20_thr022_cc20_score_sweep/score_max_ge0.65 | test | 0.22 | 20 | 25 | 0.2872 | 0.1910 | 0.840 | 1475.04 | 0.32 | |
| test_e20_thr022_cc20_score_sweep/score_p95_ge0.4 | test | 0.22 | 20 | 25 | 0.2869 | 0.1881 | 0.880 | 1594.72 | 1.12 | |
| test_e20_thr022_cc20_score_sweep/score_p95_ge0.75 | test | 0.22 | 20 | 25 | 0.2840 | 0.2891 | 0.640 | 1020.44 | 0.00 | |
| test_e20_thr022_cc20_score_sweep/score_max_ge0.45 | test | 0.22 | 20 | 25 | 0.2800 | 0.1861 | 0.920 | 1718.08 | 1.40 | |
| test_e20_thr022_cc20_score_sweep/score_p95_ge0.35 | test | 0.22 | 20 | 25 | 0.2793 | 0.1866 | 0.920 | 1745.20 | 2.12 | |
| test_e20_thr022_cc20_score_sweep/score_max_ge0.95 | test | 0.22 | 20 | 25 | 0.2772 | 0.2105 | 0.680 | 1217.60 | 0.00 | |
| test_e20_thr022_cc20_score_sweep/score_max_ge0.4 | test | 0.22 | 20 | 25 | 0.2746 | 0.1866 | 0.920 | 1788.80 | 2.04 | |
| test_e20_thr022_cc20_score_sweep/score_max_ge0.35 | test | 0.22 | 20 | 25 | 0.2721 | 0.1861 | 0.920 | 1841.20 | 2.84 | |
| test_e20_thr022_min_size_sweep/cc150 | test | 0.22 | 150 | 25 | 0.2713 | 0.1891 | 0.800 | 1684.88 | 0.84 | |
| test_e20_thr022_cc20_score_sweep/score_p95_ge0.3 | test | 0.22 | 20 | 25 | 0.2707 | 0.1861 | 0.920 | 1865.76 | 3.36 | |
| test_e20_thr022_cc20_score_sweep/score_mean_ge0.25 | test | 0.22 | 20 | 25 | 0.2699 | 0.1833 | 0.920 | 1864.04 | 3.40 | |
| test_e20_thr022_min_size_sweep/cc200 | test | 0.22 | 200 | 25 | 0.2697 | 0.1861 | 0.800 | 1643.84 | 0.68 | |
| test_e20_thr022_min_size_sweep/cc75 | test | 0.22 | 75 | 25 | 0.2694 | 0.1866 | 0.840 | 1795.24 | 1.64 | |
| test_e20_thr022_min_size_sweep/cc100 | test | 0.22 | 100 | 25 | 0.2694 | 0.1891 | 0.840 | 1754.36 | 1.28 | |
| test_e20_thr022_min_size_sweep/cc50 | test | 0.22 | 50 | 25 | 0.2689 | 0.1852 | 0.880 | 1864.68 | 2.68 | |
| test_e20_thr022_min_size_sweep/cc40 | test | 0.22 | 40 | 25 | 0.2657 | 0.1852 | 0.880 | 1885.96 | 3.12 | |
| test_e20_thr022_cc20_score_sweep/score_max_ge0.3 | test | 0.22 | 20 | 25 | 0.2650 | 0.1833 | 0.920 | 1901.44 | 4.12 | |
| test_e20_thr022_cc20_score_sweep/score_p95_ge0.25 | test | 0.22 | 20 | 25 | 0.2637 | 0.1833 | 0.920 | 1942.16 | 5.04 | |
| test_e20_thr022_cc20_score_sweep/score_max_ge0.25 | test | 0.22 | 20 | 25 | 0.2637 | 0.1833 | 0.920 | 1943.12 | 5.08 | |
| test_e20_thr022_min_size_sweep/cc30 | test | 0.22 | 30 | 25 | 0.2636 | 0.1843 | 0.880 | 1913.64 | 3.88 | |
| test_e20_thr022_min_size_sweep/cc20 | test | 0.22 | 20 | 25 | 0.2636 | 0.1833 | 0.920 | 1945.04 | 5.16 | |
| test_e20_thr022_cc20_score_sweep/score_max_ge0 | test | 0.22 | 20 | 25 | 0.2636 | 0.1833 | 0.920 | 1945.04 | 5.16 | |
| test_e20_thr022_cc20_score_sweep/score_max_ge0.05 | test | 0.22 | 20 | 25 | 0.2636 | 0.1833 | 0.920 | 1945.04 | 5.16 | |
| test_e20_thr022_cc20_score_sweep/score_max_ge0.1 | test | 0.22 | 20 | 25 | 0.2636 | 0.1833 | 0.920 | 1945.04 | 5.16 | |
| test_e20_thr022_cc20_score_sweep/score_max_ge0.15 | test | 0.22 | 20 | 25 | 0.2636 | 0.1833 | 0.920 | 1945.04 | 5.16 | |
| test_e20_thr022_cc20_score_sweep/score_max_ge0.2 | test | 0.22 | 20 | 25 | 0.2636 | 0.1833 | 0.920 | 1945.04 | 5.16 | |
| test_e20_thr022_cc20_score_sweep/score_mean_ge0 | test | 0.22 | 20 | 25 | 0.2636 | 0.1833 | 0.920 | 1945.04 | 5.16 | |
| test_e20_thr022_cc20_score_sweep/score_mean_ge0.05 | test | 0.22 | 20 | 25 | 0.2636 | 0.1833 | 0.920 | 1945.04 | 5.16 | |
| test_e20_thr022_cc20_score_sweep/score_mean_ge0.1 | test | 0.22 | 20 | 25 | 0.2636 | 0.1833 | 0.920 | 1945.04 | 5.16 | |
| test_e20_thr022_cc20_score_sweep/score_mean_ge0.15 | test | 0.22 | 20 | 25 | 0.2636 | 0.1833 | 0.920 | 1945.04 | 5.16 | |
| test_e20_thr022_cc20_score_sweep/score_mean_ge0.2 | test | 0.22 | 20 | 25 | 0.2636 | 0.1833 | 0.920 | 1945.04 | 5.16 | |
| test_e20_thr022_cc20_score_sweep/score_p95_ge0 | test | 0.22 | 20 | 25 | 0.2636 | 0.1833 | 0.920 | 1945.04 | 5.16 | |
| test_e20_thr022_cc20_score_sweep/score_p95_ge0.05 | test | 0.22 | 20 | 25 | 0.2636 | 0.1833 | 0.920 | 1945.04 | 5.16 | |
| test_e20_thr022_cc20_score_sweep/score_p95_ge0.1 | test | 0.22 | 20 | 25 | 0.2636 | 0.1833 | 0.920 | 1945.04 | 5.16 | |
| test_e20_thr022_cc20_score_sweep/score_p95_ge0.15 | test | 0.22 | 20 | 25 | 0.2636 | 0.1833 | 0.920 | 1945.04 | 5.16 | |
| test_e20_thr022_cc20_score_sweep/score_p95_ge0.2 | test | 0.22 | 20 | 25 | 0.2636 | 0.1833 | 0.920 | 1945.04 | 5.16 | |
| test_e20_thr022_min_size_sweep/cc300 | test | 0.22 | 300 | 25 | 0.2635 | 0.1861 | 0.720 | 1573.24 | 0.48 | |
| test_e20_thr022_min_size_sweep/cc15 | test | 0.22 | 15 | 25 | 0.2632 | 0.1821 | 0.920 | 1970.64 | 6.56 | |
| test_e20_thr022_min_size_sweep/cc10 | test | 0.22 | 10 | 25 | 0.2608 | 0.1815 | 0.920 | 2005.12 | 9.40 | |
| test_e20_thr022_cc20_score_sweep/score_p95_ge0.8 | test | 0.22 | 20 | 25 | 0.2590 | 0.1912 | 0.600 | 990.32 | 0.00 | |
| test_e20_thr022_min_size_sweep/cc5 | test | 0.22 | 5 | 25 | 0.2581 | 0.1801 | 0.920 | 2050.76 | 16.12 | |
| test_e20_thr022_min_size_sweep/cc0 | test | 0.22 | 0 | 25 | 0.2563 | 0.1787 | 0.960 | 2105.48 | 48.64 | |
| test_e20_thr022_cc20_score_sweep/score_mean_ge0.4 | test | 0.22 | 20 | 25 | 0.2477 | 0.1861 | 0.520 | 849.36 | 0.00 | |
| test_e20_thr022_cc20_score_sweep/score_p95_ge0.85 | test | 0.22 | 20 | 25 | 0.2415 | 0.1861 | 0.560 | 986.80 | 0.00 | |
| test_e20_thr022_min_size_sweep/cc500 | test | 0.22 | 500 | 25 | 0.2302 | 0.1095 | 0.640 | 1499.00 | 0.36 | |
| test_e20_thr022_cc20_score_sweep/score_p95_ge0.9 | test | 0.22 | 20 | 25 | 0.2283 | 0.1350 | 0.520 | 873.84 | 0.00 | |
| test_e20_fp_ohem_bal_thr_sweep_cc20 | test | 0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7 | 20 | 25 | 0.2098 | 0.1401 | 0.920 | 3295.92 | 8.64 | |
| test_e20_fp_ohem_bal_thr_sweep_cc20_zoom020_025 | test | 0.2,0.21,0.22,0.23,0.24,0.25 | 20 | 25 | 0.2098 | 0.1401 | 0.920 | 3295.92 | 8.64 | |
| test_e20_thr022_cc20_score_sweep/score_p95_ge0.95 | test | 0.22 | 20 | 25 | 0.2036 | 0.0000 | 0.440 | 758.24 | 0.00 | |
| test_e20_thr022_cc20_score_sweep/score_mean_ge0.45 | test | 0.22 | 20 | 25 | 0.1846 | 0.0000 | 0.360 | 550.88 | 0.00 | |
| test_e20_thr022_cc20_score_sweep/score_mean_ge0.5 | test | 0.22 | 20 | 25 | 0.1440 | 0.0000 | 0.280 | 487.16 | 0.00 | |
| test_smoke2_fromrunbest_cc20 | test | 0.7 | 20 | 25 | 0.1343 | 0.0610 | 0.960 | 8897.40 | 2.08 | |
| test_smoke2_fp_fromrunbest_cc20 | test | 0.7 | 20 | 25 | 0.1330 | 0.0762 | 0.960 | 7249.96 | 3.04 | |
| test_e20_thr022_cc20_score_sweep/score_mean_ge0.55 | test | 0.22 | 20 | 25 | 0.1218 | 0.0000 | 0.200 | 222.28 | 0.00 | |
| test_smoke2_fp_ohem_fromrunbest_cc20 | test | 0.7 | 20 | 25 | 0.0969 | 0.0000 | 0.280 | 1.68 | 0.00 | |
| test_e20_thr022_cc20_score_sweep/score_mean_ge0.6 | test | 0.22 | 20 | 25 | 0.0310 | 0.0000 | 0.040 | 39.84 | 0.00 | |
| test_e20_thr022_cc20_score_sweep/score_mean_ge0.65 | test | 0.22 | 20 | 25 | 0.0310 | 0.0000 | 0.040 | 39.84 | 0.00 | |
| test_e20_thr022_cc20_score_sweep/score_mean_ge0.7 | test | 0.22 | 20 | 25 | 0.0310 | 0.0000 | 0.040 | 39.84 | 0.00 | |
| test_e20_thr022_cc20_score_sweep/score_mean_ge0.75 | test | 0.22 | 20 | 25 | 0.0310 | 0.0000 | 0.040 | 39.84 | 0.00 | |
| test_smoke2_fp_ohem_thr_sweep_cc20 | test | 0.1,0.2,0.3,0.4,0.5,0.6,0.7 | 20 | 25 | 0.0203 | 0.0063 | 1.000 | 110596.32 | 0.00 | |
| test_e20_thr022_cc20_score_sweep/score_mean_ge0.8 | test | 0.22 | 20 | 25 | 0.0000 | 0.0000 | 0.000 | 0.00 | 0.00 | |
| test_e20_thr022_cc20_score_sweep/score_mean_ge0.85 | test | 0.22 | 20 | 25 | 0.0000 | 0.0000 | 0.000 | 0.00 | 0.00 | |
| test_e20_thr022_cc20_score_sweep/score_mean_ge0.9 | test | 0.22 | 20 | 25 | 0.0000 | 0.0000 | 0.000 | 0.00 | 0.00 | |
| test_e20_thr022_cc20_score_sweep/score_mean_ge0.95 | test | 0.22 | 20 | 25 | 0.0000 | 0.0000 | 0.000 | 0.00 | 0.00 | |

## Details

### val_postproc_patch48_dwi_adc_flair

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/val_postproc_patch48_dwi_adc_flair`
- min_size: 20
- n: 25 (gt_pos=-)
- dice: mean=0.0127 median=0.0058 min=0.0000 max=0.0655 (n>0.1=0, n>0.3=0)

### test_e20_fp_ohem_bal_fromrunbest_cc20

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_fp_ohem_bal_fromrunbest_cc20`
- model_path: `/Users/yusukefujinami/ToReBrain/pipeline/runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.5
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.4120 median=0.5391 min=0.0000 max=0.8310 (n>0.1=17, n>0.3=16)
- detection_rate_case: 0.720
- FP: mean_fp_vox=257.76 mean_fp_cc=0.92 mean_fp_cc_vox=51.16

### score_max_ge0.8

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_max_ge0.8`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.3103 median=0.3134 min=0.0000 max=0.7755 (n>0.1=19, n>0.3=13)
- detection_rate_case: 0.800
- FP: mean_fp_vox=1301.96 mean_fp_cc=0.12 mean_fp_cc_vox=57.68

### score_p95_ge0.65

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_p95_ge0.65`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.3064 median=0.3134 min=0.0000 max=0.7755 (n>0.1=19, n>0.3=13)
- detection_rate_case: 0.760
- FP: mean_fp_vox=1121.40 mean_fp_cc=0.08 mean_fp_cc_vox=33.08

### score_p95_ge0.7

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_p95_ge0.7`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.3049 median=0.3134 min=0.0000 max=0.7755 (n>0.1=18, n>0.3=13)
- detection_rate_case: 0.720
- FP: mean_fp_vox=1082.76 mean_fp_cc=0.00 mean_fp_cc_vox=0.00

### score_mean_ge0.35

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_mean_ge0.35`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.3032 median=0.3134 min=0.0000 max=0.7755 (n>0.1=18, n>0.3=13)
- detection_rate_case: 0.720
- FP: mean_fp_vox=1096.36 mean_fp_cc=0.04 mean_fp_cc_vox=12.36

### score_p95_ge0.45

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_p95_ge0.45`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2965 median=0.2516 min=0.0000 max=0.7755 (n>0.1=19, n>0.3=11)
- detection_rate_case: 0.880
- FP: mean_fp_vox=1502.28 mean_fp_cc=0.56 mean_fp_cc_vox=228.40

### score_p95_ge0.6

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_p95_ge0.6`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2941 median=0.2891 min=0.0000 max=0.7755 (n>0.1=19, n>0.3=12)
- detection_rate_case: 0.800
- FP: mean_fp_vox=1306.68 mean_fp_cc=0.20 mean_fp_cc_vox=66.04

### test_smoke2_fp_ohem_autothr_cc20

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_smoke2_fp_ohem_autothr_cc20`
- model_path: `runs/3d_unet/medseg_3d_unet_smoke2_dwi_adc_flair_fp_ohem/best.pt`
- thresholds: 0.5
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2935 median=0.2407 min=0.0000 max=0.7620 (n>0.1=15, n>0.3=11)
- detection_rate_case: 0.720
- FP: mean_fp_vox=344.64 mean_fp_cc=1.76 mean_fp_cc_vox=185.36

### test_smoke2_fp_ohem_thr0p5_cc20

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_smoke2_fp_ohem_thr0p5_cc20`
- model_path: `runs/3d_unet/medseg_3d_unet_smoke2_dwi_adc_flair_fp_ohem/best.pt`
- thresholds: 0.5, 0.6, 0.7
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2935 median=0.2407 min=0.0000 max=0.7620 (n>0.1=15, n>0.3=11)
- detection_rate_case: 0.720
- FP: mean_fp_vox=344.64 mean_fp_cc=1.76 mean_fp_cc_vox=185.36

### score_max_ge0.85

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_max_ge0.85`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2928 median=0.2366 min=0.0000 max=0.7755 (n>0.1=18, n>0.3=11)
- detection_rate_case: 0.760
- FP: mean_fp_vox=1257.12 mean_fp_cc=0.04 mean_fp_cc_vox=20.72

### score_max_ge0.5

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_max_ge0.5`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2925 median=0.1881 min=0.0000 max=0.7755 (n>0.1=20, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1629.76 mean_fp_cc=0.80 mean_fp_cc_vox=351.48

### score_p95_ge0.55

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_p95_ge0.55`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2923 median=0.2451 min=0.0000 max=0.7755 (n>0.1=19, n>0.3=12)
- detection_rate_case: 0.800
- FP: mean_fp_vox=1321.96 mean_fp_cc=0.24 mean_fp_cc_vox=81.32

### score_max_ge0.75

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_max_ge0.75`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2910 median=0.2420 min=0.0000 max=0.7755 (n>0.1=19, n>0.3=11)
- detection_rate_case: 0.800
- FP: mean_fp_vox=1401.92 mean_fp_cc=0.24 mean_fp_cc_vox=157.64

### score_max_ge0.9

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_max_ge0.9`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2901 median=0.2366 min=0.0000 max=0.7755 (n>0.1=18, n>0.3=11)
- detection_rate_case: 0.760
- FP: mean_fp_vox=1248.96 mean_fp_cc=0.04 mean_fp_cc_vox=20.72

### score_max_ge0.6

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_max_ge0.6`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2894 median=0.1910 min=0.0000 max=0.7755 (n>0.1=19, n>0.3=11)
- detection_rate_case: 0.880
- FP: mean_fp_vox=1537.56 mean_fp_cc=0.48 mean_fp_cc_vox=261.72

### score_max_ge0.55

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_max_ge0.55`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2891 median=0.1902 min=0.0000 max=0.7755 (n>0.1=19, n>0.3=11)
- detection_rate_case: 0.880
- FP: mean_fp_vox=1546.72 mean_fp_cc=0.56 mean_fp_cc_vox=268.56

### score_mean_ge0.3

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_mean_ge0.3`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2885 median=0.1902 min=0.0000 max=0.7755 (n>0.1=19, n>0.3=11)
- detection_rate_case: 0.840
- FP: mean_fp_vox=1445.64 mean_fp_cc=0.48 mean_fp_cc_vox=176.60

### score_p95_ge0.5

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_p95_ge0.5`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2875 median=0.1902 min=0.0000 max=0.7755 (n>0.1=19, n>0.3=11)
- detection_rate_case: 0.840
- FP: mean_fp_vox=1431.56 mean_fp_cc=0.40 mean_fp_cc_vox=173.20

### score_max_ge0.7

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_max_ge0.7`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2875 median=0.1910 min=0.0000 max=0.7755 (n>0.1=19, n>0.3=11)
- detection_rate_case: 0.840
- FP: mean_fp_vox=1434.24 mean_fp_cc=0.28 mean_fp_cc_vox=166.08

### score_max_ge0.65

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_max_ge0.65`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2872 median=0.1910 min=0.0000 max=0.7755 (n>0.1=19, n>0.3=11)
- detection_rate_case: 0.840
- FP: mean_fp_vox=1475.04 mean_fp_cc=0.32 mean_fp_cc_vox=203.60

### score_p95_ge0.4

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_p95_ge0.4`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2869 median=0.1881 min=0.0000 max=0.7761 (n>0.1=19, n>0.3=11)
- detection_rate_case: 0.880
- FP: mean_fp_vox=1594.72 mean_fp_cc=1.12 mean_fp_cc_vox=320.84

### score_p95_ge0.75

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_p95_ge0.75`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2840 median=0.2891 min=0.0000 max=0.7755 (n>0.1=16, n>0.3=12)
- detection_rate_case: 0.640
- FP: mean_fp_vox=1020.44 mean_fp_cc=0.00 mean_fp_cc_vox=0.00

### score_max_ge0.45

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_max_ge0.45`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2800 median=0.1861 min=0.0000 max=0.7761 (n>0.1=17, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1718.08 mean_fp_cc=1.40 mean_fp_cc_vox=439.80

### score_p95_ge0.35

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_p95_ge0.35`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2793 median=0.1866 min=0.0000 max=0.7761 (n>0.1=17, n>0.3=10)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1745.20 mean_fp_cc=2.12 mean_fp_cc_vox=458.32

### score_max_ge0.95

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_max_ge0.95`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2772 median=0.2105 min=0.0000 max=0.7755 (n>0.1=16, n>0.3=11)
- detection_rate_case: 0.680
- FP: mean_fp_vox=1217.60 mean_fp_cc=0.00 mean_fp_cc_vox=0.00

### score_max_ge0.4

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_max_ge0.4`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2746 median=0.1866 min=0.0000 max=0.7761 (n>0.1=16, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1788.80 mean_fp_cc=2.04 mean_fp_cc_vox=501.92

### score_max_ge0.35

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_max_ge0.35`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2721 median=0.1861 min=0.0000 max=0.7761 (n>0.1=16, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1841.20 mean_fp_cc=2.84 mean_fp_cc_vox=554.32

### cc150

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_min_size_sweep/cc150`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 150
- n: 25 (gt_pos=25)
- dice: mean=0.2713 median=0.1891 min=0.0000 max=0.7755 (n>0.1=17, n>0.3=9)
- detection_rate_case: 0.800
- FP: mean_fp_vox=1684.88 mean_fp_cc=0.84 mean_fp_cc_vox=428.68

### score_p95_ge0.3

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_p95_ge0.3`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2707 median=0.1861 min=0.0000 max=0.7761 (n>0.1=16, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1865.76 mean_fp_cc=3.36 mean_fp_cc_vox=578.88

### score_mean_ge0.25

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_mean_ge0.25`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2699 median=0.1833 min=0.0000 max=0.7761 (n>0.1=16, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1864.04 mean_fp_cc=3.40 mean_fp_cc_vox=577.16

### cc200

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_min_size_sweep/cc200`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 200
- n: 25 (gt_pos=25)
- dice: mean=0.2697 median=0.1861 min=0.0000 max=0.7755 (n>0.1=17, n>0.3=9)
- detection_rate_case: 0.800
- FP: mean_fp_vox=1643.84 mean_fp_cc=0.68 mean_fp_cc_vox=401.08

### cc75

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_min_size_sweep/cc75`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 75
- n: 25 (gt_pos=25)
- dice: mean=0.2694 median=0.1866 min=0.0000 max=0.7755 (n>0.1=16, n>0.3=9)
- detection_rate_case: 0.840
- FP: mean_fp_vox=1795.24 mean_fp_cc=1.64 mean_fp_cc_vox=517.88

### cc100

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_min_size_sweep/cc100`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 100
- n: 25 (gt_pos=25)
- dice: mean=0.2694 median=0.1891 min=0.0000 max=0.7755 (n>0.1=16, n>0.3=9)
- detection_rate_case: 0.840
- FP: mean_fp_vox=1754.36 mean_fp_cc=1.28 mean_fp_cc_vox=485.92

### cc50

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_min_size_sweep/cc50`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 50
- n: 25 (gt_pos=25)
- dice: mean=0.2689 median=0.1852 min=0.0000 max=0.7755 (n>0.1=16, n>0.3=9)
- detection_rate_case: 0.880
- FP: mean_fp_vox=1864.68 mean_fp_cc=2.68 mean_fp_cc_vox=579.44

### cc40

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_min_size_sweep/cc40`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 40
- n: 25 (gt_pos=25)
- dice: mean=0.2657 median=0.1852 min=0.0000 max=0.7755 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.880
- FP: mean_fp_vox=1885.96 mean_fp_cc=3.12 mean_fp_cc_vox=598.84

### score_max_ge0.3

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_max_ge0.3`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2650 median=0.1833 min=0.0000 max=0.7761 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1901.44 mean_fp_cc=4.12 mean_fp_cc_vox=614.56

### score_p95_ge0.25

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_p95_ge0.25`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2637 median=0.1833 min=0.0000 max=0.7761 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1942.16 mean_fp_cc=5.04 mean_fp_cc_vox=651.68

### score_max_ge0.25

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_max_ge0.25`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2637 median=0.1833 min=0.0000 max=0.7761 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1943.12 mean_fp_cc=5.08 mean_fp_cc_vox=652.64

### cc30

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_min_size_sweep/cc30`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 30
- n: 25 (gt_pos=25)
- dice: mean=0.2636 median=0.1843 min=0.0000 max=0.7761 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.880
- FP: mean_fp_vox=1913.64 mean_fp_cc=3.88 mean_fp_cc_vox=624.04

### cc20

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_min_size_sweep/cc20`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2636 median=0.1833 min=0.0000 max=0.7761 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1945.04 mean_fp_cc=5.16 mean_fp_cc_vox=654.56

### score_max_ge0

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_max_ge0`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2636 median=0.1833 min=0.0000 max=0.7761 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1945.04 mean_fp_cc=5.16 mean_fp_cc_vox=654.56

### score_max_ge0.05

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_max_ge0.05`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2636 median=0.1833 min=0.0000 max=0.7761 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1945.04 mean_fp_cc=5.16 mean_fp_cc_vox=654.56

### score_max_ge0.1

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_max_ge0.1`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2636 median=0.1833 min=0.0000 max=0.7761 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1945.04 mean_fp_cc=5.16 mean_fp_cc_vox=654.56

### score_max_ge0.15

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_max_ge0.15`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2636 median=0.1833 min=0.0000 max=0.7761 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1945.04 mean_fp_cc=5.16 mean_fp_cc_vox=654.56

### score_max_ge0.2

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_max_ge0.2`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2636 median=0.1833 min=0.0000 max=0.7761 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1945.04 mean_fp_cc=5.16 mean_fp_cc_vox=654.56

### score_mean_ge0

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_mean_ge0`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2636 median=0.1833 min=0.0000 max=0.7761 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1945.04 mean_fp_cc=5.16 mean_fp_cc_vox=654.56

### score_mean_ge0.05

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_mean_ge0.05`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2636 median=0.1833 min=0.0000 max=0.7761 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1945.04 mean_fp_cc=5.16 mean_fp_cc_vox=654.56

### score_mean_ge0.1

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_mean_ge0.1`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2636 median=0.1833 min=0.0000 max=0.7761 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1945.04 mean_fp_cc=5.16 mean_fp_cc_vox=654.56

### score_mean_ge0.15

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_mean_ge0.15`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2636 median=0.1833 min=0.0000 max=0.7761 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1945.04 mean_fp_cc=5.16 mean_fp_cc_vox=654.56

### score_mean_ge0.2

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_mean_ge0.2`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2636 median=0.1833 min=0.0000 max=0.7761 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1945.04 mean_fp_cc=5.16 mean_fp_cc_vox=654.56

### score_p95_ge0

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_p95_ge0`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2636 median=0.1833 min=0.0000 max=0.7761 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1945.04 mean_fp_cc=5.16 mean_fp_cc_vox=654.56

### score_p95_ge0.05

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_p95_ge0.05`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2636 median=0.1833 min=0.0000 max=0.7761 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1945.04 mean_fp_cc=5.16 mean_fp_cc_vox=654.56

### score_p95_ge0.1

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_p95_ge0.1`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2636 median=0.1833 min=0.0000 max=0.7761 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1945.04 mean_fp_cc=5.16 mean_fp_cc_vox=654.56

### score_p95_ge0.15

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_p95_ge0.15`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2636 median=0.1833 min=0.0000 max=0.7761 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1945.04 mean_fp_cc=5.16 mean_fp_cc_vox=654.56

### score_p95_ge0.2

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_p95_ge0.2`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2636 median=0.1833 min=0.0000 max=0.7761 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1945.04 mean_fp_cc=5.16 mean_fp_cc_vox=654.56

### cc300

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_min_size_sweep/cc300`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 300
- n: 25 (gt_pos=25)
- dice: mean=0.2635 median=0.1861 min=0.0000 max=0.7755 (n>0.1=16, n>0.3=9)
- detection_rate_case: 0.720
- FP: mean_fp_vox=1573.24 mean_fp_cc=0.48 mean_fp_cc_vox=354.44

### cc15

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_min_size_sweep/cc15`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 15
- n: 25 (gt_pos=25)
- dice: mean=0.2632 median=0.1821 min=0.0000 max=0.7761 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=1970.64 mean_fp_cc=6.56 mean_fp_cc_vox=677.96

### cc10

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_min_size_sweep/cc10`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 10
- n: 25 (gt_pos=25)
- dice: mean=0.2608 median=0.1815 min=0.0000 max=0.7767 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=2005.12 mean_fp_cc=9.40 mean_fp_cc_vox=711.20

### score_p95_ge0.8

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_p95_ge0.8`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2590 median=0.1912 min=0.0000 max=0.7755 (n>0.1=15, n>0.3=11)
- detection_rate_case: 0.600
- FP: mean_fp_vox=990.32 mean_fp_cc=0.00 mean_fp_cc_vox=0.00

### cc5

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_min_size_sweep/cc5`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 5
- n: 25 (gt_pos=25)
- dice: mean=0.2581 median=0.1801 min=0.0000 max=0.7765 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.920
- FP: mean_fp_vox=2050.76 mean_fp_cc=16.12 mean_fp_cc_vox=755.24

### cc0

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_min_size_sweep/cc0`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 0
- n: 25 (gt_pos=25)
- dice: mean=0.2563 median=0.1787 min=0.0000 max=0.7768 (n>0.1=15, n>0.3=9)
- detection_rate_case: 0.960
- FP: mean_fp_vox=2105.48 mean_fp_cc=48.64 mean_fp_cc_vox=809.44

### score_mean_ge0.4

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_mean_ge0.4`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2477 median=0.1861 min=0.0000 max=0.7755 (n>0.1=13, n>0.3=10)
- detection_rate_case: 0.520
- FP: mean_fp_vox=849.36 mean_fp_cc=0.00 mean_fp_cc_vox=0.00

### score_p95_ge0.85

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_p95_ge0.85`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2415 median=0.1861 min=0.0000 max=0.7755 (n>0.1=14, n>0.3=10)
- detection_rate_case: 0.560
- FP: mean_fp_vox=986.80 mean_fp_cc=0.00 mean_fp_cc_vox=0.00

### cc500

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_min_size_sweep/cc500`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 500
- n: 25 (gt_pos=25)
- dice: mean=0.2302 median=0.1095 min=0.0000 max=0.7755 (n>0.1=14, n>0.3=8)
- detection_rate_case: 0.640
- FP: mean_fp_vox=1499.00 mean_fp_cc=0.36 mean_fp_cc_vox=308.80

### score_p95_ge0.9

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_p95_ge0.9`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2283 median=0.1350 min=0.0000 max=0.7755 (n>0.1=13, n>0.3=9)
- detection_rate_case: 0.520
- FP: mean_fp_vox=873.84 mean_fp_cc=0.00 mean_fp_cc_vox=0.00

### test_e20_fp_ohem_bal_thr_sweep_cc20

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_fp_ohem_bal_thr_sweep_cc20`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2098 median=0.1401 min=0.0000 max=0.7905 (n>0.1=14, n>0.3=7)
- detection_rate_case: 0.920
- FP: mean_fp_vox=3295.92 mean_fp_cc=8.64 mean_fp_cc_vox=1096.20

### test_e20_fp_ohem_bal_thr_sweep_cc20_zoom020_025

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_fp_ohem_bal_thr_sweep_cc20_zoom020_025`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.2, 0.21, 0.22, 0.23, 0.24, 0.25
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2098 median=0.1401 min=0.0000 max=0.7905 (n>0.1=14, n>0.3=7)
- detection_rate_case: 0.920
- FP: mean_fp_vox=3295.92 mean_fp_cc=8.64 mean_fp_cc_vox=1096.20

### score_p95_ge0.95

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_p95_ge0.95`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.2036 median=0.0000 min=0.0000 max=0.7755 (n>0.1=11, n>0.3=8)
- detection_rate_case: 0.440
- FP: mean_fp_vox=758.24 mean_fp_cc=0.00 mean_fp_cc_vox=0.00

### score_mean_ge0.45

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_mean_ge0.45`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.1846 median=0.0000 min=0.0000 max=0.7755 (n>0.1=9, n>0.3=8)
- detection_rate_case: 0.360
- FP: mean_fp_vox=550.88 mean_fp_cc=0.00 mean_fp_cc_vox=0.00

### score_mean_ge0.5

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_mean_ge0.5`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.1440 median=0.0000 min=0.0000 max=0.7755 (n>0.1=7, n>0.3=6)
- detection_rate_case: 0.280
- FP: mean_fp_vox=487.16 mean_fp_cc=0.00 mean_fp_cc_vox=0.00

### test_smoke2_fromrunbest_cc20

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_smoke2_fromrunbest_cc20`
- model_path: `runs/3d_unet/medseg_3d_unet_smoke2_dwi_adc_flair/best.pt`
- thresholds: 0.7
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.1343 median=0.0610 min=0.0000 max=0.8036 (n>0.1=10, n>0.3=4)
- detection_rate_case: 0.960
- FP: mean_fp_vox=8897.40 mean_fp_cc=2.08 mean_fp_cc_vox=1780.92

### test_smoke2_fp_fromrunbest_cc20

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_smoke2_fp_fromrunbest_cc20`
- model_path: `runs/3d_unet/medseg_3d_unet_smoke2_dwi_adc_flair_fp/best.pt`
- thresholds: 0.7
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.1330 median=0.0762 min=0.0000 max=0.7802 (n>0.1=11, n>0.3=3)
- detection_rate_case: 0.960
- FP: mean_fp_vox=7249.96 mean_fp_cc=3.04 mean_fp_cc_vox=2205.88

### score_mean_ge0.55

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_mean_ge0.55`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.1218 median=0.0000 min=0.0000 max=0.7755 (n>0.1=5, n>0.3=5)
- detection_rate_case: 0.200
- FP: mean_fp_vox=222.28 mean_fp_cc=0.00 mean_fp_cc_vox=0.00

### test_smoke2_fp_ohem_fromrunbest_cc20

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_smoke2_fp_ohem_fromrunbest_cc20`
- model_path: `runs/3d_unet/medseg_3d_unet_smoke2_dwi_adc_flair_fp_ohem/best.pt`
- thresholds: 0.7
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.0969 median=0.0000 min=0.0000 max=0.6832 (n>0.1=7, n>0.3=4)
- detection_rate_case: 0.280
- FP: mean_fp_vox=1.68 mean_fp_cc=0.00 mean_fp_cc_vox=0.00

### score_mean_ge0.6

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_mean_ge0.6`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.0310 median=0.0000 min=0.0000 max=0.7755 (n>0.1=1, n>0.3=1)
- detection_rate_case: 0.040
- FP: mean_fp_vox=39.84 mean_fp_cc=0.00 mean_fp_cc_vox=0.00

### score_mean_ge0.65

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_mean_ge0.65`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.0310 median=0.0000 min=0.0000 max=0.7755 (n>0.1=1, n>0.3=1)
- detection_rate_case: 0.040
- FP: mean_fp_vox=39.84 mean_fp_cc=0.00 mean_fp_cc_vox=0.00

### score_mean_ge0.7

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_mean_ge0.7`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.0310 median=0.0000 min=0.0000 max=0.7755 (n>0.1=1, n>0.3=1)
- detection_rate_case: 0.040
- FP: mean_fp_vox=39.84 mean_fp_cc=0.00 mean_fp_cc_vox=0.00

### score_mean_ge0.75

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_mean_ge0.75`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.0310 median=0.0000 min=0.0000 max=0.7755 (n>0.1=1, n>0.3=1)
- detection_rate_case: 0.040
- FP: mean_fp_vox=39.84 mean_fp_cc=0.00 mean_fp_cc_vox=0.00

### test_smoke2_fp_ohem_thr_sweep_cc20

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_smoke2_fp_ohem_thr_sweep_cc20`
- model_path: `runs/3d_unet/medseg_3d_unet_smoke2_dwi_adc_flair_fp_ohem/best.pt`
- thresholds: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.0203 median=0.0063 min=0.0005 max=0.2657 (n>0.1=1, n>0.3=0)
- detection_rate_case: 1.000
- FP: mean_fp_vox=110596.32 mean_fp_cc=0.00 mean_fp_cc_vox=0.00

### score_mean_ge0.8

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_mean_ge0.8`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.0000 median=0.0000 min=0.0000 max=0.0000 (n>0.1=0, n>0.3=0)
- detection_rate_case: 0.000
- FP: mean_fp_vox=0.00 mean_fp_cc=0.00 mean_fp_cc_vox=0.00

### score_mean_ge0.85

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_mean_ge0.85`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.0000 median=0.0000 min=0.0000 max=0.0000 (n>0.1=0, n>0.3=0)
- detection_rate_case: 0.000
- FP: mean_fp_vox=0.00 mean_fp_cc=0.00 mean_fp_cc_vox=0.00

### score_mean_ge0.9

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_mean_ge0.9`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.0000 median=0.0000 min=0.0000 max=0.0000 (n>0.1=0, n>0.3=0)
- detection_rate_case: 0.000
- FP: mean_fp_vox=0.00 mean_fp_cc=0.00 mean_fp_cc_vox=0.00

### score_mean_ge0.95

- path: `/Users/yusukefujinami/ToReBrain/pipeline/results/3d_unet_medseg/test_e20_thr022_cc20_score_sweep/score_mean_ge0.95`
- model_path: `runs/3d_unet/medseg_3d_unet_e20_dwi_adc_flair_fp_ohem_balanced/best.pt`
- thresholds: 0.22
- min_size: 20
- n: 25 (gt_pos=25)
- dice: mean=0.0000 median=0.0000 min=0.0000 max=0.0000 (n>0.1=0, n>0.3=0)
- detection_rate_case: 0.000
- FP: mean_fp_vox=0.00 mean_fp_cc=0.00 mean_fp_cc_vox=0.00
