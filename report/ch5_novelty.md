# 5 창의성 및 차별성

이 초안의 새 모델 수치는 **개발 교차검증** 결과다. 2026-10-01 18:00 KST 전에는 최종 테스트를 실행하지 않는다. 사전 검증에서 고정 설정으로 테스트를 한 번 본 이력이 있으며 verification/의 성능은 신규 모델의 성능으로 재사용하지 않는다.

CBL 현업 기준선, 공휴일 채택 판정, 이분산 분위수 보정, 확률·여유 경보, 예측거리별 조치, 2026 시간대 재평가를 각각 성능 변화로 평가한다. 선택 기법은 사전 기준을 만족한 경우에만 채택하며 기각도 공개한다.

### coverage_by_fold

[전체 표](../outputs/tables/coverage_by_fold.csv)

| model | fold | top_n | n | top_q95_coverage | tau | top_edge | ci_low | ci_high | target_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lgbm_quantile_a | 0 | 172 | 2746 | 0.9942 | 176.5000 | 166.0001 | 0.9806 | 1.0000 | 19 |
| lgbm_quantile_a | 1 | 477 | 2746 | 0.8365 | 177.0000 | 163.2952 | 0.8004 | 0.8742 | 20 |
| lgbm_quantile_a | 2 | 185 | 2059 | 0.5892 | 175.0000 | 171.2784 | 0.4554 | 0.7688 | 9 |
| lgbm_quantile_b | 0 | 172 | 2746 | 0.9942 | 176.5000 | 166.0001 | 0.9806 | 1.0000 | 19 |
| lgbm_quantile_b | 1 | 477 | 2746 | 0.9623 | 177.0000 | 163.2952 | 0.9437 | 0.9812 | 20 |
| lgbm_quantile_b | 2 | 185 | 2059 | 0.7838 | 175.0000 | 171.2784 | 0.6968 | 0.9053 | 9 |
| lgbm_quantile_raw | 0 | 172 | 2746 | 0.9709 | 176.5000 | 166.0001 | 0.9378 | 1.0000 | 19 |
| lgbm_quantile_raw | 1 | 477 | 2746 | 0.8742 | 177.0000 | 163.2952 | 0.8379 | 0.9118 | 20 |
| lgbm_quantile_raw | 2 | 185 | 2059 | 0.4703 | 175.0000 | 171.2784 | 0.3551 | 0.6266 | 9 |

### coverage_comparison

[전체 표](../outputs/tables/coverage_comparison.csv)

| model | alpha | segment | coverage | coverage_error | scoring_n | total_n | ci_low | ci_high | target_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lgbm_quantile_raw | 0.9000 | 전체 | 0.8400 | 0.0600 | 7551 | 7551 | 0.8188 | 0.8593 | 85 |
| lgbm_quantile_raw | 0.9000 | 상위 예측 | 0.7266 | 0.1734 | 834 | 7551 | 0.6471 | 0.8051 | 48 |
| lgbm_quantile_raw | 0.9500 | 전체 | 0.9142 | 0.0358 | 7551 | 7551 | 0.8972 | 0.9294 | 85 |
| lgbm_quantile_raw | 0.9500 | 상위 예측 | 0.8046 | 0.1454 | 834 | 7551 | 0.7319 | 0.8747 | 48 |
| lgbm_quantile_a | 0.9000 | 전체 | 0.8578 | 0.0422 | 7551 | 7551 | 0.8341 | 0.8789 | 85 |
| lgbm_quantile_a | 0.9000 | 상위 예측 | 0.7218 | 0.1782 | 834 | 7551 | 0.6535 | 0.7925 | 48 |
| lgbm_quantile_a | 0.9500 | 전체 | 0.9298 | 0.0202 | 7551 | 7551 | 0.9160 | 0.9419 | 85 |
| lgbm_quantile_a | 0.9500 | 상위 예측 | 0.8141 | 0.1359 | 834 | 7551 | 0.7522 | 0.8728 | 48 |
| lgbm_quantile_b | 0.9000 | 전체 | 0.8780 | 0.0220 | 7551 | 7551 | 0.8580 | 0.8953 | 85 |
| lgbm_quantile_b | 0.9000 | 상위 예측 | 0.8813 | 0.0187 | 834 | 7551 | 0.8187 | 0.9357 | 48 |
| lgbm_quantile_b | 0.9500 | 전체 | 0.9400 | 0.0100 | 7551 | 7551 | 0.9268 | 0.9509 | 85 |
| lgbm_quantile_b | 0.9500 | 상위 예측 | 0.9293 | 0.0207 | 834 | 7551 | 0.8917 | 0.9638 | 48 |

### coverage_top_by_day

[전체 표](../outputs/tables/coverage_top_by_day.csv)

| fold | date | n | missed_upper | actual_peak |
| --- | --- | --- | --- | --- |
| 0 | 2021-04-14 | 1 | 0 | 0 |
| 0 | 2021-04-15 | 18 | 0 | 4 |
| 0 | 2021-04-16 | 16 | 0 | 4 |
| 0 | 2021-04-19 | 12 | 0 | 0 |
| 0 | 2021-04-20 | 2 | 0 | 0 |
| 0 | 2021-04-21 | 3 | 0 | 0 |
| 0 | 2021-04-22 | 2 | 0 | 0 |
| 0 | 2021-04-23 | 2 | 0 | 0 |
| 0 | 2021-04-26 | 21 | 0 | 0 |
| 0 | 2021-04-27 | 4 | 0 | 0 |
| 0 | 2021-04-28 | 2 | 0 | 0 |
| 0 | 2021-04-30 | 2 | 0 | 0 |
| 0 | 2021-05-03 | 20 | 0 | 4 |
| 0 | 2021-05-04 | 13 | 1 | 3 |
| 0 | 2021-05-06 | 19 | 0 | 4 |

### horizon_advantage

[전체 표](../outputs/tables/horizon_advantage.csv)

| horizon | minutes | model | selected_model | comparison_role | baseline_family | baseline_model | paired_peak_n | peak_mae_gain | ci_low | ci_high | model_beats_baseline_ci |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 15 | lgbm_no_holiday | p1_latest | development_comparator_not_selected | persistence | p1_latest | 416 | 1.0536 | -0.5151 | 2.8394 | False |
| 1 | 15 | lgbm_no_holiday | p1_latest | development_comparator_not_selected | seasonal | s2_week | 416 | 8.5728 | 3.7894 | 16.0750 | True |
| 1 | 15 | lgbm_no_holiday | p1_latest | development_comparator_not_selected | cbl | c2a_max_4_5_adjusted | 416 | 4.4685 | 0.9717 | 9.3612 | True |
| 4 | 60 | lgbm_no_holiday_weight_2 | lgbm_no_holiday_weight_2 | selected_point_model | persistence | p1_latest | 425 | 19.8157 | 15.6979 | 24.9716 | True |
| 4 | 60 | lgbm_no_holiday_weight_2 | lgbm_no_holiday_weight_2 | selected_point_model | seasonal | s2_week | 425 | 4.5428 | 0.3364 | 10.8742 | True |
| 4 | 60 | lgbm_no_holiday_weight_2 | lgbm_no_holiday_weight_2 | selected_point_model | cbl | c2a_max_4_5_adjusted | 425 | 3.8130 | -0.9260 | 9.8691 | False |
| 16 | 240 | lgbm_no_holiday | c3_holiday_hybrid | development_comparator_not_selected | persistence | p3_recent_mean | 425 | 27.1629 | 10.5730 | 43.2763 | True |
| 16 | 240 | lgbm_no_holiday | c3_holiday_hybrid | development_comparator_not_selected | seasonal | s2_week | 425 | -19.7736 | -30.0836 | -9.4318 | False |
| 16 | 240 | lgbm_no_holiday | c3_holiday_hybrid | development_comparator_not_selected | cbl | c3_holiday_hybrid | 425 | -23.4743 | -33.7912 | -14.1760 | False |
| 96 | 1440 | lgbm_no_holiday | c3_holiday_hybrid | development_comparator_not_selected | persistence | p1_latest | 369 | 20.6558 | -11.9933 | 57.3683 | False |
| 96 | 1440 | lgbm_no_holiday | c3_holiday_hybrid | development_comparator_not_selected | seasonal | s2_week | 369 | -21.4797 | -29.2761 | -11.7848 | False |
| 96 | 1440 | lgbm_no_holiday | c3_holiday_hybrid | development_comparator_not_selected | cbl | c3_holiday_hybrid | 369 | -24.9783 | -33.1659 | -15.4961 | False |

### horizon_curve

[전체 표](../outputs/tables/horizon_curve.csv)

| horizon | minutes | model | selected | n | peak_n | mae | peak_mae | peak_mae_ci_low | peak_mae_ci_high | episode_f1 | tp | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 15 | c1_mid_6_10 | False | 7564 | 416 | 43.1586 | 14.2796 | 11.8914 | 17.7848 | 0.4151 | 88 | 192 | 56 |
| 1 | 15 | c1a_mid_6_10_adjusted | False | 7564 | 416 | 13.1693 | 15.7087 | 12.5076 | 20.3294 | 0.5064 | 99 | 148 | 45 |
| 1 | 15 | c2_max_4_5 | False | 7564 | 416 | 41.4019 | 11.8341 | 8.7569 | 17.0921 | 0.3713 | 75 | 185 | 69 |
| 1 | 15 | c2a_max_4_5_adjusted | False | 7564 | 416 | 13.0114 | 14.6601 | 11.4731 | 19.1093 | 0.4921 | 94 | 144 | 50 |
| 1 | 15 | c3_holiday_hybrid | False | 7564 | 416 | 20.1272 | 15.3590 | 12.4745 | 20.3203 | 0.4916 | 88 | 126 | 56 |
| 1 | 15 | c3_holiday_mid_4_6 | False | 7564 | 416 | 20.1272 | 15.3590 | 12.4745 | 20.3203 | 0.4916 | 88 | 126 | 56 |
| 1 | 15 | lgbm | False | 7564 | 416 | 4.5847 | 9.8367 | 7.5764 | 11.6574 | 0.6383 | 120 | 112 | 24 |
| 1 | 15 | lgbm_no_holiday | False | 7564 | 416 | 4.8422 | 10.1916 | 8.5340 | 11.5575 | 0.6253 | 116 | 111 | 28 |
| 1 | 15 | lgbm_no_holiday_weight_2 | False | 7564 | 416 | 4.6537 | 9.6242 | 7.1332 | 11.5092 | 0.6167 | 107 | 96 | 37 |
| 1 | 15 | lgbm_no_holiday_weight_4 | False | 7564 | 416 | 4.4717 | 9.1128 | 6.2373 | 11.3347 | 0.6409 | 116 | 102 | 28 |
| 1 | 15 | lgbm_quantile_a | False | 7564 | 416 | 4.3284 | 9.7590 | 8.1498 | 11.0929 | 0.6624 | 104 | 66 | 40 |
| 1 | 15 | lgbm_quantile_b | False | 7564 | 416 | 4.3284 | 9.7590 | 8.1498 | 11.0929 | 0.6481 | 105 | 75 | 39 |
| 1 | 15 | lgbm_quantile_raw | False | 7564 | 416 | 4.3284 | 9.7590 | 8.1498 | 11.0929 | 0.6879 | 108 | 62 | 36 |
| 1 | 15 | lgbm_weight_2 | False | 7564 | 416 | 4.6082 | 9.3269 | 6.8700 | 11.2611 | 0.6087 | 105 | 96 | 39 |
| 1 | 15 | lgbm_weight_4 | False | 7564 | 416 | 4.4716 | 9.1424 | 6.3205 | 11.3580 | 0.6484 | 118 | 102 | 26 |

### T2-1_fva

[전체 표](../outputs/tables/T2-1_fva.csv)

| stage | model | adopted | metric_domain | previous_model | selected_final | is_selected_point_model | n | peak_n | mae | rmse | mape_nonzero | peak_mae | position_tp | position_fp | position_fn | position_f1 | episode_tp | episode_fp | episode_fn | episode_f1 | false_alarms_positions | timing_mae_minutes | magnitude_mae | vs_cbl_gain | vs_cbl_gain_ci_low | vs_cbl_gain_ci_high | vs_cbl_improvement_fraction | vs_cbl_improvement_ci_low | vs_cbl_improvement_ci_high | vs_cbl_paired_peak_n | vs_previous_gain | vs_previous_gain_ci_low | vs_previous_gain_ci_high | vs_previous_improvement_fraction | vs_previous_improvement_ci_low | vs_previous_improvement_ci_high | vs_previous_paired_peak_n | brier | pr_auc | coverage_0.1 | top_coverage_0.1 | pinball_0.1 | coverage_0.5 | top_coverage_0.5 | pinball_0.5 | coverage_0.9 | top_coverage_0.9 | pinball_0.9 | coverage_0.95 | top_coverage_0.95 | pinball_0.95 | coverage_0.975 | top_coverage_0.975 | pinball_0.975 | pinball_mean | vs_previous_top_coverage_error_reduction_0.9 | vs_previous_top_coverage_error_reduction_0.95 | vs_previous_pinball_improvement_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CBL 대표 | c2a_max_4_5_adjusted | True | point | — | False | False | 7551 | 425 | 16.0055 | 26.9167 | 46.5614 | 18.2561 | 369 | 1156 | 56 | 0.3785 | 97 | 145 | 51 | 0.4974 | 1156 | 26.4433 | 12.4809 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 425.0000 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Persistence 대표 | p1_latest | True | point | c2a_max_4_5_adjusted | False | False | 7551 | 425 | 16.2647 | 27.4600 | 17.2647 | 34.2588 | 322 | 1843 | 103 | 0.2486 | 69 | 162 | 79 | 0.3641 | 1843 | 64.1304 | 2.2029 | -16.0027 | -19.5702 | -12.2520 | -0.8766 | -1.3288 | -0.5338 | 425.0000 | -16.0027 | -19.5702 | -12.2520 | -0.8766 | -1.3288 | -0.5338 | 425.0000 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| LightGBM 기본 | lgbm_no_holiday | True | point | p1_latest | False | False | 7551 | 425 | 8.6683 | 13.7791 | 15.5904 | 19.1784 | 366 | 687 | 59 | 0.4953 | 93 | 107 | 55 | 0.5345 | 687 | 35.3226 | 15.0732 | -0.9223 | -6.3837 | 6.1673 | -0.0505 | -0.4569 | 0.2654 | 425.0000 | 15.0804 | 10.0417 | 21.1926 | 0.4402 | 0.3229 | 0.5509 | 425.0000 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 공휴일 특징 후보 | lgbm | False | point | lgbm_no_holiday | False | False | 7551 | 425 | 8.5871 | 13.6616 | 15.6297 | 18.5847 | 383 | 762 | 42 | 0.4879 | 103 | 109 | 45 | 0.5722 | 762 | 26.6505 | 14.4711 | -0.3286 | -5.8513 | 6.8984 | -0.0180 | -0.4097 | 0.2944 | 425.0000 | 0.5937 | 0.1199 | 1.1161 | 0.0310 | 0.0064 | 0.0582 | 425.0000 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 피크 가중 2 후보 | lgbm_no_holiday_weight_2 | True | point | lgbm_no_holiday | False | True | 7551 | 425 | 8.1848 | 13.1640 | 14.3281 | 14.4431 | 379 | 593 | 46 | 0.5426 | 98 | 87 | 50 | 0.5886 | 593 | 33.6735 | 12.0515 | 3.8130 | -0.9260 | 9.8691 | 0.2089 | -0.0627 | 0.4185 | 425.0000 | 4.7353 | 3.4284 | 5.8204 | 0.2469 | 0.1898 | 0.2870 | 425.0000 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 피크 가중 4 후보 | lgbm_no_holiday_weight_4 | True | point | lgbm_no_holiday | False | False | 7551 | 425 | 8.7779 | 13.9929 | 16.6034 | 16.4439 | 394 | 606 | 31 | 0.5530 | 104 | 108 | 44 | 0.5778 | 606 | 27.1154 | 12.7856 | 1.8122 | -4.2231 | 9.3768 | 0.0993 | -0.2913 | 0.4108 | 425.0000 | 2.7345 | 1.7982 | 3.8410 | 0.1426 | 0.0849 | 0.2230 | 425.0000 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 분위수 원본 | lgbm_quantile_raw | True | uncertainty | — | False | False | 7551 | 425 | 7.2370 | 12.9313 | 11.4936 | 17.9631 | 386 | 457 | 39 | 0.6088 | 104 | 95 | 44 | 0.5994 | 457 | 23.3654 | 14.6555 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.0297 | 0.7107 | 0.1588 | 0.0252 | 1.8125 | 0.5316 | 0.4592 | 3.6185 | 0.8400 | 0.7266 | 1.9996 | 0.9142 | 0.8046 | 1.3261 | 0.9529 | 0.8549 | 0.9305 | 1.9374 | — | — | — |
| 전역 보정 A | lgbm_quantile_a | True | uncertainty | lgbm_quantile_raw | False | False | 7551 | 425 | 7.2370 | 12.9313 | 11.4936 | 17.9631 | 407 | 669 | 18 | 0.5423 | 103 | 120 | 45 | 0.5553 | 669 | 30.1456 | 15.6253 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.0294 | 0.7145 | 0.1588 | 0.0252 | 1.8125 | 0.5316 | 0.4592 | 3.6185 | 0.8578 | 0.7218 | 1.9863 | 0.9298 | 0.8141 | 1.2873 | 0.9676 | 0.8837 | 0.8747 | 1.9158 | -0.0048 | 0.0096 | 0.0111 |
| 구간별 보정 B | lgbm_quantile_b | True | uncertainty | lgbm_quantile_a | False | False | 7551 | 425 | 7.2370 | 12.9313 | 11.4936 | 17.9631 | 383 | 383 | 42 | 0.6432 | 94 | 60 | 54 | 0.6225 | 383 | 26.4894 | 15.6419 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.0290 | 0.7093 | 0.1588 | 0.0252 | 1.8125 | 0.5316 | 0.4592 | 3.6185 | 0.8780 | 0.8813 | 1.8705 | 0.9400 | 0.9293 | 1.1909 | 0.9800 | 0.9976 | 0.8167 | 1.8618 | 0.1595 | 0.1151 | 0.0282 |
| 최종 점예측 | lgbm_no_holiday_weight_2 | True | point | lgbm_no_holiday_weight_4 | True | True | 7551 | 425 | 8.1848 | 13.1640 | 14.3281 | 14.4431 | 379 | 593 | 46 | 0.5426 | 98 | 87 | 50 | 0.5886 | 593 | 33.6735 | 12.0515 | 3.8130 | -0.9260 | 9.8691 | 0.2089 | -0.0627 | 0.4185 | 425.0000 | 2.0008 | -0.1148 | 3.7229 | 0.1217 | -0.0083 | 0.1990 | 425.0000 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

### T5-1_adoption

[전체 표](../outputs/tables/T5-1_adoption.csv)

| criterion | adopted | holiday_mae_improvement | overall_mae_improvement | peak_mae_improvement | base_false_alarms | weighted_false_alarms | top_coverage_a | top_coverage_b | pinball_a | pinball_b | spearman | ci95 | episodes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h1_holiday | True | {'estimate': 0.07014355514630065, 'ci95': [0.004192334691893758, 0.14331761581065677], 'n': 2695} | {'estimate': 0.25747389027887974, 'ci95': [0.1399992398883545, 0.3945923632501202], 'n': 7564} | — | — | — | — | — | — | — | — | — | — |
| h1_weight_2 | False | — | — | {'estimate': 0.5098504817683618, 'ci95': [0.2690929117442732, 0.7652856385409028], 'n': 416} | 607.0000 | 800.0000 | — | — | — | — | — | — | — |
| h1_weight_4 | True | — | — | {'estimate': 0.6942737271912911, 'ci95': [0.19248731000430958, 1.3493060479374777], 'n': 416} | 607.0000 | 504.0000 | — | — | — | — | — | — | — |
| h1_mondrian | True | — | — | — | — | — | 0.8731 | 0.9281 | 1.1469 | 1.1397 | — | — | — |
| h1_expected_exceedance | True | — | — | — | — | — | — | — | — | — | 0.8629 | [0.7965698753257224, 0.8956253690827658] | 144.0000 |
| h4_holiday | False | {'estimate': 0.01879818270721785, 'ci95': [-0.08001616394884321, 0.11167175580434711], 'n': 2696} | {'estimate': 0.08125264113071506, 'ci95': [-0.010435211595326389, 0.17957655901660485], 'n': 7551} | — | — | — | — | — | — | — | — | — | — |
| h4_weight_2 | True | — | — | {'estimate': 4.73533082606913, 'ci95': [3.428436648065865, 5.820442849277651], 'n': 425} | 687.0000 | 593.0000 | — | — | — | — | — | — | — |
| h4_weight_4 | True | — | — | {'estimate': 2.7345212951950355, 'ci95': [1.7981638811118952, 3.840991954397303], 'n': 425} | 687.0000 | 606.0000 | — | — | — | — | — | — | — |
| h4_mondrian | True | — | — | — | — | — | 0.8141 | 0.9293 | 1.9158 | 1.8618 | — | — | — |
| h4_expected_exceedance | True | — | — | — | — | — | — | — | — | — | 0.6731 | [0.524720260672653, 0.7835914948017733] | 148.0000 |
| h16_holiday | True | {'estimate': 0.9234390440880079, 'ci95': [0.2874811896458482, 1.6757587611287568], 'n': 2700} | {'estimate': 0.911417353457678, 'ci95': [0.18270045161675774, 1.8472727679556782], 'n': 7498} | — | — | — | — | — | — | — | — | — | — |
| h16_weight_2 | False | — | — | {'estimate': -0.40920779633114684, 'ci95': [-2.906557416273317, 1.8889946547128331], 'n': 425} | 852.0000 | 914.0000 | — | — | — | — | — | — | — |
| h16_weight_4 | True | — | — | {'estimate': 2.9908814595155664, 'ci95': [-2.1899768058922042, 7.659058363432565], 'n': 425} | 852.0000 | 851.0000 | — | — | — | — | — | — | — |
| h16_mondrian | True | — | — | — | — | — | 0.7997 | 0.9134 | 3.4220 | 3.3518 | — | — | — |
| h16_expected_exceedance | True | — | — | — | — | — | — | — | — | — | 0.7008 | [0.5796627414863255, 0.780505539806841] | 148.0000 |

선정의 전체 근거는 2장과 outputs/logs/development_selection.json에 있다. 실현되지 않은 개선을 주장하지 않는다.
