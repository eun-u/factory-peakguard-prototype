# F0-2 MSTL 원인 점검과 수정 후보

상태: 기존 Phase C 개발 산출물과 소스에 대한 읽기 전용 진단. Phase F 실제 데이터 재학습·평가 결과가 아니다. 최종 holdout 및 `verification/results`는 열지 않았다.

## 확인된 현상

`outputs/phase_c/tables/model_auc_summary.csv`의 기존 B3(MSTL) AUC-MAE는 D1 113.7312, D2 161.0431이다. 동일 표에서 B5는 D1 13.0367이다. 이것은 Phase C에서 이미 공개된 개발 결과이며 Phase F 개선 수치가 아니다.

`phase_c/statistical.py`의 B3은 fold의 마지막 28 fit 일을 한 번 적합한다. 해당 창의 추세와 계절 성분은 fit 범위에서 계산하고, 그 뒤 전체 score 종료까지 `MSTL.predict(h=n_future)`를 한 번 호출하여 고정 base path를 만든다. 예측 시에는 그 base path와 현재 관측값의 AR(1) 잔차만 결합한다. 하루마다 재적합한 과거 P4 MSTL과 다른 규격이다.

로컬 개발 모델 cache `outputs/phase_c/models/B3_h16_f{0,1,2}.joblib`의 base path 통계만 읽었다. 정답·최종 산출물은 읽지 않았다.

| h16 fold | fit 종료 | 마지막 fit 주 base 중앙값 | base 끝 주 중앙값 | base 최대값 | fit 창 forward-fill 수 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | 2021-03-13 14:00 | 93.38 | 1106.10 | 1239.39 | 0 |
| 1 | 2021-04-17 19:15 | 99.71 | 90.75 | 202.74 | 0 |
| 2 | 2021-05-23 00:30 | 82.63 | 85.46 | 212.40 | 0 |

fold 0의 고정 base가 현저히 증가했다. fit 창에는 forward-fill이 없으므로 그 fold의 폭주는 결측 보간 탓으로 설명되지 않는다. 소스에서 적합 trend forecaster는 `AutoETS(model="ZZN")`이다. 고정 base의 원거리 외삽이 큰 오차의 직접적인 구조적 원인이다. cache에는 외삽 후 trend와 seasonal을 분리한 값이 없어 AutoETS 추세와 seasonal 중 어느 성분이 얼마를 기여했는지는 이 자료만으로 단정하지 않는다. 인덱스는 15분 간격으로 fit 창부터 score 종료까지 연속 생성되어, 지금 확인한 코드에는 명백한 길이 오프셋이 없다.

## 별도 수정 후보

새 `phase_f/models/statistical.py`의 `mstl_stationary`는 fit 구간의 마지막 28일에서 MSTL 계절 성분을 한 번 학습한다. 마지막 한 주 패턴을 고정하여 미래 시각의 계절성에 배치하고, 관측된 전력과 계절 성분의 차이로 수준 상태를 origin까지 인과적으로 갱신한다. 수준 갱신 계수는 fit 관측만으로 선택한다. 원거리 `MSTL.predict` 추세 외삽은 사용하지 않는다. 기존 B3 코드·모델·수치는 그대로 보존한다.

이 수정이 실제 MAE/PeakMAE를 낮추는지는 아직 모른다. 고정 cohort의 EXPLORE에서 B3/B5와 paired 비교하고, 미래값 교란·fit 경계 검사를 통과한 뒤 판단한다.
