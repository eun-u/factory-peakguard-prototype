# 추가 기술: 결론, 한계, 향후 과제

## 7.1 결론

이 연구는 제조공장의 15분 전력을 1시간 앞서 예측하고, 그 예측을 최대수요 피크 경보와 생산량 이동으로 연결했다. 핵심 결과는 세 가지다.

1. **가장 강한 기준선 대비 개선.** 직전 15분 persistence 대비 LightGBM의 피크 위치 MAE는 27.73 → 17.77로 35.9% [19.6, 50.1] 줄었다. 모델의 가치는 피크를 더 많이 맞히는 데보다 **오경보를 줄이는 데**(774 → 305) 있었다.
2. **실패 조건의 분리.** 미탐은 저녁 16–24시(재현율 0.388 [0.250, 0.561])와 가동 전환 시각, 장기 휴무 후 첫 가동일에, 오경보는 주간 고생산 시간에 집중되었다. 저녁 피크는 짧고 낮으며, 모델은 이미 높은 부하를 평균 25.7 낮게 예측했다.
3. **현장 가치의 조건부 확인.** 조치 비용이 피크 손실의 1/4~2/5인 현장에서 경보의 상대 경제가치는 0.16~0.20이었다(구간 하한 > 0).

익일 일간 최대 예측에서는 기준선보다 낫다는 근거를 얻지 못했고(MAPE 28.3% vs 18.6%), 0.95 분위수는 과소 커버리지(0.911)였다.

## 7.2 한계

| 한계 | 영향 | 대응 |
| :-- | :-- | :-- |
| 전력 단위·15분 구간 경계 미확인 (A3, A4) | 금액 환산 불가, 구간 해석이 가정에 의존 | 비율·민감도로만 보고. 확인되면 4.4절 식으로 바로 환산 |
| 완전한 하계·동계 부재 (K5-a 해당) | 다른 계절·연중 최대수요로 외삽 불가 | 결론을 1~9월로 한정 |
| 설비·제품·교대 정보 부재 | 조건 분석이 대리변수에 의존, 저감 시뮬레이션의 이동 가능 비율 미상 | 대리변수임을 명시, 시뮬레이션을 가정 기반으로 한정 |
| 테스트 구간 재열람 | 낙관 편향 가능성 | 모델 설정 고정 시점과 열람 이력 공개(표 1-5) |
| 하이퍼파라미터 미튜닝 | 성능을 덜 끌어냈을 수 있음 | 개발 폴드 튜닝을 향후 과제로 |
| 테스트 38일, 저녁 피크 49개 위치 | 조건별 구간이 넓음, 다중 비교 미보정 | 탐색적 결과로 표기 |
| 장기 휴무 후 첫 가동일 1일 | 일반화 불가 | 운영 규칙은 제안으로만 제시 |
| 생산량–전력 관계는 관측 연관 | 이동 효과는 인과 추정이 아님 | 조치 기록으로 현장 검증(4.6절) |

## 7.3 향후 과제

1. 휴무·교대 일정을 입력에 추가해 장기 휴무 후 첫 가동일과 저녁 연장 운전을 직접 표현한다.
2. 개발 폴드에서 하이퍼파라미터와 피크 가중 손실을 튜닝하고, 새 기간의 자료로 한 번만 평가한다.
3. conformal 보정을 운영에 적용하고 월별로 커버리지를 감시한다.
4. 조치 기록을 쌓아 생산량 이동의 실제 효과를 추정한다.

## 7.4 참고문헌

1. Recht, B., Roelofs, R., Schmidt, L., Shankar, V. (2019). Do ImageNet Classifiers Generalize to ImageNet? *Proceedings of ICML*, PMLR 97. — 테스트 재사용과 대안 설명 배제 절차
2. Engstrom, L. et al. (2020). Identifying Statistical Bias in Dataset Replication. *Proceedings of ICML*, PMLR 119. — 측정 절차 편향
3. Zeng, A., Chen, M., Zhang, L., Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? *AAAI 2023*. — 단순 기준선의 강함
4. Makridakis, S., Spiliotis, E., Assimakopoulos, V. (2018). Statistical and Machine Learning forecasting methods: Concerns and ways forward. *PLOS ONE* 13(3). — 기준선과의 공정 비교
5. Liu, C. et al. (2026). LLM-enhanced embodied multi-agent manufacturing system. *Journal of Manufacturing Systems* 84, 357–382. — 보고서 구성 참고
6. 〔추가 예정〕 Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*.
7. 〔추가 예정〕 Vovk, V., Gammerman, A., Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. — conformal 예측
8. 〔추가 예정〕 Richardson, D. S. (2000). Skill and relative economic value of the ECMWF ensemble prediction system. *QJRMS* 126. — 비용-손실 모형과 REV
9. 〔추가 예정〕 Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *Annals of Statistics* 17(3). — 블록 부트스트랩
10. 〔추가 예정〕 한국전력공사 기본공급약관 및 시행세칙 — 최대수요전력과 요금적용전력

6~10번은 서지 정보를 원문으로 대조한 뒤 확정한다.

## 7.5 경진대회 만족도 조사 완료 화면 (필수)

〔캡처 이미지 삽입: 만족도 조사 완료 페이지. 원본은 `submission/survey/`에 보관〕
