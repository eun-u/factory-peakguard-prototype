# FG-R9 사전 고정 (2026-10-07, 후보 결과 확인 전 작성)

배경: FG-R8 최종 테스트(1회)에서 정확 복제 게이트 작동률 1.5%, Chronos-2 단독이 평균 MAE에서 우세.
이번 라운드는 테스트 결과를 본 뒤의 개선이며, 테스트 재평가는 "2차 평가"로만 보고한다.

## 근거한 공개 연구·코드
- GIFT-Eval 상위 파운데이션 모델: Chronos-2(amazon/chronos-2), TimesFM-2.5(google/timesfm-2.5-200m-pytorch),
  TimesFM-3(google/timesfm-3.0-pytorch, 2026-08), TiRex(NX-AI/TiRex). Moirai-2(uni2ts)와 Toto(toto-ts)는
  Windows/Python 3.13에서 의존성(구 scikit-learn 소스 빌드) 설치 실패로 제외.
- 파운데이션 모델 앙상블이 단일 모델보다 낫다는 연구(arXiv 2508.16641, 2507.22053, 2512.16022). 튜닝 없는 평균/중앙값만 사용.

## 선정 규칙
- 평가: 개발 16주(EXPLORE+CONFIRM) score 행 중 **비게이트 행**(테스트 유사 조건)의 13지평 pooled MAE 평균.
- 동점(차이 ≤ 0.05)은 같은 행의 Peak MAE가 낮은 후보.
- 후보: Chronos-2, TimesFM-3, TimesFM-2.5, TiRex 단독; 4개 평균; 4개 중앙값; Chronos+각 모델 2개 평균(3개).
  설치·추론 실패 모델은 해당 후보에서 제외하고 기록.
- 게이트: 오늘 관측 슬롯 수 ≥ k. k는 개발 score 게이트 행 정밀도(|유사일−실측|<0.5) ≥ 0.95가 되는 최소 구간 하한 = 5
  (개발 정밀도: 1~2칸 0.585, 3~4칸 0.887, 5~8칸 0.966, 9칸 이상 ≥0.989).
- 피크 상향 보정(B8, margin 25, CAL 선택)과 CQR 위험 추정은 FG-R8 절차 유지(분위수는 Chronos).
- 최종 잠금은 FG-R8과 같은 28일 보정 구간. 테스트 2차 평가는 1회.

## 추가 후보 (1차 후보 결과 확인 후, 추가 후보 결과 확인 전 작성)

1차 결과: 비게이트 개발 MAE Chronos-2 6.456 < Chronos+TimesFM-3 평균 6.607 < 4개 중앙값 6.975 < TimesFM-3 7.093
< TiRex 8.684 < TimesFM-2.5 9.143. 앙상블은 모두 Chronos 단독보다 나빴다.
추가 후보(같은 선정 규칙): Chronos-2 + 알려진 미래 달력 공변량(시각 sin/cos, 요일, 주말·공휴일);
위 + 과거 전용 공변량 production_known(시간 종료 후 확정 생산량, A8); Chronos-2 문맥 1024.
