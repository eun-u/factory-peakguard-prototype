# ③ 사전 타당성 검증

상태: 실행 완료. 세부 수치는 `03_press.json` 및 아래 CSV를 참조.

## P0 구조 진단
- 사용 행: 20600 / 전체 20600; candidate segment 수: 620
- 표본 간격 중앙값: 0.1초; gap 기준: 0.5초 (중앙값의 5배).
- 정상/이상 segment 수: 599/21. 측정 범위와 간격 분포는 `03_press.json`, `03_gaps.csv` 참조.
- 원파형 여부: 추정만 가능. 부호·분포·자기상관은 JSON 참조.

## P1 세션 shortcut
- 최고 단변량: AI1_Vibration_min AUROC 0.879 (95% CI 0.746–0.989). 라벨로 점수 방향을 정한 진단용 값이다.
- 전체 표: `03_p1_univariate.csv`.

## P2 정규화 불변성
- 사전 고정 raw Mahalanobis: AUROC 0.999 (95% CI 0.996–1.000); PR-AUC 0.985 (95% CI 0.953–1.000).
- 정상 segment 앞 60%로만 학습하고 뒤 40%와 모든 이상 segment로 평가. 전체 12개 조합: `03_p2_models.csv`.
- 정규화 후 하락은 세션 차이와 실제 고장 레벨 변화 양쪽으로 해석될 수 있다.

## P3 3채널 관계 잔차
- 합산 잔차 AUROC 0.907 (95% CI 0.788–1.000); PR-AUC 0.875 (95% CI 0.710–1.000).
- `03_p3_relation.csv`, 산점도 `03_p3_level_relation.png`.

## P4 합성 주입
- offset·scale 및 채널 교체별 레벨/관계 AUROC: `03_p4_injections.csv`. 가짜 이상은 실제 고장 증거가 아니다.

## P5 경보 KPI
- 정상 학습 점수 분위수만으로 α=1%, 5% 임계값을 설정했다. 규칙별 정상 운전시간당 경보 사건·채터링·이상 segment 탐지율·첫 경보 지연: `03_p5_alarms.csv`.
- 지연은 이상 파일 시작 기준이다. 독립 고장 시작 시각과 해제 시각이 확인되지 않아 지연 CI·해제 시간은 판정불가.

## 사전 경고 기준 판정
- K3-a: 비해당
- K3-b: 해당
- K3-c: 비해당

## 해석 한계
- Candidate segments are defined by timestamp gaps, not verified press cycles.
- Abnormal segments may belong to one recording session; segment CIs do not capture independent-failure uncertainty.
- Normalized performance loss can reflect session shift or genuine fault-related level change.
- No fault onset or return-to-normal timestamp is verified; release time is unmeasurable.
- P1 score orientation uses both labels for diagnosis and is not a deployable classifier.
- P5 detection delay is from abnormal file start; it is not validated early-warning lead time.
- Channel shuffle interpolates donor segments when lengths differ; this also alters shape and autocorrelation.
- Sequential chatter and one-session detection delay lack a defensible independent-event bootstrap CI.
