# 논문형 연구 관리

1. [기본 정의](charter.json): Background부터 Contribution까지 12항목, RQ1~4.
2. [주장·근거](claims.json): 주장 상태, 근거 역할, 비교 행/JSON 위치.
3. [실험 등록부](../experiments/registry.json): 기존 연구와 새 계획을 구분.
4. [현재 현황](generated/status.md): 정의·선정·주장·실험을 자동 연결.
5. [논문 본문](../paper/manuscript.md): 수동 서술과 생성 결과를 결합.

```bash
python -m research status
python -m research validate
python -m research show R2-EX02
python -m research build
python -m research check
python -m unittest discover -s research/tests -v
```

Python 3.10+ 표준 라이브러리만 필요하다. raw data, 학습 패키지, GPU 없이 관리·집필 가능하다. `build`는 지정된 네 생성 파일만 갱신하며 학습·최종 평가·승인·예약 작업을 호출하지 않는다. `check`는 소스와 생성물의 불일치를 실패로 반환한다. 알려진 작은 결과 파일의 해시·선정·키·근거 관계를 검증하며 과학적 우월성 자체를 자동 판정하지 않는다.

실험 상태는 planned/running/completed/blocked/rejected, 주장 상태는 supported_development/exploratory/hypothesis/design_only/simulation_only다. 완료 실험은 근거 파일이 필요하고 탐색 근거를 개발 확증 주장으로 승격할 수 없다. 기각된 결과도 보존한다.

주장은 근거·실험 ID와 수동 집필 절을 연결한다. 실험 후 등록부 상태·근거 경로·본문을 갱신하고 `build`와 `check`를 실행한다. GitHub Actions도 생성물 최신성과 관리 계층 테스트를 검사한다. 원자료가 필요한 모델 실험의 성공 여부를 이 검사로 대신하지 않는다.

연구 관리 모듈은 `src/`의 과학적 계산과 분리하여 기존 모델 지문을 보존한다. 새 모델/정책 구현은 정상적으로 기존 해시와 캐시를 무효화해야 하며 이를 회피해서는 안 된다.
