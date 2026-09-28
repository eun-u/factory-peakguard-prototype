# P4 모델 보강 실험 (개발 CV 전용)

사전 고정: `outputs/logs/preregistration_0928_P4.md`. 봉인 개발 prefix만 읽는다(`src.session_data.load_development_history`).
기존 `src/`, `configs/default.yaml`은 수정하지 않는다. 결과는 `outputs/p4/`, 캐시는 `_validation/p4_cache/`(Git 제외).

실행 순서(저장소 루트, Python 3.13, requirements-pipeline.lock.txt + statsforecast 2.1.1, optuna 5.0.0, numba, chronos-forecasting 2.3.2, torch CPU):

    python -m research_p4.lgbm_component    # 기존 LGBM 재학습·OOF 일치 검사
    python -m research_p4.stat_models dshw  # P4-1
    python -m research_p4.stat_models mstl  # P4-2 (일간 MSTL 약 10분)
    python -m research_p4.blends            # P4-3, P4-4
    python -m research_p4.lgbm_deep         # P4-5, P4-6 (약 25분)
    python -m research_p4.dense_quantile    # E4
    python -m research_p4.chronos_ref       # E5 참고 (CPU 약 40분, HF 모델 다운로드)
    python -m research_p4.leakage_check     # 미래 교란 불변 검사
    python -m research_p4.evaluate_p4       # 사전 규칙 채점·Holm

LightGBM은 다른 무거운 작업과 동시에 돌리면 OpenMP 스레드 경합으로 크게 느려진다. 순차 실행한다.
정확한 패키지 버전은 `outputs/p4/p4_environment_freeze.txt`.
