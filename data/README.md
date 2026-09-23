# 원본 데이터 위치와 무결성

아래 파일은 내용 변경 없이 이동했다. SHA-256은 이동 전후 동일함을 확인했다. data/raw/는 Git 추적에서 제외하므로 다른 환경에서는 별도로 받은 원본을 같은 위치에 배치한다. 전처리·시각 복원·특징 생성 결과를 이 폴더에 덮어쓰지 않는다.

| 현재 경로 | 이전 경로 | SHA-256 |
| :-- | :-- | :-- |
| data/raw/task02_welding/scaled_data.csv | 2. 용접기 AI 데이터셋/2. 용접기 AI 데이터셋/scaled_data.csv | a267d1eaab5ca7c81205e38dbb7fb024ebbcb25b866d8cc0907890526e4edfb6 |
| data/raw/task02_welding/Welding Data Set_01.xlsx | 2. 용접기 AI 데이터셋/2. 용접기 AI 데이터셋/Welding Data Set_01.xlsx | d514d6aaa121630c04d7d51c97a56025e78e2c86868813f7722ba5db922c1f33 |
| data/raw/task03_press/outlier_data.csv | 3. 소성가공 예지보전 AI 데이터셋/3. 소성가공 예지보전 AI 데이터셋/outlier_data.csv | 9fad8c238e63dba3257d8be32a32362548955c772171a95c0d2af61b0435ed2e |
| data/raw/task03_press/press_data_normal.csv | 3. 소성가공 예지보전 AI 데이터셋/3. 소성가공 예지보전 AI 데이터셋/press_data_normal.csv | f2d61cb3b3108f49a3305d2966b3a31f26942ac8af8ffc9eb531b432b945b0a7 |
| data/raw/task05_power/okm_augumented_2021.csv | 5. 자원 최적화 AI 데이터셋/5. 자원 최적화 AI 데이터셋/okm_augumented_2021.csv | 8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830 |
| data/raw/task05_power/source.zip | data/source.zip | 02640b4cc90967c26a897953e43f42be0c9b0ca971364fa20b587cb50466b63a |

원본 ZIP에 든 okm_augumented_2021.csv와 별도 CSV는 바이트 단위로 같으며 SHA-256은 8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830이다. ZIP은 출처 보존과 화면 시제품 실행에 사용하고, 검증 코드는 별도 CSV를 읽는다.

기존 verification/results/01_inventory.md, 02_inventory.md, 03_·05_ 결과의 이전 경로는 당시 실행 스냅샷이므로 수정하지 않았다. 이 표를 경로 대응표로 사용한다.
