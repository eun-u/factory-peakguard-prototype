# ② 로봇용접 데이터 인벤토리

## 발견한 파일

파일 | 바이트 | SHA-256 앞 12자리
:-- | --: | :--
2. 용접기 AI 데이터셋/2. 용접기 AI 데이터셋/scaled_data.csv | 639,945 | a267d1eaab5c
2. 용접기 AI 데이터셋/2. 용접기 AI 데이터셋/Welding Data Set_01.xlsx | 551,445 | d514d6aaa121

## XLSX 시트: Raw data

- 크기: 11,939행 × 10열; 완전 중복 행: 0개

| 컬럼 | dtype | 결측 | 고유값 |
| :-- | :-- | --: | --: |
| idx | float64 | 0 | 2,000 |
| Machine_Name | object | 0 | 1 |
| Item No | object | 0 | 1 |
| working time | float64 | 0 | 9 |
| Thickness 1(mm) | float64 | 0 | 1 |
| Thickness 2(mm) | float64 | 0 | 1 |
| weld force(bar) | float64 | 0 | 177 |
| weld current(kA) | float64 | 0 | 43 |
| weld Voltage(v) | float64 | 0 | 113 |
| weld time(ms) | float64 | 0 | 8 |

- `working time` 엑셀 일련번호 해석: 2020-03-24~2020-04-07; 날짜 9개; 날짜 파싱 실패 0개

## XLSX 시트: result

- 크기: 23행 × 7열; 완전 중복 행: 0개

| 컬럼 | dtype | 결측 | 고유값 |
| :-- | :-- | --: | --: |
| idx | float64 | 0 | 23 |
| Machine_Name | object | 0 | 1 |
| Item No | object | 0 | 1 |
| working time | float64 | 0 | 8 |
| defect | float64 | 0 | 4 |
| defect type | float64 | 0 | 3 |
| Unnamed: 7 | object | 20 | 3 |

- `working time` 엑셀 일련번호 해석: 2020-03-24~2020-04-07; 날짜 8개; 날짜 파싱 실패 0개

## XLSX 시트: data set

- 크기: 10행 × 5열; 완전 중복 행: 0개

| 컬럼 | dtype | 결측 | 고유값 |
| :-- | :-- | --: | --: |
| Data | object | 0 | 10 |
| 항목 설명 | object | 0 | 10 |
| 수집 범위 | object | 0 | 6 |
| Unnamed: 4 | object | 10 | 0 |
| Unnamed: 5 | object | 10 | 0 |

## CSV: scaled_data.csv

- 크기: 11,939행 × 5열; 완전 중복 행: 0개

| 컬럼 | dtype | 결측 | 고유값 |
| :-- | :-- | --: | --: |
| Unnamed: 0 | int64 | 0 | 11,939 |
| 용접 가압력 | float64 | 18 | 212 |
| 전류 | float64 | 15 | 84 |
| 전압 | float64 | 15 | 48 |
| 통전시간 | float64 | 23 | 39 |

## 시트 내 데이터 사전

항목 | 설명 | 문서상 수집 범위
:-- | :-- | :--
idx | 생산순번 | -
Machine_Name | 생산설비 | -
Item No | 생산품목 | -
working time | 작업시간 | -
Thickness 1(mm) | 소재 두께 1 | 0.3-2.3
Thickness 2(mm) | 소재 두께 2 | 0.3-2.3
weld force(bar) | 용접 가압력 | 1.00~12.00
weld current(kA) | 전류  | 12.00~18.00
weld Voltage(v) | 전압 | 1.50~3.50
weld time(ms) | 통전시간 | 30~120

`data set` 시트의 두께는 컬럼명에 mm, 가압력은 bar, 전류는 kA, 전압은 V, 통전시간은 ms가 명시되어 있다.
`working time`은 실제로 날짜 일련번호만 있으며 시각은 없다. 용접속도는 제공되지 않았다.
`result`의 `defect`는 0~3 정수이며 날짜·유형별 보고 건수로 추정된다(데이터 사전에 정의 없음). `defect type`은 1~3 코드다. G열은 제목 없이 첫 세 행에서만 각각 파임불량, 용접부족, 크랙발생을 적는다. 불량 유형 외에 개별 용접 양불 정의나 검사 규격은 없다.
2020-03-31에는 유형 3 결과 행이 없으므로 그날의 유형 3을 0건으로 해석하지 않는다.

## 행 단위와 결합 가능성

`Raw data`는 날짜별로 `idx`가 다시 시작하고 행마다 용접조건 한 세트가 기록되어 있어 용접 1회의 요약 행으로 **추정**된다. 파형 시계열과 초 단위 타임스탬프는 없다.
`result`는 `idx`가 1~23인 날짜·유형별 건수 집계 행이다. 두 시트의 `idx`는 서로 다른 순번이며, 동일 날짜에는 원자료 행이 수백~수천 개 있다. 개별 검사 결과를 원자료의 한 행에 결합할 키가 없다.
`scaled_data.csv`에는 불량 라벨·날짜·용접 ID가 없고 변환 절차도 없다. 동일 행 수만으로 원자료와 정렬되었다고 가정하지 않는다.
따라서 결과 건수를 해당 날짜의 모든 용접 행에 방송하거나 누락 날짜를 정상으로 채우지 않는다.
