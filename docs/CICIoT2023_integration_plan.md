# CICIoT2023 데이터셋 추가 실험 계획서

- 작성일: 2026-10-17
- 대상: FedSAD-Implementation (`main` @ `2215fe6`)
- 목적: 기존 5개 데이터셋(KDD99, NSL-KDD, InSDN, CSE-CIC-IDS2018, UNSW-NB15)과 **동일한 방식**으로 CICIoT2023을 전처리하고, FedSAD(TAAE + GSAD + DS Fusion), RNEP/FedAvg/CML, 베이스라인(AE-LSTM, CNN-LSTM, MM-FewShot-IDS, MV-IDS) 실험을 재현한다.

---

## 1. 요약 (TL;DR)

| 항목 | 결론 |
| :--- | :--- |
| 공식 CSV로 바로 가능한 것 | **TAAE / RNEP / FedAvg / CML / AE-LSTM / CNN-LSTM** (행 단위 오토인코더·분류기) |
| 공식 CSV로 **불가능**한 것 | **GSAD, 그리고 GSAD를 쓰는 FedSAD 통합 평가** — `Timestamp`, `Dst Port` 컬럼이 없음 |
| PCAP 재추출이 필요한 것 | GSAD/FedSAD용 flow 테이블 (Timestamp, Src/Dst IP·Port 포함), MM-FewShot-IDS·MV-IDS용 NetFlow 형식 피처 |
| 권장 방향 | **PCAP → CICFlowMeter로 CIC2018과 동일한 78개 flow 피처(+Timestamp)를 새로 추출해 TAAE·GSAD가 같은 원천을 쓰게 한다.** 공식 CSV는 보조 실험(문헌 비교용)으로 사용 |
| 샘플링 | Normal 200,000 / Anomaly 200,000 (CSE-CIC-IDS2018과 동일 규모). 단, CIC2018식 "희소 클래스 as-is + 비례 다운샘플링"을 그대로 쓰면 중간 규모 클래스가 수백 개로 붕괴 → **클래스별 상한(cap) 방식 권장** (§4.3) |
| 코드 변경 범위 | `utils/arguments.py`, `fedsad.py`, `gsad.py`, `rnep.py`, `utils/get_datasets.py`, `scripts/test_*.py`, `models/MMFEWSHOTSIDS/load_data.py`, `models/MVIDS/preprocess.py`, `.gitignore`, `README.md` |

> ⚠️ **즉시 조치 권장:** `CICIoT/` 폴더(PCAP 548 GB, CSV 8.4 GB)가 `.gitignore`에 없어 현재 untracked 상태입니다. 실수로 `git add .` 하면 대형 파일이 스테이징되므로 `.gitignore`에 `/CICIoT`를 먼저 추가하세요.

---

## 2. 현재 파이프라인 분석

### 2.1 데이터 흐름

```text
원본(PCAP / CSV)
   │  (레포 외부 전처리: packet → session → SDP log-consistent feature)
   ▼
/data/SDP_Dataset/Unified_model/
   ├── cic_rnep/   CIC_ae_normal.csv, CIC_anomaly_ae_{1..14}.csv   ← TAAE 입력 (행 단위 피처)
   └── cic_graph/  CIC_ae_normal.csv, CIC_anomaly_ae_{1..14}.csv   ← GSAD 입력 (Timestamp + Dst Port 필수)
   ▼
taae.py / rnep.py  → results/<DS>/taae/taae_*_weights.h5
gsad.py            → results/<DS>/gsad/normal_stats.pkl
fedsad.py          → results/<DS>/fedsad/{fedsad.pkl, fedsad_cm.png, fedsad_server.txt}
```

### 2.2 모델별 데이터 요구사항

| 모듈 | 파일 | 입력 형식 | 필수 컬럼 | 분할 |
| :--- | :--- | :--- | :--- | :--- |
| TAAE (RNEP 학습) | `rnep.py`, `taae.py`, `utils/get_datasets.py:get_datasets_cic_sam` | normal / anomaly CSV, 모두 수치형 | 없음 (라벨 컬럼 제외 전 컬럼 사용) | normal 80:20 (unified) / 50:50 (FL 비교) |
| GSAD | `gsad.py` | normal / anomaly CSV | **`Timestamp`** (dayfirst 파싱), **`Dst Port`** | normal `sample(frac=0.8, random_state=123)` |
| FedSAD 통합 | `fedsad.py`, `utils/get_datasets.py:fedsad_data_preprocessing` | TAAE용 + GSAD용 CSV 두 벌 | 위 두 조건 모두 | 1분(`'1T'`) 윈도우 단위 판정 |
| AE-LSTM / CNN-LSTM | `scripts/test_aelstm.py`, `scripts/test_cnnlstm.py`, `models/AE_LSTM.py`, `models/CNN_LSTM.py` | `*_normal.csv`, `*_anomaly_{i}.csv`, 병합본 `*_anomaly.csv` | 없음 (`Label` 있으면 drop) | 80:20 |
| MM-FewShot-IDS | `models/MMFEWSHOTSIDS/load_data.py` | 단일 CSV + `Label` | **NetFlow v3 컬럼** (`L4_DST_PORT`, `IN_BYTES`, …) | preset 기반 |
| MV-IDS | `models/MVIDS/preprocess.py`, `load_data.py` | 단일 CSV + `label/Label` | **NetFlow v3 컬럼** (view schema) | preset 기반 |

### 2.3 FedSAD 통합 평가의 핵심 특성 (데이터 설계에 영향)

1. **GSAD 그래프**: 1분 윈도우 안의 고유 `Dst Port`를 노드로 하는 **완전 그래프**를 만들고 `num_nodes, num_edges, density, avg_degree` 4개 통계를 정상 분포(mean ± `threshold_std`·std)와 비교 (`gsad.py:17-46`).
2. **TAAE ↔ GSAD 정렬 방식**: `fedsad.py:evaluate()`는 GSAD 윈도우 크기(행 수)만큼 TAAE 테이블에서 **순서대로** 행을 잘라 매칭합니다. 즉 두 테이블이 같은 행 순서를 공유할 필요는 없지만, **같은 원천에서 나온 데이터일수록 융합 결과가 의미 있습니다.**
3. **파일 탐색 규칙**: anomaly 파일은 `anomaly_prefix`로 시작하는 CSV를 모두 읽습니다. 병합 파일명이 prefix와 겹치면 **이중 집계**되므로 CIC2018처럼 `CIC_ae_anomaly.csv`(병합) vs `CIC_anomaly_ae_`(개별)로 분리해야 합니다.

---

## 3. CICIoT2023 원본 현황 (로컬 `CICIoT/` 확인 결과)

### 3.1 디렉터리

```text
CICIoT/
├── CSV/CSV/<클래스명>/<원본pcap명>.pcap.csv   # 34개 폴더 (Benign 1 + Attack 33), 8.4 GB
└── PCAP/<클래스명>/<클래스명>[N].pcap          # 34개 폴더, 548 GB
```

- 폴더명 표기 불일치 주의: CSV는 `DDoS-PSHACK_FLOOD`, `DDoS-RSTFINFLOOD` / PCAP은 `DDoS-PSHACK_Flood`, `DDoS-RSTFINFlood` → 클래스 매핑 테이블에서 정규화 필요.

### 3.2 공식 CSV 스키마 (39 컬럼)

```text
Header_Length, Protocol Type, Time_To_Live, Rate,
fin/syn/rst/psh/ack/ece/cwr_flag_number, ack/syn/fin/rst_count,
HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC, TCP, UDP, DHCP, ARP, ICMP, IGMP, IPv, LLC,
Tot sum, Min, Max, AVG, Std, Tot size, IAT, Number, Variance
```

- **라벨 컬럼 없음** → 폴더명이 라벨.
- **Timestamp, IP, Port 없음** → 각 행은 flow가 아니라 *고정 개수 패킷 윈도우의 평균 통계* (예: `Number=10`). 따라서 시간 윈도우 그래프(GSAD)를 만들 수 없음.
- 모든 컬럼이 수치형 → TAAE/AE-LSTM/CNN-LSTM 입력으로는 그대로 사용 가능.

### 3.3 클래스별 행 수 (공식 CSV 기준, 직접 집계)

| 카테고리 | 클래스 | 행 수 |
| :--- | :--- | ---: |
| Benign | Benign_Final | 1,098,191 |
| DDoS | DDoS-ICMP_Flood | 7,200,501 |
| | DDoS-UDP_Flood | 5,412,231 |
| | DDoS-TCP_Flood | 4,497,649 |
| | DDoS-PSHACK_FLOOD | 4,094,772 |
| | DDoS-SYN_Flood | 4,059,179 |
| | DDoS-RSTFINFLOOD | 4,045,279 |
| | DDoS-SynonymousIP_Flood | 3,598,133 |
| | DDoS-ICMP_Fragmentation | 452,490 |
| | DDoS-UDP_Fragmentation | 286,925 |
| | DDoS-ACK_Fragmentation | 285,075 |
| | DDoS-HTTP_Flood | 28,790 |
| | DDoS-SlowLoris | 23,426 |
| DoS | DoS-UDP_Flood | 3,072,990 |
| | DoS-TCP_Flood | 2,671,430 |
| | DoS-SYN_Flood | 2,028,836 |
| | DoS-HTTP_Flood | 71,861 |
| Mirai | Mirai-greeth_flood | 991,834 |
| | Mirai-udpplain | 890,574 |
| | Mirai-greip_flood | 751,646 |
| Recon | VulnerabilityScan | 373,351 |
| | Recon-HostDiscovery | 134,378 |
| | Recon-OSScan | 98,259 |
| | Recon-PortScan | 82,284 |
| | Recon-PingSweep | 2,262 |
| Spoofing | MITM-ArpSpoofing | 307,560 |
| | DNS_Spoofing | 178,898 |
| Web-based | BrowserHijacking | 5,859 |
| | CommandInjection | 5,409 |
| | SqlInjection | 5,245 |
| | XSS | 3,846 |
| | Uploading_Attack | 1,252 |
| | Backdoor_Malware | 3,218 |
| Brute Force | DictionaryBruteForce | 13,064 |
| **합계** | Anomaly 33종 | **45,678,506** |

→ Normal : Anomaly ≈ 1 : 41. CSE-CIC-IDS2018(≈ 5 : 1)과 정반대로 **공격이 압도적으로 많은** 구조이므로 샘플링 전략이 핵심입니다.

---

## 4. 전처리 계획

### 4.1 트랙 구성

| 트랙 | 원천 | 산출물 | 사용 모델 |
| :--- | :--- | :--- | :--- |
| **Track A (주 실험)** | PCAP → CICFlowMeter | flow CSV (Timestamp, Src/Dst IP·Port, ~80개 flow 피처, Label) | TAAE, GSAD, **FedSAD**, RNEP/FedAvg/CML, AE-LSTM, CNN-LSTM |
| Track B (보조 실험) | 공식 CSV (39 피처) | 패킷-윈도우 통계 CSV | TAAE, RNEP, AE-LSTM, CNN-LSTM (문헌 수치와 비교용) |
| Track C (선택) | PCAP → nProbe / NFStream | NetFlow v3 형식 CSV | MM-FewShot-IDS, MV-IDS |

**Track A를 주로 권장하는 이유**

- README에 명시된 "packet-level datasets → log-consistent features aligned with SDP logs" 파이프라인과 CSE-CIC-IDS2018/UNSW-NB15 처리 방식을 그대로 따를 수 있음.
- CSE-CIC-IDS2018 원본 CSV도 CICFlowMeter 출력이므로 컬럼명(`Timestamp`, `Dst Port`)·타임스탬프 형식이 GSAD 코드와 그대로 호환됨.
- TAAE와 GSAD가 **같은 flow 테이블**에서 나오므로 FedSAD 융합의 일관성이 보장됨.

#### 기준 스키마: CSE-CIC-IDS2018 TAAE 입력 (확인 완료)

`CIC_anomaly_ae_1.csv`(normal과 동일한 스키마)를 확인한 결과, **CICFlowMeter 출력 컬럼 78개**로 구성되어 있습니다.

```text
Dst Port, Protocol, Flow Duration, Tot Fwd Pkts, Tot Bwd Pkts, TotLen Fwd Pkts, TotLen Bwd Pkts,
Fwd/Bwd Pkt Len Max·Min·Mean·Std, Flow Byts/s, Flow Pkts/s, Flow IAT Mean·Std·Max·Min,
Fwd IAT Tot·Mean·Std·Max·Min, Bwd IAT Tot·Mean·Std·Max·Min, Fwd/Bwd PSH·URG Flags,
Fwd/Bwd Header Len, Fwd/Bwd Pkts/s, Pkt Len Min·Max·Mean·Std·Var,
FIN·SYN·RST·PSH·ACK·URG Flag Cnt, CWE Flag Count, ECE Flag Cnt, Down/Up Ratio, Pkt Size Avg,
Fwd/Bwd Seg Size Avg, Fwd/Bwd Byts/b Avg·Pkts/b Avg·Blk Rate Avg, Subflow Fwd/Bwd Pkts·Byts,
Init Fwd/Bwd Win Byts, Fwd Act Data Pkts, Fwd Seg Size Min, Active Mean·Std·Max·Min, Idle Mean·Std·Max·Min
```

- CSE-CIC-IDS2018 공개 CSV(80컬럼)에서 **`Timestamp`와 `Label`만 제거한 형태**입니다 (`Dst Port`는 TAAE 피처로 포함).
- GSAD용(`cic_graph`)은 별도 스키마입니다 (아래 참조).
- 따라서 CICIoT2023도 **PCAP → CICFlowMeter → 위 78개 컬럼(+Timestamp)** 으로 맞추면 CIC2018과 완전히 같은 피처 공간이 되어, TAAE 입력 차원(78)·전처리 코드를 그대로 재사용할 수 있습니다.
- 이 헤더를 스키마 기준 파일(`preprocess/ciciot_schema.txt` 등)로 저장해 두고, 추출 결과의 컬럼명·순서를 검증하는 데 사용합니다.
- `CIC_anomaly_ae_1.csv` = 36,897행, 전부 `Dst Port=80`, `Protocol=6` → README 샘플링 표의 **DDoS Attack-HOIC(36,897)** 와 일치. 즉 파일 번호는 `scripts/test_taae.py`의 `attack_map` 순서(1=HOIC)를 따릅니다.

#### 기준 스키마: CSE-CIC-IDS2018 GSAD 입력 (확인 완료)

`cic_graph/CIC_ae_normal.csv` 분석 결과:

| 항목 | 값 |
| :--- | :--- |
| 컬럼 | **`Dst Port`, `Timestamp` 두 개뿐** |
| 행 수 | 200,000 (TAAE normal과 동일 규모) |
| Timestamp 형식 | `dd/mm/yyyy hh:mm:ss` (예: `20/02/2018 01:36:05`), 파싱 실패 0건 |
| 기간 | 2018-02-14 ~ 2018-03-02 중 10일, 날짜별 비중은 원본 benign 양에 비례 (02-20이 109,467행) → **전체 benign에서 행 단위 무작위 샘플링** |
| 정렬 | 시간순 정렬 안 됨 (GSAD 코드에서 정렬) |
| 1분 윈도우 | 비어있지 않은 윈도우 4,531개, 윈도우당 평균 44행(중앙값 20), 고유 포트 평균 13.8(최대 1,994) |
| 참고 | 시각이 01:00~12:59 범위뿐 → CICFlowMeter의 12시간제 출력에서 AM/PM이 빠진 CIC2018 원본 특성이 그대로 남아 있음 |

**CICIoT2023 적용 방침**

- GSAD용 파일도 **`Dst Port`, `Timestamp` 2컬럼**으로 동일하게 생성하고, 형식은 `dd/mm/yyyy HH:MM:SS`(24시간제)로 씁니다. AM/PM 손실은 CIC2018 고유의 결함이므로 재현하지 않습니다 (`dayfirst=True` 파싱과 호환).
- 샘플링은 CIC2018과 똑같이 **행 단위 무작위 샘플링**을 씁니다 (§4.3).
- TAAE 파일과 GSAD 파일은 **같은 행을 같은 순서로** 뽑아 저장합니다 (i번째 행이 같은 flow). 두 파일에 같은 `sample(frac=0.8, random_state=123)`를 적용하면 train/test가 같은 flow로 나뉩니다.
- 결과 해석 시 참고: CICIoT2023 benign은 pcap 4개뿐이라 캡처 기간이 CIC2018(10일)보다 짧을 가능성이 큽니다. 그러면 윈도우당 행 수가 CIC2018보다 훨씬 많아집니다. GSAD는 같은 데이터셋의 normal로 통계를 학습하므로 데이터셋 안에서는 일관되지만, `threshold_std`는 다시 조정해야 합니다.

### 4.2 Track A 처리 절차

1. **PCAP 선별** (548 GB 전체 처리는 비현실적)
   - Benign: 4개 pcap 전부 처리 (6.6 GB).
   - Attack: 대형 클래스는 분할된 pcap 중 1–2개만 처리 (예: `DDoS-ICMP_Flood.pcap`, `DDoS-ICMP_Flood1.pcap`). 어차피 §4.3에서 수천 건으로 다운샘플링하므로 충분.
   - 소형 클래스(Web-based, Recon, BruteForce 등)는 전부 처리.
2. **flow 추출**: CICFlowMeter(Java 원본 권장, CIC2018과 동일 도구) 실행 → 클래스별 flow CSV.
   - 공식 CICIoT2023 CSV에 Timestamp만 "붙이는" 방식은 불가능합니다. 공식 CSV는 패킷 윈도우 통계(39개 피처)라 CIC2018 피처와 다르고, PCAP의 어느 시점과 대응되는지 행 단위 키(IP/Port/시간)가 없기 때문입니다. → **TAAE용·GSAD용 모두 PCAP에서 새로 추출**합니다.
   - Java CICFlowMeter 출력 헤더는 CIC2018 공개 CSV의 약어 형식(`Dst Port`, `TotLen Fwd Pkts` …)과 다를 수 있고, Python `cicflowmeter`는 snake_case(`dst_port`, `flow_duration` …)를 씁니다. 어떤 도구를 쓰든 **rename 매핑으로 §4.1 기준 78개 컬럼명·순서에 맞춥니다.**
   - 추출 결과에서 `Timestamp`(GSAD용), `Dst Port`(공통)를 확보하고, `Flow ID`, `Src IP`, `Src Port`, `Dst IP`는 라벨 검증 후 제거.
3. **라벨링**: 폴더명 → `Label` (Benign / 33종 attack → 최종 27종, §11.3), 대분류 `Category` 컬럼도 함께 저장(카테고리별 recall 분석용).
4. **정제**: `Flow ID`, `Src IP`, `Dst IP`, `Src Port` 등 식별자 컬럼 제거(TAAE용) / GSAD용은 `Timestamp`, `Dst Port` 유지. `inf/NaN → 0`, 음수 → 0 (`sanitize_numeric`와 동일).
5. **타임스탬프 검증**: `pd.to_datetime(..., dayfirst=True)` 파싱 결과가 올바른지 확인 (CICFlowMeter Java 버전은 `dd/MM/yyyy hh:mm:ss a` 형식, Python 포트는 ISO 형식일 수 있음). 파싱 실패 행은 GSAD 코드에서 조용히 drop되므로 **반드시 drop 비율을 로그로 확인**.
6. **산출물 저장** (§5 디렉터리 규약).

### 4.3 샘플링 전략 (Normal 200k / Anomaly 200k)

**Normal**: Benign 1,098,191 중 200,000 샘플링 (CIC2018과 동일 규모).

- CIC2018 `cic_graph`가 행 단위 무작위 샘플링을 썼으므로(§4.1) 동일하게 적용합니다. TAAE용과 GSAD용은 같은 샘플 행을 공유합니다.

**Anomaly**: CIC2018 방식("희소 클래스 as-is + 나머지 비례 다운샘플링")을 그대로 적용하면 아래처럼 중간 규모 클래스가 수백 건으로 붕괴합니다.

| 방식 | 설명 | 결과 예 (공식 CSV 기준 계산) |
| :--- | :--- | :--- |
| ① CIC2018 방식 그대로 (as-is 임계 30k) | ≤30k 10개 클래스 as-is(92,371) + 나머지 107,629를 원본 비율로 배분 | DoS-HTTP_Flood **170**, Recon-PortScan **194**, Recon-OSScan **232**, DDoS-ICMP_Flood 17,000 |
| ② **클래스별 상한(cap) 방식 — 권장** | 각 클래스 `min(원본, cap)`, 합이 200k가 되도록 cap 결정 | cap ≈ **6,650** → 희소 7개 클래스는 as-is, 나머지 26개는 6,650씩 |

권장 이유: "rare attack은 as-is, 빈도 높은 공격은 다운샘플링해 다양성 보존"이라는 **CIC2018 샘플링 의도**를 그대로 유지하면서, CICIoT2023의 극단적 불균형(최대/최소 ≈ 5,750배)에서도 클래스 다양성이 무너지지 않습니다. 논문/README에는 "capped sampling"으로 명시하면 됩니다.

> 위 cap 값(6,650)은 공식 CSV 33종 기준 추정치입니다. 실제 산출물은 flow 단위 27종 기준으로 다시 계산해 **cap = 8,673~8,674**입니다 (§11.1).

> Track A(flow 단위)의 클래스별 행 수는 공식 CSV(패킷 윈도우 단위)와 다르므로 flow 추출 후 cap을 **재계산**해야 합니다. 샘플링 스크립트는 `random_state=123` 고정.

### 4.4 분할 규칙 (기존과 동일)

| 실험 | Normal 분할 | Anomaly |
| :--- | :--- | :--- |
| FL 비교 (CML / FedAvg / RNEP) | 50 : 50 (train : test) | 전부 test |
| Unified / packet-level (TAAE, GSAD, FedSAD, 베이스라인) | 80 : 20 | 전부 test |

---

## 5. 산출물 디렉터리 규약

기존 `cic_rnep` / `cic_graph` 구조를 그대로 따릅니다.

```text
/data/SDP_Dataset/Unified_model/
├── ciciot_rnep/                       # TAAE / RNEP / FedSAD-TAAE 입력
│   ├── CICIoT_ae_normal.csv
│   ├── CICIoT_anomaly_ae_1.csv ... CICIoT_anomaly_ae_27.csv
│   └── CICIoT_ae_anomaly.csv          # 병합본 (prefix와 겹치지 않게!)
└── ciciot_graph/                      # GSAD / FedSAD-GSAD 입력 (Timestamp, Dst Port 포함)
    ├── CICIoT_ae_normal.csv
    └── CICIoT_anomaly_ae_1.csv ... CICIoT_anomaly_ae_27.csv

./CICIoT2023/ae_datas_sampled/         # 베이스라인(AE-LSTM, CNN-LSTM, test_taae)용 로컬 경로
results/CICIoT2023/{taae,gsad,fedsad,rnep,avg,cml,aelstm,cnnlstm,mmfewshotsids,mvids}/
```

- 공격 번호 ↔ 이름 매핑(1–27)은 `attack_map`으로 코드에 고정하고, 매핑 CSV(`ciciot_attack_map.csv`)도 함께 저장.
- 원본 데이터는 로컬 `D:\...\CICIoT`, 학습은 서버(`/data/SDP_Dataset`)이므로 **전처리 완료된 CSV만 서버로 전송**.

---

## 6. 코드 변경 계획

### 6.1 공통 설정

| 파일 | 변경 내용 |
| :--- | :--- |
| `.gitignore` | `/CICIoT`, `/CICIoT2023` 추가 |
| `utils/arguments.py` | `get_args`, `get_fedsad_args`, `get_gsad_args`의 `--eval_dataset` choices에 `"ciciot"` 추가. `get_fedsad_args`의 `dataset_configs`에 `ciciot` 항목 추가 (`taae_dir=ciciot_rnep`, `gsad_dir=ciciot_graph`, `taae_model=results/CICIoT2023/taae/taae_ciciot_weights.h5`, `gsad_model=results/CICIoT2023/gsad/normal_stats.pkl`, `normal_file=CICIoT_ae_normal.csv`, `anomaly_prefix=CICIoT_anomaly_ae_`) |
| `utils/get_datasets.py` | `get_datasets_ciciot(split="80_20" \| "50_50")` 추가 — `get_datasets_cic_sam`과 동일 로직, 경로만 변경 |

### 6.2 FedSAD / GSAD / RNEP

| 파일 | 변경 내용 |
| :--- | :--- |
| `fedsad.py` | `__main__`의 `dataset_configs`에 `ciciot` 추가. (권장) 중복 정의 대신 `utils.arguments.get_fedsad_args()`의 config를 사용하도록 정리 |
| `gsad.py` | `dataset_file_configs`에 `ciciot` 추가 |
| `scripts/test_gsad.py` | choices·config에 `ciciot` 추가 |
| `rnep.py` | 현재 `get_datasets_cic_sam()` 하드코딩 → `--eval_dataset` 인자로 로더 선택. 결과 경로를 `results/<DS>/rnep/`으로 일반화. (`taae.py:FLClient.evaluate`의 `./FedSAD_Results/client.txt` 하드코딩도 동일하게 정리) |
| `models/FedAVG.py`, `models/CML.py` | 데이터 로더를 `get_datasets_ciciot(split="50_50")`로 교체 가능하게 인자화. 참고: `FedAVG.py`가 import하는 `get_datasets_cic`는 현재 주석 처리되어 있어 **실행 시 ImportError** — 함께 수정 필요 |

### 6.3 베이스라인

| 파일 | 변경 내용 |
| :--- | :--- |
| `scripts/test_taae.py` | `DATASET_CONFIG["CICIoT2023"]` 추가 (base_dir, 파일명, 27종 `attack_map`) |
| `scripts/test_aelstm.py`, `models/AE_LSTM.py` | config + `get_datasets_ciciot_semi()` 추가 |
| `scripts/test_cnnlstm.py`, `models/CNN_LSTM.py` | config + `model_params` 지정. `timesteps × features ≥ 피처 수`로 padding됨 (`reshape_for_sequence`). 예) Track B 39피처 → `(timesteps=5, features=8)`=40, Track A 78피처 → `(10, 8)`=80 (CIC2018 config와 동일하게 맞추면 됨) |
| `models/MMFEWSHOTSIDS/load_data.py` | `DATASET_PRESETS["ciciot"]` 추가 (Track C NetFlow 컬럼 기준) |
| `models/MVIDS/preprocess.py`, `load_data.py` | `CICIOT_VIEW_SCHEMA` 정의 + `VIEW_SCHEMAS["ciciot"]` 등록 |

### 6.4 성능 이슈 (사전 대응)

- **GSAD 완전 그래프 폭증**: `create_gsad_from_window()`는 고유 포트 n개에 대해 `combinations(nodes, 2)`로 n(n−1)/2개 엣지를 실제로 생성합니다. CICIoT2023의 Recon-PortScan / VulnerabilityScan / DDoS 윈도우는 1분에 수천~수만 개 포트가 나올 수 있어 메모리·시간이 폭증합니다.
  - 완전 그래프이므로 결과는 해석적으로 동일하게 계산 가능: `num_nodes=n`, `num_edges=n(n−1)/2`, `density=1 (n>1)`, `avg_degree=n−1`.
  - **결과값은 바뀌지 않으므로** 기존 데이터셋 수치 재현성에 영향 없이 최적화 가능 (적용 전 CIC2018에서 동일 출력 확인 권장).
- 대용량 CSV 로드 시 `dtype` 지정 / `usecols` 사용으로 메모리 절감.

---

## 7. 하이퍼파라미터 재조정 항목

| 파라미터 | 현재값 | 비고 |
| :--- | :--- | :--- |
| `--threshold_std` (GSAD) | 0.25 (CIC), 5 (UNSW) | 데이터셋 의존성이 큼 → CICIoT용 그리드 탐색 필요 (예: 0.25, 0.5, 1, 2, 5) |
| `--percentile` (TAAE threshold) | 82 | 80–95 범위 탐색 |
| `TransformerAAE(beta=0.3)` | CIC용으로 0.3 조정됨 (`taae.py` 주석) | 0.1 / 0.3 / 0.5 비교 |
| `TIME_WINDOW` | `'1T'` | IoT 공격 캡처 길이가 짧은 클래스(Uploading_Attack 등)는 윈도우 수가 매우 적을 수 있음 → 윈도우 수 통계 확인 후 `'30S'` 등 검토. 단, 변경 시 다른 데이터셋과 비교 조건이 달라지므로 **기본값 유지 + 부록 실험** 권장 |
| `--client_nums`, `--client_epochs`, `--server_rounds` | 15 / 30 / 10 | 기존과 동일 유지 (비교 공정성) |

> 하이퍼파라미터 탐색은 **normal 검증 셋**(train normal의 일부)으로만 하고 test anomaly를 보고 조정하지 않도록 주의.

---

## 8. 실행 순서 (체크리스트)

### Phase 0 — 준비
- [ ] `.gitignore`에 `/CICIoT` 추가
- [x] CIC2018 TAAE 입력 피처 목록 확인 — CICFlowMeter 78컬럼 (§4.1)
- [ ] CICFlowMeter 출력 → 78컬럼 rename 매핑 작성
- [ ] 클래스명 정규화·카테고리 매핑 테이블 작성 (§3.1, §3.3)

### Phase 1 — 데이터 생성
- [ ] Track A: PCAP 선별 → CICFlowMeter → 라벨링 → 정제
- [ ] 클래스별 flow 수 집계 → cap 재계산 → 샘플링 (`random_state=123`)
- [ ] `ciciot_rnep/`, `ciciot_graph/` 산출 및 서버 전송
- [ ] (보조) Track B: 공식 CSV 기반 동일 구조 산출
- [ ] 검증: 컬럼 수, NaN/inf 비율, Timestamp 파싱 실패율, 윈도우 수(normal/anomaly)

### Phase 2 — 코드 확장
- [ ] §6.1–6.3 변경 적용 (`--eval_dataset ciciot`)
- [ ] §6.4 GSAD 최적화 (CIC2018 결과 동일성 확인 후)

### Phase 3 — 학습/평가
```bash
python rnep.py --eval_dataset ciciot                        # TAAE 연합학습 → taae_ciciot_weights.h5
python gsad.py --eval_dataset ciciot --ae_data_dir /data/SDP_Dataset/Unified_model/ciciot_graph --result_path results/CICIoT2023/gsad
python fedsad.py --eval_dataset ciciot                      # 통합 평가
python scripts/test_taae.py                                 # (CICIoT2023 config 선택)
python scripts/test_aelstm.py && python scripts/test_cnnlstm.py
```
- [ ] FL 비교: CML / FedAvg / RNEP (normal 50:50)
- [ ] 베이스라인: AE-LSTM, CNN-LSTM, (Track C 확보 시) MM-FewShot-IDS, MV-IDS

### Phase 4 — 정리
- [ ] `results/CICIoT2023/` 결과 저장 (cm, roc, report, weights)
- [ ] `results/TAAE_ROC/CICIoT2023_roc_data.csv` 추가 → `combined_roc_curves.png` 갱신
- [ ] README 데이터셋 표·설명·샘플링 표에 CICIoT2023 추가

---

## 9. 리스크 및 확인 사항

| 리스크 | 영향 | 대응 |
| :--- | :--- | :--- |
| PCAP 548 GB 처리 시간/디스크 | Track A 지연 | 대형 클래스는 pcap 1–2개만 처리, 클래스별 병렬 실행 |
| CIC2018 전처리 스크립트가 레포에 없음 | "동일한 방식" 재현성 저하 | 78컬럼 스키마는 확인됨. CICIoT 전처리 스크립트는 `preprocess/`로 레포에 포함 |
| CICFlowMeter 버전/구현 차이 | 컬럼명·일부 피처 계산 방식 차이 | Java 버전 사용, 컬럼명은 rename 매핑으로 통일 |
| 공식 CSV에 Timestamp/Port 없음 | 공식 CSV만으로는 FedSAD 불가 | Track A 필수 |
| Benign이 4개 pcap(캡처 기간 제한) | GSAD 정상 통계 윈도우 수 부족 가능 | normal 윈도우 수 사전 집계, 부족 시 `TIME_WINDOW` 부록 실험 |
| 공격별 캡처가 독립 세션 | 한 윈도우에 정상+공격이 섞이지 않음 (CIC2018과 동일한 한계) | 논문에 평가 가정으로 명시 |
| 극단적 클래스 불균형 | 비례 샘플링 시 클래스 붕괴 | cap 샘플링 (§4.3) |
| GSAD 완전 그래프 O(n²) | Recon/DDoS 윈도우에서 OOM | 해석적 계산 (§6.4) |
| NetFlow 피처 부재 | MM-FewShot-IDS, MV-IDS 미실행 | Track C(nProbe/NFStream) 또는 CICFlowMeter 컬럼 기반 preset 정의 |

---

## 10. 결정이 필요한 사항

1. **주 트랙 확정**: Track A(PCAP flow) 주 실험 + Track B 보조 — 이 구성으로 갈지.
2. **샘플링 방식**: cap 방식(권장) vs CIC2018 비례 방식 그대로.
3. **MM-FewShot-IDS / MV-IDS 포함 여부**: 포함 시 Track C(NetFlow 추출) 작업 추가.
4. **TIME_WINDOW**: 기본 `'1T'` 고정 여부.

---

## 11. 진행 결과 (2026-10-17)

### 11.1 완료 항목

| 항목 | 결과 |
| :--- | :--- |
| 추출 도구 | Docker 이미지 `cicflowmeter:4.0` (공식 Java CICFlowMeter-4.0 빌드, `preprocess/cicflowmeter/Dockerfile`) |
| 추출 대상 | pcap 37개 / 31.1 GB (Benign 4개 전부 + 공격 클래스별 가장 작은 pcap 1개) → flow CSV 5.9 GB (`CICIoT/FLOW/`) |
| 소요 시간 | pcap을 Docker 볼륨으로 복사 약 3분 + 추출 약 15분. 단, Windows 드라이브를 직접 병렬로 읽으면 수십 배 느려지므로 볼륨에 먼저 복사하는 방식을 사용 |
| 메모리 | DoS-UDP_Flood16, DDoS-SYN_Flood15는 동시 flow가 많아 `-Xmx12g`로 다시 실행 |
| 산출물 | `CICIoT/processed/ciciot_rnep`, `ciciot_graph`, `ciciot_attack_map.csv`, `ciciot_build_stats.csv` |
| 규모 | Normal 200,000 (Benign flow 398,198 중) / Anomaly 200,000, **27종** (cap 8,673~8,674, 이보다 적은 7개 클래스는 전량 사용) |
| NaN/Inf 행 제거 | 클래스별 0~38,735행 (DoS-HTTP_Flood가 가장 많음) |
| 검증 | 78개 컬럼명과 순서가 CIC2018과 같음 · rnep와 graph 행 정렬 일치 · `get_datasets_ciciot()` 로드 OK · GSAD normal 1분 윈도우 1,681개 (2022-10-07 17:15 ~ 10-08 21:15, 28시간), 윈도우당 고유 포트 최대 1,051개 |
| 코드 | `--eval_dataset ciciot` 지원 (`utils/arguments.py`, `fedsad.py`, `gsad.py`, `scripts/test_gsad.py`, `rnep.py`), `get_datasets_ciciot()`·`CICIOT_ATTACK_MAP` 추가, `scripts/test_taae.py`·`test_aelstm.py`·`test_cnnlstm.py`에 `CICIoT2023` config 추가 |

### 11.2 발견된 문제: 폴더 단위 라벨의 오염

CICIoT2023 공격 pcap에는 **공격 중에도 계속 흐르던 IoT 기기의 정상 트래픽이 함께 들어 있습니다.** 폴더명으로 라벨을 붙이면 이 트래픽도 공격으로 라벨링됩니다 (공식 CSV도 같은 방식).

#### CICFlowMeter가 TCP/UDP 이외 공격을 표현하지 못하는 이유 (소스 확인)

`PacketReader.getIpv4Info()`(CICFlowMeter-4.0)는 IPv4 패킷이면 모두 받지만, 포트·플래그·프로토콜 번호는 **TCP 또는 UDP 헤더가 있을 때만** 채웁니다. 그 외 패킷은 `protocol=0`, `srcPort=dstPort=0`, TCP 플래그 없음으로 남고, flow ID가 `srcIP-dstIP-0-0-0`이 됩니다.

| 공격 | 패킷 형태 | CICFlowMeter 결과 |
| :--- | :--- | :--- |
| ICMP Flood / PingSweep | ICMP (포트 없음) | 같은 IP 쌍의 ICMP 수백만 개가 **flow 몇 개로 합쳐짐** |
| ICMP / UDP Fragmentation | 두 번째 조각부터 L4 헤더 없음 | 조각들이 protocol 0 flow로 합쳐짐 |
| Mirai greeth / greip | GRE 캡슐화 (IP proto 47, 안쪽 패킷은 해석하지 않음) | protocol 0 flow로 합쳐짐 |
| MITM-ArpSpoofing | ARP는 IP 패킷이 아님 → 버림. 중계되는 트래픽은 IP 계층에서 정상과 같음 | 공격 흔적이 flow 피처에 거의 없음 |

실측 (추출한 pcap 기준, `protocol 0` flow):

| 클래스 | 전체 flow | protocol 0 flow | protocol 0이 차지하는 패킷 비중 |
| :--- | ---: | ---: | ---: |
| Benign (참고) | 398,330 | 8,242 (2.1%) | 2.5% |
| DDoS-ICMP_Flood | 5,432 | 225 (4.1%) | **99.9%** (2,680만 패킷) |
| DDoS-ICMP_Fragmentation | 10,963 | 356 (3.2%) | **96.4%** |
| Mirai-greeth_flood | 8,451 | 365 (4.3%) | **98.1%** |
| Mirai-greip_flood | 1,592 | 60 (3.8%) | **99.2%** |
| DDoS-UDP_Fragmentation | 10,729 | 354 (3.3%) | 47.4% |
| Recon-PingSweep | 1,950 | 108 (5.5%) | 13.6% |
| MITM-ArpSpoofing | 9,349 | 169 (1.8%) | 1.2% |

→ 공격 패킷은 전부 들어오지만 **수백 개의 거대한 flow로 뭉칩니다.** flow 단위로 샘플링하면 남는 것은 대부분(95% 이상) 같은 pcap 안의 배경 정상 flow입니다. 또 공격 flow도 포트가 0이라 GSAD 그래프(Dst Port 노드)에서는 포트 0 노드 하나로만 보입니다.

클래스별 `Protocol`, `Dst Port` 분포 점검 결과:

| 구분 | 클래스 | 근거 |
| :--- | :--- | :--- |
| 공격 트래픽이 뚜렷함 | DDoS TCP/SYN/PSHACK/RSTFIN/SynonymousIP Flood, DoS TCP/SYN/HTTP Flood, DDoS HTTP Flood, SlowLoris, DDoS/DoS UDP Flood, Recon HostDiscovery/OSScan/PortScan | 특정 포트 집중(예: RSTFIN→554 100%) 또는 flow당 패킷 수가 큼(UDP Flood 중앙값 1,000) |
| **CICFlowMeter가 표현하지 못하는 공격** | DDoS-ICMP_Flood, DDoS-ICMP_Fragmentation, Recon-PingSweep (ICMP), Mirai-greeth_flood, Mirai-greip_flood (GRE), MITM-ArpSpoofing (ARP) | 공격 패킷이 protocol 0 flow 수백 개로 뭉침(위 표) → 샘플된 flow의 대부분은 배경 정상 트래픽, 포트 분포가 normal과 같음(53/443/32100) |
| 섞여 있음 | Mirai-udpplain, DDoS-UDP/ACK_Fragmentation, DNS_Spoofing, VulnerabilityScan, DictionaryBruteForce, Web 계열 6종 | 일부 공격 flow가 있지만 대부분 normal과 비슷한 분포 |

참고: IP는 캡처마다 DHCP로 바뀌어서(benign IP와 겹침이 적음) IP 기준 필터링은 신뢰하기 어렵습니다.

**결정 (2026-10-17): 6개 클래스를 제외하고 27종으로 구성** → §11.3

### 11.3 결정: 6개 공격 클래스 제외 (33종 → 27종)

| 제외 클래스 | 프로토콜 | 제외 이유 |
| :--- | :--- | :--- |
| DDoS-ICMP_Flood | ICMP | 포트가 없어 2,680만 패킷이 protocol 0 flow 225개로 뭉침. 샘플된 flow의 약 96%가 배경 정상 트래픽 |
| DDoS-ICMP_Fragmentation | ICMP 조각 | 두 번째 조각부터 L4 헤더가 없어 protocol 0 flow 356개로 뭉침(패킷의 96.4%) |
| Recon-PingSweep | ICMP | 포트가 없음. pcap 전체 flow 1,950개 중 공격 flow는 108개뿐 |
| Mirai-greeth_flood | GRE | CICFlowMeter가 GRE 안쪽 패킷을 해석하지 않아 protocol 0 flow 365개로 뭉침(패킷의 98.1%) |
| Mirai-greip_flood | GRE | 위와 같음. protocol 0 flow 60개(패킷의 99.2%) |
| MITM-ArpSpoofing | ARP | ARP는 IP 패킷이 아니라서 flow에서 빠짐. 공격자를 거쳐 중계되는 트래픽은 IP 계층에서 정상과 같아 flow 피처에 공격 흔적이 거의 없음 |

**왜 제외했는가**

1. **평가 오염 방지:** 폴더 라벨을 그대로 쓰면 이 클래스에서 뽑힌 flow의 대부분(정상 트래픽)이 "공격"으로 라벨링됩니다. 모델이 정상으로 맞게 판단해도 미탐으로 계산되어, 실제 탐지 능력과 상관없이 recall이 낮아집니다.
2. **"탐지 불가"가 아니라 "표현 불가":** 공격 패킷 자체는 flow에 들어가 있고, 남은 소수의 공격 flow는 패킷 수가 극단적이라 오히려 탐지가 쉬울 수 있습니다. 문제는 CICFlowMeter의 flow 정의(5-tuple: IP·포트·프로토콜)가 포트 없는 프로토콜을 표현하지 못하는 것입니다. MITM만은 flow 피처로 실제 구분이 어렵습니다.
3. **CIC2018 피처 공간 유지:** 같은 78개 CICFlowMeter 피처를 쓰는 것이 "동일한 방식"의 전제입니다. 이 6종을 살리려면 ICMP type, GRE 안쪽 헤더 같은 별도 피처가 필요해 피처 공간이 달라집니다.
4. **대안을 쓰지 않은 이유:** protocol 0 flow만 공격으로 남기면 라벨은 정확해지지만 pcap당 수십~수백 개뿐이라 pcap을 훨씬 많이 처리해야 하고, MITM은 이 방법으로도 해결되지 않습니다.

**논문/README 기재 예시**

> CICIoT2023 contains 33 attack types. We exclude six attacks whose packets carry no TCP/UDP header (ICMP-based: DDoS-ICMP_Flood, DDoS-ICMP_Fragmentation, Recon-PingSweep; GRE-based: Mirai-greeth_flood, Mirai-greip_flood) or are not IP traffic (MITM-ArpSpoofing). CICFlowMeter merges such packets into a small number of port-less flows, so flow records sampled from these captures are dominated by background benign traffic. The remaining 27 attack types are used.

**구현 위치:** `preprocess/build_ciciot_dataset.py`의 `EXCLUDED` (사유 포함). 공격 번호 1–27은 공식 CSV 행 수 내림차순이며 `utils/get_datasets.py:CICIOT_ATTACK_MAP`과 `scripts/test_*.py`의 `attack_map`이 `ciciot_attack_map.csv`와 일치합니다.

**남은 주의 사항:** 27종 중에도 배경 트래픽이 섞인 클래스가 있습니다 (Mirai-udpplain, DDoS-UDP/ACK_Fragmentation, DNS_Spoofing, VulnerabilityScan, DictionaryBruteForce, Web 계열 6종, §11.2 표). 이 클래스들은 공식 데이터셋과 같은 폴더 라벨을 유지하고, 결과 분석에서 클래스별 recall을 따로 보고할 때 이 점을 함께 적는 것을 권장합니다.
