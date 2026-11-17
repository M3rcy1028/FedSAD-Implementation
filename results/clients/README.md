# GeNIS client 수 비교

**5·10·15·20·25·30 clients의 학습과 평가가 모두 완료됐다.**
최종 성능 표와 결과 해석은 [프로젝트 README](../../README.md#effect-of-client-count-on-genis)에 정리했다.
전체 수치는 [client_comparison.csv](client_comparison.csv), 요약 표는 [final_comparison_table.md](final_comparison_table.md)에서 확인할 수 있다.

비교 대상은 5, 10, 15, 20, 25, 30 clients다. 원래 실행 순서는 5 → 15 → 20 → 25 → 30이며,
나중에 추가한 10 clients는 기존 실행이 끝나 잠금이 해제된 뒤 이어서 학습했다.
10-client 실행 상태는 `clients_10/status.json`, 대기 및 실행 관리 로그는 `clients_10/queue.log`에 저장했다.
모든 실험은 server 10 rounds,
previous TAAE(latent 64), 로컬 70 epochs, batch 16, learning rate 3e-5,
seed 123, dropout 0.1로 실행한다. 이 값은 기존 게시된 GeNIS TAAE 설정에서 읽었다.
이 기존 비교 연구의 로컬 epoch는 70이며, 새 통합 인터페이스의 기본 epoch 30과 구분된다.

정상 학습 31,123행의 같은 shuffle 순서를 `numpy.array_split`으로 분할한다.
공통 검증 데이터는 7,630행, 테스트 데이터는 8,746행이며 정상:공격이 1:1이다.
데이터와 실행 소스는 `experiment_config.json`에 기록한 SHA256으로 고정했다.
기존 RNEP 집계와 홀수 client의 미짝지어진 client 처리 방식은 유지한다.

## 저장 파일

- `clients_<N>/status.json`: 해당 실험의 현재 상태와 완료 round 수
- `clients_<N>/rounds/round_01/validation_metrics.json`: round별 검증 성능
- `clients_<N>/rounds/round_01/checkpoint.h5`: 같은 round 가중치에 대한 내부 상대경로 링크
- `clients_<N>/round_metrics.csv`, `round_metrics.png`: round 0(학습 전)부터 10까지 검증 성능
- `clients_<N>/GeNIS_*.png`, `GeNIS_*metrics.json`: 완료 후 최종 CM·ROC·분포와 성능
- `clients_<N>/rnep_frame_genis_weights.h5`: 완료 후 최종 가중치
- `clients_<N>/metrics_summary.csv`: 최종 테스트 성능, 고정/검증 선택 임계값을 각각 기록
- `clients_<N>/run/previous/training.log`: 실제 모델 학습 로그
- `client_comparison.csv`, `client_comparison.png`: 완료된 실험들의 최종 테스트 성능 비교
- `round_comparison.csv`, `round_comparison.png`: 실험별 round 검증 성능 비교
- `status.json`, `sweep.log`: 전체 순차 실행 상태와 로그

Round별 평가는 고정 percentile 82의 정상 학습 오차를 임계값으로 사용한다.
최종 그림의 임계값은 기존과 같이 검증 macro F1으로 선택하며,
동일 percentile 82의 최종 테스트 성능도 함께 저장해 비교할 수 있게 했다.
테스트 데이터는 학습, round 평가, 임계값 선택에 사용하지 않는다.

검증 성능과 최종 테스트 성능은 각각 다른 CSV에 기록한다.

GitHub에는 최종 가중치·그림·지표·round별 검증 결과와 완료 검증 기록을 포함한다.
중간 체크포인트, checkpoint 링크, 학습 로그와 중복 실행 파일은 로컬에만 유지한다.

## 완료 상태와 결과 확인

```bash
cat results/clients/status.json
cat results/clients/final_comparison_table.md
```

백그라운드 실행을 관리하는 코드는 `experiment/_legacy/scripts/run_genis_client_sweep.py`다.
실험이 실행 중일 때 중복으로 시작하면 파일 잠금으로 거부한다.
추가 실험용 `genis_clients_10` 세션은 `--wait-lock`으로 기존 작업이 끝나기를 기다린 뒤 10 clients를 실행했다.
완료된 기존 client 수는 다시 학습하지 않았으며, 10-client 결과도 같은 비교 CSV와 PNG에 합쳤다.
실행이 종료된 뒤 기존 결과 표·그림만 다시 수집하려면 다음 명령을 사용한다.

```bash
python experiment/_legacy/scripts/run_genis_client_sweep.py --collect-only
```
