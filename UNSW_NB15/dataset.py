import pandas as pd
import os

# 0. 생성할 디렉터리 이름
output_dir = 'ae_datas'

# 1. 'ae_datas' 디렉터리 생성 (이미 존재하면 무시)
os.makedirs(output_dir, exist_ok=True)
print(f"'{output_dir}' 디렉터리를 준비했습니다.")

# 2. CSV 파일 이름 정의
benign_file = 'UNSW_NB15_benign.csv'
attack_file = 'UNSW_NB15_attack.csv'

# 3. 제거할 컬럼 목록
cols_to_drop = ['srcip', 'dstip', 'attack_cat', 'label']

# 4. Benign 파일 처리
try:
    # CSV 파일에 헤더가 있으므로 header=0 (기본값) 사용
    df_benign = pd.read_csv(benign_file)
    
    # 불필요한 열 제거
    df_benign_processed = df_benign.drop(columns=cols_to_drop, errors='ignore')
    
    # 파일로 저장
    benign_output_path = os.path.join(output_dir, 'UNSW_NB15_normal.csv')
    df_benign_processed.to_csv(benign_output_path, index=False)
    
    # [수정됨] 저장된 파일의 데이터 개수 출력
    print(f"'{benign_output_path}' 파일 저장 완료. (데이터 개수: {len(df_benign_processed)})")

except FileNotFoundError:
    print(f"오류: '{benign_file}' 파일을 찾을 수 없습니다. 건너뜁니다.")
except Exception as e:
    print(f"Benign 파일 처리 중 오류 발생: {e}")

print("-" * 30)

# 5. Attack 파일 처리
try:
    # CSV 파일에 헤더가 있으므로 header=0 (기본값) 사용
    df_attack = pd.read_csv(attack_file)
    
    # 5-1. 통합 Anomaly 파일 생성
    df_anomaly_combined = df_attack.drop(columns=cols_to_drop, errors='ignore')
    
    # 파일로 저장
    combined_output_path = os.path.join(output_dir, 'UNSW_NB15_anomaly.csv')
    df_anomaly_combined.to_csv(combined_output_path, index=False)
    
    # [수정됨] 저장된 파일의 데이터 개수 출력
    print(f"'{combined_output_path}' 파일 저장 완료. (데이터 개수: {len(df_anomaly_combined)})")
    
    # 5-2. 공격 유형별 분리 및 인코딩
    unique_attacks = df_attack['attack_cat'].dropna().unique()
    attack_types = sorted([str(att).strip() for att in unique_attacks])
    attack_encoding = {attack_name: i for i, attack_name in enumerate(attack_types)}
    
    print("\n--- ⚔️ 공격 유형 인코딩 정보 ---")
    for name, code in attack_encoding.items():
        print(f"  {name}: {code}")
    print("---------------------------------")
    
    # 5-3. 각 공격 유형별로 파일 저장
    for attack_name, encoding_id in attack_encoding.items():
        # 해당 공격 유형만 필터링
        df_specific_attack = df_attack[df_attack['attack_cat'].str.strip() == attack_name]
        
        # 불필요한 열 제거
        df_specific_processed = df_specific_attack.drop(columns=cols_to_drop, errors='ignore')
        
        # 파일 이름 정의 및 저장
        specific_output_filename = f'UNSW_NB15_anomaly_{encoding_id}.csv'
        specific_output_path = os.path.join(output_dir, specific_output_filename)
        
        df_specific_processed.to_csv(specific_output_path, index=False)
        
        # [수정됨] 저장된 파일의 데이터 개수 출력
        print(f"'{attack_name}'(Code {encoding_id}) -> '{specific_output_path}' 저장 완료. (데이터 개수: {len(df_specific_processed)})")

except FileNotFoundError:
    print(f"오류: '{attack_file}' 파일을 찾을 수 없습니다. 처리를 중단합니다.")
except KeyError as e:
    print(f"오류: '{e}' 컬럼을 찾을 수 없습니다. 원본 CSV 파일의 컬럼명을 확인해주세요.")
except Exception as e:
    print(f"Attack 파일 처리 중 오류 발생: {e}")

print("\n모든 작업이 완료되었습니다.")