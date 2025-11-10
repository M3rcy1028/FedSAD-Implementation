import seaborn as sns
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

dataset_cms = {
    "InSDN": [
        [11173, 2512],  # Actual Normal (TN, FP)
        [13, 275452]    # Actual Anomaly (FN, TP)
    ],
    "NSL-KDD": [
        [29236, 1485],  # Actual Normal (TN, FP)
        [2774, 57353]   # Actual Anomaly (FN, TP)
    ],
    "KDD99": [
        [14626, 737],   # Actual Normal (TN, FP)
        [1740, 111512]  # Actual Anomaly (FN, TP)
    ],
    "CSE-CIC-IDS2018": [
        [32882, 7118],  # Actual Normal (TN, FP)
        [22789, 177211] # Actual Anomaly (FN, TP)
    ],
    # 'cm_sdn'은 'InSDN'과 동일한 데이터셋으로 보이므로 하나만 사용합니다.
    "UNSW-NB15": [
        [352787, 39168], # Actual Normal (TN, FP)
        [180, 99463]    # Actual Anomaly (FN, TP)
    ]
}

def plot_specific_cm(cm_data, labels, title, save_path):
    """
    주어진 데이터를 사용하여 'evaluate.py' 스타일의
    Confusion Matrix를 그리고 저장합니다.
    """
    
    # 1. 캔버스 설정
    plt.figure(figsize=(8, 6))
    
    # 2. Seaborn 히트맵 생성
    # annot=True: 각 셀에 숫자 표시
    # fmt='d': 숫자를 정수(decimal) 형태로 표시
    # cmap='Blues': 'evaluate.py'와 동일한 파란색 계열 색상맵
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_data,
        annot=True,
        fmt='d',  
        cmap='Blues',
        xticklabels=[f'Predicted {labels[0]}', f'Predicted {labels[1]}'],
        yticklabels=[f'Actual {labels[0]}', f'Actual {labels[1]}']
    )
    
    # 3. 라벨 및 타이틀 설정
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    
    # 4. 레이아웃 최적화 및 저장
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"✅ Confusion Matrix가 '{save_path}'에 저장되었습니다.")
    print("데이터:")
    print(np.array(cm_data))

DATA_FILES_TO_PLOT = {
    "InSDN": "./Results/InSDN_roc_data.csv",
    "KDD99": "./Results/KDD99_roc_data.csv",
    "UNSW_NB15": "./Results/UNSW_NB15_roc_data.csv",
    # "CSE-CIC-IDS2018": "./Results/CSE-CIC-IDS2018_roc_data.csv"
    # NSL-KDD
}

# 저장할 최종 이미지 파일 이름
SAVE_PATH = "all_datasets_roc_curves_combined.png"
# -----------------


def create_combined_roc_plot():
    """
    지정된 _roc_data.csv 파일들을 읽어 하나의 ROC 차트에 그립니다.
    """
    
    # 1. 캔버스 설정
    plt.figure(figsize=(10, 8))
    
    # 2. "무작위 추측" 기준선 (대각선)
    # (사용자님이 첨부한 InSDN 이미지 스타일 참조)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='No Skill (AUC = 0.5)')

    # 3. 각 데이터셋의 ROC 곡선 계산 및 플롯
    # matplotlib의 기본 컬러 사이클 사용
    colors = plt.cm.get_cmap('tab10', len(DATA_FILES_TO_PLOT))

    found_files = 0
    
    for i, (dataname, file_path) in enumerate(DATA_FILES_TO_PLOT.items()):
        
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: '{file_path}' 파일을 찾을 수 없습니다. 건너뜁니다.")
            continue
            
        print(f"Loading '{file_path}'...")
        
        try:
            # CSV 파일 로드
            df = pd.read_csv(file_path)
            y_true = df['y_true']
            y_scores = df['anomaly_score']
            
            # ROC 커브 계산
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            # 차트에 플롯
            plt.plot(fpr, tpr, color=colors(i), lw=2, 
                     label=f'{dataname} (AUC = {roc_auc:.4f})')
            
            found_files += 1
            
        except Exception as e:
            print(f"❌ Error processing '{file_path}': {e}")

    if found_files == 0:
        print("❌ Error: 플롯할 ROC 데이터를 하나도 찾지 못했습니다. 파일 경로를 확인하세요.")
        plt.close()
        return

    # 4. 차트 설정
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curves Comparison by Dataset')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    # 5. 저장
    plt.savefig(SAVE_PATH, dpi=300)
    plt.close()

    print(f"\n✅ 통합 ROC 곡선 차트가 '{SAVE_PATH}'에 저장되었습니다.")

dataname = "NSL-KDD"
cm_data_to_plot =  [
                [11899, 390],  # Actual Normal (TN, FP)
                [3451, 56676]    # Actual Anomaly (FN, TP)
            ]

# --- 실행 ---
if __name__ == "__main__":
    
    plot_specific_cm(cm_data_to_plot, ['Normal', 'Anomaly'], 
                    f"Confusion Matrix ({dataname})", 
                    f"cm_{dataname}.png")
    # create_combined_roc_plot()
