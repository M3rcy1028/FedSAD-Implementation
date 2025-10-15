import kagglehub
import pandas as pd
import os
import glob

# Kaggle 데이터셋 다운로드
path = kagglehub.dataset_download("badcodebuilder/insdn-dataset")
print("Downloaded path:", path)

# 서브폴더 내용 확인
subfolders = os.listdir(path)
print("Files/Folders in dataset folder:", subfolders)

# InSDN_DatasetCSV 폴더로 진입
csv_folder = os.path.join(path, "InSDN_DatasetCSV")

# 해당 폴더 안의 CSV 파일 찾기
csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV file found in {csv_folder}")
csv_path = csv_files[0]
print("Reading CSV file:", csv_path)

# CSV 파일 읽기
df = pd.read_csv(csv_path)
print("✅ Data loaded successfully!")
print("Shape:", df.shape)
print(df.head())
