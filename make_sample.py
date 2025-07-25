import pandas as pd
import os

# 원본 데이터 경로
input_csv = "data/matches_serie_A.csv"
# 예측용 데이터 저장 경로
output_csv = "data/test_sample.csv"

# CSV 로드
df = pd.read_csv(input_csv)

# 2025년 데이터만 추출
df_2025 = df[df['Season'] == 2025].copy()

# 실제 결과는 제거
if 'Result' in df_2025.columns:
    df_2025.drop(columns=['Result'], inplace=True)

# 저장
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df_2025.to_csv(output_csv, index=False)

print(f"[INFO] 2025년 예측용 데이터가 '{output_csv}'에 저장되었습니다.")

#마지막에 예측된 결과와 실제 결과를 비교하기 위해 25년도 자료만 따로 저장
df = pd.read_csv("data/matches_serie_A.csv")
df_2025 = df[df['Season'] == 2025].reset_index(drop=True)
df_2025.to_csv("data/true_2025.csv", index=False)