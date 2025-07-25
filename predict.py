import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 모듈 import
from utils.preprocessing import (
    encode_chracter,
    process_date_features,
    remove_and_missing,
    split_features
)
from models.MLP_model import MLP

# 예측 결과 숫자 → 문자 변환
label_map_rev = {0: 'L', 1: 'D', 2: 'W'}

def preprocess_test_data(df):
    df = encode_chracter(df)
    df = process_date_features(df)
    df = remove_and_missing(df)
    drop_cols = ["Unnamed: 0", "Date", "Time", "Captain", "Formation", "Opp Formation", "Referee", "Match Report", "Notes", "Comp", "Opponent", "Team"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    return df

def predict(model_path, scaler_path, input_csv, output_csv, hidden_dim=256, dropout=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 1. 데이터 불러오기 및 전처리
    df = pd.read_csv(input_csv)
    raw_df = df.copy()
    X = preprocess_test_data(df)

    # 2. 정규화 (학습 시 저장한 scaler 사용)
    scaler = joblib.load(scaler_path)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)

    # 3. 텐서 변환
    X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32).to(device)

    # 4. 모델 로드
    input_dim = X_tensor.shape[1]
    output_dim = 3
    model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 5. 예측
    with torch.no_grad():
        outputs = model(X_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    # 6. 예측 결과 문자로 변환 및 저장
    raw_df['Predicted_Label'] = preds
    raw_df['Predicted_Result'] = raw_df['Predicted_Label'].map(label_map_rev)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    raw_df.to_csv(output_csv, index=False)
    print(f"[INFO] Prediction saved to: {output_csv}")

    return raw_df

if __name__ == "__main__":
    result_df = predict(
        model_path="best_model.pth",
        scaler_path="scaler.pkl",
        input_csv="data/test_sample.csv",
        output_csv="results/predicted_results.csv",
        hidden_dim=256
    )
