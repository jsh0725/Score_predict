import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 모듈 import
from utils.preprocessing import (
    load_and_split_by_season,
    encode_number,
    encode_chracter,
    process_date_features,
    remove_and_missing,
    split_features
)
from datasets.datasets import create_dataloaders
from models.MLP_model import MLP

def preprocess_pipeline(df: pd.DataFrame) -> tuple:
    df = encode_number(df)
    df = encode_chracter(df)
    df = process_date_features(df)
    df = remove_and_missing(df)
    X, y = split_features(df)
    return X, y

def evaluate_model(model, dataloader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            pred = torch.argmax(outputs, dim=1)

            preds.extend(pred.cpu().numpy())
            targets.extend(y_batch.numpy())

    return accuracy_score(targets, preds)

def main():
    # 설정
    csv_path = "data/matches_serie_A.csv"
    train_seasons = [2020, 2021, 2022]
    test_seasons = [2023, 2024]
    batch_size = 64
    epochs = 30
    learning_rate = 1e-3
    hidden_dim = 256
    output_dim = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 데이터 로딩 및 전처리
    train_df, test_df = load_and_split_by_season(csv_path, train_seasons, test_seasons)
    X_train, y_train = preprocess_pipeline(train_df)
    X_test, y_test = preprocess_pipeline(test_df)

    #정규화(StandardScaler)
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

    # DataLoader 생성
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, batch_size=batch_size)

    # 모델 초기화
    input_dim = X_train.shape[1]
    model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=0.3).to(device)

    # 손실 함수 & 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 학습 루프
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        test_acc = evaluate_model(model, test_loader, device)

        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Test Accuracy: {test_acc:.4f}")

        # Best 모델 저장
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_model.pth")
            joblib.dump(scaler, "scaler.pkl")
            print("→ Best model saved.")

    print(f"\n[FINAL] Best Test Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()


