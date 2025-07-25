#predict.py 실행 결과로 나온 예측 결과인 predicted_results.csv와 25년도의 실제 경기 결과가 들어 있는 true_2025.csv를 비교해 예측 정확도 평가
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# 1. 파일 로드
true_df = pd.read_csv("data/true_2025.csv")
pred_df = pd.read_csv("results/predicted_results.csv")

# 2. 두 값 모두 문자열로 강제 변환
true_labels = true_df['Result'].astype(str).values
pred_labels = pred_df['Predicted_Result'].astype(str).values

# 3. 정확도 및 평가 리포트
accuracy = accuracy_score(true_labels, pred_labels)
report = classification_report(true_labels, pred_labels, target_names=['L', 'D', 'W'])

print(f"\n[Evaluation on 2025 Season Matches]")
print(f"Accuracy: {accuracy:.4f}")
print(report)
