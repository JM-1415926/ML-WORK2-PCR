import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load
import xgboost as xgb

# 文件路径
file_path = "TrainDataset2024_preprocessed.xlsx"  # 已处理的测试数据集路径

# 加载测试数据
data = pd.read_excel(file_path, engine='openpyxl')

# 提取 ID、特征和真实标签
IDs = data['ID']
X_test = data.drop(columns=['ID', 'pCR (outcome)'], errors='ignore')  # 假设 pCR (outcome) 是真实标签列
y_true = data['pCR (outcome)']

# 加载保存的模型和处理器
model = xgb.Booster()
model.load_model("Model-xgboost.json")  # 使用的训练好的 xgboost 模型

# 模型预测
dtest = xgb.DMatrix(X_test)
predictions = model.predict(X_test)

# 混淆矩阵
cm = confusion_matrix(y_true, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 分类报告
report = classification_report(y_true, predictions, target_names=['Class 0', 'Class 1'])
print("Classification Report:")
print(report)

# 平衡分类准确率
balanced_accuracy = balanced_accuracy_score(y_true, predictions)
print(f"Balanced Classification Accuracy: {balanced_accuracy:.2f}")


