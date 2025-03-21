#!/usr/bin/env python3

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import os
import pandas as pd
import joblib

import datetime

with open("/home/poc/main/IGA/logs/cron_log.txt", "a") as f:
    f.write(f"Script ran at {datetime.datetime.now()}\n")





# Define file paths
auth_log_path = '/home/poc/main/IGA/data/authentication_log.csv'
operation_log_path = '/home/poc/main/IGA/data/operation_log.csv'
org_role_path = '/home/poc/main/IGA/data/org_role.csv'
system_role_path = '/home/poc/main/IGA/data/system_role.csv'
organization_path = '/home/poc/main/IGA/data/orgnazation.csv'

# Load the CSV files into pandas DataFrames
auth_log_df = pd.read_csv(auth_log_path)
operation_log_df = pd.read_csv(operation_log_path)
org_role_df = pd.read_csv(org_role_path)
system_role_df = pd.read_csv(system_role_path)
organization_df = pd.read_csv(organization_path)

# Define file paths for the anomaly text files
auth_log_anomaly_path = '/home/poc/main/IGA/data/authentication_log_anomaly.txt'
operation_log_anomaly_path = '/home/poc/main/IGA/data/operation_log_anomaly.txt'

# Read the text files into pandas DataFrames
auth_log_anomaly_df = pd.read_csv(auth_log_anomaly_path)
operation_log_anomaly_df = pd.read_csv(operation_log_anomaly_path)

# Merge authentication_log.csv and authentication_log_anomaly.txt
auth_anomaly_ids = [int(num.strip("[] ")) for num in auth_log_anomaly_df.columns]
auth_log_df['is_anomaly'] = 0
auth_log_df.loc[auth_log_df['id'].isin(auth_anomaly_ids), 'is_anomaly'] = 1

# Merge operation_log.csv and operation_log_anomaly.txt
operation_anomaly_ids = [int(num.strip("[] ")) for num in operation_log_anomaly_df.columns]
operation_log_df['is_anomaly'] = 0
operation_log_df.loc[operation_log_df['id'].isin(operation_anomaly_ids), 'is_anomaly'] = 1
import pandas as pd
import numpy as np

# 假日列表區
holidays = pd.to_datetime(['2024-01-01', '2024-02-28', '2024-04-04', '2024-05-01', '2024-06-25', '2024-09-28', '2024-10-10'])

def identify_day_type(date):
    if date in holidays:
        return 'holiday'
    elif date.weekday() >= 5:  # 5 表示星期六，6 表示星期天
        return 'weekend'
    else:
        return 'weekday'
    
    # 特徵工程
# 1. 將類別型特徵轉換為數值
auth_log_df_ex = pd.DataFrame()
le = LabelEncoder()

auth_log_df_ex['user_action'] = auth_log_df['user_account'].astype(str) + '_' + auth_log_df['action'].astype(str)
auth_log_df_ex['user_ip'] = auth_log_df['user_account'].astype(str) + '_' + auth_log_df['IP'].astype(str)
# 對新的組合進行特徵編碼
auth_log_df_ex['user_action'] = le.fit_transform(auth_log_df_ex['user_action'])
auth_log_df_ex['user_ip'] = le.fit_transform(auth_log_df_ex['user_ip'])

auth_log_df_ex['user_account'] = le.fit_transform(auth_log_df['user_account'])
auth_log_df_ex['role_id'] = le.fit_transform(auth_log_df['role_id'])
auth_log_df_ex['user_id'] = le.fit_transform(auth_log_df['user_id'])
auth_log_df_ex['action'] = le.fit_transform(auth_log_df['action'])
auth_log_df_ex['description'] = le.fit_transform(auth_log_df['description'])
auth_log_df_ex['IP'] = le.fit_transform(auth_log_df['IP'])

# 2. 從 timestamp 中提取特徵 The day of the week with Monday=0, Sunday=6.
auth_log_df_ex['hour'] = pd.to_datetime(auth_log_df['timestamp']).dt.hour
auth_log_df_ex['minute'] = pd.to_datetime(auth_log_df['timestamp']).dt.minute
auth_log_df_ex['second'] = pd.to_datetime(auth_log_df['timestamp']).dt.second
auth_log_df_ex['dayofweek'] = pd.to_datetime(auth_log_df['timestamp']).dt.dayofweek
auth_log_df_ex['day_type'] = pd.to_datetime(auth_log_df['timestamp']).map(identify_day_type)
auth_log_df_ex['day_type'] = le.fit_transform(auth_log_df_ex['day_type'])

def detect_outliers(df, features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[c], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[c], 75)
        # IQR
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature c
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        
        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = list(set([x for x in outlier_indices if outlier_indices.count(x) > 2]))
    return outlier_indices

# 檢查 'hour', 'minute' 和 'second' 的異常值
outliers_to_remove = detect_outliers(auth_log_df_ex, ['hour', 'minute', 'second'])
auth_log_df_ex = auth_log_df_ex.drop(outliers_to_remove, axis=0)

# 3. 填補缺失值 (若有)
auth_log_df_ex = auth_log_df_ex.fillna(0)  # 或其他適當的填補方式

# 4. 取出 X and Y
X = auth_log_df_ex
Y = auth_log_df['is_anomaly'].values

# 選擇特徵和目標
features = X[['user_action', 'user_ip', 'user_account', 'IP', 'user_id', 'role_id', 'action', 'hour', 'minute', 'second', 'dayofweek', 'day_type']]
target = Y

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 分類特徵的處理
numeric_features = ['hour', 'minute', 'second']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_features = ['user_action', 'user_ip', 'user_account', 'IP', 'user_id', 'role_id', 'action', 'dayofweek', 'day_type']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 建立處理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 分割訓練集和測試集
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.3, random_state=42)

from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier())
])
param_grid = {
    'classifier__max_depth': [3, 6, 10],
    'classifier__min_child_weight': [1, 5, 10],
    'classifier__gamma': [0.5, 1, 1.5, 2],
    'classifier__subsample': [0.6, 0.8, 1.0],
    'classifier__colsample_bytree': [0.6, 0.8, 1.0],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__n_estimators': [100, 200, 300],
    'classifier__scale_pos_weight': [sum(Y_train == 0) / sum(Y_train == 1)]
}

optimal_params = {
    'max_depth': 14,
    'min_child_weight': 10,
    'gamma': 1,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'learning_rate': 0.2,
    'n_estimators': 300,
    'scale_pos_weight': 21.580645161290324,
    'objective': 'binary:logistic'  # Assuming a binary classification problem
}

# model = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', XGBClassifier(**optimal_params, n_jobs=-1))
# ])
MODEL_PATH = '/home/poc/main/IGA/models/clf_model_update.pkl'

# 如果模型存在，就讀進來並繼續訓練；否則建立新模型
if os.path.exists(MODEL_PATH):
    print("Incremental training....")
    clf = joblib.load(MODEL_PATH)
    clf.fit(X_train, Y_train, xgb_model=clf.get_booster())  # <-- 增量訓練
else:
    print("First time training....")
    clf = XGBClassifier(**optimal_params, n_jobs=-1)
    clf.fit(X_train, Y_train)

# clf = XGBClassifier(**optimal_params, n_jobs=-1)
# clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

joblib.dump(clf, '/home/poc/main/IGA/models/clf_model_update.pkl')
print("`.pkl` 存檔測試成功！")


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import plotly.figure_factory as ff
from sklearn.metrics import classification_report
# 計算正確率
accuracy = accuracy_score(Y_test, Y_pred)
print(f"RandomForest 的正確率: {accuracy*100:.2f}%")
print("模型已更新！")
cm = confusion_matrix(Y_test, Y_pred)
cm_labels = [0, 1]

# Create the confusion matrix heatmap
fig = ff.create_annotated_heatmap(z=cm, x=cm_labels, y=cm_labels, colorscale='Blues')

# Update layout for better readability
fig.update_layout(
    title='Confusion Matrix',
    xaxis=dict(title='Predicted Label'),
    yaxis=dict(title='True Label')
)

probabilities = clf.predict_proba(X_test)

import numpy as np
indices_FN = np.where((Y_pred == 0) & (Y_test == 1))[0]
indices_TN = np.where((Y_pred == 1) & (Y_test == 1))[0]

# Extract the corresponding samples
y_true_selected_FN = X_test.iloc[indices_FN]
y_true_selected_TN = X_test.iloc[indices_TN]

FN = auth_log_df.loc[y_true_selected_FN.index]
TN = auth_log_df.loc[y_true_selected_TN.index]

import plotly.express as px

# 分析密碼錯誤的登入失敗
password_errors = FN[FN['description'] == 'wrong password']

# 查看不同IP地址的登入嘗試次數
ip_attempts = FN['IP'].value_counts().reset_index()
ip_attempts.columns = ['IP', 'Counts']

# 查看不同角色的登入嘗試次數
role_attempts = FN['role_id'].value_counts().reset_index()
role_attempts.columns = ['Role_ID', 'Counts']

# 登入嘗試的時間分佈
FN['Hour'] = pd.to_datetime(FN['timestamp']).dt.hour
time_analysis = FN['Hour'].value_counts().sort_index().reset_index()
time_analysis.columns = ['Hour', 'Counts']