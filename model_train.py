import pandas as pd 
import numpy as np 
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

import xgboost as xgb



# Data preparation

df = pd.read_csv('/home/timur/work_hub/ML_zoomcamp_projects/HR_turnover_prediciton/employee_churn_data.csv')
df['left'] = [1 if value == 'yes' else 0 for value in df['left']]


categorical_features = [
    'department',
    'salary'
]
numerical_features = [
    'promoted',
    'review',
    'projects',
    'tenure',
    'satisfaction',
    'bonus',
    'avg_hrs_month'
]

X = df[categorical_features+numerical_features]
y = df['left']

# Train/test split
df_train, df_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# DictVectorizer

train_dict = df_train.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dict)

test_dict = df_test.to_dict(orient='records')
X_test = dv.transform(test_dict)

# Training model
xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

params = {'colsample_bytree': 1.0,
 'learning_rate': 0.1,
 'max_depth': 7,
 'n_estimators': 50,
 'subsample': 0.8}

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    colsample_bytree=params['colsample_bytree'],
    learning_rate=params['learning_rate'],
    max_depth=params['max_depth'],
    n_estimators=params['n_estimators'],
    subsample=params['subsample']
)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict_proba(X_test)[:, 1]

roc_auc_xbg = roc_auc_score(y_test, y_pred)

# Save the model
output_file = "xgb_model.model"
xgb_model.save_model(output_file)

# Save dv
dv_output_file = "dv.pkl"
joblib.dump(dv, dv_output_file)