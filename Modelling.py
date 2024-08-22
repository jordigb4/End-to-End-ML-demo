######################################## Import Libraries #######################################################

import pandas as pd
import os
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, recall_score,roc_auc_score,precision_score,f1_score,roc_curve,auc
from catboost import CatBoostClassifier, Pool

######################################## Clean Data #######################################################

DATA_DIR = os.path.join('data','heart_attack_dataset.csv')
df = pd.read_csv(DATA_DIR)
df = df.drop(columns=['Patient ID'])
df[['systolic', 'diastolic']] = df['Blood Pressure'].str.split('/', expand=True)
df = df.drop(columns=['Blood Pressure'])
df.columns = ['age', 'sex','cholesterol','heart_rate','diabetes','fam_hist','smoking','obesity','alcohol','exersice'
              , 'diet', 'heart_problems', 'medication','stress','sedentary_hours','income','BMI','triglycerides',
              'physical_activity_days','sleep','country','continent','hemisphere','risk','systolic','diastolic']
num_cols = {'age', 'cholesterol', 'heart_rate', 'exersice', 'sedentary_hours', 'income', 'BMI', 'triglycerides',
            'physical_activity_days', 'sleep', 'systolic', 'diastolic'}
cat_cols = set(df.columns) - num_cols
num_cols = list(num_cols)
cat_cols = list(cat_cols)
df[cat_cols] = df[cat_cols].astype('category')
df[num_cols] = df[num_cols].apply(pd.to_numeric)

cat_cols = list(set(cat_cols)- set(['risk']))

################################################## StratifiedShuffleSplit ############################################
strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=32)

train_index, test_index = next(strat_split.split(df, df["risk"]))

# Create train and test sets
strat_train_set = df.loc[train_index]
strat_test_set = df.loc[test_index]

X_train = strat_train_set.drop("risk", axis=1)
y_train = strat_train_set["risk"].copy()

X_test = strat_test_set.drop("risk", axis=1)
y_test = strat_test_set["risk"].copy()

################################################## CATBOOST ##################################################

cat = CatBoostClassifier(verbose=False, auto_class_weights= 'Balanced',random_state=32)
cat.fit(X_train,y_train,cat_features = cat_cols)
y_pred = cat.predict(X_test)
accuracy, recall, roc_auc, precision = [round(metric(y_test, y_pred), 4) for metric in [accuracy_score,
                                                                                        recall_score, roc_auc_score,
                                                                                        precision_score]]
model_names = ['CatBoost_Model']
result = pd.DataFrame({'Accuracy': accuracy, 'Recall': recall, 'Roc_Auc': roc_auc, 'Precision': precision},
                      index=model_names)
print(result)

# Save model

MODEL_DIR = os.path.join('models')

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

MODEL_PATH = os.path.join(MODEL_DIR,'catboost_model.cbm')
cat.save_model(MODEL_PATH)