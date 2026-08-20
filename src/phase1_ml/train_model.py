# train_model.py
# Purpose: Train and compare RF and XGBoost site classifiers, explain with SHAP.
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

FEATURES = ['dist_road_m', 'dist_hospital_m', 'flood_risk']
TARGET = 'label'

def train_and_evaluate():
    df = pd.read_csv('data/processed/labelled_sites.csv')
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f'Training on {len(X_train)} sites, testing on {len(X_test)}')

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_score = rf.score(X_test, y_test)
    print(f'Random Forest Accuracy: {rf_score:.3f}')
    print(classification_report(y_test, rf.predict(X_test)))

    xg = xgb.XGBClassifier(n_estimators=100, random_state=42,
                           eval_metric='logloss')
    xg.fit(X_train, y_train)
    xg_score = xg.score(X_test, y_test)
    print(f'XGBoost Accuracy: {xg_score:.3f}')

    best = rf if rf_score >= xg_score else xg
    print(f'Best model: {"RandomForest" if best is rf else "XGBoost"}')

    explainer = shap.TreeExplainer(best)
    shap_vals = explainer.shap_values(X_test)
    # SHAP >= 0.45 returns a 3-D array (samples, features, classes) for
    # sklearn binary classifiers; older versions returned a list of two
    # arrays. Reduce to the positive-class matrix in either case.
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    elif getattr(shap_vals, 'ndim', 2) == 3:
        shap_vals = shap_vals[:, :, 1]
    shap.summary_plot(shap_vals, X_test, feature_names=FEATURES, show=False)
    plt.savefig('outputs/reports/shap_importance.png', bbox_inches='tight')
    plt.close()
    print('SHAP plot saved to outputs/reports/shap_importance.png')

    os.makedirs('models/saved', exist_ok=True)
    joblib.dump(best, 'models/saved/site_scorer_model.pkl')
    print('Model saved to models/saved/site_scorer_model.pkl')
    return best

if __name__ == '__main__':
    train_and_evaluate()