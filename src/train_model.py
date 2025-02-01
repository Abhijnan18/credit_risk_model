import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier  # CatBoost
import shap
import joblib
import os

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Load processed data
X_train = pd.read_csv('data/processed/train_test_data/X_train.csv')
y_train = pd.read_csv('data/processed/train_test_data/y_train.csv')
X_test = pd.read_csv('data/processed/train_test_data/X_test.csv')
y_test = pd.read_csv('data/processed/train_test_data/y_test.csv')

# Train Logistic Regression (optional, for comparison)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
joblib.dump(lr, 'models/logistic_regression.pkl')

# Train CatBoost
cat_model = CatBoostClassifier(
    random_state=42,  # For reproducibility
    verbose=0,        # Disable training logs
    iterations=100,   # Number of boosting iterations
    learning_rate=0.1,  # Learning rate
    depth=6           # Depth of the trees
)
cat_model.fit(X_train, y_train)
joblib.dump(cat_model, 'models/catboost_model.pkl')  # Save CatBoost model

# Evaluate AUC-ROC
# Predicted probabilities for class 1
y_pred_cat = cat_model.predict_proba(X_test)[:, 1]
auc_roc = roc_auc_score(y_test, y_pred_cat)
print(f"AUC-ROC (CatBoost): {auc_roc:.2f}")

# SHAP explainability
explainer = shap.TreeExplainer(cat_model)
shap_values = explainer.shap_values(X_test)

# SHAP summary plot
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('reports/shap_summary.png', dpi=300)
plt.close()

# Feature importance plot (CatBoost built-in)
feature_importance = cat_model.get_feature_importance()
plt.figure(figsize=(10, 6))
plt.barh(X_train.columns, feature_importance)
plt.title("CatBoost Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig('reports/catboost_feature_importance.png', dpi=300)
plt.close()
