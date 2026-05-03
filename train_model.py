import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("heart.csv")

X = df.drop("condition", axis=1)
y = df["condition"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Strong model
model = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric='logloss'
)

# Stratified 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_accuracy = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
cv_auc = cross_val_score(model, X_scaled, y, cv=skf, scoring='roc_auc')

print("Average Cross Validation Accuracy:", cv_accuracy.mean() * 100)
print("Average ROC-AUC Score:", cv_auc.mean() * 100)

# Train final model on full data
model.fit(X_scaled, y)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model saved successfully!")