import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()
X, y = data.data, data.target

models = {
    "Tək ağac": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean():.3f} (±{scores.std():.3f})")

# Xüsusiyyətlərin əhəmiyyəti
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
top = np.argsort(importances)[::-1][:5]

print("\nƏn vacib 5 xüsusiyyət:")
for i in top:
    print(f"{data.feature_names[i]}: {importances[i]:.3f}")