from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)

# Hamısı üçün eyni bölgü
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)
    ),
    "SVM": make_pipeline(StandardScaler(), SVC()),
}

print(f"{'Model':<20} | {'orta':>5} | {'±':>5}")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"{name:<20} | {scores.mean():.3f} | {scores.std():.3f}")