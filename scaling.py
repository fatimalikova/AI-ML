from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)

models = {
    "KNN (miqyassız)": KNeighborsClassifier(n_neighbors=5),
    "KNN + StandardScaler": make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=5)
    ),
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean():.3f} (±{scores.std():.3f})")

print("\nXüsusiyyətlərin orta qiymətləri (ilk 5):")
for i in range(5):
    print(f"sütun {i}: {X[:, i].mean():.3f}")