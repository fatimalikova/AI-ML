from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = fetch_openml("titanic", version=1, as_frame=True).frame

num_cols = ["pclass", "age", "sibsp", "parch", "fare"]
cat_cols = ["sex", "embarked"]

X = df[num_cols + cat_cols].copy()
X[cat_cols] = X[cat_cols].astype(object)
y = df["survived"].astype(int)

print("Boş qiymətlər:")
print(X.isna().sum())
print("\nSağ qalma nisbəti:", round(y.mean(), 3))

# Rəqəmlər: boşları median ilə doldur, sonra miqyasla
num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
# Mətnlər: boşları ən çox rast gəlinənlə doldur, sonra 0/1 sütunlara çevir
cat_pipe = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)

preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

models = {
    "Baseline (həmişə çoxluq)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n{'Model':<26} | {'orta':>5} | {'±':>5}")
for name, model in models.items():
    pipe = make_pipeline(preprocess, model)
    scores = cross_val_score(pipe, X, y, cv=cv)
    print(f"{name:<26} | {scores.mean():.3f} | {scores.std():.3f}")