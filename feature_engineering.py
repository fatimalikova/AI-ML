from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = fetch_openml("titanic", version=1, as_frame=True).frame

cat_cols = ["sex", "embarked"]
base_cols = ["pclass", "age", "sibsp", "parch", "fare"]

X = df[base_cols + cat_cols].copy()
X[cat_cols] = X[cat_cols].astype(object)
X["family_size"] = X["sibsp"] + X["parch"] + 1
y = df["survived"].astype(int)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def build_preprocess(num_cols, add_indicator):
    num_pipe = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=add_indicator),
        StandardScaler(),
    )
    cat_pipe = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


variants = {
    "Əvvəlki (baza)": (base_cols, False),
    "+ family_size": (base_cols + ["family_size"], False),
    "+ family_size + boşluq göstəricisi": (base_cols + ["family_size"], True),
}

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

print(f"{'Variant':<36} | {'Model':<20} | {'orta':>5} | {'±':>5}")
for v_name, (cols, indicator) in variants.items():
    for m_name, model in models.items():
        pipe = make_pipeline(build_preprocess(cols, indicator), model)
        scores = cross_val_score(pipe, X, y, cv=cv)
        print(f"{v_name:<36} | {m_name:<20} | {scores.mean():.3f} | {scores.std():.3f}")