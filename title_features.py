from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = fetch_openml("titanic", version=1, as_frame=True).frame

# Adın içindən titulu çıxar: "Braund, Mr. Owen" -> "Mr"
title = df["name"].str.extract(r",\s*([^\.]+)\.")[0].str.strip()
title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
title = title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")

print("Titul paylanması:")
print(title.value_counts())

base_num = ["pclass", "age", "sibsp", "parch", "fare"]
base_cat = ["sex", "embarked"]

X = df[base_num + base_cat].copy()
X[base_cat] = X[base_cat].astype(object)
X["title"] = title
X["is_alone"] = ((X["sibsp"] + X["parch"]) == 0).astype(int)
y = df["survived"].astype(int)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def build_preprocess(num_cols, cat_cols):
    num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    cat_pipe = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


variants = {
    "Əvvəlki (baza)": (base_num, base_cat),
    "+ title": (base_num, base_cat + ["title"]),
    "+ title + is_alone": (base_num + ["is_alone"], base_cat + ["title"]),
}

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

print(f"\n{'Variant':<22} | {'Model':<20} | {'orta':>5} | {'±':>5}")
for v_name, (num_cols, cat_cols) in variants.items():
    for m_name, model in models.items():
        pipe = make_pipeline(build_preprocess(num_cols, cat_cols), model)
        scores = cross_val_score(pipe, X, y, cv=cv)
        print(f"{v_name:<22} | {m_name:<20} | {scores.mean():.3f} | {scores.std():.3f}")