import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = fetch_openml("titanic", version=1, as_frame=True).frame

title = df["name"].str.extract(r",\s*([^\.]+)\.")[0].str.strip()
title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
title = title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")

num_cols = ["pclass", "age", "sibsp", "parch", "fare"]
base_cat = ["sex", "embarked"]

X = df[num_cols + base_cat].copy()
X[base_cat] = X[base_cat].astype(object)
X["title"] = title
y = df["survived"].astype(int)

# Hər iki variant üçün eyni bölgülər
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)


def make_model(cat_cols):
    num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    cat_pipe = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
    return make_pipeline(pre, LogisticRegression(max_iter=1000))


s_base = cross_val_score(make_model(base_cat), X, y, cv=cv)
s_title = cross_val_score(make_model(base_cat + ["title"]), X, y, cv=cv)

diff = s_title - s_base
print(f"Baza:    {s_base.mean():.3f} (±{s_base.std():.3f})")
print(f"+ title: {s_title.mean():.3f} (±{s_title.std():.3f})")
print(f"\nHər bölgü üzrə fərq (title - baza), cəmi {len(diff)} bölgü:")
print(f"orta fərq: {diff.mean():+.3f}")
print(f"title daha yaxşı olan bölgü sayı: {(diff > 0).sum()} / {len(diff)}")
print(f"title daha pis olan bölgü sayı:   {(diff < 0).sum()} / {len(diff)}")