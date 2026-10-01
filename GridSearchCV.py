from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

param_grid = {
    "max_depth": [1, 2, 3, 4, 5, 7, 10, None],
    "min_samples_leaf": [1, 5, 10],
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
)
grid.fit(X_train, y_train)

print("Ən yaxşı parametrlər:", grid.best_params_)
print("CV dəqiqliyi:", round(grid.best_score_, 3))
print("Test dəqiqliyi:", round(grid.score(X_test, y_test), 3))