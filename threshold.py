from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42, stratify=data.target
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# sütun 0 = malignant ehtimalı
p_malignant = model.predict_proba(X_test)[:, 0]

print("hədd | recall | precision | qaçan | yalan həyəcan")
for t in [0.5, 0.4, 0.3, 0.2, 0.1]:
    pred = (p_malignant >= t).astype(int)   # 1 = malignant (bizim üçün)
    true = (y_test == 0).astype(int)
    r = recall_score(true, pred)
    p = precision_score(true, pred)
    fn = ((true == 1) & (pred == 0)).sum()
    fp = ((true == 0) & (pred == 1)).sum()
    print(f"{t:>4} | {r:.3f}  | {p:.3f}     | {fn:>5} | {fp:>5}")