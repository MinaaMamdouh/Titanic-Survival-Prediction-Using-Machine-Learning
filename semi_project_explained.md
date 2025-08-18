# شرح مفصل لملف Jupyter: `semi_project_preproccessing.ipynb`

## نظرة عامة
- عدد الخلايا الإجمالي: 23
- عدد خلايا الكود: 20
- عدد خلايا الشرح (Markdown): 3

## المكتبات المستخدمة (مكتشفة تلقائيًا)
matplotlib, numpy, pandas, sklearn

## ملفات البيانات المذكورة
- /content/Titanic-Dataset (3).csv

## تفصيل الخلايا (حسب الاكتشاف التلقائي)

### خلية رقم 1
- ماذا تفعل الخلية (تقريبي): ترميز الفئات، تقييس/تطبيع، تقسيم البيانات
- عدد الأسطر: 4

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
```

### خلية رقم 2
- ماذا تفعل الخلية (تقريبي): تحميل البيانات
- عدد الأسطر: 1

```python
df = pd.read_csv("/content/Titanic-Dataset (3).csv")
```

### خلية رقم 3
- ماذا تفعل الخلية (تقريبي): عام/غير محدد
- عدد الأسطر: 2

```python
print("First 5 columns before cleaning data:")
print(df.head())
```

### خلية رقم 4
- ماذا تفعل الخلية (تقريبي): عام/غير محدد
- عدد الأسطر: 1

```python
df = df.drop(["Cabin"], axis=1)
```

### خلية رقم 5
- ماذا تفعل الخلية (تقريبي): التعامل مع القيم المفقودة
- عدد الأسطر: 1

```python
df["Age"].fillna(df["Age"].median(), inplace=True)
```

### خلية رقم 6
- ماذا تفعل الخلية (تقريبي): التعامل مع القيم المفقودة
- عدد الأسطر: 1

```python
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
```

### خلية رقم 7
- ماذا تفعل الخلية (تقريبي): ترميز الفئات
- عدد الأسطر: 3

```python
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])          # Male = 1, Female = 0 (غالبًا)
df["Embarked"] = le.fit_transform(df["Embarked"])
```

### خلية رقم 8
- ماذا تفعل الخلية (تقريبي): عام/غير محدد
- عدد الأسطر: 2

```python
X = df.drop(["Survived", "PassengerId", "Name", "Ticket"], axis=1)
y = df["Survived"]
```

### خلية رقم 9
- ماذا تفعل الخلية (تقريبي): تقسيم البيانات
- عدد الأسطر: 3

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### خلية رقم 10
- ماذا تفعل الخلية (تقريبي): تقييس/تطبيع
- عدد الأسطر: 3

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### خلية رقم 11
- ماذا تفعل الخلية (تقريبي): عام/غير محدد
- عدد الأسطر: 3

```python
print("\nAfter Preprocessing:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
```

### خلية رقم 13
- ماذا تفعل الخلية (تقريبي): نماذج تقليدية، مقاييس تقييم، رسومات واستكشاف بصري
- عدد الأسطر: 5

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
```

### خلية رقم 14
- ماذا تفعل الخلية (تقريبي): نماذج تقليدية
- عدد الأسطر: 5

```python
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}
```

### خلية رقم 15
- ماذا تفعل الخلية (تقريبي): تدريب/توقع/تقييم، رسومات واستكشاف بصري
- عدد الأسطر: 7

```python
plt.figure(figsize=(8,6))

auc_scores = {}

for name, model in models.items():
    # تدريب
    model.fit(X_train, y_train)
```

### خلية رقم 16
- ماذا تفعل الخلية (تقريبي): تدريب/توقع/تقييم، مقاييس تقييم، رسومات واستكشاف بصري
- عدد الأسطر: 15

```python
plt.figure(figsize=(8,6))

auc_scores = {}

for name, model in models.items():
    # تدريب
    model.fit(X_train, y_train)

    # التنبؤ بالاحتمالات (probabilities)
    y_prob = model.predict_proba(X_test)[:,1]

    # حساب ROC & AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    auc_scores[name] = roc_auc
```

### خلية رقم 17
- ماذا تفعل الخلية (تقريبي): مقاييس تقييم، رسومات واستكشاف بصري
- عدد الأسطر: 1

```python
plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
```

### خلية رقم 19
- ماذا تفعل الخلية (تقريبي): رسومات واستكشاف بصري
- عدد الأسطر: 6

```python
plt.plot([0,1],[0,1],'k--')  # خط الـ random guess
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.show()
```

### خلية رقم 20
- ماذا تفعل الخلية (تقريبي): عام/غير محدد
- عدد الأسطر: 3

```python
print("AUC Scores:")
for name, score in auc_scores.items():
    print(f"{name}: {score:.3f}")
```

### خلية رقم 21
- ماذا تفعل الخلية (تقريبي): تدريب/توقع/تقييم، مقاييس تقييم
- عدد الأسطر: 17

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

print("Model Evaluation:\n")
for name, model in models.items():
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"{name}:")
    print(f"  Accuracy  = {acc:.3f}")
    print(f"  Precision = {prec:.3f}")
    print(f"  Recall    = {rec:.3f}")
    print(f"  F1-Score  = {f1:.3f}")
    print("-"*40)
```

### خلية رقم 22
- ماذا تفعل الخلية (تقريبي): تدريب/توقع/تقييم، مقاييس تقييم
- عدد الأسطر: 9

```python
acc_scores = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    acc_scores[name] = accuracy_score(y_test, y_pred)

best_model_name_acc = max(acc_scores, key=acc_scores.get)
best_model_score_acc = acc_scores[best_model_name_acc]

print(f"The best model is: {best_model_name_acc} with Accuracy = {best_model_score_acc:.3f}")
```