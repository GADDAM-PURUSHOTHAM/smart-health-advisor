import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import sklearn

print("Using sklearn version:", sklearn.__version__)

# ===================== Load Dataset =====================
data = pd.read_csv("Training.csv")

# Features & target
X = data.drop(columns=["prognosis"])
y = data["prognosis"]

print("Number of features used for training:", X.shape[1])

# Save feature names (VERY IMPORTANT)
with open("feature_names.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

# ===================== Train-Test Split =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===================== Train SVC =====================
svc = SVC(kernel="rbf", probability=True, random_state=42)
svc.fit(X_train, y_train)
print("SVC Accuracy:", accuracy_score(y_test, svc.predict(X_test)))

with open("svc.pkl", "wb") as f:
    pickle.dump(svc, f)

# ===================== Train Decision Tree =====================
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
print("DT Accuracy:", accuracy_score(y_test, dt.predict(X_test)))

with open("dt.pkl", "wb") as f:
    pickle.dump(dt, f)

print("✅ Models trained & feature names saved")

