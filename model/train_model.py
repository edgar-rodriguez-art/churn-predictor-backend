import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Cargar dataset de Kaggle
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Preprocesar (versión simple)
df = df[["tenure", "MonthlyCharges", "Contract", "Churn"]]
df["Contract"] = df["Contract"].astype("category").cat.codes
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df[["tenure", "MonthlyCharges", "Contract"]]
y = df["Churn"]

# 3. Entrenar modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced') 
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)


# 4. Guardar modelo
joblib.dump(clf, "churn_model.pkl")

'''
test_inputs = [
    {"tenure": 0, "MonthlyCharges": 25, "Contract": 0},
    {"tenure": 5, "MonthlyCharges": 75, "Contract": 0},
    {"tenure": 20, "MonthlyCharges": 90, "Contract": 0},
    {"tenure": 50, "MonthlyCharges": 35, "Contract": 1},
     {"tenure": 0, "MonthlyCharges": 70, "Contract": 0},   # Cliente nuevo, contrato mes a mes
    {"tenure": 5, "MonthlyCharges": 70, "Contract": 1},   # Cliente reciente, contrato de un año
    {"tenure": 12, "MonthlyCharges": 70, "Contract": 2},  # Cliente de 1 año, contrato de 2 años
    {"tenure": 24, "MonthlyCharges": 70, "Contract": 0},  # Cliente leal, pero contrato mes a mes
    {"tenure": 36, "MonthlyCharges": 70, "Contract": 1},  # Cliente más antiguo, contrato 1 año
    {"tenure": 60, "MonthlyCharges": 70, "Contract": 2},  # Cliente muy leal, contrato 2 años
]

for test in test_inputs:
    print(test, "=>", clf.predict(pd.DataFrame([test]))[0])
    print("proba ", clf.predict_proba(pd.DataFrame([test])))
'''