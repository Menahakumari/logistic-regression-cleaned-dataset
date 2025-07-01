import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
df = pd.read_csv("/content/new.csv")

# Separate numeric and categorical columns
df_numeric = df.select_dtypes(include=['float64', 'int64'])
df_categorical = df.select_dtypes(include=['object', 'category'])

df_numeric = df_numeric.fillna(df_numeric.mean(numeric_only=True))

# Fill categorical columns with mode
df_categorical = df_categorical.fillna(df_categorical.mode().iloc[0])

df_clean = pd.concat([df_numeric, df_categorical], axis=1)
df_encoded = pd.get_dummies(df_clean, drop_first=True)

X = df_encoded.iloc[:, :-1]
y = df_encoded.iloc[:, -1]

#Final NaN fix: Fill any leftover missing values with 0
X = X.fillna(0)

# Train/test split and standardize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate with confusion matrix, precision, recall, ROC-AUC
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

#Threshold tuning
threshold = 0.4
y_pred_custom = (y_prob >= threshold).astype(int)
print("\nConfusion Matrix @ threshold 0.4:\n", confusion_matrix(y_test, y_pred_custom))
print("Precision @ threshold 0.4:", precision_score(y_test, y_pred_custom))
print("Recall @ threshold 0.4:", recall_score(y_test, y_pred_custom))

# Sigmoid explanation
print("\nSigmoid Function: S(z) = 1 / (1 + e^-z) — used in logistic regression to convert output to probability.")
