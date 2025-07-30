# Elevate-Labs-AI-ML-internship-PROJECT-SUBMISSION
🔐 Fraud Detection &amp; 🧠 Resume Ranker Detects credit card fraud and ranks resumes using ML. Uses TF-IDF, Logistic Regression, Random Forest, and ROC-AUC.  Tools: Python, Scikit-learn, Pandas

Fraud Detection code:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
df = pd.read_csv("/content/creditcard_2023.csv")  # replace with your actual file name

# Check and drop NaN in the target column
df = df.dropna(subset=['Class'])  # replace 'Class' with your actual target column if different

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Optional: fill missing values in features if any
X = X.fillna(X.mean())

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


Resume Ranker code:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
df = pd.read_csv("/content/Resume.csv")  # Replace with your actual file name

# Check the name of the target column (replace 'Class' if needed)
target_column = 'Class'

# Drop rows with NaN in the target column
df = df.dropna(subset=[target_column])

# Separate features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Fill any missing values in features
X = X.fillna(X.mean())

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
