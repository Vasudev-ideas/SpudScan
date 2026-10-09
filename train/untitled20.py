# PhishStop with Visualization

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv(r"C:\Users\vasan\Videos\vasu\archive (3)\Phishing_Email.csv")  # change path

df.columns = ["text", "label"]

# -------------------------------
# 2. Preprocessing
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', ' URL ', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    return text

df["text"] = df["text"].apply(clean_text)

# -------------------------------
# 3. Feature Extraction
# -------------------------------
vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# -------------------------------
# 4. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5. Train Model
# -------------------------------
model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)

# -------------------------------
# 6. Predictions
# -------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# 7. Confusion Matrix Visualization
# -------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i][j], ha='center', va='center')

plt.show()

# -------------------------------
# 8. ROC Curve
# -------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="AUC = %0.2f" % roc_auc)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# -------------------------------
# 9. Class Distribution Graph
# -------------------------------
df["label"].value_counts().plot(kind='bar')
plt.title("Class Distribution")
plt.xlabel("Class (0=Legit, 1=Phishing)")
plt.ylabel("Count")
plt.show()