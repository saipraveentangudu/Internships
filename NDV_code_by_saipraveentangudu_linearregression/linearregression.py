# Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Load the Dataset
data_frame = pd.read_csv("spam_mail.csv")
print(data_frame.head())

# 2. Label Encoding for Text and Label Columns
encoder = LabelEncoder()
data_frame['encoded_text'] = encoder.fit_transform(data_frame['text'])
data_frame['encoded_label'] = encoder.fit_transform(data_frame['label'])

# 3. Decision Tree Classification
features_tree = data_frame[['encoded_text', 'encoded_label']]
target_tree = data_frame['label_num']

X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(
    features_tree, target_tree, test_size=0.3, random_state=42
)

tree_model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
tree_model.fit(X_train_dt, y_train_dt)

# 4. Predictions and Evaluation - Decision Tree
predictions_dt = tree_model.predict(X_test_dt)

print("\n--- Decision Tree Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test_dt, predictions_dt)}")
print(f"Precision: {precision_score(y_test_dt, predictions_dt)}")
print(f"Recall: {recall_score(y_test_dt, predictions_dt)}")
print(f"F1 Score: {f1_score(y_test_dt, predictions_dt)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test_dt, predictions_dt)}")

# 5. Decision Tree Visualization
plt.figure(figsize=(10, 6))
plot_tree(tree_model, filled=True, feature_names=['encoded_text', 'encoded_label'], class_names=['Not Spam', 'Spam'])
plt.title("Decision Tree Structure")
plt.show()

# 6. Logistic Regression on Encoded Text Only
features_lr = data_frame[['encoded_text']]
target_lr = data_frame['label_num']

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    features_lr, target_lr, test_size=0.3, random_state=42
)

log_reg = LogisticRegression()
log_reg.fit(X_train_lr, y_train_lr)

# 7. Predictions and Evaluation - Logistic Regression
predictions_lr = log_reg.predict(X_test_lr)

print("\n--- Logistic Regression Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test_lr, predictions_lr)}")
print(f"Precision: {precision_score(y_test_lr, predictions_lr)}")
print(f"Recall: {recall_score(y_test_lr, predictions_lr)}")
print(f"F1 Score: {f1_score(y_test_lr, predictions_lr)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test_lr, predictions_lr)}")

# 8. Visualization of Logistic Regression Predictions
plt.scatter(features_lr, target_lr, color='green', label='Actual')
plt.plot(features_lr, log_reg.predict(features_lr), color='orange', label='Predicted')
plt.title("Logistic Regression Fit for Spam Detection")
plt.xlabel("Encoded Text")
plt.ylabel("Label")
plt.legend()
plt.grid(True)
plt.show()
