from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
train_data = pd.read_csv("./dataset/train.csv")
test_data = pd.read_csv("./dataset/test.csv")

# Prepare the data for modeling
y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

# Initialize logistic regression model with regularization
model = LogisticRegression(C=1.0, penalty='l2', random_state=1)
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nClassification Report:")
print(classification_report(y_val, y_pred))

accuracy = accuracy_score(y_val, y_pred)
print("\nAccuracy:", accuracy)

# Plot confusion matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Make predictions on the test set
predictions = model.predict(X_test)

# Save the output
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

# # # Distribution of Survival
# # plt.figure(figsize=(8, 4))
# # sns.countplot(x='Survived', data=train_data)
# # plt.title('Distribution of Survival')
# # plt.show()

# # # Survival by Gender
# # plt.figure(figsize=(8, 4))
# # sns.countplot(x='Survived', hue='Sex', data=train_data)
# # plt.title('Survival by Gender')
# # plt.show()

# # # Survival by Class
# # plt.figure(figsize=(8, 4))
# # sns.countplot(x='Survived', hue='Pclass', data=train_data)
# # plt.title('Survival by Class')
# # plt.show()