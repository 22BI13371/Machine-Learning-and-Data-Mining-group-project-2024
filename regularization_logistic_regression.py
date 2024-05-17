from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
train_data = pd.read_csv("./dataset/train.csv")
test_data = pd.read_csv("./dataset/test.csv")

# # Display the first few rows of the train data
# print(train_data.head())

# # Distribution of Survival
# plt.figure(figsize=(8, 4))
# sns.countplot(x='Survived', data=train_data)
# plt.title('Distribution of Survival')
# plt.show()

# # Survival by Gender
# plt.figure(figsize=(8, 4))
# sns.countplot(x='Survived', hue='Sex', data=train_data)
# plt.title('Survival by Gender')
# plt.show()

# # Survival by Class
# plt.figure(figsize=(8, 4))
# sns.countplot(x='Survived', hue='Pclass', data=train_data)
# plt.title('Survival by Class')
# plt.show()

# Prepare the data for modeling
y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# Initialize logistic regression model with regularization. The C parameter controls the strength of regularization. 
model = LogisticRegression(C=1.0, penalty='l2', random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

# Save the output
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
