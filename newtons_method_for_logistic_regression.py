# Made by Luong Ngoc Phuc

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

# Load the datasets
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

passengerid = test_df.PassengerId

# Handle missing values
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

# Drop columns that won't be used in the model
train_df.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'], inplace=True)
test_df.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'], inplace=True)

# Separate features and target variable from training data
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

# One-hot encode categorical variables and scale numerical variables
categorical_features = ['Sex', 'Embarked']
numerical_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Create a pipeline that combines the preprocessor with logistic regression
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='newton-cg', random_state=42))
])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = pipeline.predict(X_val)

# Evaluate the model
# accuracy = (y_pred == y_val).mean()
print(classification_report(y_val, y_pred))


"""------------------------------"""
# Load the submission file and the gender_submission file for comparison
# predicted_df = pd.read_csv('/mnt/data/submission.csv')
# actual_df = pd.read_csv('/mnt/data/gender_submission.csv')

# # Merge the two dataframes on PassengerId
# comparison_df = predicted_df.merge(
#     actual_df, on='PassengerId', suffixes=('_predicted', '_actual'))

# # Calculate the accuracy of the predictions
# accuracy = (comparison_df['Survived_predicted'] ==
#             comparison_df['Survived_actual']).mean()
# accuracy


"""------------------------------"""
# Fit the pipeline to the entire training data
# pipeline.fit(X, y)

# Make predictions on the test data
test_predictions = pipeline.predict(test_df)

# Prepare the submission file
submission = pd.DataFrame({
    'PassengerId': passengerid,
    'Survived': test_predictions
})

# Save the submission file
submission.to_csv('submission.csv', index=False)
