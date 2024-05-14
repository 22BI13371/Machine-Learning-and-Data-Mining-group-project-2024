from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
import random
import numpy as np
import pandas as pd


train_data_file = "dataset/train.csv"
# valid_data_file = "dataset/valid.csv"
test_data_file = "dataset/test.csv"

data = pd.read_csv(train_data_file)
# validate_data = pd.read_csv(valid_data_file)
test_data = pd.read_csv(test_data_file)


def preprocess(df):
    df = df.copy()

    def normalize_name(x):
        return " ".join([v.strip(",()[].\"'") for v in x.split(" ")])

    def ticket_number(x):
        return x.split(" ")[-1]

    def ticket_item(x):
        items = x.split(" ")
        if len(items) == 1:
            return "NONE"
        return "_".join(items[0:-1])

    df["Name"] = df["Name"].apply(normalize_name)
    df["Ticket_number"] = df["Ticket"].apply(ticket_number)
    df["Ticket_item"] = df["Ticket"].apply(ticket_item)
    return df


preprocessed_train_data = preprocess(data)
preprocessed_test_data = preprocess(test_data)


# print(preprocessed_train_data.head(5))

preprocessed_train_data = preprocessed_train_data[[
    'Survived', 'Age', 'Sex', 'Pclass']]
preprocessed_train_data = pd.get_dummies(
    preprocessed_train_data, columns=['Sex', 'Pclass'])
preprocessed_train_data.dropna(inplace=True)
# preprocessed_train_data.head()
# print(preprocessed_train_data)

preprocessed_test_data = preprocessed_test_data[['Age', 'Sex', 'Pclass']]
preprocessed_test_data = pd.get_dummies(
    preprocessed_test_data, columns=['Sex', 'Pclass'])
preprocessed_test_data.dropna(inplace=True)
# preprocessed_train_data.head()
# print(preprocessed_test_data)


X = preprocessed_train_data.drop('Survived', axis=1)
Y = preprocessed_train_data['Survived']


# Spliting dataset to train and valid sets
X_train, x_valid, Y_train, y_valid = train_test_split(
    X, Y, test_size=0.25, random_state=16)

# print(X_train.head(10))
# print(Y_train.head(10))
# print(x_valid.head(5))
# print(Y_train.head(5))

# init model
model = LogisticRegression(random_state=16, penalty=None, max_iter=50)

# fitting model
model.fit(X_train, Y_train)

y_pred_valid = model.predict(x_valid)
y_pred_test = model.predict(preprocessed_test_data)
# print(y_pred_test)

# print(y_pred)
score = model.score(x_valid, y_valid)
print(cross_val_score(model, X, Y, cv=5).mean())

# disp = ConfusionMatrixDisplay.from_predictions(y_valid, y_pred)
# plt.show()

# print(classification_report(y_valid, y_pred_valid))
# print(y_pred)

# create final result dataframe
# test = np.array(['PassengerId', 'Survived'],
#                 [[test_data['PassengerID'], y_pred]])
# print(preprocessed_test_data)
# csv = pd.DataFrame(data=test[1:, 1:],
#                    columns=test[0, 1:])

# result = {
#     'PassengerId': test_data["PassengerId"],
#     'Survived': y_pred_test
# }
print(y_pred_test.shape)
print(test_data['PassengerId'].shape)

# result_df = pd.DataFrame(result)
# print(result_df)
