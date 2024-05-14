import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


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

data = data[['Survived', 'Age', 'Sex', 'Pclass']]
data = pd.get_dummies(data, columns=['Sex', 'Pclass'])
data.dropna(inplace=True)
data.head()


X = data.drop('Survived', axis=1)
Y = data['Survived']


# Spliting dataset to train and valid sets
X_train, x_test, Y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=16)

# init model
model = LogisticRegression(random_state=16)

# fitting model
model.fit(X_train, Y_train)

y_pred = model.predict(x_test)

# print(y_pred)
score = model.score(x_test, y_test)
print(cross_val_score(model, X, Y, cv=5).mean())


def initialize_theta(dim):
    theta = np.random.rand(dim)
    return theta
