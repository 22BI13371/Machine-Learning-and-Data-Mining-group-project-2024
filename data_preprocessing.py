"""Made with referenced to  "A Statistical Analysis & ML workflow of Titanic" By MASUM RUMI 
https://www.kaggle.com/code/masumrumi/a-statistical-analysis-ml-workflow-of-titanic/notebook#Part-8:-Submit-test-predictions"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import SGDClassifier


train_file = "dataset/train.csv"
test_file = "dataset/test.csv"

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
passengerid = test.PassengerId


def missing_percentage(df):
    """This function takes a DataFrame(df) as input and returns two columns, total missing values and total missing values percentage"""
    total = df.isnull().sum().sort_values(ascending=False)
    percent = round(df.isnull().sum().sort_values(
        ascending=False)/len(df)*100, 2)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


def percent_value_counts(df, feature):
    """This function takes in a dataframe and a column and finds the percentage of the value_counts"""
    percent = pd.DataFrame(
        round(df.loc[:, feature].value_counts(dropna=False, normalize=True)*100, 2))
    # creating a df with th
    total = pd.DataFrame(df.loc[:, feature].value_counts(dropna=False))
    # concating percent and total dataframe

    total.columns = ["Total"]
    percent.columns = ['Percent']
    return pd.concat([total, percent], axis=1)


# Replacing the null values in the Embarked column with the mode.
train.fillna({"Embarked": "C"}, inplace=True)

# Concat train and test into a variable "all_data"
survivers = train.Survived

train.drop(["Survived"], axis=1, inplace=True)

all_data = pd.concat([train, test], ignore_index=False)

# Assign all the null values to N
all_data.fillna({'Cabin': 'N'}, inplace=True)

all_data.Cabin = [i[0] for i in all_data.Cabin]


def cabin_estimator(i):
    """Grouping cabin feature by the first letter"""
    a = 0
    if i < 16:
        a = "G"
    elif i >= 16 and i < 27:
        a = "F"
    elif i >= 27 and i < 38:
        a = "T"
    elif i >= 38 and i < 47:
        a = "A"
    elif i >= 47 and i < 53:
        a = "E"
    elif i >= 53 and i < 54:
        a = "D"
    elif i >= 54 and i < 116:
        a = 'C'
    else:
        a = "B"
    return a


with_N = all_data[all_data.Cabin == "N"]

without_N = all_data[all_data.Cabin != "N"]

# applying cabin estimator function.
with_N.loc[:, 'Cabin'] = with_N.Fare.apply(lambda x: cabin_estimator(x))

# getting back train.
all_data = pd.concat([with_N, without_N], axis=0)

# PassengerId helps us separate train and test.
all_data.sort_values(by='PassengerId', inplace=True)

# Separating train and test from all_data.
train = all_data[:891].copy()

test = all_data[891:].copy()

# adding saved target variable with train.
# print(train.head(5))
train.loc[:, 'Survived'] = survivers

missing_value = test[(test.Pclass == 3) &
                     (test.Embarked == "S") &
                     (test.Sex == "male")].Fare.mean()

train.fillna({"Fare": missing_value}, inplace=True)


# dropping the three outliers where Fare is over $500
train = train[train.Fare < 500]


# Placing 0 for female and
# 1 for male in the "Sex" column.
train.loc[:, 'Sex'] = train.Sex.apply(lambda x: 0 if x == "female" else 1)
test.loc[:, 'Sex'] = test.Sex.apply(lambda x: 0 if x == "female" else 1)


# Creating a new colomn with a
train['name_length'] = [len(i) for i in train.Name]
# test.loc[:, 'name_length'] = [len(i) for i in test.Name]

# Create a copy of the test DataFrame
test_copy = test.copy()

# Calculate the length of each name and assign it to the 'name_length' column
test_copy.loc[:, 'name_length'] = test_copy['Name'].apply(len)

# If you need to update the original test DataFrame with the changes,
# you can do so by assigning the modified copy back to it
test = test_copy


def name_length_group(size):
    a = ''
    if (size <= 20):
        a = 'short'
    elif (size <= 35):
        a = 'medium'
    elif (size <= 45):
        a = 'good'
    else:
        a = 'long'
    return a


# Create a copy of the test DataFrame
test_copy = test.copy()

# Perform the mapping and assignment on the copied DataFrame
test_copy.loc[:, 'nLength_group'] = test_copy['name_length'].map(
    name_length_group)

# Now, if you need to update the original test DataFrame with the changes,
# you can do so by assigning the modified copy back to it
test = test_copy

train['nLength_group'] = train['name_length'].map(name_length_group)
# test.loc[:, 'nLength_group'] = test['name_length'].map(name_length_group)

# Here "map" is python's built-in function.
# "map" function basically takes a function and
# returns an iterable list/tuple or in this case series.
# However,"map" can also be used like map(function) e.g. map(name_length_group)
# or map(function, iterable{list, tuple}) e.g. map(name_length_group, train[feature]]).
# However, here we don't need to use parameter("size") for name_length_group because when we
# used the map function like ".map" with a series before dot, we are basically hinting that series
# and the iterable. This is similar to .append approach in python. list.append(a) meaning applying append on list.


# cuts the column by given bins based on the range of name_length
# group_names = ['short', 'medium', 'good', 'long']
# train['name_len_group'] = pd.cut(train['name_length'], bins = 4, labels=group_names)


# get the title from the name
train["title"] = [i.split('.')[0] for i in train.Name]
train["title"] = [i.split(',')[1] for i in train.title]
# Whenever we split like that, there is a good change that we will end up with while space around our string values. Let's check that.

train.title = train.title.apply(lambda x: x.strip())


# We can also combile all three lines above for test set here
test['title'] = [i.split('.')[0].split(',')[1].strip() for i in test.Name]


def name_converted(feature):
    """
    This function helps modifying the title column
    """

    result = ''
    if feature in ['the Countess', 'Capt', 'Lady', 'Sir', 'Jonkheer', 'Don', 'Major', 'Col', 'Rev', 'Dona', 'Dr']:
        result = 'rare'
    elif feature in ['Ms', 'Mlle']:
        result = 'Miss'
    elif feature == 'Mme':
        result = 'Mrs'
    else:
        result = feature
    return result


test.title = test.title.map(name_converted)
train.title = train.title.map(name_converted)


train['family_size'] = train.SibSp + train.Parch+1
test['family_size'] = test.SibSp + test.Parch+1

# bin the family size.


def family_group(size):
    """
    This funciton groups(loner, small, large) family based on family size
    """

    a = ''
    if (size <= 1):
        a = 'loner'
    elif (size <= 4):
        a = 'small'
    else:
        a = 'large'
    return a


# apply the family_group function in family_size
train['family_group'] = train['family_size'].map(family_group)
test['family_group'] = test['family_size'].map(family_group)


train['is_alone'] = [1 if i < 2 else 0 for i in train.family_size]
test['is_alone'] = [1 if i < 2 else 0 for i in test.family_size]


''' Author's Note: "I have yet to figureout how to best manage ticket feature. So, any suggestion would be truly appreciated. For now, I will get rid off the ticket feature."'''
train.drop(['Ticket'], axis=1, inplace=True)

test.drop(['Ticket'], axis=1, inplace=True)


# Calculating fare based on family size.
train['calculated_fare'] = train.Fare/train.family_size
test['calculated_fare'] = test.Fare/test.family_size


def fare_group(fare):
    """
    This function creates a fare group based on the fare provided
    """

    a = ''
    if fare <= 4:
        a = 'Very_low'
    elif fare <= 10:
        a = 'low'
    elif fare <= 20:
        a = 'mid'
    elif fare <= 45:
        a = 'high'
    else:
        a = "very_high"
    return a


train['fare_group'] = train['calculated_fare'].map(fare_group)
test['fare_group'] = test['calculated_fare'].map(fare_group)

# train['fare_group'] = pd.cut(train['calculated_fare'], bins = 4, labels=groups)


# Need further considering
train.drop(['PassengerId'], axis=1, inplace=True)
test.drop(['PassengerId'], axis=1, inplace=True)


train = pd.get_dummies(train, columns=['title', "Pclass", 'Cabin', 'Embarked',
                       'nLength_group', 'family_group', 'fare_group'], drop_first=False)
test = pd.get_dummies(test, columns=['title', "Pclass", 'Cabin', 'Embarked',
                      'nLength_group', 'family_group', 'fare_group'], drop_first=False)
train.drop(['family_size', 'Name', 'Fare',
           'name_length'], axis=1, inplace=True)
test.drop(['Name', 'family_size', "Fare", 'name_length'], axis=1, inplace=True)


# rearranging the columns so that I can easily use the dataframe to predict the missing age values.
train = pd.concat([train[["Survived", "Age", "Sex", "SibSp",
                  "Parch"]], train.loc[:, "is_alone":]], axis=1)
test = pd.concat([test[["Age", "Sex"]], test.loc[:, "SibSp":]], axis=1)


# create bins for age
def age_group_fun(age):
    """
    This function creates a bin for age
    """
    a = ''
    if age <= 1:
        a = 'infant'
    elif age <= 4:
        a = 'toddler'
    elif age <= 13:
        a = 'child'
    elif age <= 18:
        a = 'teenager'
    elif age <= 35:
        a = 'Young_Adult'
    elif age <= 45:
        a = 'adult'
    elif age <= 55:
        a = 'middle_aged'
    elif age <= 65:
        a = 'senior_citizen'
    else:
        a = 'old'
    return a


# Applying "age_group_fun" function to the "Age" column.
train['age_group'] = train['Age'].map(age_group_fun)
test['age_group'] = test['Age'].map(age_group_fun)

# Creating dummies for "age_group" feature.
train = pd.get_dummies(train, columns=['age_group'], drop_first=True)
test = pd.get_dummies(test, columns=['age_group'], drop_first=True)


# writing a function that takes a dataframe with missing values and outputs it by filling the missing values.
def completing_age(df):
    # gettting all the features except survived
    age_df = df.loc[:, "Age":]

    temp_train = age_df.loc[age_df.Age.notnull()]  # df with age values
    temp_test = age_df.loc[age_df.Age.isnull()]  # df without age values

    y = temp_train.Age.values  # setting target variables(age) in y
    x = temp_train.loc[:, "Sex":].values

    rfr = RandomForestRegressor(n_estimators=1500, n_jobs=-1)
    rfr.fit(x, y)

    predicted_age = rfr.predict(temp_test.loc[:, "Sex":])

    df.loc[df.Age.isnull(), "Age"] = predicted_age

    return df


# Implementing the completing_age function in both train and test dataset.
completing_age(train)
completing_age(test)


# separating our independent and dependent variable
X = train.drop(['Survived'], axis=1)
y = train["Survived"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.33, random_state=0)

headers = X_train.columns


st_scale = StandardScaler()

# transforming "train_x"
X_train = st_scale.fit_transform(X_train)
# transforming "test_x"
X_test = st_scale.transform(X_test)

# transforming "The testset"
test = st_scale.transform(test)


# call on the model object
# logreg = LogisticRegression(solver='lbfgs', penalty=None, random_state=42)
logreg = SGDClassifier(loss='log_loss', penalty=None, alpha=0.0, max_iter=50000, tol=None,
                       random_state=42, learning_rate='invscaling', eta0=0.001, shuffle=True, early_stopping=False)

# fit the model with "train_x" and "train_y"
logreg.fit(X_train, y_train)
# logreg.fit(X, y)

# Once the model is trained we want to find out how well the model is performing, so we test the model.
# we use "X_test" portion of the data(this data was not used to fit the model) to predict model outcome.
y_pred = logreg.predict(X_test)


# Once predicted we save that outcome in "y_pred" variable.
# Then we compare the predicted value( "y_pred") and actual value("test_y") to see how well our model is performing.


accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

# print sumarized classification report
print(classification_report(y_test, y_pred))
print(logreg.score(X_test, y_test))
# print("Accuracy: ", accuracy, "\nRecall: ", recall, "\nPrecision: ", precision)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

class_names = np.array(['not_survived', 'survived'])

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


""" ROC for Titanic survivors """

# plt.style.use('seaborn-pastel')
y_score = logreg.decision_function(X_test)

# FPR, TPR, _ = roc_curve(y_test, y_score)
# ROC_AUC = auc(FPR, TPR)
# print(ROC_AUC)

# plt.figure(figsize=[11, 9])
# plt.plot(FPR, TPR, label='ROC curve(area = %0.2f)' % ROC_AUC, linewidth=4)
# plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate', fontsize=18)
# plt.ylabel('True Positive Rate', fontsize=18)
# plt.title('ROC for Titanic survivors', fontsize=18)
# plt.show()


""" Precision Recall Curve for Titanic survivors """

# y_score = logreg.decision_function(X_test)

# precision, recall, _ = precision_recall_curve(y_test, y_score)
# PR_AUC = auc(recall, precision)

# plt.figure(figsize=[11, 9])
# plt.plot(recall, precision, label='PR curve (area = %0.2f)' %
#          PR_AUC, linewidth=4)
# plt.xlabel('Recall', fontsize=18)
# plt.ylabel('Precision', fontsize=18)
# plt.title('Precision Recall Curve for Titanic survivors', fontsize=18)
# plt.legend(loc="lower right")
# plt.show()


""" Using StratifiedShuffleSplit """
sc = st_scale

# run model 10x with 60/30 split intentionally leaving out 10%
cv = StratifiedShuffleSplit(n_splits=10, test_size=.25, random_state=0)
# Using standard scale for the whole dataset.

# saving the feature names for decision tree display
column_names = X.columns

X = sc.fit_transform(X)
accuracies = cross_val_score(
    LogisticRegression(solver='lbfgs', penalty=None),  X, y, cv=cv)
# print("Cross-Validation accuracy scores:{}".format(accuracies))
# print("Mean Cross-Validation accuracy score: {}".format(round(accuracies.mean(), 5)))


""" Using grid search for logistic regression """
# # C_vals is the alpla value of lasso and ridge regression(as alpha increases the model complexity decreases,)
# # remember effective alpha scores are 0<alpha<infinity
# C_vals = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4,
#           5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 16.5, 17, 17.5, 18]
# # Choosing penalties(Lasso(l1) or Ridge(l2))
# penalties = ['l1', 'l2']
# # Choose a cross validation strategy.
# cv = StratifiedShuffleSplit(n_splits=10, test_size=.25)

# # setting param for param_grid in GridSearchCV.
# param = {'penalty': penalties, 'C': C_vals}

# logreg = LogisticRegression(solver='liblinear')
# # Calling on GridSearchCV object.
# grid = GridSearchCV(estimator=LogisticRegression(),
#                     param_grid=param,
#                     scoring='accuracy',
#                     n_jobs=-1,
#                     cv=cv
#                     )
# # Fitting the model
# grid.fit(X, y)


test = test.astype(int)
test_prediction = logreg.predict(test)
submission = pd.DataFrame({
    "PassengerId": passengerid,
    "Survived": test_prediction
})

submission.PassengerId = submission.PassengerId.astype(int)
submission.Survived = submission.Survived.astype(int)

submission.to_csv("titanic1_submission.csv", index=False)
