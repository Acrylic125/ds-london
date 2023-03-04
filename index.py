import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('data/train.csv', header=None)
train_label_data = pd.read_csv('data/trainLabels.csv', header=None)

X = train_data
y = train_label_data

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, 
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)

scores = {}

for n_estimators in range(1, 1000, 100):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=0)

    model.fit(X_train_full, y_train.values.ravel())
    score = model.score(X_valid_full, y_valid)    

    scores[n_estimators] = score

print(scores)

# ctype = X.dtypes
# print(ctype)
# print(X.shape)
# print(y.shape)
