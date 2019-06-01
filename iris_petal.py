from sklearn import datasets
from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load dataset

iris = datasets.load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(data={'SepalLength': X[:, 0], 'SepalWidth': X[:, 1], 'PetalLength': X[:, 2], 'PetalWidth': X[:, 3],
                        'y': y})
df = df.sample(frac=1.0)
train = df.sample(frac=0.5)
val = df[~df.index.isin(train.index)]
test = val.sample(frac=0.7)
val = val[~val.index.isin(test.index)]
test['y_predicted'] = None

# split dataset into training, validation and testing sets

X_train = np.array([train['PetalLength'], train['PetalWidth']]).transpose()
y_train = np.array(train['y'].values)
X_val = np.array([val['PetalLength'], val['PetalWidth']]).transpose()
X_test = np.array([test['PetalLength'], test['PetalWidth']]).transpose()

# train svm model

clf = svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)
clf.fit(X_train, y_train)

# calculate validation and testing accuracies

val['y_predicted'] = clf.predict(X_val)
val_errors = sum(val['y'] != val['y_predicted'])
val_accuracy = (1 - (val_errors / len(val['y'])))
test['y_predicted'] = clf.predict(X_test)
test_errors = sum(test['y'] != test['y_predicted'])
test_accuracy = (1 - (test_errors / len(test['y'])))
print("val_accuracy", val_accuracy)
print("test_accuracy", test_accuracy)

# plot decision boundary

padding = 0.15
res = 0.01
x_min, x_max = min(df['PetalLength'].tolist()), max(df['PetalLength'].tolist())
y_min, y_max = min(df['PetalWidth'].tolist()), max(df['PetalWidth'].tolist())
x_range = x_max - x_min
y_range = y_max - y_min
x_min -= x_range * padding
y_min -= y_range * padding
x_max += x_range * padding
y_max += y_range * padding
xx, yy = np.meshgrid(np.arange(x_min, x_max, res), np.arange(y_min, y_max, res))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.style.use('ggplot')
plt.figure(figsize=(8, 6))
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=1.0)
plt.scatter(df['PetalLength'], df['PetalWidth'], c=df['y'], cmap=plt.cm.Spectral, edgecolors='black')
plt.xlabel('Petal Length', size=14)
plt.ylabel('Petal Width', size=14)
val_accuracy = str(round(val_accuracy, 3))
test_accuracy = str(round(test_accuracy, 3))
title = "iris-petal-linear: val_acc = " + val_accuracy + " , test_acc = " + test_accuracy
plt.title(title)
plt.savefig("iris-petal-linear.png")
