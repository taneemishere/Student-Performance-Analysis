import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

data = pd.read_csv("student-mat.csv", sep=';')
# print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# print(data.head())

predict = "G3"

X = np.array(data.drop([predict], 1))
# print(X)
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

'''best = 0
for _ in range(30):

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    model = LinearRegression()

    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)

    print(acc)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", 'wb') as f:
            pickle.dump(model, f)'''

pickle_in = open("studentmodel.pickle", 'rb')
model = pickle.load(pickle_in)

print("Co-efficients: ", model.coef_)
print("Intercept: ", model.intercept_)

predictions = model.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = "G2"
style.use("ggplot")
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grades")
plt.show()
